#!/usr/bin/env python3
"""Exercise a local server/client round trip using disposable mutual-TLS certificates."""

import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run(command, expected=0, timeout=15):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if (result.returncode == 0) != (expected == 0):
        raise RuntimeError(
            f"command returned {result.returncode}, expected "
            f"{'success' if expected == 0 else 'failure'}:\n"
            f"{' '.join(map(str, command))}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def issue_certificate(openssl, directory, name, san, usage, ca_key, ca_cert):
    key = directory / f"{name}.key"
    request = directory / f"{name}.csr"
    certificate = directory / f"{name}.crt"
    extensions = directory / f"{name}.ext"
    extensions.write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        f"extendedKeyUsage={usage}\n"
        f"subjectAltName={san}\n",
        encoding="ascii",
    )
    run(
        [
            openssl, "req", "-new", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(request), "-subj", f"/CN={name}",
        ]
    )
    run(
        [
            openssl, "x509", "-req", "-in", str(request), "-CA", str(ca_cert),
            "-CAkey", str(ca_key), "-CAcreateserial", "-days", "1",
            "-extfile", str(extensions), "-out", str(certificate),
        ]
    )
    return certificate, key


def wait_for_listener(process, host, port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"server exited early:\nstdout: {stdout}\nstderr: {stderr}")
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("server did not start listening within 10 seconds")


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: test_mtls_round_trip.py SERVER CLIENT OPENSSL [--gpu-enabled]")
    server_executable, client_executable, openssl = sys.argv[1:4]
    gpu_enabled = len(sys.argv) == 5 and sys.argv[4] == "--gpu-enabled"
    if len(sys.argv) == 5 and not gpu_enabled:
        raise SystemExit("unknown test option")

    with tempfile.TemporaryDirectory(prefix="tuna-mtls-test-") as temporary:
        directory = Path(temporary)
        ca_key = directory / "ca.key"
        ca_cert = directory / "ca.crt"
        run(
            [
                openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(ca_key), "-out", str(ca_cert), "-days", "1",
                "-subj", "/CN=Tuna-Test-CA",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            ]
        )
        rogue_ca_key = directory / "rogue-ca.key"
        rogue_ca_cert = directory / "rogue-ca.crt"
        run(
            [
                openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(rogue_ca_key), "-out", str(rogue_ca_cert), "-days", "1",
                "-subj", "/CN=Tuna-Untrusted-Test-CA",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            ]
        )
        server_cert, server_key = issue_certificate(
            openssl, directory, "server", "DNS:localhost", "serverAuth", ca_key, ca_cert
        )
        client_cert, client_key = issue_certificate(
            openssl, directory, "client", "DNS:sample-client", "clientAuth", ca_key, ca_cert
        )
        bad_client_cert, bad_client_key = issue_certificate(
            openssl, directory, "unapproved-client", "DNS:unapproved-client",
            "clientAuth", ca_key, ca_cert,
        )
        untrusted_client_cert, untrusted_client_key = issue_certificate(
            openssl, directory, "untrusted-client", "DNS:sample-client",
            "clientAuth", rogue_ca_key, rogue_ca_cert,
        )

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]

        address = f"127.0.0.1:{port}"
        server = subprocess.Popen(
            [
                server_executable, "--listen", address, "--cert", str(server_cert),
                "--key", str(server_key), "--client-ca", str(ca_cert),
                "--allowed-client-san", "sample-client",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_for_listener(server, "127.0.0.1", port)

            valid = run(
                [
                    client_executable, "--server", f"localhost:{port}",
                    "--ca", str(ca_cert), "--cert", str(client_cert),
                    "--key", str(client_key), "3", "4",
                ]
            )
            if "sum_of_squares=25" not in valid.stdout:
                raise RuntimeError(f"unexpected valid response: {valid.stdout}")

            if gpu_enabled:
                gpu = run(
                    [
                        client_executable, "--server", f"localhost:{port}",
                        "--ca", str(ca_cert), "--cert", str(client_cert),
                        "--key", str(client_key), "--matrix-size", "128",
                    ],
                    timeout=30,
                )
                match = re.search(
                    r"matrix_size=128 result_checksum=([-+0-9.eE]+) "
                    r"result_0_0=([-+0-9.eE]+)",
                    gpu.stdout,
                )
                if not match:
                    raise RuntimeError(f"unexpected GPU matrix response: {gpu.stdout}")

                size = 128
                left_column_sums = [
                    sum((row + column) % 5 + 1 for row in range(size))
                    for column in range(size)
                ]
                right_row_sums = [
                    sum((row * 2 + column) % 7 + 1 for column in range(size))
                    for row in range(size)
                ]
                expected_checksum = sum(
                    left_column_sums[index] * right_row_sums[index]
                    for index in range(size)
                )
                expected_first_value = sum(
                    (index % 5 + 1) * (index * 2 % 7 + 1)
                    for index in range(size)
                )
                if abs(float(match.group(1)) - expected_checksum) > 0.5:
                    raise RuntimeError(
                        f"unexpected GPU checksum {match.group(1)}, expected {expected_checksum}"
                    )
                if abs(float(match.group(2)) - expected_first_value) > 0.01:
                    raise RuntimeError(
                        f"unexpected first matrix value {match.group(2)}, "
                        f"expected {expected_first_value}"
                    )
            else:
                no_gpu = run(
                    [
                        client_executable, "--server", f"localhost:{port}",
                        "--ca", str(ca_cert), "--cert", str(client_cert),
                        "--key", str(client_key), "--matrix-size", "128",
                    ],
                    expected=1,
                )
                if "requires a CUDA-enabled server build" not in no_gpu.stderr:
                    raise RuntimeError(
                        "CPU-only server did not explicitly reject GPU work: "
                        f"{no_gpu.stderr}"
                    )

            unauthorized = run(
                [
                    client_executable, "--server", f"localhost:{port}",
                    "--ca", str(ca_cert), "--cert", str(bad_client_cert),
                    "--key", str(bad_client_key), "3",
                ],
                expected=1,
            )
            if "client certificate identity is not authorized" not in unauthorized.stderr:
                raise RuntimeError(f"unauthorized SAN was not rejected by the server: {unauthorized.stderr}")

            run(
                [
                    client_executable, "--server", f"localhost:{port}",
                    "--ca", str(ca_cert), "--cert", str(untrusted_client_cert),
                    "--key", str(untrusted_client_key), "3",
                ],
                expected=1,
            )
            run(
                [
                    client_executable, "--server", f"127.0.0.1:{port}",
                    "--ca", str(ca_cert), "--cert", str(client_cert),
                    "--key", str(client_key), "3",
                ],
                expected=1,
            )
            overflow = run(
                [
                    client_executable, "--server", f"localhost:{port}",
                    "--ca", str(ca_cert), "--cert", str(client_cert),
                    "--key", str(client_key), "4294967295", "4294967295",
                ],
                expected=1,
            )
            if "overflows uint64" not in overflow.stderr:
                raise RuntimeError(f"overflow was not reported as a remote error: {overflow.stderr}")
        finally:
            server.terminate()
            try:
                server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.communicate()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

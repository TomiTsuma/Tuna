#!/usr/bin/env python3
"""GPU-free end-to-end test of Track B CUDA remoting (Phase 3b-3d).

The full production path, with only the GPU replaced:

  tuna_cuda_e2e_client  (links libcudart = Tuna's shim)
      -> TCP (+ mTLS) -> tuna_server --foreground -> per-session worker
      -> CudaSession -> mock CUDA stack (tuna_mock_cuda, as runtime_library)

Checks:
  * plaintext loopback: registration, kernel launch, __constant__ symbols,
    streams/events, a 40 MiB bulk round trip, async error reporting;
  * mTLS: the same program with certificates from tuna_server's own tooling;
  * no TUNA_SERVER: the shim reports cudaErrorNoDevice instead of failing;
  * a server whose runtime_library is the shim itself refuses to load it;
  * with --cublas-client: a cuBLAS program (libcublas = Tuna's cuBLAS shim)
    over plaintext and mTLS, and its no-server behavior.

Usage: test_cuda_e2e.py TUNA_SERVER E2E_CLIENT MOCK_CUDA_LIB SHIM_DIR [--cublas-client CLIENT]
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

IS_WINDOWS = os.name == "nt"


class Server:
    def __init__(self, binary, config_path):
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS else {}
        self.process = subprocess.Popen([binary, "--foreground", "--config", str(config_path)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)
        self.port = None
        ready = threading.Event()

        def reader():
            for line in self.process.stdout:
                match = re.search(r"listening on .*:(\d+)", line)
                if match:
                    self.port = int(match.group(1))
                    ready.set()
            ready.set()

        threading.Thread(target=reader, daemon=True).start()
        if not ready.wait(20) or self.port is None:
            self.process.kill()
            raise RuntimeError("server never reported its port; stderr: " + self.process.stderr.read())

    def stop(self):
        self.process.send_signal(signal.CTRL_BREAK_EVENT if IS_WINDOWS else signal.SIGINT)
        return self.process.wait(timeout=20)


def client_env(shim_dir, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TUNA_")}
    # The client must load the shim as its libcudart, never NVIDIA's.
    key = "PATH" if IS_WINDOWS else "LD_LIBRARY_PATH"
    env[key] = str(shim_dir) + os.pathsep + env.get(key, "")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_client(client, env, *args):
    result = subprocess.run([client, *args], capture_output=True, text=True, timeout=60, env=env)
    print(result.stdout.rstrip())
    if result.returncode != 0:
        print(result.stderr.rstrip())
    return result


def write_config(path, **fields):
    path.write_text(json.dumps({"listen_port": 0, "log_level": "debug", **fields}))
    return path


def test_plaintext(server_bin, clients, mock, shim_dir, tmp):
    server = Server(server_bin, write_config(tmp / "plain.json", cuda={"runtime_library": str(mock)}))
    try:
        for client in clients:
            result = run_client(client, client_env(shim_dir, TUNA_SERVER=f"127.0.0.1:{server.port}"))
            assert result.returncode == 0 and "PASS" in result.stdout, f"plaintext e2e failed: {client}"
            print(f"ok: {Path(client).name} ran over plaintext loopback")
    finally:
        assert server.stop() == 0


def test_mtls(server_bin, clients, mock, shim_dir, tmp):
    pki = tmp / "pki"

    def tool(*args):
        result = subprocess.run([server_bin, *map(str, args)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"{args}: {result.stderr}"

    tool("--gen-ca", pki / "ca")
    tool("--issue-server-cert", "localhost", "--ca-dir", pki / "ca", "--out", pki / "server",
         "--san", "DNS:localhost,IP:127.0.0.1")
    tool("--issue-client-cert", "alice", "--ca-dir", pki / "ca", "--out", pki / "alice")
    tls = {"ca_cert": str(pki / "ca/ca.crt"), "cert": str(pki / "server/localhost.crt"),
           "key": str(pki / "server/localhost.key"), "allowed_client_names": ["alice"]}
    server = Server(server_bin, write_config(tmp / "tls.json", tls=tls, cuda={"runtime_library": str(mock)}))
    try:
        env = client_env(shim_dir, TUNA_SERVER=f"127.0.0.1:{server.port}", TUNA_SERVER_NAME="localhost",
                         TUNA_CA_CERT=pki / "ca/ca.crt", TUNA_CLIENT_CERT=pki / "alice/alice.crt",
                         TUNA_CLIENT_KEY=pki / "alice/alice.key", TUNA_LOG=1)
        for client in clients:
            result = run_client(client, env)
            assert result.returncode == 0 and "PASS" in result.stdout, f"mTLS e2e failed: {client}"
            assert "(mTLS)" in result.stderr, result.stderr
            print(f"ok: {Path(client).name} over mTLS")

        # Without a client certificate the shim cannot connect: no device.
        result = run_client(clients[0], client_env(shim_dir, TUNA_SERVER=f"127.0.0.1:{server.port}"),
                            "--expect-no-server")
        assert result.returncode == 0, "a plaintext client got a device from an mTLS server"
        print("ok: a client without a certificate sees no device")
    finally:
        assert server.stop() == 0


def test_no_server(client, shim_dir):
    result = run_client(client, client_env(shim_dir), "--expect-no-server")
    assert result.returncode == 0, "no-server behavior wrong"
    assert "TUNA_SERVER is not set" in result.stderr, result.stderr
    print("ok: without TUNA_SERVER the shim reports cudaErrorNoDevice")


def test_shim_refused_as_runtime(server_bin, client, shim_dir, tmp):
    shim = next(p for p in Path(shim_dir).iterdir() if p.name.startswith(("libcudart.so", "cudart64")))
    server = Server(server_bin, write_config(tmp / "loop.json", cuda={"runtime_library": str(shim)}))
    try:
        result = run_client(client, client_env(shim_dir, TUNA_SERVER=f"127.0.0.1:{server.port}"),
                            "--expect-no-server")
        assert result.returncode == 0, "server loaded the client shim as its runtime"
        print("ok: a server pointed at the shim refuses it (no self-forwarding loop)")
    finally:
        assert server.stop() == 0


def test_cublas_no_server(cublas_client, shim_dir):
    result = run_client(cublas_client, client_env(shim_dir), "--expect-no-server")
    assert result.returncode == 0, "cuBLAS no-server behavior wrong"
    print("ok: without TUNA_SERVER, cublasCreate reports CUBLAS_STATUS_NOT_INITIALIZED")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("server")
    parser.add_argument("client")
    parser.add_argument("mock")
    parser.add_argument("shim_dir")
    parser.add_argument("--cublas-client")
    args = parser.parse_args()
    clients = [args.client] + ([args.cublas_client] if args.cublas_client else [])
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        test_no_server(args.client, args.shim_dir)
        test_plaintext(args.server, clients, args.mock, args.shim_dir, tmp)
        test_mtls(args.server, clients, args.mock, args.shim_dir, tmp)
        test_shim_refused_as_runtime(args.server, args.client, args.shim_dir, tmp)
        if args.cublas_client:
            test_cublas_no_server(args.cublas_client, args.shim_dir)
    print("all CUDA end-to-end checks passed")


if __name__ == "__main__":
    main()

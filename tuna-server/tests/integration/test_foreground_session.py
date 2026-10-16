#!/usr/bin/env python3
"""End-to-end Track B check of the real binaries (Phase 2a-2c).

Runs `tuna_server --foreground` and drives it with `tuna_ping`:
  * a plaintext server refuses a non-loopback listen_address;
  * every session runs in its own worker process (distinct PIDs);
  * bulk payloads larger than one frame round-trip over the bulk socket;
  * killing one worker ends only that session, and clients then reconnect;
  * max_sessions is enforced;
  * mTLS: an allowlisted client connects, while a non-allowlisted one, a
    plaintext one and one presenting no certificate are all rejected;
  * a client retries with backoff until a late-starting server is up;
  * the certificate tooling refuses to overwrite files;
  * the server stops gracefully, with exit code 0.

Usage: test_foreground_session.py TUNA_SERVER TUNA_PING
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

IS_WINDOWS = os.name == "nt"


class Server:
    def __init__(self, binary, config_path):
        kwargs = {}
        if IS_WINDOWS:
            # Own process group so CTRL_BREAK_EVENT reaches only the server.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            [binary, "--foreground", "--config", str(config_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)
        self.lines = []
        self.port = None
        ready = threading.Event()

        def reader():
            for line in self.process.stdout:
                self.lines.append(line.strip())
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
        try:
            return self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            raise RuntimeError("server did not stop within 20 s of a stop signal")


def ping(tool, port, *args, timeout=60):
    return subprocess.run([tool, "--port", str(port), *map(str, args)],
                          capture_output=True, text=True, timeout=timeout)


def worker_pid(output):
    match = re.search(r"worker (\d+)", output)
    assert match, f"no worker pid in: {output}"
    return int(match.group(1))


def held_session(tool, port, hold_ms, *args):
    """Starts a ping that holds its session open; returns (process, worker pid)."""
    process = subprocess.Popen([tool, "--port", str(port), "--count", "1", "--hold-ms", str(hold_ms), *args],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first = process.stdout.readline()
    return process, worker_pid(first)


def kill_pid(pid):
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    else:
        os.kill(pid, signal.SIGKILL)


def write_config(path, **fields):
    path.write_text(json.dumps({"listen_port": 0, "log_level": "debug", **fields}))
    return path


def test_plaintext(server_bin, tool, tmp):
    # Plaintext must never be network-exposed.
    result = subprocess.run([server_bin, "--foreground", "--config",
                             str(write_config(tmp / "exposed.json", listen_address="0.0.0.0"))],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 1, f"non-loopback plaintext bind not refused: {result}"
    assert "without a \"tls\" config section" in result.stderr, result.stderr
    print("ok: plaintext non-loopback listen_address refused")

    server = Server(server_bin, write_config(tmp / "plain.json", max_sessions=3))
    try:
        port = server.port
        result = ping(tool, port, "--count", 50)
        assert result.returncode == 0, f"ping failed: {result.stdout} {result.stderr}"
        assert "50 echo(s)" in result.stdout and "bulk socket yes" in result.stdout, result.stdout
        print("ok: " + result.stdout.strip().splitlines()[-1])

        # Concurrent sessions run in distinct worker processes.
        results = [None] * 3

        def run(i):
            results[i] = ping(tool, port, "--count", 25, "--hold-ms", 300)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(results))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        failed = [r for r in results if r is None or r.returncode != 0]
        assert not failed, f"concurrent pings failed: {failed}"
        pids = {worker_pid(r.stdout) for r in results}
        assert len(pids) == len(results), f"sessions shared a worker: {pids}"
        print(f"ok: {len(results)} concurrent sessions in distinct workers {sorted(pids)}")

        # Payloads larger than the 16 MiB frame cap, with and without a bulk socket.
        for extra in ([], ["--no-bulk-socket"]):
            result = ping(tool, port, "--count", 1, "--bulk-mb", 40, *extra)
            assert result.returncode == 0, f"bulk echo failed: {result.stdout} {result.stderr}"
            assert ("bulk socket no" in result.stdout) == bool(extra), result.stdout
            print("ok: " + result.stdout.strip().splitlines()[-1] + (" (main connection)" if extra else ""))

        # Killing one worker ends only its session; the listener and the other
        # sessions carry on, and a client can reconnect.
        victim, victim_pid = held_session(tool, port, 3000)
        survivor, survivor_pid = held_session(tool, port, 3000)
        kill_pid(victim_pid)
        assert victim.wait(timeout=30) != 0, "session survived its worker being killed"
        assert survivor.wait(timeout=30) == 0, "an unrelated session died with another worker: " + \
            survivor.stderr.read()
        result = ping(tool, port, "--count", 3)
        assert result.returncode == 0, f"reconnect after worker kill failed: {result.stderr}"
        print(f"ok: killed worker {victim_pid}; worker {survivor_pid} and the listener kept going; reconnected")

        # max_sessions (3): three held sessions fill the server, a fourth is refused.
        holders = [held_session(tool, port, 2500) for _ in range(3)]
        result = ping(tool, port, "--count", 1)
        assert result.returncode != 0, "max_sessions was not enforced"
        for process, _ in holders:
            assert process.wait(timeout=30) == 0
        result = ping(tool, port, "--count", 1)
        assert result.returncode == 0, f"server did not recover from max_sessions: {result.stderr}"
        print("ok: max_sessions enforced and released")
    finally:
        if server.process.poll() is None:
            code = server.stop()
            assert code == 0, f"server exit code {code}; stderr: {server.process.stderr.read()}"
            print("ok: graceful stop")


def test_thread_mode(server_bin, tool, tmp):
    server = Server(server_bin, write_config(tmp / "thread.json", worker_mode="thread"))
    try:
        result = ping(tool, server.port, "--count", 10, "--bulk-mb", 20)
        assert result.returncode == 0, f"thread-mode ping failed: {result.stdout} {result.stderr}"
        print("ok: worker_mode thread serves sessions and bulk")
    finally:
        assert server.stop() == 0


def test_mtls(server_bin, tool, tmp):
    pki = tmp / "pki"

    def tool_run(*args):
        result = subprocess.run([server_bin, *map(str, args)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"{args}: {result.stderr}"

    tool_run("--gen-ca", pki / "ca")
    tool_run("--issue-server-cert", "localhost", "--ca-dir", pki / "ca", "--out", pki / "server",
             "--san", "DNS:localhost,IP:127.0.0.1")
    tool_run("--issue-client-cert", "alice", "--ca-dir", pki / "ca", "--out", pki / "alice")
    tool_run("--issue-client-cert", "mallory", "--ca-dir", pki / "ca", "--out", pki / "mallory")
    again = subprocess.run([server_bin, "--issue-client-cert", "alice", "--ca-dir", str(pki / "ca"),
                            "--out", str(pki / "alice")], capture_output=True, text=True, timeout=30)
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr, again.stderr
    print("ok: certificate tooling issues a CA, server and client certs, and refuses overwrites")

    tls = {"ca_cert": str(pki / "ca/ca.crt"), "cert": str(pki / "server/localhost.crt"),
           "key": str(pki / "server/localhost.key"), "allowed_client_names": ["alice"]}
    # With TLS, non-loopback addresses are allowed. Binding a wildcard address
    # on a Windows developer machine triggers a firewall prompt, so that is
    # only exercised on POSIX.
    address = "127.0.0.1" if IS_WINDOWS else "0.0.0.0"
    server = Server(server_bin, write_config(tmp / "tls.json", listen_address=address, tls=tls))
    try:
        def as_client(name, *extra):
            return ping(tool, server.port, "--host", "127.0.0.1", "--server-name", "localhost",
                        "--ca", pki / "ca/ca.crt", "--cert", pki / name / f"{name}.crt",
                        "--key", pki / name / f"{name}.key", *extra)

        result = as_client("alice", "--count", 20, "--bulk-mb", 40)
        assert result.returncode == 0, f"mTLS ping failed: {result.stdout} {result.stderr}"
        assert "mTLS" in result.stdout and "bulk socket yes" in result.stdout, result.stdout
        print("ok: mTLS session with bulk socket: " + result.stdout.strip().splitlines()[-1])

        result = as_client("mallory", "--count", 1)
        assert result.returncode != 0, "a client outside allowed_client_names was accepted"
        print("ok: non-allowlisted client rejected")

        result = ping(tool, server.port, "--count", 1)
        assert result.returncode != 0, "a plaintext client was accepted by a TLS server"
        print("ok: plaintext client rejected by the TLS server")

        # A client that presents no certificate: point --cert/--key at files
        # that do not exist, so the client cannot even build its context.
        result = ping(tool, server.port, "--count", 1, "--ca", pki / "ca/ca.crt",
                      "--cert", tmp / "none.crt", "--key", tmp / "none.key")
        assert result.returncode != 0
        print("ok: client without a certificate cannot connect")
    finally:
        assert server.stop() == 0


def test_retry(server_bin, tool, tmp):
    # Reserve a port, start the client first, then the server on that port.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    client = subprocess.Popen([tool, "--port", str(port), "--count", "3", "--connect-deadline-ms", "15000"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(1.0)
    server = Server(server_bin, write_config(tmp / "late.json", listen_port=port))
    try:
        out, err = client.communicate(timeout=30)
        assert client.returncode == 0, f"client did not retry until the server was up: {out} {err}"
        print("ok: client retried with backoff until a late server came up")
    finally:
        assert server.stop() == 0


def dump_logs():
    """On failure, show what the server and the newest workers logged."""
    log_dir = Path(tempfile.gettempdir()) / "tuna-server-logs"
    logs = sorted(log_dir.glob("tuna-worker-*.log"), key=lambda p: p.stat().st_mtime)[-4:]
    for log in [log_dir / "tuna-server.log", *logs]:
        if log.exists():
            print(f"----- {log.name} (last lines)")
            for line in log.read_text(errors="replace").splitlines()[-15:]:
                print(line)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_foreground_session.py TUNA_SERVER TUNA_PING")
    server_bin, tool = sys.argv[1:3]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            test_plaintext(server_bin, tool, tmp)
            test_thread_mode(server_bin, tool, tmp)
            test_mtls(server_bin, tool, tmp)
            test_retry(server_bin, tool, tmp)
        except Exception:
            dump_logs()
            raise


if __name__ == "__main__":
    main()

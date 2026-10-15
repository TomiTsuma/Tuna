#!/usr/bin/env python3
"""End-to-end Track B check of the real binaries.

Starts `tuna_server --foreground` on an ephemeral loopback port, pings it with
`tuna_ping` (sequentially and concurrently), verifies that a non-loopback
listen_address is refused, and stops the server gracefully.

Usage: test_foreground_session.py TUNA_SERVER TUNA_PING
"""

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


def start_server(server, config_path):
    kwargs = {}
    if IS_WINDOWS:
        # Own process group so CTRL_BREAK_EVENT reaches only the server.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        [server, "--foreground", "--config", str(config_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)


def wait_for_port(process, timeout=20):
    """Reads stdout until the server prints 'listening on HOST:PORT'."""
    found = {}

    def reader():
        for line in process.stdout:
            match = re.search(r"listening on .*:(\d+)", line)
            if match:
                found["port"] = int(match.group(1))
                return

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout)
    if "port" not in found:
        process.kill()
        raise RuntimeError("server never reported its port; stderr: " + process.stderr.read())
    return found["port"]


def stop_gracefully(process):
    process.send_signal(signal.CTRL_BREAK_EVENT if IS_WINDOWS else signal.SIGINT)
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError("server did not stop within 10 s of a stop signal")


def ping(tool, port, count):
    return subprocess.run([tool, "--port", str(port), "--count", str(count)],
                          capture_output=True, text=True, timeout=30)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_foreground_session.py TUNA_SERVER TUNA_PING")
    server, tool = sys.argv[1:3]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. Non-loopback bind is refused while there is no TLS.
        exposed = tmp / "exposed.json"
        exposed.write_text(json.dumps({"listen_port": 0, "listen_address": "0.0.0.0"}))
        result = subprocess.run([server, "--foreground", "--config", str(exposed)],
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 1, f"non-loopback bind not refused: {result}"
        assert "until TLS exists" in result.stderr, result.stderr
        print("ok: non-loopback listen_address refused")

        # 2. Serve sessions on an ephemeral loopback port.
        config = tmp / "server.json"
        config.write_text(json.dumps({"listen_port": 0, "log_level": "debug"}))
        process = start_server(server, config)
        try:
            port = wait_for_port(process)
            print(f"ok: server listening on port {port}")

            result = ping(tool, port, 50)
            assert result.returncode == 0, f"ping failed: {result.stdout} {result.stderr}"
            assert "50 echo(s)" in result.stdout, result.stdout
            print("ok: " + result.stdout.strip().splitlines()[-1])

            results = [None] * 4

            def run(i):
                results[i] = ping(tool, port, 25)

            threads = [threading.Thread(target=run, args=(i,)) for i in range(len(results))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            failed = [r for r in results if r is None or r.returncode != 0]
            assert not failed, f"concurrent pings failed: {failed}"
            print(f"ok: {len(results)} concurrent sessions")

            # The listener survives sessions ending; a new one still works.
            assert ping(tool, port, 1).returncode == 0
            print("ok: listener still accepting after sessions ended")
        finally:
            if process.poll() is None:
                code = stop_gracefully(process)
                assert code == 0, f"server exit code {code}; stderr: {process.stderr.read()}"
                print("ok: graceful stop")


if __name__ == "__main__":
    main()

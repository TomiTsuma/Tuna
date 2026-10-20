#!/usr/bin/env python3
"""Real-GPU milestone test: unmodified CUDA programs through Tuna.

Compiles each sample in cuda/samples/ with nvcc exactly as a user would
(`nvcc -cudart shared`, so it links NVIDIA's libcudart.so.12), then runs it
twice:
  native - with NVIDIA's runtime on the local GPU (the reference);
  tuna   - with LD_LIBRARY_PATH pointing at Tuna's shim, against a local
           tuna_server whose workers run NVIDIA's runtime on the same GPU.
Each run must print PASS. Wall times are reported for both, which makes the
per-call overhead of the remoting visible (on loopback, so no network cost).

Samples that include cublas_v2.h also link NVIDIA's libcublas (--cublas-dir,
e.g. the nvidia-cublas-cu12 wheel's lib/): natively they run on it, and
through Tuna they load Tuna's libcublas.so.12 while the server runs NVIDIA's.
Without --cublas-dir those samples are skipped.

Needs an NVIDIA GPU and a CUDA 12 nvcc; it runs in the GPU lane / by hand,
not in hosted CI.

Usage: test_gpu_samples.py --server TUNA_SERVER --shim-dir DIR --nvcc NVCC
                           --cudart-dir DIR [--cublas-dir DIR] [--arch sm_89]
                           [--samples DIR] [--mtls]
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
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Server:
    def __init__(self, binary, config_path):
        self.process = subprocess.Popen([binary, "--foreground", "--config", str(config_path)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
            raise RuntimeError("server never reported its port: " + self.process.stderr.read())

    def stop(self):
        self.process.send_signal(signal.SIGINT)
        return self.process.wait(timeout=30)


def run(cmd, env, timeout=300):
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--shim-dir", required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--cudart-dir", required=True, help="directory with NVIDIA's libcudart.so.12")
    parser.add_argument("--cublas-dir", help="directory with NVIDIA's libcublas.so.12 (headers in ../include)")
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--samples", default=str(HERE.parent.parent / "samples"))
    parser.add_argument("--mtls", action="store_true", help="run the Tuna side over mTLS")
    args = parser.parse_args()

    base_env = {k: v for k, v in os.environ.items() if not k.startswith("TUNA_")}
    samples = sorted(Path(args.samples).glob("*.cu"))
    failures = []
    rows = []
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        binaries = {}
        for source in samples:
            binary = tmp / source.stem
            extra = []
            if "cublas_v2.h" in source.read_text():
                if not args.cublas_dir:
                    print(f"skip: {source.name} needs cuBLAS (--cublas-dir)")
                    continue
                cublas_dir = Path(args.cublas_dir)
                extra = ["-I", str(cublas_dir.parent / "include"), "-Xlinker", str(cublas_dir / "libcublas.so.12")]
            compile_result = subprocess.run([args.nvcc, f"-arch={args.arch}", "-cudart", "shared", "-O2",
                                             "-o", str(binary), str(source), *extra], capture_output=True, text=True)
            if compile_result.returncode != 0:
                raise SystemExit(f"nvcc failed for {source.name}:\n{compile_result.stderr}")
            binaries[source.stem] = binary

        cudart = Path(args.cudart_dir) / "libcudart.so.12"
        config = {"listen_port": 0, "log_level": "info", "cuda": {"runtime_library": str(cudart)}}
        if args.cublas_dir:
            config["cuda"]["cublas_library"] = str(Path(args.cublas_dir) / "libcublas.so.12")
        tuna_env = dict(base_env)
        if args.mtls:
            pki = tmp / "pki"
            for cmd in (["--gen-ca", pki / "ca"],
                        ["--issue-server-cert", "localhost", "--ca-dir", pki / "ca", "--out", pki / "server",
                         "--san", "DNS:localhost,IP:127.0.0.1"],
                        ["--issue-client-cert", "alice", "--ca-dir", pki / "ca", "--out", pki / "alice"]):
                subprocess.run([args.server, *map(str, cmd)], check=True, capture_output=True)
            config["tls"] = {"ca_cert": str(pki / "ca/ca.crt"), "cert": str(pki / "server/localhost.crt"),
                             "key": str(pki / "server/localhost.key"), "allowed_client_names": ["alice"]}
            tuna_env.update(TUNA_SERVER_NAME="localhost", TUNA_CA_CERT=str(pki / "ca/ca.crt"),
                            TUNA_CLIENT_CERT=str(pki / "alice/alice.crt"),
                            TUNA_CLIENT_KEY=str(pki / "alice/alice.key"))
        config_path = tmp / "server.json"
        config_path.write_text(json.dumps(config))
        server = Server(args.server, config_path)
        try:
            tuna_env["TUNA_SERVER"] = f"127.0.0.1:{server.port}"
            tuna_env["LD_LIBRARY_PATH"] = args.shim_dir + os.pathsep + base_env.get("LD_LIBRARY_PATH", "")
            native_env = dict(base_env)
            native_dirs = [args.cudart_dir] + ([args.cublas_dir] if args.cublas_dir else [])
            native_env["LD_LIBRARY_PATH"] = os.pathsep.join(native_dirs + [base_env.get("LD_LIBRARY_PATH", "")])

            for name, binary in binaries.items():
                native, native_s = run([str(binary)], native_env)
                tuna, tuna_s = run([str(binary)], tuna_env)
                ok = native.returncode == 0 and "PASS" in native.stdout and \
                    tuna.returncode == 0 and "PASS" in tuna.stdout
                rows.append((name, native_s, tuna_s, ok))
                print(f"===== {name} (through Tuna)")
                print(tuna.stdout.rstrip())
                if not ok:
                    failures.append(name)
                    print(f"--- native rc={native.returncode}\n{native.stdout}\n{native.stderr}")
                    print(f"--- tuna rc={tuna.returncode}\n{tuna.stderr}")
        finally:
            server.stop()

    print("\n| sample | native (s) | through Tuna (s) | result |")
    print("|---|---|---|---|")
    for name, native_s, tuna_s, ok in rows:
        print(f"| {name} | {native_s:.2f} | {tuna_s:.2f} | {'PASS' if ok else 'FAIL'} |")
    if failures:
        raise SystemExit(f"failed: {failures}")
    print(f"\nall {len(rows)} samples passed natively and through Tuna{' (mTLS)' if args.mtls else ''}")


if __name__ == "__main__":
    main()

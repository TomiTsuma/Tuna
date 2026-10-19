#!/usr/bin/env python3
"""Real-GPU milestone test: unmodified PyTorch programs through Tuna (plan
milestones 4 and 5).

Runs each cuda/samples/torch_*.py twice with the same Python + PyTorch:
  native - on the local GPU (the reference);
  tuna   - with Tuna's libcudart.so.12, libcublas.so.12 and libcuda.so.1
           preloaded, against a local tuna_server whose worker runs NVIDIA's
           runtime and cuBLAS (from the same PyTorch install) on that GPU.

By default the Tuna run also has **no GPU of its own**: on Linux the client is
started in a user + mount namespace with /dev/dxg (WSL) or /dev/nvidia* bind-
mounted to /dev/null, so anything that bypassed Tuna would fail instead of
silently using the local GPU. --keep-gpu turns that off.

Each run must print PASS, and the lines that carry results ("checksum:",
"loss step=...") must be identical natively and through Tuna: same GPU, same
libraries, same inputs, so remoting must not change a single bit.

PyTorch's libraries find CUDA through RPATH, which LD_LIBRARY_PATH cannot
override, hence LD_PRELOAD (the shims carry NVIDIA's sonames, so the real
libraries are then never loaded).

Usage: test_torch.py --server TUNA_SERVER --shim-dir DIR [--python PY]
                     [--samples DIR] [--only NAME ...] [--keep-gpu] [--mtls]
"""

import argparse
import glob
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
RESULT_LINE = re.compile(r"^(checksum:|loss step=)")


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


def nvidia_lib_dir(python, package):
    """lib/ of an NVIDIA wheel PyTorch depends on (nvidia.cuda_runtime, nvidia.cublas)."""
    code = f"import nvidia.{package} as p; print(list(p.__path__)[0])"
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    return Path(out) / "lib"


def gpu_devices():
    return [d for d in ["/dev/dxg", *sorted(glob.glob("/dev/nvidia*"))] if os.path.exists(d)]


def without_gpu(cmd):
    """cmd in a user + mount namespace where the GPU device nodes are /dev/null."""
    mounts = " && ".join(f"mount --bind /dev/null {d}" for d in gpu_devices()) or "true"
    return ["unshare", "-rm", "sh", "-c", f'{mounts} && exec "$@"', "sh", *cmd]


def run(cmd, env, timeout=900):
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return result, time.perf_counter() - start


def results(stdout):
    return [line for line in stdout.splitlines() if RESULT_LINE.match(line)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True)
    parser.add_argument("--shim-dir", required=True)
    parser.add_argument("--python", default=sys.executable, help="a Python with a CUDA build of PyTorch")
    parser.add_argument("--samples", default=str(HERE.parent.parent / "samples"))
    parser.add_argument("--only", nargs="*", help="sample names (without .py) to run")
    parser.add_argument("--keep-gpu", action="store_true", help="let the Tuna run see the local GPU")
    parser.add_argument("--mtls", action="store_true", help="run the Tuna side over mTLS")
    args = parser.parse_args()

    samples = sorted(Path(args.samples).glob("torch_*.py"))
    if args.only:
        samples = [s for s in samples if s.stem in args.only]
    shim = Path(args.shim_dir)
    preload = ":".join(str(shim / name) for name in ("libcudart.so.12", "libcublas.so.12", "libcuda.so.1"))
    hide_gpu = not args.keep_gpu
    if hide_gpu and subprocess.run(without_gpu(["true"]), capture_output=True).returncode != 0:
        raise SystemExit("cannot hide the GPU (unprivileged user namespaces unavailable); use --keep-gpu")

    base_env = {k: v for k, v in os.environ.items() if not k.startswith("TUNA_")}
    failures, rows = [], []
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        config = {"listen_port": 0, "log_level": "info", "cuda": {
            "runtime_library": str(nvidia_lib_dir(args.python, "cuda_runtime") / "libcudart.so.12"),
            "cublas_library": str(nvidia_lib_dir(args.python, "cublas") / "libcublas.so.12")}}
        tuna_env = dict(base_env, LD_PRELOAD=preload)
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
            for sample in samples:
                cmd = [args.python, str(sample)]
                native, native_s = run(cmd, base_env)
                tuna, tuna_s = run(without_gpu(cmd) if hide_gpu else cmd, tuna_env)
                passed = native.returncode == 0 and "PASS" in native.stdout and \
                    tuna.returncode == 0 and "PASS" in tuna.stdout
                identical = results(native.stdout) == results(tuna.stdout)
                ok = passed and identical
                rows.append((sample.stem, native_s, tuna_s, ok))
                print(f"===== {sample.stem} (through Tuna{', no local GPU' if hide_gpu else ''})")
                print(tuna.stdout.rstrip())
                if not ok:
                    failures.append(sample.stem)
                    if passed:
                        print("--- results differ from the native run:")
                        for a, b in zip(results(native.stdout), results(tuna.stdout)):
                            print(f"    native {a}\n    tuna   {b}")
                    print(f"--- native rc={native.returncode}\n{native.stdout}\n{native.stderr[-3000:]}")
                    print(f"--- tuna rc={tuna.returncode}\n{tuna.stderr[-3000:]}")
        finally:
            server.stop()

    print("\n| sample | native (s) | through Tuna (s) | result |")
    print("|---|---|---|---|")
    for name, native_s, tuna_s, ok in rows:
        print(f"| {name} | {native_s:.2f} | {tuna_s:.2f} | {'PASS' if ok else 'FAIL'} |")
    if failures:
        raise SystemExit(f"failed: {failures}")
    print(f"\nall {len(rows)} PyTorch samples passed natively and through Tuna{' (mTLS)' if args.mtls else ''}, "
          "with identical results")


if __name__ == "__main__":
    main()

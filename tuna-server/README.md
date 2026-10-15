# Tuna Server

One `tuna_server` binary, two tracks (see the [repository README](../README.md)):

| Track | What it is | How it is invoked | Status |
|---|---|---|---|
| **A — explicit workload API** | gRPC + mutual TLS service: `RemoteSumOfSquares`, `RemoteMatrixMultiply` (CUDA). Source: `src/workload/`. | `tuna_server --cert … --key … --client-ca … --allowed-client-san …` | Working MVP (CPU path CI-tested; GPU path awaiting RTX 3060 qualification). |
| **B — CUDA driver-API remoting** | Server side of the `nvcuda.dll` shim design in `../docs/plans/2026-09-23-tuna-execution-plan.md`. Source: `src/core/`. | `tuna_server --foreground [--config PATH]` | Phase 1 (config + logging); transport arrives in Phase 2. |

Track A is compiled in when `TUNA_SERVER_WORKLOAD_API=ON` (default on Linux, off on Windows because it needs gRPC). Everything below up to *Track B* documents Track A.

## Track A — Linux workload server

The Linux server implements the versioned `tuna.v1` gRPC API. The initial product slice includes a CPU sum-of-squares connectivity check and a CUDA matrix-multiplication workload. A first-release GPU deployment requires an NVIDIA RTX 3060 (compute capability 8.6) and CUDA Toolkit 12.8 or newer within the validated support matrix. There is no CPU fallback for GPU requests.

## Supported deployment baseline

- Ubuntu 22.04 or 24.04 x64; release qualification must be run on both.
- NVIDIA RTX 3060 12 GB or a separately qualified GPU with compute capability 8.6 or newer.
- NVIDIA Linux driver 570.26 or newer for CUDA 12.8 GA, installed and tested using NVIDIA's official compatibility and installation instructions.
- C++20 compiler, CMake 3.21+, gRPC C++/Protocol Buffers development packages, and CUDA Toolkit 12.8.
- Server certificate, private key, and dedicated client CA provisioned by the operator.

The 570.26 minimum is the CUDA 12.8 GA toolkit driver version published by NVIDIA; newer compatible drivers are preferred. NVIDIA documents a lower CUDA 12.x minor-version compatibility floor (525.60.13 on Linux), but that is not the selected production baseline. Confirm the current NVIDIA compatibility documentation before each deployment.

The current CI verifies the CPU contract and mTLS path but does not have an NVIDIA GPU. Do not mark a GPU release qualified until CUDA 12.8 builds and the correctness, load, and service-device-access tests pass on the target GPU and both Ubuntu versions.

## Build and test

Install the distribution build dependencies and install CUDA using NVIDIA's official instructions for the exact Ubuntu release and driver. Then, from the repository root:

```bash
sudo apt update
sudo apt install -y build-essential cmake libgrpc++-dev libprotobuf-dev \
  protobuf-compiler protobuf-compiler-grpc python3 openssl
nvidia-smi
nvcc --version
cmake -S tuna-server -B build-server \
  -DCMAKE_BUILD_TYPE=Release \
  -DTUNA_ENABLE_CUDA=ON
cmake --build build-server --parallel
ctest --test-dir build-server --output-on-failure
```

The production CUDA configuration fails at startup if no compatible GPU is visible. It targets CUDA architecture 8.6. Its CTest suite includes a GPU multiplication correctness test (dimension 130, exercising partial CUDA tiles), so that test must run on a real compatible GPU. For CPU-only protocol development and non-GPU tests, omit `-DTUNA_ENABLE_CUDA=ON`; such a server deliberately returns `UNAVAILABLE` for matrix multiplication and is **not a first-release GPU server**.

Install the executable, operator guide, and systemd unit:

```bash
sudo cmake --install build-server --prefix /usr/local
```

The shared client/server contract and mTLS integration tests can also be built from the repository root. They validate transport and the sum-of-squares smoke test; they do not prove GPU execution.

## API and workload limits

- `RemoteSumOfSquares`: 1–4,096 unsigned 64-bit values; checked arithmetic; used as a CPU connectivity smoke test.
- `RemoteMatrixMultiply`: two square row-major float32 matrices, each 128×128 through 512×512; all inputs and outputs must be finite. The deterministic sample CLI builds matrices locally, sends them over mTLS, and displays a result checksum and first value.
- GPU requests are executed on CUDA device 0. The server checks compute capability 8.6 or newer at startup, serializes GPU requests, and rejects concurrent GPU submissions with `RESOURCE_EXHAUSTED` instead of queueing unbounded device-memory work.
- Requests/results are not persisted. Messages are capped at 4 MiB. Matrix multiplication is not an arbitrary CUDA interface and accepts no client code or device pointers.
- The listener defaults to `127.0.0.1:50051`. Bind to a network interface only with a deliberate firewall policy.

The sample matrices are synthetic diagnostic data, not an application offload SDK or transparent acceleration of another program. GPU throughput, multi-user scheduling, and per-tenant resource isolation are not yet production qualified.

## Service account, certificates, and device access

Install using a dedicated unprivileged account and strict certificate permissions. The service unit expects `tuna` and the `video` group:

```bash
sudo useradd --system --user-group --home-dir /nonexistent --shell /usr/sbin/nologin tuna
sudo install -d -o root -g tuna -m 0750 /etc/tuna
sudo install -o root -g tuna -m 0640 server.crt /etc/tuna/server.crt
sudo install -o tuna -g tuna -m 0600 server.key /etc/tuna/server.key
sudo install -o root -g tuna -m 0640 client-ca.crt /etc/tuna/client-ca.crt
```

The systemd unit must access the GPU device nodes; it does not hide all devices. It grants the service the `video` group and restricts the device cgroup to `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm`. Confirm the nodes and group ownership are created by the installed NVIDIA driver before starting the service. Adjust the allowlist deliberately if the qualified host uses different device nodes; do not grant broad device access by default.

Create `/etc/tuna/tuna-server.env` with an intentional listener and the exact client DNS SAN value (without `DNS:`):

```ini
TUNA_LISTEN_ADDRESS=0.0.0.0:50051
TUNA_ALLOWED_CLIENT_SAN=sample-client
```

Protect the file and start the service:

```bash
sudo chown root:tuna /etc/tuna/tuna-server.env
sudo chmod 0640 /etc/tuna/tuna-server.env
sudo systemctl daemon-reload
sudo systemctl enable --now tuna-server.service
sudo systemctl status tuna-server.service
sudo journalctl -u tuna-server.service
```

Open TCP/50051 only to trusted client networks. Validate certificate name/chain, device visibility as the `tuna` user, and firewall access before accepting requests.

## Production release blockers

- TLS 1.3 minimum enforcement has not been proven; do not expose this as a production service until tests show older TLS versions are rejected.
- Certificate issuance, renewal, revocation, and recovery are manual and not implemented as an operational lifecycle.
- The service authorizes one configured client SAN, not a managed fleet of independent identities.
- GPU integration has not yet been built or tested on a physical RTX 3060 in this workspace. Performance, numerical tolerance, 12 GB memory behavior, thermal/power limits, driver upgrade, and CUDA fault recovery need target-hardware qualification.
- CUDA device access and the systemd sandbox need validation on clean Ubuntu 22.04 and 24.04 hosts.
- No signed package/update/rollback system, health/metrics endpoint, structured operational telemetry, or independent security review exists.
- A single active GPU request is bounded; general user quotas, fair scheduling, cross-tenant isolation, and denial-of-service controls remain incomplete.

NVIDIA references: [CUDA 12.8 Toolkit Release Notes — driver versions](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html) and [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/).

## Track B — remoting server core (`--foreground`)

Builds anywhere, with no gRPC dependency (pass `-DTUNA_SERVER_WORKLOAD_API=OFF` on Linux to build only this track). On Windows the documented toolchain is the Qt-bundled MinGW 13.1 + Ninja:

```bash
cmake -S tuna-server -B tuna-server/build-ninja -G Ninja -DTUNA_SERVER_WORKLOAD_API=OFF
cmake --build tuna-server/build-ninja
ctest --test-dir tuna-server/build-ninja --output-on-failure
```

`tuna_server --foreground --config tuna-server.json` loads a JSON config, writes a rotating log to `<temp>/tuna-server-logs/tuna-server.log`, and serves sessions until Ctrl-C (SIGINT/SIGTERM, or Ctrl-Break on Windows):

```json
{ "listen_port": 7443, "listen_address": "127.0.0.1", "log_level": "info" }
```

- `listen_port` is required, and `0` picks a free port. The server prints `listening on HOST:PORT` on stdout.
- `listen_address` defaults to `127.0.0.1`. **Non-loopback addresses are refused** until the TLS transport lands in Phase 2b, because plaintext GPU RPC must never reach a network.
- Each connection completes a `Hello` handshake (v1 frame protocol, `../proto/tuna_frames.proto`) and is then served on its own thread, with a 3 s heartbeat and a 10 s silence timeout. Phase 2c moves sessions into separate worker processes.
- Phase 2a serves only the reserved diagnostic echo API (`api_id 0`). Every other `api_id` returns `CUDA_ERROR_NOT_SUPPORTED` until Phase 3's generated CUDA dispatchers.

Check a running server from the client side with `tuna_ping --port 7443 --count 20`. `tuna_rpc_bench` measures RPC latency and batched throughput in-process. Both are built from the shared `../tunnel/` library.

Its security model is the *Track B* section of `../tuna-guest-agent/docs/security_model.md`.

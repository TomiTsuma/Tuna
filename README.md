# Tuna

Tuna lets a machine without a capable GPU use a remote server's GPU. Development runs on two tracks that share one repository, one protocol directory and one server binary.

| | **Track A — explicit workload API** | **Track B — CUDA driver-API remoting** |
|---|---|---|
| Idea | An application calls a typed Tuna RPC (`RemoteSumOfSquares`, `RemoteMatrixMultiply`), and the server runs a fixed, bounded workload. | Unmodified CUDA apps (CUDA samples, PyTorch) load a replacement `nvcuda.dll`. Their driver API calls are forwarded to a per-session worker that runs them on the server's real GPU. |
| Transport | gRPC over mutual TLS | v1 length-prefixed protobuf frames over TCP + TLS 1.3 (msquic later) |
| Contract | [`proto/tuna/v1/workload.proto`](proto/tuna/v1/workload.proto) | [`proto/tuna_frames.proto`](proto/tuna_frames.proto) |
| Status | **Working MVP**: Windows client and GUI, Linux server, CPU path and mTLS tested. The GPU path still needs qualification on RTX 3060 hardware. | **Phase 2 done**: `tunnel/` (framing, TCP + mTLS 1.3 transports, handshake, heartbeat, batched RPC, bulk channel), certificate tooling, per-session worker processes. **Phase 3a in progress** (`cuda/`): CUDA runtime-API codegen + marshalling, and a first call driving a real GPU through the generated dispatch. Approach is **B2** (replace `libcudart`, Linux client first) — see `docs/plans/2026-10-16-phase3-spike-findings.md`. |
| Plan | [`tuna-guest-agent/docs/implementation_plan.md`](tuna-guest-agent/docs/implementation_plan.md) | [`docs/plans/2026-09-23-tuna-execution-plan.md`](docs/plans/2026-09-23-tuna-execution-plan.md) |

The tracks are complementary. Track A gives a secure, usable product today, for workloads that can be expressed as explicit calls. Track B is the longer bet on transparent acceleration. It reuses Track A's security posture (mTLS, a dedicated CA, identity allowlists), and it goes further because clients there run arbitrary GPU code. Both are covered in [the security model](tuna-guest-agent/docs/security_model.md).

## Layout

```text
proto/                 Wire contracts for both tracks
cmake/                 Shared CMake helpers (protobuf + frame-proto library)
tuna-guest-agent/      Windows client: Qt GUI, gRPC sample client, Track B core, tests, docs
tuna-server/           Server: one `tuna_server` binary, Track A (src/workload) + Track B (src/core, src/remoting)
tunnel/                Track B transport + RPC runtime shared by client and server, plus tuna_ping / tuna_rpc_bench
docs/plans/            Track B execution plan and earlier design history
docs/changelog/        Per-change engineering log
.github/workflows/     CI: Windows client, Windows server core, Linux client+server (Ubuntu 22.04/24.04)
```

## Quick start

| Goal | Where to look |
|---|---|
| Build and run the Windows client / GUI | [`tuna-guest-agent/README.md`](tuna-guest-agent/README.md) |
| Deploy the Linux server | [`tuna-server/README.md`](tuna-server/README.md) |
| Build Track B only, with no gRPC (Windows: MinGW + Ninja) | `cmake -S tuna-guest-agent -B tuna-guest-agent/build-ninja -G Ninja -DTUNA_BUILD_SAMPLE_CLIENT=OFF` and `cmake -S tuna-server -B tuna-server/build-ninja -G Ninja -DTUNA_SERVER_WORKLOAD_API=OFF`, then `cmake --build` and `ctest` in each |
| Run everything that CI's Linux job runs | `cmake -S tuna-guest-agent -B build-all -DTUNA_BUILD_SAMPLE_CLIENT=ON -DTUNA_BUILD_LINUX_SERVER=ON && cmake --build build-all -j && ctest --test-dir build-all` |

## License

MIT. See [`tuna-guest-agent/LICENSE`](tuna-guest-agent/LICENSE).

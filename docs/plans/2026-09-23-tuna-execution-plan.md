# Tuna — Execution Plan (API Remoting Revision)

**Date:** 2026-09-23
**Status:** **Approved / canonical** (2026-10-14) — supersedes §4 of `2026-07-31-tuna-guest-and-server-completion-design.md`. The CUDA-first vertical slice and the v1 wire protocol in §4 govern all implementation; the July file-first slice is retained for history only.
**Author:** Thomas Tsuma (with Claude)

---

## 1. Where the project stands today

| Area | State |
|---|---|
| Phase 0 housekeeping | Largely done: `minhook` is a real submodule, Catch2/CTest wired in both repos, `tuna-server` scaffolded, `src/` restructured into `core/ daemon/ hooks/ tunnel/ gui/`. |
| Phase 1 core plumbing | Started: `proto/tuna_frames.proto` exists and builds in both repos with a round-trip test. `core/config_manager.cpp`, `logger.cpp`, `error_handler.cpp` are still 0-byte stubs. |
| Tunnel | Not started. Every file in `src/tunnel/` is empty. |
| Hooks | Not started. `hook_handlers.h` declares `HookedNtCreateFile`; the only working hook code is the reference prototype `tests/createfile_hook.c`. |
| Daemon | Not started. Every file in `src/daemon/win/` is empty. |
| Server | Prints a version string and exits. |
| GUI | Qt mockup driven by fake `QTimer` metrics. |
| Docs | `security_model.md` is empty, and so is `gui_design_mockups.png`. All `scripts/*` are empty. |

The four design docs describe a very broad system: file, memory, D3D, OpenGL, Vulkan, CUDA, OpenCL, sockets, USB, and process hooks, plus a kernel driver. The July completion design ordered that work sensibly. It still picked `CreateFileW` as the first vertical slice and treated every API family as roughly equal in value.

This revision keeps the July plan's architecture, testing discipline, and repo layout. It changes **what gets intercepted first, at which layer, and why**, based on how the existing remote-GPU systems actually work (see §2).

---

## 2. What existing systems teach us

### 2.1 Landscape

| System | What it remotes | Interception layer | Transport | Notes |
|---|---|---|---|---|
| **rCUDA** (UPV, 2010–) | CUDA | CUDA Runtime API shim library | TCP / InfiniBand verbs | This is the reference academic system. Its papers show that network bandwidth dominates overhead, and that InfiniBand gets close to native performance for many HPC apps. |
| **GVirtuS** | CUDA (and more) | Frontend/backend split with a runtime-level shim | TCP, VMSocket | Extends the same model to VMs and ARM. |
| **Cricket** (RWTH) | CUDA | `LD_PRELOAD` of a `cudart` replacement | ONC RPC / XDR | The app must link `cudart` dynamically (`-cudart shared`). Supports checkpoint/restart. Tied to specific CUDA versions. |
| **SCUDA → LUPINE** | CUDA driver API, NVML, NCCL | Replacement `libcuda` / `nvcuda` driver shim. The app's own runtime and libraries run locally. | One long-lived HTTP/2 stream with mandatory LZ4 | Clients run on Linux, macOS and Windows. Hundreds of stubs are **generated**, not written by hand. |
| **Thunder Compute** | CUDA | CUDA layer, below PyTorch and other frameworks | TCP | Reports being within about 10% of native for AI workloads. Each GPU has one tenant at a time, and memory is wiped between users. |
| **Juice Labs** | CUDA plus some graphics | Client/agent shims launched with `juice run <app>` | TCP/IP | Supports Windows and Linux clients. Uses a launcher model, not system-wide hooks. |
| **Venus** (Mesa / virglrenderer) | Vulkan | Guest Vulkan driver that serializes calls | virtio-gpu shared-memory rings | The protocol is **generated from `vk.xml`**. Host-visible memory is shared as blobs. |
| **Neptune** (UTM, 2026) | Direct3D 11 / DXGI | Replacement `d3d11.dll` / `dxgi.dll` that serialize COM calls | Shared-memory ring. The host runs DXVK. | Its serializer is **generated from MIDL**. It uses per-thread rings, async fences, and zero-copy frame handoff. |
| **HTCondor standard universe** (historic) | POSIX syscalls | App relinked against a syscall library that forwards to a "shadow" process | TCP | This is the classic "forward syscalls to another machine" design. It was later dropped because keeping it working was expensive and it had to be relinked for every platform. |

### 2.2 Lessons that change the plan

1. **Remote the accelerator API, not raw syscalls.** None of the working systems forward kernel syscalls for GPU work. On Windows, the GPU is reached through WDDM `D3DKMT*` thunks. Those carry driver-private command buffers and GPU virtual-address mappings, and they cannot be forwarded over a network. Every system that works sits at a **documented API boundary**: the CUDA driver or runtime API, Vulkan, or D3D COM interfaces. Tuna's hooks should target these boundaries too.
   - *Consequence for CPU work:* ordinary CPU computation, such as a Photoshop filter running on the CPU, does not show up as syscalls, so no hook can offload it. Remote CPU execution is only practical as explicit RPC (the `@tuna_remote` decorator idea) or by running the whole app remotely.

2. **The CUDA driver API is the best first target.** LUPINE shims the driver API (`nvcuda.dll` on Windows) and lets the application ship its own `cudart`, cuBLAS and cuDNN. This means:
   - one small, stable, versioned C ABI to cover, instead of the runtime plus every library;
   - library kernels (cuBLAS/cuDNN fatbins) still reach the server through `cuModuleLoadData` / `cuLibraryLoadData`;
   - it works for PyTorch/TensorFlow unchanged, and that is where the demand is.

3. **Generate the stubs.** Venus generates from `vk.xml`, Neptune from MIDL, and LUPINE uses codegen with pinned headers. Writing hundreds of marshalling stubs by hand is where bugs come from. Tuna should parse `cuda.h`, add a small hand-written annotation file for pointer semantics, and emit both the client stubs and the server dispatchers.

4. **Latency, not bandwidth, is the first thing that breaks.** Every synchronous call costs one network round trip: roughly 0.2–1 ms on a LAN and much more over the internet. Frameworks make thousands of calls per training step. The systems that perform well:
   - send calls that cannot fail synchronously as **fire-and-forget** (kernel launches, async memcpys, event records) and report errors later, the same way CUDA already reports async errors;
   - **batch** calls per thread or stream (Neptune uses per-thread rings);
   - keep bulk data transfers on a **separate channel**, so a 500 MB memcpy doesn't block small control calls behind it.

5. **Bandwidth sets the ceiling, so be honest about fit.** 1 GbE carries about 0.125 GB/s and PCIe 4.0 x16 about 32 GB/s. Workloads whose data stays on the GPU (training, inference, simulation) work well. Workloads that copy a lot between host and device, and real-time graphics that need raw frames back, do not work without compression or encoding.

6. **Launch-scoped injection beats system-wide hooking.** Juice uses `juice run <app>`. Cricket and SCUDA use `LD_PRELOAD`. A launcher that injects into one process is simpler and safer, and it is less likely to trip antivirus than a service that injects into everything. `AppInit_DLLs` is disabled when Secure Boot is on.

7. **Use one server process per client session.** Thunder uses sole tenancy and wipes memory between users. A crashing kernel or a malicious client then only takes down its own session, and GPU memory can be cleaned up when the session ends.

---

## 3. Revised architecture (deltas from the July design)

```
Client (weak PC, no NVIDIA GPU needed)              Server (GPU box)
┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
│ app.exe (python/torch, cuda sample)  │            │ tuna-server (listener, auth)          │
│   cudart64_12.dll / cublas / cudnn   │            │   └─ spawns per-session worker proc   │
│   └─> tuna_nvcuda.dll  (driver shim) │  TLS/QUIC  │        tuna-worker.exe                │
│        generated cu* stubs ──────────┼──────────► │         generated dispatchers         │
│        async batcher / handle table  │ ctrl chan  │         └─> real nvcuda.dll → GPU     │
│        bulk-copy channel ────────────┼──────────► │         module cache (by hash)        │
│ tuna.exe run <app>  (launcher/inject)│ bulk chan  │         resource accounting / wipe    │
│ tuna daemon (optional: config, GUI)  │            │                                       │
└──────────────────────────────────────┘            └──────────────────────────────────────┘
```

Changes from the July design:

| July design | Revised | Reason |
|---|---|---|
| First vertical slice: `CreateFileW` → file op on server | First vertical slice: **CUDA driver API** (`cuInit` … `cuLaunchKernel`) | Delivers the stated goal (use a remote GPU) directly. Remote file I/O doesn't make the client faster by itself. |
| MinHook trampolines as the main mechanism | **Proxy DLL** (`nvcuda.dll` replacement) as the main mechanism. MinHook only redirects `LoadLibrary*` / `GetProcAddress` when the proxy isn't picked up. | Exporting the same symbols is more robust than patching code, and it needs no hook on hot paths. |
| System-wide Windows service injects hooks | **`tuna run <app>`** launcher (create suspended → inject → resume). The daemon becomes optional (config, status, GUI backend). | Simpler, safer, and matches Juice/Cricket. |
| Hand-written Context Object per API | **Generated** RPC stubs from `cuda.h` plus an annotations file | Scale and correctness. |
| `Frame { header, encrypted_payload }` | `Frame { header, payload }`. Encryption is TLS's job. Add `RpcRequest` / `RpcResponse` / `Hello`. | Encrypting inside TLS is redundant. |
| QUIC via msquic from day one | A **`ITransport` interface**: TCP+TLS first (fastest route to the milestone), then msquic behind the same interface. | Keeps the vertical slice independent of msquic integration. QUIC's advantage (independent streams) matters most in Phase 4. |
| D3D/DXGI hooks in Phase 4 | Graphics moves to a **separate research track** (§6) | Most expensive and least proven part over a real network. |

---

## 4. Wire protocol (v1)

Update `proto/tuna_frames.proto`:

```proto
syntax = "proto3";
package tuna;

enum FrameType { FRAME_TYPE_UNSPECIFIED = 0; SYS = 1; GPU = 2; IO = 3; HEARTBEAT = 4; CONTROL = 5; }

message FrameHeader {
  FrameType type        = 1;
  uint64    frame_id    = 2;   // 64-bit: avoid the wraparound bugs Neptune hit
  uint32    channel     = 3;   // 0 = control, 1 = rpc, 2 = bulk
  uint64    timestamp_ns= 4;
}
message Frame { FrameHeader header = 1; bytes payload = 2; }

message Hello {                // CONTROL, first frame each direction
  uint32 protocol_version = 1;
  string client_os        = 2;
  uint32 cuda_driver_version = 3;   // server fills: what cuDriverGetVersion returns
  repeated DeviceInfo devices = 4;  // server fills
  bytes  resume_token     = 5;
}
message DeviceInfo { string name = 1; uint64 total_mem = 2; int32 cc_major = 3; int32 cc_minor = 4; bytes uuid = 5; }

message RpcRequest {           // GPU
  uint64 call_id  = 1;
  uint32 api_id   = 2;         // generated enum, stable across versions
  uint32 flags    = 3;         // bit0 = ASYNC (no reply expected), bit1 = HAS_BULK
  uint64 thread_id= 4;         // client thread → server thread affinity (CUDA contexts are per-thread)
  bytes  args     = 5;         // generated compact encoding (not per-call protobuf messages)
  uint64 bulk_ref = 6;         // id of a payload on the bulk channel, if any
}
message RpcResponse {
  uint64 call_id  = 1;
  int32  result   = 2;         // CUresult
  bytes  out_args = 3;
  uint64 bulk_ref = 4;
}
message RpcBatch { repeated RpcRequest calls = 1; }   // async calls coalesced
```

Rules:
- The framing is length-prefixed protobuf. `args` uses a flat, generated little-endian encoding, because protobuf-per-call is too slow for hot paths.
- Async calls return `CUDA_SUCCESS` locally right away. The first deferred error gets latched and returned from the next synchronizing call (`cuStreamSynchronize`, `cuCtxSynchronize`, `cuMemcpy*` without `Async`, `cuEventSynchronize`), which matches CUDA's own async error model.
- Payloads above a threshold (for example 64 KB) go on the bulk channel, LZ4 compressed when the ratio is worth it, chunked, with an xxHash per chunk.

---

## 5. Phased execution plan

Durations assume one developer working part-time. The phases are dependency-ordered, and each one ends with a **demo-able milestone and an integration test**, following the July design's testing strategy.

### Phase 1 — Finish core plumbing *(≈1 week, in progress)*
Complete the existing `2026-07-31-phase-1-core-plumbing.md` tasks: `config_manager`, `logger` (spdlog), and `error_handler`, for both agent and server. **Defer** the Windows SCM daemon work to Phase 5.
- Update `tuna_frames.proto` to v1 (§4). Update the round-trip tests.
- Write `docs/security_model.md` (draft: trust model, mTLS, per-session worker, what the server will execute).
- **Exit:** `ctest` is green in both repos, and config and logging work in a foreground `tuna_server --foreground`.

### Phase 2 — Transport + RPC runtime *(≈2–3 weeks)*
- `tunnel/transport.h`: `ITransport` with `connect/listen/send(channel, bytes)/on_receive`. The first implementation is **TCP + TLS 1.3** (OpenSSL, or Schannel on Windows), with `TCP_NODELAY` and two sockets (control+rpc, bulk). Keep the msquic implementation as a later drop-in.
- mTLS with self-signed CA tooling: `tuna-server --gen-ca`, `--issue-client-cert`.
- Hello/capabilities exchange, heartbeat every 3 s, 10 s timeout, reconnect with backoff. These already appear in `tunneling_protocol.md`.
- `rpc/`: call-id correlation, a blocking `call()` and a non-blocking `post()`, per-thread batching queues flushed on sync calls or on a timer (~0.5 ms), and deferred-error latching.
- Server: accept → authenticate → **spawn `tuna-worker` process per session**, and hand the socket to it (on Windows, `WSADuplicateSocket`; on Linux, `SCM_RIGHTS`).
- **Exit (integration test):** agent ↔ server on localhost do the handshake. 10k echo RPCs: measure p50/p99 RTT, and batched throughput in calls/sec. Kill the worker and confirm the listener survives.

### Phase 3 — CUDA driver API vertical slice *(≈4–5 weeks — the core milestone)*
**3a. Code generator (`tools/codegen/`, Python + libclang):**
- Parse `cuda.h` from a pinned CUDA Toolkit version (start with 12.x, matching the server's driver).
- `annotations.yaml`: for each function, the direction and size expression of every pointer parameter. For example, `cuMemcpyHtoD_v2: srcHost: {in: bytes, size: ByteCount}`, and `cuDeviceGetName: name: {out: bytes, size: len}`. Handles (`CUcontext`, `CUmodule`, `CUfunction`, `CUstream`, `CUevent`, `CUdeviceptr`) pass through as opaque `uint64`. The server owns the real values, and the client never dereferences them.
- Emit: client stubs (`tuna_nvcuda.dll` exports + `.def` file), server dispatch table, the `api_id` enum, and per-function marshalling tests.
- Functions without an annotation are generated as stubs that return `CUDA_ERROR_NOT_SUPPORTED` and log the name, so coverage grows as real apps show what they use.

**3b. Hand-written special cases (the hard ones):**
- `cuGetProcAddress` / `cuGetProcAddress_v2`: modern `cudart` gets most driver entry points through this rather than through imports. It **must** return Tuna's own stubs, or calls will skip the shim.
- `cuGetExportTable`: undocumented internal tables used by `cudart`. This is **the main technical risk**. Investigate early (week 1 of this phase): log which table UUIDs `cudart` requests, and check how LUPINE and ZLUDA handle them.
- `cuLaunchKernel`: `kernelParams` is an array of pointers to args of unknown size. Use `cuFuncGetParamInfo` on the server (CUDA ≥ 12.4) to get offsets and sizes, cache them per `CUfunction`, fetch them once to the client, and then marshal args by value.
- `cuModuleLoadData` / `cuModuleLoadFatBinary` / `cuLibraryLoadData`: find the image size from the fatbin/ELF header, hash it, and upload only if the server's module cache misses.
- Host memory: `cuMemAllocHost` / `cuMemHostAlloc` return ordinary local memory. Mark it "pinned" in a client-side table so async copies from it can be treated as async. `cuMemAllocManaged` (unified memory) is **not supported in v1**: return `CUDA_ERROR_NOT_SUPPORTED`. A socket can't back `mmap`, which is the same limit SCUDA ran into.
- `cuDriverGetVersion` / `cuDeviceGet*`: return the server's values, taken from `Hello`.

**3c. Loading the shim on Windows:**
- First try: put `nvcuda.dll` (our shim) in a private directory and have `tuna run` launch the app with that directory first in the DLL search path.
- Fallback: `tuna run` creates the process suspended and injects `tuna_loader.dll` (Detours `DetourCreateProcessWithDllEx`, or the existing MinHook path). That DLL hooks `LoadLibraryExW` / `LoadLibraryW` and redirects `nvcuda.dll` to the shim, so System32-only lookups are covered too.
- This is where the existing MinHook work and `createfile_hook.c` pay off, as the loader's hooking technique.

**Milestones (in order; each one is an integration test):**
1. `deviceQuery` from cuda-samples, run on the client with no NVIDIA GPU, prints the server's GPU.
2. `vectorAdd` produces correct results.
3. `matrixMul` plus a cuBLAS `sgemm` sample (proves library fatbins load remotely).
4. Python: `torch.cuda.is_available() == True`, `torch.randn(4096,4096,device='cuda') @ ...` matches a CPU reference.
5. A small PyTorch training loop (MNIST/ResNet-18) runs to completion.

**CI:** the codegen and marshalling tests, plus a **mock driver** on the server side that returns canned results, run on GitHub-hosted runners. Real-GPU milestone tests run nightly on the GPU box as a self-hosted runner. This follows the July design's GPU test lane.

### Phase 4 — Performance *(≈2–3 weeks)*
- Turn on async classification for launch, async memcpy, event record, and stream wait. Measure the fraction of calls that are synchronous for each workload.
- Bulk channel: chunking, LZ4, and overlapping transfer with compute. Measure achieved throughput against link rate.
- Switch the transport to **msquic** (a QUIC stream per CUDA stream, plus a bulk stream) behind `ITransport`. Compare against TCP. Keep whichever wins.
- Server-side caches: modules by hash, `cuFuncGetParamInfo` results, device attributes (answered locally after first fetch).
- **Benchmarks to publish in `docs/benchmarks.md`:** per-call RTT, memcpy GB/s against `iperf3`, and ResNet-18 step time for native / Tuna-over-localhost / Tuna-over-LAN. Target: within 20% of native on LAN for GPU-resident training.

### Phase 5 — Productization: launcher, daemon, GUI *(≈2–3 weeks)*
- `tuna.exe` CLI: `tuna connect <server>`, `tuna run <app> [args]`, `tuna status`, `tuna doctor` (checks certs, RTT, driver version match).
- Windows service (existing `daemonization_workflow.md`, now slimmed down): holds config and certs, exposes a named-pipe API, and reports session metrics. **It does not inject into processes.**
- Rewire the Qt GUI (`mainwindow.cpp`) from the `QTimer` fakes to the named-pipe API: live RTT, bytes moved, active sessions, server GPU utilization (a small NVML query on the server).
- NVML shim (`nvml.dll`) with a minimal set (device count, name, memory info, utilization), so `nvidia-smi`-style tools and PyTorch memory queries work.
- Packaging: real `scripts/build.ps1` and the NSIS installer.

### Phase 6 — Security hardening *(≈1–2 weeks, overlaps Phase 5)*
- Threat model in `security_model.md`: an authenticated client can run **arbitrary GPU code** on the server. That's the whole point. So isolation comes from the per-session worker, a low-privilege service account, GPU memory scrubbed on session end (`cuMemsetD8` on freed allocations, or relying on context teardown, to be verified), and per-client quotas (memory, concurrent sessions).
- Cert pinning, revocation list, and an audit log of sessions and modules loaded (hashes).
- Fuzz the server dispatcher with the generated marshalling tests plus libFuzzer on `RpcRequest.args` decoding. This is the server's main attack surface.

### Phase 7 — Expansion *(ongoing, pick by demand)*
- **Linux / macOS clients:** the generated stubs are platform-neutral. Linux needs a `libcuda.so.1` shim loaded via `LD_PRELOAD` or library path. This is cheap once Phase 3 exists.
- **NCCL / multi-GPU:** multiple server GPUs exposed as devices. Cross-server peer copies have to stage through the client (a limitation LUPINE documents too).
- **`@tuna_remote` Python decorator** for explicit CPU/RAM offload (pickle the function and its args, run on the server). This is the only practical path to "use the server's CPU/RAM".
- **OpenCL / HIP:** the same codegen with a different header and annotation file.

---

## 6. Research track (parallel, time-boxed): graphics and file I/O

These stay in scope as **spikes with go/no-go decisions**, not committed phases.

**6.1 Graphics (D3D / Vulkan / OpenGL).** There are two realistic designs:
- **API remoting via Vulkan only.** On the client, run DXVK (D3D9–11 → Vulkan) and vkd3d-proton (D3D12 → Vulkan), and remote **only Vulkan**. The protocol can be generated from `vk.xml` the way Venus does it. The server renders, and `Present` returns frames **encoded on the server with NVENC** (H.264/HEVC/AV1). The client decodes and blits. Raw 1080p60 frames are about 0.5 GB/s, far more than any LAN can carry, so encoding is mandatory. Neptune shows that per-thread rings, async fences, and generated serializers are needed even over *shared memory*. Over a network, draw-call-heavy apps will be limited by latency.
- **Whole-app streaming** (the Parsec/Sunshine/Moonlight model): run the app on the server and stream frames and input. It is much simpler and proven, but it is a different product (remote desktop) rather than interception.
- *Spike:* forward a Vulkan triangle, then `vkcube`, over localhost with generated stubs. Measure frame time at 1 ms of added RTT. Decide whether to continue.

**6.2 File I/O.** Hooking `CreateFileW` so that files are opened on the server does not speed up the client. The file redirection that is actually useful goes the **other way**: a remoted GPU job that needs the client's dataset. Better options than per-call hooks:
- the bulk channel with content-addressed caching (already needed for memcpy);
- if a real filesystem view is needed, WinFsp or the Windows Projected File System (ProjFS) on the relevant side, instead of hooking every file API.
- Keep `createfile_hook.c` as the reference for the Phase 3c loader hooks.

**Drop from scope:** the kernel-mode WDK driver (§3 of `syscall_interception_design.md`), USB/device redirection, and Winsock hooks. None of them help the GPU goal, and each one is a large project on its own.

---

## 7. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `cuGetExportTable` hidden tables are needed by `cudart` and are undocumented | Blocks the runtime/PyTorch path | Investigate in week 1 of Phase 3. Study how LUPINE/ZLUDA handle it. The fallback is a runtime-API shim like rCUDA/Cricket, which needs `cudart` linked dynamically (PyTorch on Windows ships it as a DLL). |
| Client `cudart` newer than the server driver | `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` / init failures | `Hello` reports the server driver version, and `tuna doctor` warns. Document minimum server driver. |
| Latency makes chatty apps unusable over WAN | Poor experience | Measure the sync-call fraction, and aggressively batch async calls. Recommend LAN or a nearby server, and publish benchmarks honestly. |
| Unified/managed memory in apps | App failures | Return `NOT_SUPPORTED` clearly and log it. Revisit with page-fault-driven emulation only if a key workload needs it. |
| Antivirus flags injection | Install friction | Prefer the proxy DLL over injection. Code-sign binaries. Only inject in launch-scoped processes. |
| Scope creep from the original 9-category API list | Nothing finishes | Nothing outside §5 starts until the Phase 3 milestone 4 (PyTorch matmul) passes. |

---

## 8. Immediate next steps (next 2 weeks)

1. Finish Phase 1 tasks 2–4 (config, logger, error handler). The plan already exists.
2. Update `tuna_frames.proto` to v1 (§4) and update both `test_frame_proto.cpp` files.
3. **Spike (2–3 days):** write a hand-written `nvcuda.dll` proxy that exports only `cuInit`, `cuDriverGetVersion`, `cuGetProcAddress(_v2)`, and `cuGetExportTable`, logs every call and requested export-table UUID, and forwards to the real driver on a machine with a GPU. Run `deviceQuery` and `python -c "import torch; torch.cuda.init()"` under it. This settles the biggest unknown before building the codegen.
4. Start `tools/codegen/`: parse `cuda.h` with libclang and dump the function list with parameter types. Draft `annotations.yaml` for the ~60 functions `deviceQuery` + `vectorAdd` + PyTorch init actually call (taken from the spike's log).

---

## 9. References

- rCUDA — [middleware whitepaper](https://network.nvidia.com/pdf/whitepapers/rCUDA_Middleware_and_Applications.pdf); [Influence of InfiniBand FDR on remote GPU virtualization](https://www.mcs.anl.gov/papers/P5004-0813_1.pdf); [Performance comparison of CUDA remote GPU virtualization frameworks](https://www.researchgate.net/publication/284646474_A_Performance_Comparison_of_CUDA_Remote_GPU_Virtualization_Frameworks)
- Cricket — [RWTH-ACS/cricket](https://github.com/RWTH-ACS/cricket); [Eiling et al., CCPE 2022](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.6474)
- SCUDA / LUPINE — [kevmo314/scuda](https://github.com/kevmo314/scuda) (now [lupinemachines/lupine](https://github.com/lupinemachines/lupine)); [HN discussion](https://news.ycombinator.com/item?id=41787547)
- Thunder Compute — [How Thunder Compute works (GPU-over-TCP)](https://www.thundercompute.com/blog/how-thunder-compute-works-gpu-over-tcp)
- Juice Labs — [docs](https://docs.juicelabs.co/docs/juice/intro); [wiki](https://github.com/Juice-Labs/Juice-Labs/wiki/)
- Venus — [Mesa docs](https://docs.mesa3d.org/drivers/venus.html); [Collabora: state of virglrenderer](https://www.collabora.com/news-and-blog/blog/2025/01/15/the-state-of-gfx-virtualization-using-virglrenderer/)
- Neptune — [Direct3D virtualization for QEMU (UTM, 2026)](https://blog.getutm.app/2026/introducing-neptune-direct3d-virtualization-for-qemu/)
- HTCondor — [Standard universe wiki](https://htcondor-wiki.cs.wisc.edu/index.cgi/wiki/index.cgi/wiki?p=StandardUniverse)

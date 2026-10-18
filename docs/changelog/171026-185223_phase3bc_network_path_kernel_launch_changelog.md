# Phase 3b/3c — CUDA over the network, kernel launch, real GPU milestones 1–3

**Date:** 2026-10-17
**Branch:** `phase-2-transport`
**Author:** Thomas Tsuma (with Claude)
**Builds on:** `171026-020905_phase3a_cuda_codegen_slice_changelog.md`
**Execution plan:** `docs/plans/2026-09-23-tuna-execution-plan.md` §5 Phase 3, milestones 1–3

## Outcome

Unmodified CUDA programs, compiled the normal way (`nvcc -cudart shared`),
run on a remote GPU through Tuna: the program loads Tuna's `libcudart.so.12`
instead of NVIDIA's, and every runtime call, kernel launch and memory copy
goes over the Phase 2 tunnel (TCP + mTLS 1.3, bulk socket) to a per-session
`tuna_server` worker that runs NVIDIA's runtime on the GPU.

Real-GPU run (WSL Ubuntu 24.04, RTX 4050 Laptop, driver 571.96 / CUDA 12.8,
loopback; `cuda/tests/integration/test_gpu_samples.py`):

| sample | what it exercises | native (s) | through Tuna, plaintext (s) | through Tuna, mTLS (s) |
|---|---|---|---|---|
| device_query | properties, attributes, versions | 0.37 | 0.15 | 0.19 |
| vector_add | NVIDIA's vectorAdd: `<<<>>>`, memcpy, `cudaGetLastError` | 0.16 | 0.20 | 0.21 |
| matrix_mul | templated shared-memory kernel, pinned memory, non-blocking stream, 300 async launches, events (811 GFlop/s through Tuna) | 0.54 | 0.31 | 0.40 |
| symbols_streams | `__constant__`/`__device__` vars, struct-by-value + double params, atomics, dynamic shared memory, two streams + event ordering, occupancy, func attributes, async error | 0.18 | 0.27 | 0.36 |

All four print `PASS` natively and through Tuna. (Wall times include process
start and CUDA context creation; on loopback the remoting overhead is small.
Network benchmarks are Phase 4.)

## What changed

### Code generator (`tools/codegen/`)
- **Stable api ids**: `api_ids.yaml` is an append-only registry; new names
  get the next id, existing ids never move. (3a assigned ids by sorted name,
  so adding a function renumbered others: a client/server version skew bug.)
- New parameter role `in_struct`; typed handles (`cudaStream_t`,
  `cudaEvent_t`) instead of `void*`; flags `_async` (fire-and-forget),
  `_sync` (reports latched async errors), `_wrap` (static generated stub that a
  hand-written export wraps). Every parameter must now be annotated.
- **Full ABI**: every libcudart 12 export not implemented is generated as a
  stub returning `cudaErrorNotSupported` (`cudart12_exports.txt`, 435 symbols),
  so any program loads against the shim.
- `cudaGetErrorName`/`String` tables are generated from the `cudaError` enum's
  doc comments (134 codes), so the shim answers them locally.
- Works on Linux (pip libclang + gcc's system include dirs).
- 41 simple remoted functions (device, memory, streams, events), 12 special.

### Client shim (`cuda/src/client*.cpp`) — now a real `libcudart.so.12`
- Built as `tuna_cudart` → `<build>/shim/libcudart.so.12` (Linux, version node
  `libcudart.so.12`, only the CUDA API exported via `libcudart.map`) or
  `cudart64_12.dll` (Windows/MinGW, also builds and passes the e2e test).
- **Lazy connect** on the first CUDA call from `TUNA_SERVER`, `TUNA_CA_CERT`,
  `TUNA_CLIENT_CERT`, `TUNA_CLIENT_KEY`, `TUNA_SERVER_NAME`. No server →
  `cudaErrorNoDevice`, which apps already treat as "no GPU here".
- **Kernel launch**: `__cudaRegisterFatBinary/Function/Var` are recorded
  locally; a fatbin is uploaded on first use of one of its kernels (lazy, since
  programs embed far more device code than they run), each kernel's parameter
  layout is fetched once, and `cudaLaunchKernel` is fire-and-forget with
  parameters packed by value. `__cudaPush/PopCallConfiguration` implement `<<<>>>`.
- Symbols: `cudaMemcpyTo/FromSymbol(Async)`, `cudaGetSymbolAddress/Size`.
- Memcpy family: direction-aware; H2D/D2H payloads on the bulk channel;
  `cudaMemcpyDefault` inferred from tracked allocations; async H2D/D2D are
  fire-and-forget; copies > 1 GiB go in pieces.
- Per-thread last error (`cudaGetLastError`/`PeekAtLastError` merge the
  client's own error with the server's and with latched async failures).
- Local: pinned host memory (page-aligned local memory), host registration,
  error strings, runtime version, profiler start/stop.
- Device properties and attributes cached after first fetch (frameworks
  query them per operation).

### Server (`cuda/src/dispatch.cpp`, `tuna-server/src/remoting/dispatch.cpp`)
- `CudaSession` replaces the stateless `CudaDispatch`: per-session fatbin
  images, function ids, and per-CUDA-context loaded modules / resolved
  functions (so multi-device and `cudaDeviceReset` re-resolve lazily).
- Kernels go through the driver API in the runtime's primary context:
  `cuModuleLoadData`, `cuModuleGetFunction`, `cuFuncGetParamInfo` (layout),
  `cuLaunchKernel`, `cuModuleGetGlobal`, `cuFuncGet/SetAttribute`, occupancy.
- `backend_real` opens NVIDIA's libcudart **and** libcuda; it **refuses to
  load Tuna's own shim** as the "real" runtime (no self-forwarding loop if the
  server's library path is wrong).
- `tuna_server` routes CUDA api ids to the session's `CudaSession`, opening the
  CUDA stack on the first CUDA call (pings/health checks never create a
  context). New config section: `"cuda": {"enabled", "runtime_library", "driver_library"}`.

### Tests
- `tuna_cuda_tests` (17 cases, 1137 assertions, no GPU): props cache,
  memcpyDefault, streams/events/async copies, registration + launch, `<<<>>>`
  path, async launch error latching, unknown kernels, `__constant__` symbols,
  func attributes/occupancy, device reset reloading modules, host memory, error
  strings, not-supported stubs, malformed requests (lying name length, bad
  fatbin, unknown function id).
- `tuna_mock_cuda`: a mock CUDA stack (device memory in host buffers, three
  kernels executed on the host with real grid bounds, a `__constant__`).
- `tuna_cuda_e2e` (ctest): real `tuna_server` + real shim + `tuna_cuda_e2e_client`
  with the mock as the server's runtime: plaintext, mTLS, 40 MiB bulk, no-server
  behavior, and the self-load guard. Runs in CI.
- `test_gpu_samples.py` (GPU lane / by hand): the table above.
- CI: new `linux-cuda-remoting` job (NVIDIA redistributable headers only)
  builds the shim and runs everything above except the GPU samples, and fails
  if the committed generated code is stale.

## Verification (2026-10-17)

| Check | Where | Result |
|---|---|---|
| Full ctest (incl. CUDA unit + e2e) | WSL Ubuntu 24.04 / GCC 13.3 | 68/68 |
| Full ctest (incl. CUDA unit + e2e, Windows shim DLL) | Windows MinGW 13.1 | 68/68 |
| Server without CUDA shim (unchanged path) | Windows MinGW | 50/50 |
| GPU samples, plaintext + mTLS | WSL + RTX 4050 | 4/4 + 4/4 |

## Design decisions and tradeoffs

- **Runtime API, not driver API** (B2, decided after the spike): NVIDIA's
  cudart cannot run over a driver shim (integrity-checked export tables).
  Cost: the shim must cover the runtime API surface; mitigated by generated
  stubs and full-ABI not-supported exports.
- **Driver API on the server for modules**: using the real runtime's
  `__cudaRegisterFatBinary` server-side would hide parameter layouts; the
  driver API gives them (`cuFuncGetParamInfo`) at the cost of requiring a
  CUDA 12.4+ server driver.
- **Function ids instead of `CUfunction` handles** cross the wire, so the
  client cache survives device switches and resets; the server maps
  (context, id) → `CUfunction`.
- **Lazy fatbin upload per session**: the right default for PyTorch-size
  binaries. A content-addressed cross-session module cache is Phase 4.
- **Async launch errors** surface at the next synchronizing call (CUDA's own
  model for kernel faults). Deviation: a launch-configuration error is also
  returned by the next sync, not only by the launch.
- **Single server thread per session**: correct for single-device programs;
  per-client-thread device state is future work.

## Not included (next)
- cuBLAS / cuDNN API shims (needed for plan milestone 3's sgemm and for
  PyTorch, milestones 4–5).
- `tuna run` launcher (sets the env + library path), `tuna doctor`.
- Phase 4 performance: network benchmarks, content-addressed module cache,
  LZ4 on bulk.

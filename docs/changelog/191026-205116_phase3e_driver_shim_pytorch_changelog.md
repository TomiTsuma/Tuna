# Phase 3e — driver-API shim; PyTorch on a GPU-less client (plan milestone 4)

**Date:** 2026-10-19
**Branch:** `phase-2-transport`
**Author:** Thomas Tsuma (with Claude)
**Builds on:** `181026-182236_phase3d_cublas_remoting_changelog.md`
**Execution plan:** `docs/plans/2026-09-23-tuna-execution-plan.md` §5 Phase 3, milestone 4
("`torch.cuda.is_available() == True`, `torch.randn(4096,4096,device='cuda') @ ...`
matches a CPU reference")

## Why this was the next step

Phase 3d closed milestone 3. The plan's next gate is milestone 4, and nothing
outside Phase 3 starts until it passes. Phase 3d expected cuBLASLt and the
other CUDA libraries to be the blockers. A spike showed that the first one is
the **driver API** instead.

## How the blockers were found (spike)

PyTorch 2.8.0 (cu128 wheel) was installed in a WSL venv and run against Tuna's
`libcudart.so.12` + `libcublas.so.12` with `TUNA_LOG=1`. The client had **no
GPU**: it ran in an unprivileged user + mount namespace with `/dev/dxg`
bind-mounted to `/dev/null` (`unshare -rm`, no root needed).

1. `torch.cuda.is_available()` was **already True** through the Phase 3d shims.
2. `torch.cuda.init()` then failed with "CUDA driver error: unknown error" in
   `_hasPrimaryContext` → `cuDevicePrimaryCtxGetState`. PyTorch makes that
   call, and its whole driver usage, through `libcaffe2_nvrtc.so`, which links
   `libcuda.so.1` directly. That is the only torch library that imports the
   driver (23 functions).
3. A throwaway logging `libcuda.so.1` (in scratch, not committed) listed the
   rest of what milestone 4 needs: `cuDevicePrimaryCtxGetState` (called before
   **every op**), `cuCtxGetCurrent`, `cuDevicePrimaryCtxRetain` and
   `cuCtxSetCurrent` (before creating the cuBLAS handle), plus one runtime
   call, `cudaStreamIsCapturing`. With those faked, the 4096² matmul passed.

## Outcome

Real GPU (WSL Ubuntu 24.04, RTX 4050 Laptop, driver 571.96 / CUDA 12.8,
loopback). PyTorch 2.8.0+cu128 is unmodified, and the client has no GPU:

| `cuda/samples/torch_matmul.py` | native | through Tuna, plaintext | through Tuna, mTLS |
|---|---|---|---|
| `torch.cuda.is_available()` | True | **True** | **True** |
| device | RTX 4050 Laptop | **RTX 4050 Laptop** (the server's) | same |
| 4096² fp32 matmul, max abs error vs fp64 CPU | 1.03e-3 | 1.03e-3 | 1.03e-3 |
| checksum of the result | -127810.83994067139 | **identical** | **identical** |

**Milestone 4 passes.** Milestone 5 (training) is not done yet; see "Not
included (next)".

## What changed

### Driver shim: `libcuda.so.1` / `nvcuda.dll` (new)

A third library shim, built like the cuBLAS one. It exports the full driver
ABI (652 symbols, listed from the r571 driver into
`tools/codegen/cuda12_driver_exports.txt`) and sends everything through the
runtime shim's bridge (`tuna_cudart_call`). Driver calls therefore share the
session, worker, context and ordering with the runtime and cuBLAS calls.

- **Generated** from `cuda.h` with `tools/codegen/annotations_driver.yaml`
  (9 functions): `cuInit`, `cuDriverGetVersion`, `cuDeviceGet`,
  `cuDeviceGetCount`, `cuDeviceGetAttribute` (cached), `cuDeviceTotalMem_v2`,
  `cuDeviceGetUuid_v2`, `cuDevicePrimaryCtxGetState` (cached once active) and
  `cuDevicePrimaryCtxSetFlags_v2`.
- **Hand-written, remoted**: `cuDevicePrimaryCtxRetain` and `cuDeviceGetName`
  (whose OUT buffer is sized by `len`).
- **Client-local**: contexts (`cuCtxGetCurrent` / `SetCurrent` / `PushCurrent`
  / `PopCurrent` / `GetDevice` / `Synchronize`, `cuDevicePrimaryCtxRelease`),
  `cuGetErrorName` / `cuGetErrorString` (tables generated from cuda.h's
  `CUresult` enum), and `cuGetProcAddress_v2`.
- Everything else returns `CUDA_ERROR_NOT_SUPPORTED`, logged once with
  `TUNA_LOG=1`.

### Server (`cuda/src/dispatch_driver.cpp`)

- `CudaSession::DispatchDriver`: the generated cases plus primary-context
  retain and device name. Before its first driver call the worker runs
  `cuInit(0)` (`DriverInit`), because the runtime may not have initialized the
  driver yet.
- Each primary context is retained once per session and kept for the worker's
  lifetime. A client "release" changes nothing, since the runtime holds the
  context anyway.
- `backend_real.cpp` refuses a `driver_library` that is Tuna's own shim
  (`tuna_cuda_driver_shim` marker), as it already did for the runtime and
  cuBLAS.

### Codegen (`tools/codegen/gen_cuda.py`)

- The runtime client emitter is parameterized by library (export macro,
  status type, call path) so the driver reuses it. The runtime output is
  byte-identical apart from the one new function.
- `generate_driver` always runs, because `cuda.h` ships with every toolkit and
  with the `cuda_cudart` redistributable. It emits `driver_client_stubs.inc`,
  `driver_server_dispatch.inc`, `driver_unsupported_stubs.inc` and
  `driver_error_strings.inc`.
- **Safety rule enforced by the generator:** a generated driver call may only
  take `CUdeviceptr` / `CUstream` handles. Contexts, modules and functions are
  host pointers in the worker and must never reach the server from the wire.
- Driver ids are appended to the shared `api_ids.yaml` registry, after
  cuBLAS's.

### Runtime shim

- `cudaStreamIsCapturing` is remoted (torch asks before allocator work).
  Capture APIs are unsupported, so the answer is always "not capturing".
- New bridge export `tuna_cudart_context_generation()`. The driver shim uses
  it to drop its cached "primary context active" state on `cudaDeviceReset`.

### Tests

- `cuda/tests/test_driver.cpp` (8 cases, in-process through the mock):
  - device queries;
  - attribute caching;
  - primary-context state following the runtime, cached once active,
    refetched after a reset;
  - retain/set/push/pop selecting the runtime device;
  - tokens validated;
  - per-thread current context;
  - error strings local;
  - malformed requests rejected by the server.
- `cuda/tests/e2e_driver_client.cpp`: a program linked `-lcuda -lcudart`, run
  by `test_cuda_e2e.py` over plaintext and mTLS against the mock server, and
  without a server (`cuInit` → `CUDA_ERROR_NO_DEVICE`). It covers
  `cuGetProcAddress` resolving the shim's own (newest-variant) exports.
- The mock gained a driver: `cuInit` (device queries fail before it, like the
  real driver), device queries, and primary-context state/retain/flags. Its
  `cudaMalloc` now initializes the context, as the real runtime does.
- `cuda/tests/integration/test_torch.py` (new, real GPU): runs each
  `cuda/samples/torch_*.py` natively and through Tuna with the client's GPU
  hidden (user + mount namespace), requires PASS from both, and requires the
  result lines to be **identical**. Supports `--mtls`.
- `cuda/samples/torch_matmul.py` (milestone 4) and
  `cuda/samples/torch_train_mlp.py` (milestone 5, part 1; see below).
- CI: the stale-codegen check now also fails on generated files that were
  never committed (`git status --porcelain`), not only on modified ones.

## Verification (2026-10-19)

| Check | Where | Result |
|---|---|---|
| Full ctest (incl. 8 driver cases + driver e2e) | WSL Ubuntu 24.04 / GCC 13.3 | 85/85 |
| Full ctest (incl. `nvcuda.dll`) | Windows MinGW 13.1, CUDA 12.8 headers | 85/85 |
| `test_torch.py --only torch_matmul`, plaintext and mTLS | WSL + RTX 4050, no client GPU | PASS, identical checksum |
| Regenerated codegen output, runtime stubs unchanged except `cudaStreamIsCapturing` | WSL | ok |

## Design decisions and tradeoffs

- **A driver shim at all, rather than only patching around PyTorch.** The
  driver is how frameworks and NVIDIA's libraries check for the GPU. Owning
  `libcuda.so.1` also guarantees that nothing in the process reaches a local
  driver: `cuGetProcAddress` lookups resolve to Tuna too. The cost is one more
  library in the shim set, and it adds little code because it reuses the
  bridge and codegen.
- **Context tokens on the client, never server pointers.** This adds no
  server attack surface: the server never receives a context value, and the
  generator enforces that for every future annotation. The cost is that only
  primary contexts are supported. That is all the runtime API (and therefore
  PyTorch) uses; `cuCtxCreate` remains unsupported.
- **Caching `cuDevicePrimaryCtxGetState` once active.** PyTorch calls it per
  op, so remoting it naively would add one RTT to every kernel. "Active" can
  only end through a device reset, which the context generation tracks, so the
  cache is exact. "Inactive" answers are not cached.
- **Hiding the GPU in tests with user namespaces.** On a dev box with a GPU,
  a library that bypassed Tuna would otherwise succeed silently on the local
  GPU. Bind-mounting the device nodes away makes the "no GPU on the client"
  claim testable without root or a second machine.
- **Windows DLL search order.** System32's `nvcuda.dll` beats `PATH`, so the
  driver e2e client gets its own directory with the shims beside it. A real
  application needs the same from the Phase 5 launcher (`tuna run`), which
  plan §3c anticipated.

## Not included (next)

- **cuBLASLt, which blocks milestone 5.** `torch_train_mlp.py` reaches its
  first `nn.Linear`: `addmm` with a bias goes to cuBLASLt (`gemm_and_bias`).
  NVIDIA's `libcublasLt` is still loaded in the client, and it segfaults in
  `cublasLtMatmulAlgoGetHeuristic` without a local GPU. It needs its own shim,
  with server-side tables for its descriptor objects (matmul desc, layouts,
  preferences) and validation of the algo blobs it returns.
- cuDNN (convolutions; ResNet-18 for milestone 5), cuRAND/cuFFT/cuSPARSE/NCCL
  as workloads need them.
- Driver-loaded modules (`cuModuleLoadData` / `cuLaunchKernel`), used by
  PyTorch's jiterator kernels. The server already has the module machinery
  for runtime fatbins.
- Phase 4 performance work, and Phase 6 handle tables for runtime streams and
  events, as listed in 3d.

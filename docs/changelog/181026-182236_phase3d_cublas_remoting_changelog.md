# Phase 3d — cuBLAS remoting (plan milestone 3, cuBLAS sgemm)

**Date:** 2026-10-18
**Branch:** `phase-2-transport`
**Author:** Thomas Tsuma (with Claude)
**Builds on:** `171026-185223_phase3bc_network_path_kernel_launch_changelog.md`
**Execution plan:** `docs/plans/2026-09-23-tuna-execution-plan.md` §5 Phase 3, milestone 3
("`matrixMul` plus a cuBLAS `sgemm` sample (proves library fatbins load remotely)")

## Why this was the next step

Phase 3b/3c finished milestones 1–2 and the `matrixMul` half of milestone 3.
The plan says nothing outside Phase 3 starts until milestone 4 (PyTorch
matmul) passes, and every PyTorch matmul goes through cuBLAS. So cuBLAS comes
first: it closes milestone 3 and is the first of the libraries PyTorch needs.

The plan expected cuBLAS to "just work" because it assumed a *driver*-API shim
(library kernels would reach the server through `cuModuleLoadData`). The
Phase 3 spike moved Tuna to a *runtime*-API shim (B2), and that changes the
picture: NVIDIA's libcublas statically links its own runtime and calls libcuda
directly, so on a client with no GPU it cannot initialize, whatever libcudart
is loaded. cuBLAS has to be remoted **at the cuBLAS API**, the way rCUDA and
Cricket do it.

## Outcome

An ordinary cuBLAS program, built with `nvcc -cudart shared ... -lcublas`,
runs unchanged through Tuna: it loads Tuna's `libcublas.so.12` and
`libcudart.so.12`, and cuBLAS executes in the session's worker on the server's
GPU.

Real GPU (WSL Ubuntu 24.04, RTX 4050 Laptop, driver 571.96 / CUDA 12.8,
loopback; `cuda/tests/integration/test_gpu_samples.py --cublas-dir ...`):

| sample | native | through Tuna, plaintext | through Tuna, mTLS |
|---|---|---|---|
| cublas_sgemm: 50 × SGEMM 1024², GPU-timed | **5658 GFlop/s** | **5647 GFlop/s** | **5493 GFlop/s** |
| cublas_sgemm: max abs error vs CPU (SGEMM / fp16 GemmEx) | 2.83e-06 / 5.58e-05 | identical | identical |
| all 5 samples (device_query, vector_add, matrix_mul, symbols_streams, cublas_sgemm) | PASS | PASS | PASS |

Throughput is native because BLAS calls are fire-and-forget: the client
queues 50 GEMMs without waiting, and the GPU never starves. (Loopback only;
network benchmarks are Phase 4.)

## What changed

### Architecture: one session, many libraries
- **`tuna_cudart_call`** (`cuda/include/tuna/cuda/bridge.h`): a plain C bridge
  exported by Tuna's libcudart. Library shims send their marshalled calls
  through it, so they share the process's single session. That matters for
  correctness, not just economy: a `cublasSgemm` must run in the same server
  context as the `cudaMalloc`'d buffers it uses, and in order after the
  `cudaMemcpy` that filled them.
  - *Alternative considered:* a second connection per library. Rejected,
    because it would need a server-side context share and cross-connection
    ordering, and it would double the handshakes.
  - *Alternative considered:* linking cuBLAS client code into libcudart.
    Impossible, because the app's loader resolves `cublas*` from a library
    whose soname is `libcublas.so.12`.
- The cuBLAS shim carries **no transport of its own**. It is small, it links
  only against Tuna's libcudart, and the C ABI means the two libraries do
  not have to share a C++ runtime.

### Code generator (`tools/codegen/gen_cuda.py`)
- `--cublas CUBLAS_INCLUDE_DIR` parses `cublas_v2.h` with libclang and reads
  `annotations_cublas.yaml` and `cublas12_exports.txt` (742 libcublas exports).
  It emits `cublas_client_stubs.inc`, `cublas_server_dispatch.inc` and
  `cublas_unsupported_stubs.inc`.
- New roles for cuBLAS scalars, whose meaning depends on the handle's pointer
  mode:
  - `in_scalar_ref` (alpha/beta): the value is copied in host mode, the device
    address is sent in device mode. `in_scalar_ref(expr)` sizes a `const void*`
    scalar; GemmEx uses the compute type, and Ctype when it is complex.
  - `out_scalar_ref` (a dot product's result): the value is returned in the
    reply in host mode, written on the device in device mode.
- Array parameters (`const float* const Aarray[]`) are now spelled as the
  pointers they decay to.
- cuBLAS ids are appended to the same `api_ids.yaml` registry (ids 54–96), so
  the server routes every call by id alone. The runtime's ids are unchanged.
- Async cuBLAS failures are mapped server-side to the matching `cudaError_t`:
  `CublasAsyncError` in the dispatch, and `kFlagAsync` for malformed posted
  requests.

### Client (`cuda/src/cublas_client.cpp`) → `libcublas.so.12` / `cublas64_12.dll`
- 39 generated stubs: handle and configuration, S/D level 1 and level 2,
  S/D gemm, GemmEx, batched, strided-batched, geam. BLAS and `cublasSet*`
  calls are posted (fire-and-forget).
- Hand-written:
  - create/destroy/`SetPointerMode`, tracking each handle's pointer mode;
  - `GetPointerMode` (local);
  - `Set/GetMatrix(Async)`: the host side is packed densely, so any lda
    works, the payload rides the bulk channel, and pieces are capped at 1 GiB;
  - `Set/GetVector(Async)` as 1 × n matrices with ld = stride;
  - status names and strings, `GetCudartVersion` and `LoggerConfigure`.
- 690 generated not-supported exports, so any cuBLAS program loads. A version
  script exports only `cublas*`.
- Connection codes are mapped: no server → `CUBLAS_STATUS_NOT_INITIALIZED`;
  an old server without cuBLAS → `NOT_SUPPORTED`; connection lost →
  `INTERNAL_ERROR`.

### Server
- `CudaSession::DispatchCublas` (`cuda/src/dispatch_cublas.cpp`) runs the
  generated cases plus `MatrixCopy`. `MatrixCopy` validates dimensions and
  caps the size before allocating, checks that the payload size matches, and
  waits on the stream for async device → host copies.
- **Handle table (security).** A `cublasHandle_t` is a pointer into cuBLAS's
  host memory inside the worker. Without checks, a malicious authenticated
  client could get arbitrary host reads and writes in the worker in two ways:
  - by sending a forged handle;
  - by sending a "device" scalar while the server's handle is in host mode,
    so cuBLAS dereferences a client-chosen address as host memory. A dot
    product's result would then be *written* there.

  The session therefore records the handles it created and each one's pointer
  mode. Every generated case refuses an unknown handle and a scalar whose
  host/device tag disagrees with the mode. Create, destroy and
  set-pointer-mode maintain the table and are hand-written
  (`_server_by_hand`).
- `OpenRealCuda` now takes `CudaLibraries {runtime, driver, cublas}`. libcublas
  is opened **lazily on the first cuBLAS call**, and a library exporting
  `tuna_cublas_shim` is refused. New config key:
  `"cuda": {"cublas_library": ...}`.
- Built without cuBLAS headers, the dispatcher compiles as before and cuBLAS
  calls report "not supported" (`TUNA_CUDA_HAVE_CUBLAS`).

### Tests
- `tests/test_cublas.cpp`: 9 GPU-free cases through the real chain (shim →
  bridge → `CudaSession` → mock cuBLAS). They cover:
  - handles, streams, versions and modes;
  - leading dimensions and strides in matrix and vector copies, plus their
    async forms;
  - S/D gemm with all four transpose combinations against a CPU reference;
  - strided-batched gemm and GemmEx;
  - level-1 results in host mode;
  - **device pointer mode** (alpha/beta and dot results in device memory);
  - async failure surfacing at `cudaDeviceSynchronize`;
  - status strings and unsupported exports;
  - malformed requests: a truncated request, an over-wide scalar, a
    2^30 × 2^30 copy, a size mismatch, a forged handle, and a scalar tagged
    "device" for a host-mode handle (cuBLAS never runs).
- The mock CUDA stack gained a small cuBLAS. It computes column-major S/D gemm,
  axpy, scal, dot and nrm2 on "device" memory, and reads scalars from host or
  device memory per the pointer mode.
- `tuna_cublas_e2e_client` + `test_cuda_e2e.py --cublas-client`: a real
  program linked against both shims, against a real `tuna_server`, over
  plaintext and mTLS. It checks a 64² sgemm, a device-mode dot, an async error
  at sync, a 24 MiB matrix round trip, and no-server behavior.
- Real GPU: `samples/cublas_sgemm.cu` (table above). `test_gpu_samples.py`
  links cuBLAS samples against NVIDIA's libcublas (`--cublas-dir`) and points
  the server at it.
- CI: the `linux-cuda-remoting` job fetches only the `include/` directory of
  NVIDIA's `libcublas` 12.8.4.1 redistributable and caches it. It regenerates
  with `--cublas` and fails if the committed output is stale.

## Verification (2026-10-18)

| Check | Where | Result |
|---|---|---|
| Full ctest (incl. 9 new cuBLAS cases + cuBLAS e2e) | WSL Ubuntu 24.04 / GCC 13.3, cuBLAS headers from the `nvidia-cublas-cu12==12.8.4.1` wheel | 77/77 |
| Full ctest (incl. `cublas64_12.dll`) | Windows MinGW 13.1, CUDA 12.8 toolkit headers | 77/77 |
| GPU samples, plaintext + mTLS | WSL + RTX 4050 | 5/5 + 5/5 |
| Committed codegen output regenerated with `--cublas`; runtime stubs byte-identical | WSL | ok |
| Server pointed at Tuna's own libcublas | WSL, by hand | refused; `cublasCreate` → `NOT_SUPPORTED`, no self-forwarding |

## Design decisions and tradeoffs

- **Remote cuBLAS at its API** rather than its kernels. This was forced by
  B2, since NVIDIA's libcublas cannot run on the client. The upside is that
  the client never loads the ~700 MB cuBLAS libraries. The cost is that each
  library needs its own shim. That cost is kept small by codegen, and by the
  fact that the same annotations format covers cuBLASLt and cuDNN next.
- **Pointer mode tracked on the client**, not inferred from addresses. It is
  exact (it mirrors cuBLAS's own per-handle state), costs no round trip, and
  works for device scalars that live in memory the client never saw allocated
  (pools, offsets).
- **Async error reporting deviation:** a posted BLAS call returns
  `CUBLAS_STATUS_SUCCESS`, and argument errors (bad lda, ...) come back as a
  `cudaError_t` from the next synchronizing CUDA call. Real cuBLAS validates
  on the host and returns `INVALID_VALUE` directly. Making every BLAS call
  synchronous would cost one RTT per GEMM, which is what the plan's latency
  lesson warns against. Client-side argument validation for the gemm family
  is a possible refinement if an application depends on the synchronous
  error.
- **Vectors as 1 × n matrices**: one server path for all eight host ↔ device
  copy functions.
- **Lazy libcublas on the server**: runtime-only sessions do not pay the
  library's load time or memory.
- `cublasHgemm` is declared only for C++ in `cublas_api.h`, so the C parse
  does not see it. It stays a not-supported export; fp16 GEMMs go through
  `cublasGemmEx`, which is what frameworks use.

## Not included (next)
- **cuBLASLt** (PyTorch's `addmm` + bias epilogues and its default matmul path
  on recent versions). Its API is descriptor-heavy (matmul / layout /
  preference descriptors), so it needs a handle-table design for the
  descriptor objects.
- cuDNN, cuFFT, cuRAND, cuSPARSE and NCCL: the rest of what `libtorch_cuda.so`
  links. Then plan milestone 4, `torch.cuda.is_available()` and a matmul.
- 64-bit (`_64`) and complex cuBLAS APIs, TRSM and other level-3 routines, as
  apps need them.
- Phase 4: zero-copy for the bridge's bulk payload (it copies once today), and
  network benchmarks.
- Phase 6 (security hardening): the runtime side still passes client-supplied
  `cudaStream_t` / `cudaEvent_t` values to the real runtime unchecked. Those
  are host pointers too. They need the same session-owned handle tables that
  cuBLAS handles now have, and fuzzing of the dispatcher.

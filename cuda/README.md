# cuda/ — Track B CUDA runtime-API remoting (Phase 3)

Replaces NVIDIA's CUDA **runtime** library (`libcudart.so.12` /
`cudart64_12.dll`) with a Tuna shim that forwards the runtime API over the
Phase 2 tunnel to a `tuna_server` worker running the real runtime on a real
GPU. Unmodified programs built with `nvcc -cudart shared` (and anything else
that loads libcudart dynamically) run on a client with **no NVIDIA GPU**.
This is option **B2** (Linux client first) from
`../docs/plans/2026-10-16-phase3-spike-findings.md`.

```
 app (nvcc -cudart shared [-lcublas])              tuna_server --foreground
   │ cudaMalloc / <<<kernel>>> / cublasSgemm         └─ worker process per session
   ▼                                                      └─ CudaSession (dispatch*.cpp)
 libcublas.so.12 = Tuna shim ─┐                                ├─ NVIDIA libcudart (cudaMalloc, ...)
   (tuna_cudart_call)         ▼                                ├─ NVIDIA libcuda   (modules, launches)
 libcudart.so.12 = Tuna shim  ── TCP + mTLS 1.3 ──►            └─ NVIDIA libcublas (opened on first use)
   client*.cpp, generated stubs   control + bulk sockets
```

Since Phase 3d the shim set also replaces **cuBLAS** (`libcublas.so.12` /
`cublas64_12.dll`): see [cuBLAS](#cublas-phase-3d) below.

## What crosses the wire

| Application does | Shim | Server |
|---|---|---|
| a simple runtime call (`cudaMalloc`, `cudaStreamCreate`, ...) | generated stub: flat little-endian args, one round trip | generated `case`: decode → real call → encode outs |
| `cudaLaunchKernel` / `kernel<<<>>>` | **fire-and-forget**, batched; params packed by value | `cuLaunchKernel` with the layout from `cuFuncGetParamInfo` |
| first launch of a kernel from a fatbin | uploads the fatbin nvcc embedded (once per session), fetches the kernel's param layout (once) | `cuModuleLoadData` per CUDA context, `cuModuleGetFunction` |
| `cudaMemcpy` H2D / D2H | payload on the **bulk channel** (separate socket, 1 MiB chunks) | real `cudaMemcpy` from/into the payload |
| `cudaMemcpyAsync` H2D / D2D, `cudaMemsetAsync`, `cudaEventRecord`, `cudaStreamWaitEvent` | fire-and-forget | real async call |
| `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventSynchronize`, `cudaMemcpy`, `cudaGetLastError` | synchronizing: reports a failed fire-and-forget call, as CUDA reports async errors | |
| `cudaMemcpyToSymbol` / `__constant__` / `__device__` vars | resolves the symbol's device address once per context | `cuModuleGetGlobal` |
| `cudaMallocHost`, `cudaHostRegister`, error strings, `cudaRuntimeGetVersion` | **local**, no round trip | |
| device properties / attributes | fetched once, then cached | |
| any other libcudart export (graphs, IPC, textures, ...) | exported stub returning `cudaErrorNotSupported` (801), logged with `TUNA_LOG=1` | |

Device pointers, streams and events are opaque `uint64` handles the server
issued; the client never dereferences them. The shim exports the **full**
libcudart 12 ABI (435 symbols, version node `libcudart.so.12`), so any program
loads; only what it calls needs to be remoted. Nothing else is exported: the
statically linked tunnel and protobuf cannot clash with an application's.

## cuBLAS (Phase 3d)

NVIDIA's libcublas cannot run on the client: it links its own copy of the
runtime and reaches the GPU through libcuda directly. So Tuna remotes cuBLAS at
its own API, like the runtime: `libcublas.so.12` (Tuna's) exports the full
cuBLAS 12 ABI (742 symbols) and forwards calls to NVIDIA's cuBLAS in the
session's worker, where the app's device pointers and streams are valid as
they are.

- **One session, one order.** The cuBLAS shim has no transport of its own; it
  sends through the runtime shim's exported C bridge `tuna_cudart_call`
  (`include/tuna/cuda/bridge.h`). A `cublasSgemm` is therefore ordered after
  the `cudaMemcpy` that filled its inputs and before the kernel that reads its
  output, exactly as on a local GPU.
- **BLAS calls are fire-and-forget** (`_async`), like kernel launches: gemm,
  gemv, axpy, scal, geam, batched/strided-batched/`Ex` variants, and the
  `cublasSet*` configuration calls. A server-side failure surfaces at the next
  synchronizing CUDA call as the matching `cudaError_t`
  (`CUBLAS_STATUS_INVALID_VALUE` → `cudaErrorInvalidValue`, ...).
- **Scalars follow the pointer mode.** alpha/beta (and results such as
  `cublasSdot`'s) are host values or device addresses depending on the
  handle's `cublasSetPointerMode`, which the client tracks; host values are
  copied into the request, device addresses are passed through. The `Ex`
  functions' `const void*` scalars are sized from the compute type (and C's
  type, for complex).
- **Host ↔ device matrix copies** (`cublasSet/GetMatrix`, `...Vector`, and the
  `Async` forms) pack the host side densely, so any `lda`/`incx` works and the
  payload rides the bulk channel with no padding. A strided vector is a 1 × n
  matrix whose leading dimension is the stride.
- Remoted today (43): handle, stream, workspace, pointer/math/atomics modes,
  S/D level 1 (axpy, scal, copy, dot, nrm2, asum, iamax), S/D gemv, S/D gemm,
  `GemmEx`, S/D `gemmBatched`, `GemmBatchedEx`, S/D `gemmStridedBatched`,
  `GemmStridedBatchedEx`, S/D geam, the matrix/vector copies. Status strings,
  `cublasGetPointerMode` and `cublasGetCudartVersion` are local. Everything
  else (complex types, `_64` APIs, TRSM, ...) returns
  `CUBLAS_STATUS_NOT_SUPPORTED` and is logged with `TUNA_LOG=1`.
- The server opens NVIDIA's libcublas on the first cuBLAS call
  (`"cuda": {"cublas_library": ...}`, default `libcublas.so.12` /
  `cublas64_12.dll`), so sessions that never use it never load it, and it
  refuses Tuna's own cuBLAS shim.

Measured on an RTX 4050 Laptop (loopback, `samples/cublas_sgemm.cu`): 50
back-to-back 1024² SGEMMs run at **5658 GFlop/s natively, 5647 through Tuna
(plaintext) and 5493 (mTLS)**, with bit-identical results: posted calls keep
the GPU fed.

## Files

| File | Role |
|---|---|
| `include/tuna/cuda/marshal.h` | `ArgWriter` / `ArgReader` (fails closed on short buffers) |
| `include/tuna/cuda/client.h` | shim core API; the connection environment variables |
| `include/tuna/cuda/backend.h` | `CudaBackend` (symbol source) and `CudaSession` (server state) |
| `include/tuna/cuda/bridge.h` | `tuna_cudart_call`: the C bridge library shims send through |
| `include/tuna/cuda/cublas_client.h` | cuBLAS shim plumbing (call/post, status mapping, scalar refs) |
| `src/client.cpp` | session (lazy connect from env), call/post, last error |
| `src/client_runtime.cpp` | memcpy family, host memory, caches, error strings |
| `src/client_launch.cpp` | fatbin/kernel/variable registration, launch, symbols |
| `src/client_unsupported.cpp` | the generated not-supported exports |
| `src/cublas_client.cpp` | cuBLAS shim: handles + pointer modes, matrix/vector copies, status strings |
| `src/cublas_unsupported.cpp` | the generated not-supported cuBLAS exports |
| `src/dispatch.cpp` | server: `CudaSession::Dispatch` (generated switch + hand-written calls) |
| `src/dispatch_cublas.cpp` | server: cuBLAS (generated switch + matrix copies, async status mapping) |
| `src/backend_real.cpp` | `dlopen` of NVIDIA's libcudart + libcuda (+ libcublas on first use); refuses to load Tuna's shims |
| `libcudart.map`, `libcublas.map` | Linux version scripts (export only the CUDA / cuBLAS API) |
| `generated/` | committed codegen output (CI checks it is current) |
| `samples/*.cu` | real-GPU milestone programs (deviceQuery, vectorAdd, matrixMul, symbols/streams, cuBLAS sgemm) |
| `tests/test_marshal.cpp`, `tests/test_cublas.cpp` | GPU-free round trips through `CudaSession` + the mock |
| `tests/mock_cuda.cpp` | mock CUDA stack: device memory in host buffers, three kernels, one `__constant__`, a small cuBLAS |
| `tests/e2e_client.cpp`, `tests/e2e_cublas_client.cpp` + `tests/integration/test_cuda_e2e.py` | real binaries end to end (TCP and mTLS), mock instead of a GPU |
| `tests/integration/test_gpu_samples.py` | nvcc-built samples natively and through Tuna on a real GPU |
| `tests/real_gpu_probe.cpp` | in-process probe of the real GPU through the dispatch |

## Build and test

Needs CUDA 12 **headers** (a toolkit, or NVIDIA's redistributable
`cuda_cudart` + `cuda_nvcc` archives); no GPU to build or run the tests. The
cuBLAS shim is built when `cublas_v2.h` is found (in the toolkit include dir,
or point `TUNA_CUBLAS_INCLUDE_DIR` at the `libcublas` redistributable's or the
`nvidia-cublas-cu12` wheel's `include/`); without it, cuBLAS calls report "not
supported" and everything else is unchanged:

```bash
cmake -S tuna-server -B build -DTUNA_SERVER_WORKLOAD_API=OFF \
      -DTUNA_BUILD_CUDA_SHIM=ON -DTUNA_CUDA_INCLUDE_DIR=/path/to/cuda-12/include \
      [-DTUNA_CUBLAS_INCLUDE_DIR=/path/to/cublas/include]
cmake --build build -j
ctest --test-dir build --output-on-failure    # includes tuna_cuda_e2e (mock GPU, + cuBLAS)
```

On a machine with an NVIDIA GPU and `nvcc` (`--cublas-dir`: NVIDIA's
`libcublas.so.12`, e.g. `pip install nvidia-cublas-cu12` → `.../nvidia/cublas/lib`):

```bash
python3 cuda/tests/integration/test_gpu_samples.py --server build/tuna_server \
    --shim-dir build/shim --nvcc /usr/local/cuda/bin/nvcc --cudart-dir /usr/local/cuda/lib64 \
    [--cublas-dir DIR] [--mtls]
```

## Run an application through Tuna (by hand)

```bash
# GPU box: tuna_server --foreground --config server.json, where server.json has
#   "cuda": {"runtime_library": "/usr/local/cuda/lib64/libcudart.so.12",   (optional)
#            "cublas_library": "/usr/local/cuda/lib64/libcublas.so.12"}   (optional)
# Client:
export TUNA_SERVER=gpu-box:7000
export TUNA_CA_CERT=ca.crt TUNA_CLIENT_CERT=alice.crt TUNA_CLIENT_KEY=alice.key  # mTLS
LD_LIBRARY_PATH=/path/to/build/shim ./my_cuda_app
```

Variables: `TUNA_SERVER` (required), `TUNA_CA_CERT` / `TUNA_CLIENT_CERT` /
`TUNA_CLIENT_KEY` (all or none; none only works against a loopback server),
`TUNA_SERVER_NAME`, `TUNA_CONNECT_TIMEOUT_MS`, `TUNA_LOG=1`. Without a
reachable server every call returns `cudaErrorNoDevice`, which applications
already handle as "no GPU".

## Limitations (v1)

- **Server driver ≥ 550 (CUDA 12.4)**: kernel parameter layouts come from
  `cuFuncGetParamInfo`.
- **Relocatable device code** (`-rdc=true`, fatbin wrapper v2) is not
  supported; such kernels fail with `cudaErrorInvalidDeviceFunction`.
- **Unified/managed memory**, mapped host memory, IPC, graphs and textures
  return `cudaErrorNotSupported`. Pinned host memory is ordinary local memory.
- All client threads share one server thread (one CUDA "current device"),
  which is correct for single-device programs; per-thread device state is
  future work.
- A failed launch is reported by the next synchronizing call rather than by
  `cudaLaunchKernel` itself; a launch-configuration error is therefore also
  returned by e.g. `cudaDeviceSynchronize`, where CUDA would only return it
  from the launch or `cudaGetLastError`.
- Libraries that bundle their own GPU code paths without going through
  libcudart need their own API shims. cuBLAS has one (Phase 3d); cuBLASLt,
  cuDNN, cuFFT, cuRAND, cuSPARSE and NCCL do not yet (PyTorch needs several).
- cuBLAS: a BLAS call's argument errors are reported by the next synchronizing
  CUDA call (as a `cudaError_t`), not by the call itself, which returns
  `CUBLAS_STATUS_SUCCESS` once queued. Host↔device matrix copies validate
  their arguments locally and do report `CUBLAS_STATUS_INVALID_VALUE`.

## Add a function

1. Add it to `tools/codegen/annotations.yaml` (runtime) or
   `tools/codegen/annotations_cublas.yaml` (cuBLAS). Every parameter needs a
   role; mark fire-and-forget calls `_async`, synchronizing ones `_sync`;
   cuBLAS scalars are `in_scalar_ref` / `out_scalar_ref`.
2. Regenerate: `python tools/codegen/gen_cuda.py <cuda-12>/include
   tools/codegen/annotations.yaml cuda/generated --cublas <cublas>/include`;
   this also appends the new name to `tools/codegen/api_ids.yaml` (ids are
   append-only, never reused).
3. Commit `cuda/generated/` and `api_ids.yaml`. If the function does not fit
   the simple shapes, list it under `special:` (server + client by hand) or
   `local:` (client only).

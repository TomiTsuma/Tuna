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
`cublas64_12.dll`): see [cuBLAS](#cublas-phase-3d) below. Since Phase 3e it
also replaces the **driver library** (`libcuda.so.1` / `nvcuda.dll`): see
[Driver API](#driver-api-phase-3e). Since Phase 3f it also replaces
**cuBLASLt** (`libcublasLt.so.12` / `cublasLt64_12.dll`): see
[cuBLASLt](#cublaslt-phase-3f). With these four shims, an unmodified PyTorch
on a client with no NVIDIA GPU sees the server's GPU (plan milestone 4).
It also trains a model there, with bit-identical results (milestone 5, MLP).

## What crosses the wire

| Application does | Shim | Server |
|---|---|---|
| a simple runtime call (`cudaMalloc`, `cudaStreamCreate`, ...) | generated stub: flat little-endian args, one round trip | generated `case`: decode → real call → encode outs |
| `cudaLaunchKernel` / `kernel<<<>>>` | configuration checked locally (device limits), then **fire-and-forget**, batched; params packed by value | `cuLaunchKernel` with the layout from `cuFuncGetParamInfo` |
| first launch of a kernel from a fatbin | uploads the fatbin nvcc embedded (once per session), fetches the kernel's param layout (once) | `cuModuleLoadData` per CUDA context, `cuModuleGetFunction` |
| `cudaMemcpy` H2D / D2H | payload on the **bulk channel** (separate socket, 1 MiB chunks) | real `cudaMemcpy` from/into the payload |
| `cudaMemcpyAsync` H2D / D2D, `cudaMemsetAsync`, `cudaEventRecord`, `cudaStreamWaitEvent` | fire-and-forget | real async call |
| `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventSynchronize`, `cudaMemcpy` | synchronizing: reports a failed fire-and-forget call, as CUDA reports async errors | |
| `cudaGetLastError` / `cudaPeekAtLastError` | **local**: the thread's own error, or a fire-and-forget failure whose answer already arrived (with `CUDA_LAUNCH_BLOCKING=1`: asks the server, which flushes) | |
| `cudaGetDevice`, `cudaStreamIsCapturing` | **local** after the first answer (the device changes only through `cudaSetDevice`; graph capture is unsupported) | |
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

## cuBLASLt (Phase 3f)

PyTorch's `nn.Linear` with a bias (`addmm`) runs as one cuBLASLt matmul
with a bias epilogue. NVIDIA's libcublasLt cannot run on a GPU-less client
either, so `libcublasLt.so.12` (Tuna's) exports its full ABI (177 symbols,
version node `libcublasLt.so.12`) and remotes the 13 calls PyTorch makes:
handle create/destroy; create, set-attribute and destroy for matmul
descriptors, matrix layouts and preferences; the heuristic query; and the
matmul.

- **The client names the descriptor objects.** PyTorch builds and destroys a
  matmul descriptor, three layouts and a preference for *every* GEMM, about 15
  calls. The shim allocates each object's id itself, so create, set-attribute,
  destroy and the matmul are all fire-and-forget. A GEMM costs **one round
  trip** (the heuristic query) instead of ~15. The server maps ids to the real
  objects, checks each id is of the expected kind, and refuses unknown ids.
  A failed create therefore surfaces at the next call that uses the object.
- **Handles:** PyTorch passes its cuBLAS handle as the Lt handle, which NVIDIA
  allows. The server accepts the cuBLAS handles it created and handles from
  `cublasLtCreate`, and nothing else.
- **Algos are allowlisted.** `cublasLtMatmulAlgo_t` is an opaque blob that
  cuBLASLt interprets, so a matmul only accepts an algo the heuristic returned
  in this session (or none).
- **Scalars:** alpha/beta follow the descriptor's pointer mode and scale type.
  Both client and server track these from `cublasLtMatmulDescSetAttribute`,
  and the server refuses a scalar encoded the other way.
- Attribute values are capped at 256 bytes and copied at the exact size sent.
  Version, property and status-string queries are local. Everything else is
  `CUBLAS_STATUS_NOT_SUPPORTED`.

## Performance: what a workload pays

`TUNA_STATS=1` prints, at exit, how many calls of each API were blocking
(one round trip each) or posted (batched), and how long the blocking ones
waited. Profiling the MLP training sample with it showed where the time went.
The fixes make the client answer exactly what it already knows:

| per 200 training steps (loopback, RTX 4050) | before | after |
|---|---|---|
| `cudaGetDevice` (PyTorch asks ~280× per step) | 56,735 round trips | 1 (cached) |
| `cudaGetLastError` (after every kernel launch) | 4,026 round trips | 0 (local) |
| `cudaStreamIsCapturing` | 207 round trips | 0 (graph capture is unsupported) |
| all blocking calls | 61,659 | 692 |
| ms/step over 200 steps, incl. first-launch warm-up (native: 4.05) | 31.1 | 8.2 |
| ms/step over 400 steps, 3 runs (native: 2.0 – 2.6) | | **2.8 – 3.6** |

What remains per step is one heuristic query per `nn.Linear` forward GEMM.
Caching heuristic results by descriptor contents is the next Phase 4 step.

## Driver API (Phase 3e)

Programs rarely call the driver API, but frameworks and NVIDIA's libraries
do. PyTorch checks `cuDevicePrimaryCtxGetState` before every op, and makes the
primary context current (`cuCtxGetCurrent` / `cuDevicePrimaryCtxRetain` /
`cuCtxSetCurrent`) before creating a cuBLAS handle. It does both through
`libcaffe2_nvrtc.so`, which links `libcuda.so.1` directly. On a client without
a GPU, that library must be Tuna's too.

- `libcuda.so.1` (Tuna's) exports the full driver ABI (652 symbols, taken from
  the r571 driver) and sends through the runtime shim's bridge, like libcublas.
- Remoted (11): `cuInit`, `cuDriverGetVersion`, `cuDeviceGet`,
  `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetAttribute`,
  `cuDeviceTotalMem`, `cuDeviceGetUuid`, `cuDevicePrimaryCtxGetState`,
  `cuDevicePrimaryCtxSetFlags`, `cuDevicePrimaryCtxRetain`. Attributes are
  cached, and so is a primary-context state once it is *active*. The runtime
  shim's context generation invalidates the latter on `cudaDeviceReset`. So
  PyTorch's per-op check costs no round trip.
- **Contexts are client-side tokens.** A `CUcontext` is a host pointer in the
  server's worker, so it never crosses the wire. The generated driver calls
  take only scalars: `gen_cuda.py` refuses any handle other than
  `CUdeviceptr` / `CUstream`. Only primary contexts exist, one token per
  device. A thread makes a token current itself (`cuCtxSetCurrent`, push/pop).
  A thread that set none sees the runtime's current device's primary context
  once it is active, which is what CUDA does after the runtime's first call.
  Making a token current also switches the runtime's device, since all client
  threads share one server thread.
- `cuGetProcAddress` returns the shim's own exports (the newest `_vN` variant
  of the base name), so nothing that looks up the driver reaches past Tuna.
  Error names and strings are local. Everything else returns
  `CUDA_ERROR_NOT_SUPPORTED` and is logged with `TUNA_LOG=1`.
- The server initializes its driver (`cuInit`) before the first driver call
  and keeps each primary context it retained for the rest of the session. It
  refuses a `driver_library` that is Tuna's own shim.

## Files

| File | Role |
|---|---|
| `include/tuna/cuda/marshal.h` | `ArgWriter` / `ArgReader` (fails closed on short buffers) |
| `include/tuna/cuda/client.h` | shim core API; the connection environment variables |
| `include/tuna/cuda/backend.h` | `CudaBackend` (symbol source) and `CudaSession` (server state) |
| `include/tuna/cuda/bridge.h` | `tuna_cudart_call`: the C bridge library shims send through |
| `include/tuna/cuda/cublas_client.h` | cuBLAS shim plumbing (call/post, status mapping, scalar refs) |
| `include/tuna/cuda/driver_client.h` | driver shim plumbing (call, CUresult codes) |
| `include/tuna/cuda/cublaslt_client.h` | cuBLASLt shim limits (attribute size, heuristic results) |
| `src/client.cpp` | session (lazy connect from env), call/post, last error |
| `src/client_runtime.cpp` | memcpy family, host memory, caches, error strings |
| `src/client_launch.cpp` | fatbin/kernel/variable registration, launch, symbols |
| `src/client_unsupported.cpp` | the generated not-supported exports |
| `src/cublas_client.cpp` | cuBLAS shim: handles + pointer modes, matrix/vector copies, status strings |
| `src/cublas_unsupported.cpp` | the generated not-supported cuBLAS exports |
| `src/driver_client.cpp` | driver shim: context tokens, cached device queries, error strings, `cuGetProcAddress` |
| `src/driver_unsupported.cpp` | the generated not-supported driver exports |
| `src/cublaslt_client.cpp`, `src/cublaslt_unsupported.cpp` | cuBLASLt shim: client-named descriptor objects, scalars by pointer mode; the not-supported exports |
| `src/dispatch.cpp` | server: `CudaSession::Dispatch` (generated switch + hand-written calls) |
| `src/dispatch_cublas.cpp` | server: cuBLAS (generated switch + matrix copies, async status mapping) |
| `src/dispatch_driver.cpp` | server: driver API (generated switch + primary-context retain, device name) |
| `src/dispatch_cublaslt.cpp` | server: cuBLASLt (object table, handle/kind/algo/scalar checks) |
| `src/backend_real.cpp` | `dlopen` of NVIDIA's libcudart + libcuda (+ libcublas on first use); refuses to load Tuna's shims |
| `libcudart.map`, `libcublas.map`, `libcublasLt.map`, `libcuda.map` | Linux version scripts (export only each library's API) |
| `generated/` | committed codegen output (CI checks it is current) |
| `samples/*.cu` | real-GPU milestone programs (deviceQuery, vectorAdd, matrixMul, symbols/streams, cuBLAS sgemm) |
| `samples/torch_*.py` | PyTorch milestone programs (4: matmul; 5: a training loop) |
| `tests/test_marshal.cpp`, `tests/test_cublas.cpp`, `tests/test_cublaslt.cpp`, `tests/test_driver.cpp` | GPU-free round trips through `CudaSession` + the mock |
| `tests/mock_cuda.cpp` | mock CUDA stack: device memory in host buffers, three kernels, one `__constant__`, a small cuBLAS |
| `tests/e2e_client.cpp`, `tests/e2e_cublas_client.cpp`, `tests/e2e_driver_client.cpp` + `tests/integration/test_cuda_e2e.py` | real binaries end to end (TCP and mTLS), mock instead of a GPU |
| `tests/integration/test_gpu_samples.py` | nvcc-built samples natively and through Tuna on a real GPU |
| `tests/integration/test_torch.py` | PyTorch samples natively and through Tuna on a real GPU, with the client's own GPU hidden |
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

PyTorch (milestones 4 and 5) works with any CUDA 12 build of torch, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu128`. The
server's runtime and cuBLAS come from the same install's `nvidia-*` wheels.
The client's own GPU is hidden in a user + mount namespace, so nothing can
silently bypass Tuna:

```bash
python3 cuda/tests/integration/test_torch.py --server build/tuna_server \
    --shim-dir build/shim --python /path/to/venv/bin/python [--only torch_matmul] [--mtls]
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
# Libraries that find CUDA through RPATH (PyTorch's) ignore LD_LIBRARY_PATH:
S=/path/to/build/shim
LD_PRELOAD=$S/libcudart.so.12:$S/libcublas.so.12:$S/libcublasLt.so.12:$S/libcuda.so.1 python train.py
```

On Windows, put the shim DLLs **next to the application's .exe**. Windows
loads `nvcuda.dll` from System32 before anything on `PATH`, so on a machine
with an NVIDIA driver only the application's own directory wins. Phase 5's
`tuna run` will set this up.

Variables: `TUNA_SERVER` (required), `TUNA_CA_CERT` / `TUNA_CLIENT_CERT` /
`TUNA_CLIENT_KEY` (all or none; none only works against a loopback server),
`TUNA_SERVER_NAME`, `TUNA_CONNECT_TIMEOUT_MS`, `TUNA_LOG=1`, `TUNA_STATS=1`
(per-API call statistics at exit), `CUDA_LAUNCH_BLOCKING=1` (error queries
wait for every queued call). Without a
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
- A launch with a bad configuration (block or grid dimensions beyond the
  device's limits) fails in `cudaLaunchKernel` itself, as in CUDA: the client
  checks it against the cached device properties. Any other launch failure
  (too much shared memory, a bad kernel argument) is reported by the next
  synchronizing call. `cudaGetLastError` right after the launch sees such a
  failure only if the server's answer has already arrived; set
  `CUDA_LAUNCH_BLOCKING=1` to have it wait.
- Libraries that bundle their own GPU code paths without going through
  libcudart need their own API shims. cuBLAS (3d) and cuBLASLt (3f) have one.
  cuDNN, cuFFT, cuRAND, cuSPARSE and NCCL do not yet; PyTorch's convolutions
  need cuDNN.
- cuBLASLt: the calls PyTorch's GEMMs make are remoted; descriptor queries
  (`...GetAttribute`), `cublasLtMatmulAlgoInit` and the matrix transforms are
  not yet. A posted call's failure (a bad attribute, a refused algo) is
  reported by the next synchronizing CUDA call, as with cuBLAS.
- Driver API: primary contexts only (`cuCtxCreate` is not supported). Modules
  loaded through the driver (`cuModuleLoadData`, which PyTorch's JIT
  "jiterator" kernels use) are not supported yet. `cuGetProcAddress` ignores
  the per-thread-default-stream flag.
- cuBLAS: a BLAS call's argument errors are reported by the next synchronizing
  CUDA call (as a `cudaError_t`), not by the call itself, which returns
  `CUBLAS_STATUS_SUCCESS` once queued. Host↔device matrix copies validate
  their arguments locally and do report `CUBLAS_STATUS_INVALID_VALUE`.

## Add a function

1. Add it to `tools/codegen/annotations.yaml` (runtime),
   `tools/codegen/annotations_cublas.yaml` (cuBLAS) or
   `tools/codegen/annotations_driver.yaml` (driver). Every parameter needs a
   role; mark fire-and-forget calls `_async`, synchronizing ones `_sync`;
   cuBLAS scalars are `in_scalar_ref` / `out_scalar_ref`.
2. Regenerate: `python tools/codegen/gen_cuda.py <cuda-12>/include
   tools/codegen/annotations.yaml cuda/generated --cublas <cublas>/include`;
   this also appends the new name to `tools/codegen/api_ids.yaml` (ids are
   append-only, never reused).
3. Commit `cuda/generated/` and `api_ids.yaml`. If the function does not fit
   the simple shapes, list it under `special:` (server + client by hand) or
   `local:` (client only).

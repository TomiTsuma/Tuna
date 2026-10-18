# cuda/ — Track B CUDA runtime-API remoting (Phase 3)

Replaces NVIDIA's CUDA **runtime** library (`libcudart.so.12` /
`cudart64_12.dll`) with a Tuna shim that forwards the runtime API over the
Phase 2 tunnel to a `tuna_server` worker running the real runtime on a real
GPU. Unmodified programs built with `nvcc -cudart shared` (and anything else
that loads libcudart dynamically) run on a client with **no NVIDIA GPU**.
This is option **B2** (Linux client first) from
`../docs/plans/2026-10-16-phase3-spike-findings.md`.

```
 app (nvcc -cudart shared)                         tuna_server --foreground
   │ cudaMalloc / <<<kernel>>> / cudaMemcpy          └─ worker process per session
   ▼                                                      └─ CudaSession (dispatch.cpp)
 libcudart.so.12 = Tuna shim  ── TCP + mTLS 1.3 ──►            ├─ NVIDIA libcudart (cudaMalloc, ...)
   client*.cpp, generated stubs   control + bulk sockets       └─ NVIDIA libcuda   (modules, launches)
```

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

## Files

| File | Role |
|---|---|
| `include/tuna/cuda/marshal.h` | `ArgWriter` / `ArgReader` (fails closed on short buffers) |
| `include/tuna/cuda/client.h` | shim core API; the connection environment variables |
| `include/tuna/cuda/backend.h` | `CudaBackend` (symbol source) and `CudaSession` (server state) |
| `src/client.cpp` | session (lazy connect from env), call/post, last error |
| `src/client_runtime.cpp` | memcpy family, host memory, caches, error strings |
| `src/client_launch.cpp` | fatbin/kernel/variable registration, launch, symbols |
| `src/client_unsupported.cpp` | the generated not-supported exports |
| `src/dispatch.cpp` | server: `CudaSession::Dispatch` (generated switch + hand-written calls) |
| `src/backend_real.cpp` | `dlopen` of NVIDIA's libcudart + libcuda; refuses to load the shim itself |
| `libcudart.map` | Linux version script (export only the CUDA API) |
| `generated/` | committed codegen output (CI checks it is current) |
| `samples/*.cu` | real-GPU milestone programs (deviceQuery, vectorAdd, matrixMul, symbols/streams) |
| `tests/test_marshal.cpp` | GPU-free round trip through `CudaSession` + the mock |
| `tests/mock_cuda.cpp` | mock CUDA stack: device memory in host buffers, three kernels, one `__constant__` |
| `tests/e2e_client.cpp` + `tests/integration/test_cuda_e2e.py` | real binaries end to end (TCP and mTLS), mock instead of a GPU |
| `tests/integration/test_gpu_samples.py` | nvcc-built samples natively and through Tuna on a real GPU |
| `tests/real_gpu_probe.cpp` | in-process probe of the real GPU through the dispatch |

## Build and test

Needs CUDA 12 **headers** (a toolkit, or NVIDIA's redistributable
`cuda_cudart` + `cuda_nvcc` archives); no GPU to build or run the tests:

```bash
cmake -S tuna-server -B build -DTUNA_SERVER_WORKLOAD_API=OFF \
      -DTUNA_BUILD_CUDA_SHIM=ON -DTUNA_CUDA_INCLUDE_DIR=/path/to/cuda-12/include
cmake --build build -j
ctest --test-dir build --output-on-failure    # includes tuna_cuda_e2e (mock GPU)
```

On a machine with an NVIDIA GPU and `nvcc`:

```bash
python3 cuda/tests/integration/test_gpu_samples.py --server build/tuna_server \
    --shim-dir build/shim --nvcc /usr/local/cuda/bin/nvcc --cudart-dir /usr/local/cuda/lib64 [--mtls]
```

## Run an application through Tuna (by hand)

```bash
# GPU box: tuna_server --foreground --config server.json, where server.json has
#   "cuda": {"runtime_library": "/usr/local/cuda/lib64/libcudart.so.12"}   (optional)
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
  libcudart (cuBLAS, cuDNN) need their own API shims — next slice.

## Add a function

1. Add it to `tools/codegen/annotations.yaml` (every parameter needs a role;
   mark fire-and-forget calls `_async`, synchronizing ones `_sync`).
2. Regenerate: `python tools/codegen/gen_cuda.py <cuda-12>/include
   tools/codegen/annotations.yaml cuda/generated` — this also appends the new
   name to `tools/codegen/api_ids.yaml` (ids are append-only, never reused).
3. Commit `cuda/generated/` and `api_ids.yaml`. If the function does not fit
   the simple shapes, list it under `special:` (server + client by hand) or
   `local:` (client only).

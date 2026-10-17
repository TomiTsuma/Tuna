# cuda/ — Track B CUDA runtime-API remoting (Phase 3)

Replaces NVIDIA's CUDA **runtime** library (`libcudart.so.12` /
`cudart64_12.dll`) with a Tuna shim that forwards documented runtime API calls
over the Phase 2 tunnel to a server running the real runtime on a real GPU.
This is option **B2** (Linux client first) from
`../docs/plans/2026-10-16-phase3-spike-findings.md`.

Shared by client and server, like `../tunnel/`:

| File | Role |
|---|---|
| `include/tuna/cuda/marshal.h` | flat little-endian arg encoder/decoder |
| `src/client.cpp` | client shim: exported `cudaXxx` symbols → marshal → `RpcClient` |
| `src/dispatch.cpp` | server: decode → real runtime → encode (generated switch + hand-written `cudaMemcpy`) |
| `src/backend_real.cpp` | `dlopen` the real libcudart (server side) |
| `generated/` | committed codegen output (regenerate with `tools/codegen/gen_cuda.py`) |
| `tests/test_marshal.cpp` | GPU-free marshalling round-trip (mock backend) |
| `tests/real_gpu_probe.cpp` | drives a real GPU through the dispatch (needs hardware) |

## Build

Opt-in, needs CUDA 12 headers (not a GPU to build or to run the marshalling
test):

```bash
cmake -S tuna-server -B build -G Ninja -DTUNA_SERVER_WORKLOAD_API=OFF \
      -DTUNA_BUILD_CUDA_SHIM=ON -DTUNA_CUDA_INCLUDE_DIR=/path/to/cuda-12/include
cmake --build build --target tuna_cuda_tests
./build/tuna_cuda_tests            # 25 assertions, no GPU

# With an NVIDIA GPU:
cmake --build build --target tuna_cuda_real_probe
LD_LIBRARY_PATH=/path/to/libcudart/dir ./build/tuna_cuda_real_probe
```

## Add a function

1. Add it to `tools/codegen/annotations.yaml` with its pointer directions.
2. Regenerate: `python tools/codegen/gen_cuda.py <cuda-12>/include \
   tools/codegen/annotations.yaml cuda/generated` and commit `cuda/generated/`.
3. If it is not a simple shape (e.g. direction depends on a runtime arg), list
   it under `special:` and hand-write it in `src/dispatch.cpp` and `src/client.cpp`.

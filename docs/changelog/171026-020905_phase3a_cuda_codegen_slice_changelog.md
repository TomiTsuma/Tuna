# Phase 3a — CUDA runtime codegen + first remoted call on the real GPU (Track B / B2)

**Date:** 2026-10-17
**Branch:** `phase-2-transport` (Phase 3 work continues here until a 3 branch is cut)
**Author:** Thomas Tsuma (with Claude)
**Plan:** `docs/superpowers/plans/2026-10-17-phase-3a-cuda-codegen-slice.md`
**Direction:** owner chose option **B** (replace NVIDIA's runtime library), Linux
client first (**B2**). See `docs/plans/2026-10-16-phase3-spike-findings.md`.

## Why this shape

The Phase 3 spike proved a *driver*-API shim cannot serve NVIDIA's cudart
(integrity-checked internal export tables). B2 instead replaces the *runtime*
library (`libcudart.so.12`) and forwards its documented API to a server running
the real runtime. This slice builds the shared foundation every later Track-B
slice needs — the code generator, the marshalling, the handle model, the
dispatcher — and proves it on the real GPU with one end-to-end call.

## What landed

### Code generator (`tools/codegen/`)
- `gen_cuda.py` parses `cuda_runtime_api.h` with libclang and, cross-referenced
  with `annotations.yaml` (pointer directions/sizes libclang cannot infer),
  emits: the stable `api_id` enum, client marshalling stubs, the server dispatch
  switch, and a coverage name table. 317 runtime functions parsed; 9 remoted so
  far (8 generated, 1 hand-written).
- Encoding is the flat little-endian arg buffer from the wire plan (§4), not
  protobuf-per-call. Device pointers cross as opaque `uint64` handles the
  **server** owns; the client never dereferences them.
- Unannotated functions are meant to generate `cudaErrorNotSupported` stubs so
  coverage grows by use (the dispatcher already returns 801 for unknown ids).
- `cudaGetDeviceProperties` is remoted as its real CUDA 12 symbol
  `cudaGetDeviceProperties_v2` (the plain name is a macro).

### Shared CUDA remoting library (`cuda/`)
- `marshal.h` — `ArgWriter`/`ArgReader` (fails closed on a short buffer).
- `dispatch.cpp` — `CudaDispatch(request, bulk_in, bulk_out, backend)`: the
  generated switch, plus the hand-written `cudaMemcpy` whose direction follows
  `kind` and whose payload rides the Phase 2c bulk channel. Never throws; a
  garbled request becomes `cudaErrorInvalidValue`.
- `client.cpp` — the client shim: exports the runtime symbols (generated stubs +
  hand-written `cudaMemcpy`), marshals each call, and sends it via the Phase 2
  `RpcClient`. A `SetCallHook` test seam routes calls in-process.
- `backend_real.cpp` — the production backend: `dlopen`/`LoadLibrary` of the
  real libcudart, with a symbol cache. Tests use a mock backend instead.
- Generated stubs are **committed** under `cuda/generated/` so the build and CI
  need no libclang or CUDA headers for codegen. `-DTUNA_REGEN_CUDA=ON`
  regenerates them; `cmake/TunaCuda.cmake` wires it all behind
  `-DTUNA_BUILD_CUDA_SHIM=ON`.

### Tests
- `cuda/tests/test_marshal.cpp` (Catch2, **no GPU**): client stub → encode →
  `CudaDispatch` → mock "driver" (a handle table backed by host buffers) →
  encode outs → client decode. Covers scalar outs, the full 1032-byte
  `cudaDeviceProp` struct out, device handles, the bulk H2D/D2H path, an unknown
  api, and a truncated arg buffer. 25 assertions.
- `cuda/tests/real_gpu_probe.cpp` (needs a GPU, not a ctest): the same path with
  the **real** backend.

## Verification (2026-10-17)

| Check | Where | Result |
|---|---|---|
| Codegen | Windows python3.11 + libclang, CUDA 12.8 headers | 9 apis from 317 functions |
| Marshalling test | Windows MinGW 13.1 | 25/25 |
| Marshalling test | WSL Ubuntu 24.04 / GCC 13.3 | 25/25 |
| **Real-GPU probe** | **WSL + RTX 4050 (cc 8.9) via `/dev/dxg`, libcudart.so.12** | **PASS** |
| Existing suites unaffected | Windows server 50/50, guest 49/49 | green |

The real-GPU probe, run through Tuna's generated dispatch (not a mock):
```
device 0: NVIDIA GeForce RTX 4050 Laptop GPU (cc 8.9, 6140 MiB) via Tuna dispatch
1 MiB device round-trip OK; free 5073 -> 5075 MiB after free, total 6140 MiB
PASS: real GPU driven through Tuna's generated CUDA dispatch
```
`cudaGetDeviceProperties` returns the real GPU, `cudaMalloc` consumes real
device memory (reflected in `cudaMemGetInfo`), and a 1 MiB buffer round-trips
H2D→D2H correctly. The transport between shim and server is not in this probe
(it runs the dispatcher in-process); that path is Phase 2, already tested, and
the client stubs call through the same `RpcClient` in production.

## B2 assumption, confirmed and qualified

Linux `torch_cuda.so` dynamically links `libcudart.so.12` and the math `.so`s
(checked by import inspection earlier), so replacing those `.so`s intercepts
PyTorch's own kernels and the math libraries — the premise of B2. This slice did
not yet run PyTorch end to end; that is a later milestone once module/kernel
launch and the cuBLAS/cuDNN API shims exist.

## Not included (next slices)
- **3b — full network path:** route CUDA `api_id`s in the server worker to
  `CudaDispatch` with a per-session real backend; build the client shim as a
  real `libcudart.so.12`; prove an unmodified runtime-API program on the client
  reaching the server's GPU over mTLS with `LD_LIBRARY_PATH`.
- **Kernel launch:** `cuGetProcAddress`-style runtime entry resolution,
  fatbin/module upload + `__cudaRegisterFatBinary`, `cudaLaunchKernel` + param
  marshalling.
- **cuBLAS/cuDNN API shims**, then the PyTorch matmul / training milestones.
- **CI:** the marshalling test needs CUDA 12 headers on the runner (the pip
  runtime wheel's headers are incomplete — missing `crt/`); wiring a toolkit
  header install into CI is a follow-up, so for now it runs on boxes that have
  the headers (verified on Windows and WSL).

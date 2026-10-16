# Phase 3a — CUDA codegen + first remoted call (Track B, option B2)

**Parent plan:** `docs/plans/2026-09-23-tuna-execution-plan.md` §5 Phase 3.
**Spike that set the direction:** `docs/plans/2026-10-16-phase3-spike-findings.md`.
**Owner decision (2026-10-16/17):** option **B** (replace NVIDIA's runtime
library, not just the driver library), starting with a **Linux client (B2)**.

## Why B2, restated

The spike proved a driver-only shim cannot serve NVIDIA's cudart (six
integrity-checked internal export tables). On Windows, PyTorch statically links
cudart into `torch_cuda.dll` plus each math library, so there is no runtime DLL
to replace. On **Linux**, the pip PyTorch wheels ship `libcudart.so.12`,
`libcublas.so.12`, `libcudnn*.so.9`, etc. as **separate shared objects** that
`torch_cuda.so` links dynamically — so replacing those `.so`s (via
`LD_LIBRARY_PATH` / `LD_PRELOAD`) intercepts PyTorch's own kernels and the math
libraries at a documented API boundary.

**Assumption to verify before building (A0 below): Linux `torch_cuda.so`
resolves its CUDA runtime and math calls through the separate `.so`s, not a
static copy.** If false, B2 does not work either and we fall back to option C.

## Scope of this slice (3a)

Not all of Phase 3. This slice delivers the **shared foundation** every later
Track-B slice needs, and proves it on the real GPU with one end-to-end call:

1. **A0 — verify the assumption.** On WSL (real RTX 4050 via `/dev/dxg`):
   stock `torch.cuda` works; `ldd`/symbol inspection confirms `torch_cuda.so`
   imports `libcudart.so.12`; a trivial `LD_PRELOAD` runtime stub is entered by
   a real runtime-API program. If A0 fails, stop and report.
2. **Codegen (`tools/codegen/`, Python).** Parse `cuda_runtime_api.h` (and later
   `cuda.h`) and an `annotations.yaml` of pointer directions/sizes. Emit:
   - the `api_id` enum (stable, generated),
   - client marshalling stubs (serialize args → `RpcRequest.args`, the flat
     little-endian encoding from the wire plan §4),
   - the server dispatch table (decode → call the real function → encode outs),
   - per-function marshalling round-trip tests.
   Functions without an annotation emit a stub returning
   `cudaErrorNotSupported` and logging the name, so coverage grows by use.
3. **First remoted functions (hand-annotated):** `cudaGetDeviceCount`,
   `cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceProperties`, `cudaMalloc`,
   `cudaFree`, `cudaMemcpy` (H2D/D2H, bulk channel for the payload). Device
   pointers are opaque `uint64` handles the **server** owns; the client never
   dereferences them (same rule as the driver-API handle plan).
4. **Server dispatcher** plugged into the Phase 2 seam (`dispatch.cpp`):
   `MakeDispatcher` gains a CUDA backend that `dlopen`s the real runtime and
   maps `api_id` → real call. One CUDA context per session (the worker process
   Phase 2c already isolates).
5. **Client shim `libcudart.so.12`:** exports the generated symbols, connects
   with `ConnectSession()` (Phase 2c), forwards each call. Loaded via
   `LD_LIBRARY_PATH` ahead of torch's bundled copy.
6. **Milestone (integration test, real GPU, self-hosted/WSL only):** a small
   C program using only the runtime API — `cudaGetDeviceProperties` prints the
   server's GPU name; `cudaMalloc`+`cudaMemcpy` round-trips a buffer — run on
   the client with the shim, executing on the server's real GPU. The server's
   CI mock backend (canned results) keeps this testable on GitHub runners.

Deferred to later 3 slices: `cuGetProcAddress`-style runtime entry resolution,
fatbin/module upload + `__cudaRegisterFatBinary`, `cudaLaunchKernel` + kernel
param marshalling, the cuBLAS/cuDNN API shims, and the PyTorch matmul milestone.

## Risks specific to 3a

| Risk | Mitigation |
|---|---|
| A0 false (static cudart on Linux too) | Verify first. Fall back to option C. |
| `cudaMemcpy` of device pointers needs the server to hold real allocations keyed by the handle we gave the client | Server-side handle table per session; the client's `void*` is an index, never a real address. |
| Runtime version skew (client torch cudart 12.x vs server libcudart) | Server reports its runtime version in `Hello`; shim refuses a mismatch it can't satisfy. |
| No MSVC on this box; nvcc needs a host compiler | B2 is Linux-first: gcc 13.3 in WSL builds both the shim and the test app; the server runs real CUDA via torch's bundled `libcudart.so.12`, so no full toolkit install is required for the slice. |

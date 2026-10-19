# Phase 3f — cuBLASLt remoting; PyTorch training on a GPU-less client (plan milestone 5, MLP)

**Date:** 2026-10-19
**Branch:** `phase-2-transport`
**Author:** Thomas Tsuma (with Claude)
**Builds on:** `191026-205116_phase3e_driver_shim_pytorch_changelog.md`
**Execution plan:** `docs/plans/2026-09-23-tuna-execution-plan.md` §5 Phase 3, milestone 5
("A small PyTorch training loop (MNIST/ResNet-18) runs to completion"), plus
the first Phase 4 item ("measure the fraction of calls that are synchronous
for each workload")

## Why this was the next step

3e passed milestone 4. The milestone 5 sample (`cuda/samples/torch_train_mlp.py`,
a 3-layer MLP trained with Adam) then segfaulted on its first `nn.Linear`.
A native backtrace (a throwaway SIGSEGV-handler preload; no gdb on the box)
showed NVIDIA's real `libcublasLt` crashing in `cublasLtMatmulAlgoGetHeuristic`.
`addmm` with a bias runs as a cuBLASLt matmul with a bias epilogue, and the
real library cannot work without a local GPU.

## Outcome

Real GPU (WSL Ubuntu 24.04, RTX 4050 Laptop, driver 571.96 / CUDA 12.8,
loopback). PyTorch 2.8.0+cu128 is unmodified, and the client has **no GPU**
(user + mount namespace, see 3e):

| | native | through Tuna |
|---|---|---|
| `torch_train_mlp.py`: 200 Adam steps, 3-layer MLP, 11 sampled losses | 2.3153 → 0.00221 | **bit-identical** (all 11) |
| train accuracy | 1.0000 | 1.0000 |
| steady state, ms/step over 400 steps (3 runs each) | 2.0 – 2.6 | **2.8 – 3.6** |
| `torch_matmul.py` checksum (milestone 4) | -127810.83994067139 | identical |

The comparison holds over plaintext and mTLS. **Milestone 5 passes for an MLP.**
A conv net (the plan's ResNet-18) still needs cuDNN; see "Not included".

## What changed

### cuBLASLt shim: `libcublasLt.so.12` / `cublasLt64_12.dll` (new)

PyTorch imports 11 cuBLASLt functions: create, set-attribute and destroy for
matmul descriptors, matrix layouts and preferences; the heuristic query; and
the matmul. It never calls `cublasLtCreate`, because it casts its cuBLAS
handle to an Lt handle, which NVIDIA allows. The shim exports the full ABI
(177 symbols, version node `libcublasLt.so.12`). It remotes those 11 calls
plus `cublasLtCreate` / `cublasLtDestroy`, all by hand. Version, property and
status-string queries are answered locally.

**Design: the client names the descriptor objects.** PyTorch builds and
destroys about 15 objects and attributes for *every* GEMM. If each create
returned a server pointer, every `nn.Linear` would pay ~15 round trips.
Instead the client allocates each object's id, so create, set-attribute,
destroy and the matmul itself are all fire-and-forget. **A GEMM costs one
round trip** (the heuristic query), which the test suite asserts.

**Server checks** (`cuda/src/dispatch_cublaslt.cpp`). Every value here comes
off the wire:
- **Objects:** an id maps to the real object this session created, of the
  kind the call expects. A layout id is never used as a matmul descriptor, and
  unknown ids are refused.
- **Handles:** only the cuBLAS handles this session created, plus handles
  from `cublasLtCreate`.
- **Algos:** `cublasLtMatmulAlgo_t` is an opaque blob that cuBLASLt
  interprets. A matmul only accepts an algo the heuristic returned in this
  session (an allowlist capped at 65,536 entries), or none.
- **Scalars:** alpha/beta must be encoded the way the descriptor's pointer
  mode says (host value or device address). Client and server both track the
  pointer mode and scale type from `SetAttribute`.
- **Attribute values:** capped at 256 bytes and copied at exactly the size
  sent, so cuBLASLt's own size check sees the truth.

**Backend:** on Linux, the cuBLAS handle also resolves the Lt symbols through
its dependency. Windows' `GetProcAddress` does not search dependencies, so
the backend opens `cublasLt64_12.dll` from next to the configured cuBLAS, and
refuses Tuna's own shim.

### Performance: what a training step paid, and the fixes

A new `TUNA_STATS=1` prints per-API blocking/posted counts and blocking wait
time at exit. On the MLP (200 steps) the first run showed:

| | blocking calls | wait |
|---|---|---|
| `cudaGetDevice` (~280 per step) | 56,735 | 5.9 s |
| `cudaGetLastError` (after every kernel launch) | 4,026 | 0.55 s |
| `cudaStreamIsCapturing` | 207 | 0.02 s |
| `cublasLtMatmulAlgoGetHeuristic` | 603 | 0.20 s |
| **total** | **61,659** | 31.1 ms/step |

The fixes have the client answer exactly what it already knows:
- **`cudaGetDevice` is cached** after the first answer. The device changes
  only through `cudaSetDevice`, which the shim sees, and all client threads
  share one server thread.
- **`cudaGetLastError` / `cudaPeekAtLastError` are local.** They return the
  thread's own error, or a posted call's failure whose answer has already
  arrived (new `RpcClient::ArrivedAsyncError`, with no flush and no round
  trip). `CUDA_LAUNCH_BLOCKING=1` restores the server round trip for
  debugging, which is CUDA's own convention.
- **Launch configuration is validated in `cudaLaunchKernel`**, against the
  cached device limits (block and grid dimensions). This is what CUDA checks
  synchronously, so `kernel<<<1, 4096>>>; cudaGetLastError()` reports
  `cudaErrorInvalidConfiguration` immediately, as natively. That is *better*
  than before, when it surfaced at the next sync. The nvcc sample
  `symbols_streams` checks exactly this, and the regression run caught it.
- **`cudaStreamIsCapturing` is local:** graph capture is unsupported, so no
  stream is ever capturing.

Result: blocking calls 61,659 → **692**. The 200-step run went from 31.1 to
8.2 ms/step (native 4.05; both include first-launch warm-up), and steady
state is 2.8–3.6 ms/step vs native 2.0–2.6. What remains per step is one
heuristic query per forward `nn.Linear`.

### Codegen

- `annotations_cublaslt.yaml` lists the hand-written (`special:`) and local
  names. `generate_cublaslt` checks them against `cublasLt.h`, assigns ids
  (appended to `api_ids.yaml`) and emits `cublaslt_unsupported_stubs.inc` from
  `cublasLt12_exports.txt`.
- `cudaGetDevice` became `_wrap` (cached). `cudaStreamIsCapturing` moved to
  `local:`. Its id stays in the append-only registry, unused.

### Tests

- `cuda/tests/test_cublaslt.cpp` (4 cases, in-process through the mock):
  - PyTorch's addmm sequence: a bias epilogue with the cuBLAS handle as the Lt
    handle, results checked against the CPU, and **exactly one blocking call
    for the whole GEMM**;
  - device pointer mode with a ReLU+bias epilogue, its own Lt handle and no
    algo;
  - refusals: unknown handle, an object of the wrong kind, a forged algo (the
    mock's own tag, so only the server's allowlist stops it), a scalar encoded
    against the pointer mode, a destroyed descriptor, oversized or missing
    attribute values;
  - local queries.
- `test_marshal.cpp`: a new case asserts that `cudaGetDevice`,
  `cudaGetLastError` and `cudaStreamIsCapturing` cost no round trip over 100
  launches, that bad configurations fail at the launch and never reach the
  server, and that an arrived device-side failure is reported locally exactly
  once. The async-failure tests now use oversized dynamic shared memory: a
  failure only the device side can see, which the mock now models.
- The mock gained a cuBLASLt: fp32 matmul with transposes, bias/ReLU
  epilogues and both pointer modes, layouts, preferences, a tagged heuristic
  result. It also gained real device limits (`maxThreadsDim`, `maxGridSize`).
- `e2e_cublas_client` also runs a cuBLASLt bias GEMM through the real
  `libcublasLt.so.12` shim, over TCP and mTLS.
- `test_torch.py` preloads `libcublasLt.so.12` too.

## Verification (2026-10-19)

| Check | Where | Result |
|---|---|---|
| Full ctest (incl. 4 cuBLASLt cases, the new round-trip case, Lt in the cuBLAS e2e) | WSL Ubuntu 24.04 / GCC 13.3 | 90/90 |
| Full ctest (incl. `cublasLt64_12.dll`) | Windows MinGW 13.1, CUDA 12.8 headers | 90/90 |
| nvcc GPU samples (milestones 1–3), plaintext and mTLS | WSL + RTX 4050 | 5/5 + 5/5 |
| `test_torch.py` (milestones 4 + 5), plaintext and mTLS, no client GPU | WSL + RTX 4050 | 2/2 + 2/2, identical results |

## Design decisions and tradeoffs

- **Client-named objects instead of server handles.** This turns ~15 round
  trips per GEMM into one. The cost is that a failed create is reported late,
  at the next call that uses the object (or the next sync). That matches how
  Tuna already treats posted BLAS calls, and cuBLASLt creates only fail on
  allocation failure or bad enums.
- **The algo allowlist** costs a set lookup per matmul and a bounded set per
  session. In return, the server never hands cuBLASLt an algo blob a client
  forged. If the cap is ever hit, new algos simply stop being accepted, which
  fails closed.
- **Local `cudaGetLastError`.** This is the one semantic trade in this phase.
  A device-side launch failure (not a configuration error, which is now
  checked locally like CUDA) that is still in flight is reported by the next
  synchronizing call rather than by an immediately following
  `cudaGetLastError`. CUDA itself reports kernel *execution* errors only at
  synchronization, so only Tuna-specific failures are deferred.
  `CUDA_LAUNCH_BLOCKING=1` brings back exact reporting at one round trip per
  query. The alternative cost 4,026 round trips per 200 steps.
- **Caching `cudaGetDevice` process-wide** is exact under Tuna's documented
  model (all client threads share one server thread and thus one current
  device). It will need to become per-thread together with that model.

## Not included (next)

- **cuDNN**: convolutions, so a ResNet-18 for the second half of milestone 5.
  It is descriptor-based like cuBLASLt, and the same client-named-object
  design applies.
- **Heuristic caching**: one round trip per forward `nn.Linear` is now the
  dominant blocking call. Caching results keyed by descriptor and layout
  contents (they are pure functions of them) would remove it.
- cuBLASLt `...GetAttribute`, `cublasLtMatmulAlgoInit` and the matrix
  transforms, when a workload needs them.
- Driver-loaded modules (jiterator), cuRAND/cuFFT/cuSPARSE/NCCL; Phase 4
  zero-copy bulk and network benchmarks; Phase 6 handle tables for runtime
  streams/events (all as listed in 3d/3e).

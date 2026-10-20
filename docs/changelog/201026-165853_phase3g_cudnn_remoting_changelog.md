# Phase 3g — cuDNN remoting; ResNet-18 trains on a GPU-less client (plan milestone 5 complete)

**Date:** 2026-10-20
**Branch:** `phase-2-transport`
**Author:** Thomas Tsuma (with Claude)
**Builds on:** `191026-212328_phase3f_cublaslt_training_changelog.md`
**Execution plan:** `docs/plans/2026-09-23-tuna-execution-plan.md` §5 Phase 3, milestone 5
("A small PyTorch training loop (MNIST/ResNet-18) runs to completion")

## Why this was the next step

3f passed milestone 5 for an MLP. A conv net needs cuDNN: PyTorch runs
convolutions and batch norm through it, and NVIDIA's `libcudnn` cannot run on
a client without a GPU. So cuDNN needed a shim of its own, like cuBLAS and
cuBLASLt.

## Outcome

Real GPU (WSL Ubuntu 24.04, RTX 4050 Laptop, driver 571.96 / CUDA 12.8,
loopback). PyTorch 2.8.0+cu128 (cuDNN 9.10.2) is unmodified, and the client
has **no GPU** (user + mount namespace, see 3e):

| `torch_train_resnet.py` (ResNet-18, SGD + momentum, batch 32, 3x32x32) | native | through Tuna |
|---|---|---|
| 30 steps, 7 sampled losses, deterministic cuDNN | 2.7404 → 0.0148 | **bit-identical** (all 7) |
| ms/step over 200 steps, incl. plan building | 24.2 – 24.6 | **26.7** (+9%) |
| blocking round trips per steady-state step | | 1.4 (cuDNN: **0**) |
| one-time setup (32 execution plans) | | ~2,100 blocking calls, ~1.6 s waited |

The comparison holds over plaintext and mTLS (2 runs each). `torch_matmul`
and `torch_train_mlp` remain bit-identical, and the 5 nvcc samples pass over
both transports. **Milestone 5 now passes as written: the plan's ResNet-18
trains to completion through Tuna.**

## Step 1: measure what PyTorch calls

`tools/cudnn_trace/` (new, Linux) is a pass-through `libcudnn.so.9`. Every
export becomes a two-instruction thunk (count the call, jump to the real
function), as in `nvcuda_trace`. The six backend-API calls are wrapped in C, so
the report shows descriptor types, attribute names and element types.

PyTorch 2.8 imports 83 cuDNN functions. ResNet-18 calls 16 of them. Comparing
2- and 6-step runs:

| per training step | calls |
|---|---|
| convolutions: `cudnnBackendExecute` (20 forward, 19 backward-data (none for the input image), 20 backward-filter) | 59 |
| a variant pack each: create, 3 x `SetAttribute` (unique ids, data pointers, workspace), finalize, destroy | 59 x 6 |
| batch norm: `cudnnBatchNormalizationForwardTrainingEx` + `BackwardEx` | 20 + 20 |
| batch-norm workspace / reserve size queries | 60 |
| legacy tensor descriptors: create, `SetTensorNd`, destroy | 120 x 3 |
| `cudnnSetStream` | ~120 |

One-time setup builds 32 execution plans through the cuDNN frontend. That
takes ~2,900 descriptor creates (mostly knob choices and engine configs the
frontend pre-creates for `GetAttribute` to fill), ~1,800 `GetAttribute` and
256 non-variant-pack finalizes. Nothing reads `VOID_PTR` or handle
attributes, and nothing uses by-value tensors.

So everything a training step calls can be fire-and-forget, and the design
follows from that.

## What changed

### cuDNN shim: `libcudnn.so.9` / `cudnn64_9.dll` (new)

The shim exports the full ABI (224 symbols, version node `libcudnn.so.9`) and
remotes 22 calls by hand (`cuda/src/cudnn_client.cpp`). It answers 8 locally;
the other 194 are generated stubs returning `CUDNN_STATUS_NOT_SUPPORTED`.

- **Client-named objects** (the cuBLASLt design). Backend descriptors and
  legacy tensor descriptors get ids from the client, so create, set and
  destroy are posted. A `CUDNN_TYPE_BACKEND_DESCRIPTOR` attribute crosses as
  ids and is translated on the server. `GetAttribute` into caller-created
  descriptors (heuristic results into engine configs, a config's engine)
  sends those ids, and cuDNN fills the server's real objects.
- **Finalize is synchronous, except for variant packs.** The frontend tries
  engine configs in turn and needs each plan's real status. A variant pack
  only bundles pointers, so its finalize is posted, like `Execute`. Steady-state
  convolutions therefore cost no round trip. The client knows each id's
  descriptor type from its create call.
- **Batch-norm size queries are cached.** The client mirrors each legacy
  tensor descriptor (it set every value itself). A size is keyed by handle,
  mode, ops and the descriptors' contents, which is exact: the size is a pure
  function of those. The cache keeps at most 4,096 entries. 60 round trips per
  step become 0 after the first step.
- **The server's version.** `cudnnGetVersion`, `cudnnGetMaxDeviceVersion` and
  `cudnnGetCudartVersion` ask the server once and cache the answer, so the
  frontend's runtime version checks see the cuDNN that actually executes.
  Without a server they fall back to the build's headers. `cudnnGetProperty`
  derives from the same value.
- Local: status strings (generated from `cudnnStatus_t`), `cudnnGetStream`
  (mirrored from `cudnnSetStream`), `cudnnGetTensorNdDescriptor` (mirror),
  and the four sub-library version checks (always consistent).
  `cudnnGetLastErrorString` fetches the server's message, since it is only
  called on error paths.

### Server: `cuda/src/dispatch_cudnn.cpp` (new)

Every value comes off the wire, so the server enforces:
- **Objects:** an id must be this session's object of the expected kind. A
  legacy tensor descriptor is never accepted as a backend descriptor. cuDNN
  checks the backend descriptor *type* itself.
- **Handles:** only `cudnnHandle_t` values this session created, whether
  passed as an argument or inside a `CUDNN_TYPE_HANDLE` attribute (graph and
  plan handles).
- **Attribute values:** an element type Tuna does not know is refused (fail
  closed). The element-size table lives in `cudnn_client.h` and is shared by
  both sides, with `static_assert`s against the real enum. The payload must
  be exactly count x size bytes, and it is copied into an 8-byte-aligned
  buffer of that size.
- **Host memory:** cuDNN dereferences two kinds of pointer in the worker.
  Scalars (batch-norm alpha/beta) are copied into local storage (`ScalarRef`).
  By-value tensors (`CUDNN_ATTR_TENSOR_IS_BY_VALUE`) make a variant pack's
  "data pointer" host memory, so they are refused on both sides.
  `CUDNN_TYPE_VOID_PTR` (device addresses) is only accepted by a variant pack.
- Activation descriptors are not supported. None can be created, so every
  activation argument must be null.

Posted failures map to the closest `cudaError_t` by cuDNN 9's status category
(`CudnnAsyncError`). The app sees them at its next synchronizing CUDA call, as
with cuBLAS.

### Backend, config, build

- `backend_real.cpp` opens `libcudnn.so.9` / `cudnn64_9.dll` on the first
  cuDNN call, and refuses Tuna's own shim (`tuna_cudnn_shim`). cuDNN loads its
  sub-libraries (`_graph`, `_ops`, `_cnn`, engines) from its own directory.
- Server config: `"cuda": {"cudnn_library": ...}`.
- CMake: `TUNA_CUDNN_INCLUDE_DIR` (cuDNN 9 is checked), target `tuna_cudnn`,
  `cuda/libcudnn.map`. Without the headers, cuDNN calls report not supported
  and nothing else changes.
- Codegen: `--cudnn DIR` checks `annotations_cudnn.yaml` against `cudnn.h` and
  emits `cudnn_unsupported_stubs.inc` (from `cudnn9_exports.txt`) and
  `cudnn_status_names.inc`. New ids 122–143 were appended to `api_ids.yaml`.
- CI (`linux-cuda-remoting`): fetches and caches the cuDNN 9.10.2.21 headers
  from NVIDIA's redistributable, runs the codegen freshness check with
  `--cudnn`, and builds with the cuDNN shim.

### Tests

- `cuda/tests/test_cudnn.cpp` (4 cases, in process through the mock):
  - **A backend-API convolution as the frontend builds it:** tensors,
    convolution, operation, graph, heuristic, results read into caller-created
    engine configs, a config's engine read back (a real copy), the plan and
    its workspace size. Three executions with fresh variant packs cost **no
    blocking call**, and the output matches a CPU reference.
  - **Batch norm:** forward training, backward and inference match CPU
    references, including running statistics. The three size queries cost
    one round trip each, and the same shapes again cost none (the mock counts
    what reaches it).
  - **Refusals:** a forged descriptor element, a legacy descriptor used as a
    backend one, a foreign handle attribute, a `VOID_PTR` outside a variant
    pack, a by-value tensor (refused by the client before anything is sent,
    and by the server when sent anyway), an unknown attribute type, forged
    output descriptors, an activation descriptor, a destroyed descriptor, a
    short wire buffer, and the server's error message reaching
    `cudnnGetLastErrorString`.
  - **Local answers:** version (one round trip, then cached), properties,
    stream, tensor descriptor, status names.
- The mock gained a cuDNN (`mock_cuda.cpp`): generic attribute storage, a
  single-convolution graph executed on the host, and spatial batch norm. It
  checks pointer-typed elements, so a translation slip fails visibly.
- `test_torch.py` preloads `libcudnn.so.9` and points the server at the
  wheel's cuDNN. `torch_train_resnet.py` now runs cuDNN in deterministic mode
  and prints exact `loss step=` lines (see the decision below).
- Unrelated fix found on the way: `tuna-server/tests/test_config.cpp` wrote
  every case's config to one temp file, so the cases raced under
  `ctest -j` (one failure in a full run). Each case now gets its own file.

## Verification (2026-10-20)

| Check | Where | Result |
|---|---|---|
| Full ctest (incl. 4 cuDNN cases) | WSL Ubuntu 24.04 / GCC 13.3 | 94/94 (x4 runs) |
| Full ctest (incl. `cudnn64_9.dll`) | Windows MinGW 13.1, CUDA 12.8 + cuDNN 9.10 headers | 94/94 |
| nvcc GPU samples (milestones 1–3), plaintext and mTLS | WSL + RTX 4050 | 5/5 + 5/5 |
| `test_torch.py` (matmul, MLP, ResNet-18), plaintext and mTLS, no client GPU | WSL + RTX 4050 | 3/3 + 3/3, twice each, bit-identical |
| Committed codegen output current (libclang 18.1.1, as CI) | WSL | yes |

## Design decisions and tradeoffs

- **Deterministic cuDNN in the ResNet test, not a tolerance.** The first
  mTLS run failed a tolerance check, and Tuna's losses looked more spread
  than native ones. The hypothesis was that the server worker might lack
  NVRTC, so cuDNN would skip its runtime-compiled engines and pick different,
  noisier ones.
  With `torch.backends.cudnn.deterministic = True`, native and Tuna were
  **bit-identical** with or without NVRTC on the server. The spread was
  ordinary atomics nondeterminism, and the native runs had clustered by
  chance (more native runs spread just as wide). An exact comparison is the
  stronger test, so the sample now uses it.
- **Posted variant-pack finalize** trades error timing for zero round trips:
  a bad pack (a missing pointer) surfaces at the next synchronizing call, not
  at finalize. Plan finalizes stay synchronous because the frontend's engine
  fallback depends on their status.
- **Plan building is chatty:** ~65 blocking calls per convolution shape,
  ~2,100 for ResNet-18, ~1.6 s on loopback. Paid once per shape (PyTorch
  caches plans), but on a 1 ms LAN that is ~2 s of warm-up. Batching
  `GetAttribute` reads, or caching heuristic results by graph, is Phase 4
  work.
- **By-value tensors are refused** rather than supported by shipping the
  value: nothing PyTorch's convolutions do needs them, and refusing keeps
  every host pointer cuDNN sees inside Tuna's own storage. Fused
  conv+bias+activation graphs that pass scalars by value will report
  `CUDNN_STATUS_NOT_SUPPORTED` until then.
- The server never learns which backend descriptor type it is handed beyond
  the create call. cuDNN validates descriptor types itself, and the server
  only guarantees the pointer is a live object of this session.

## Not included (next)

- cuDNN APIs beyond what PyTorch's convolutions and batch norm use: the legacy
  convolution API (`TORCH_CUDNN_V8_API_DISABLED=1`), pooling, activation,
  dropout, RNN, CTC, spatial transformer. They are exported as not-supported
  stubs, and `TUNA_LOG=1` names them.
- A GPU-free end-to-end cuDNN client over TCP (cuDNN is covered in process
  and on the real GPU over TCP/mTLS, not in `tuna_cuda_e2e`).
- Phase 4: cheaper plan building, cuBLASLt heuristic caching (now the only
  per-step blocking call besides the app's own syncs), network benchmarks.
- Windows: the shim builds and its tests pass, but no Windows PyTorch run has
  been made (that needs Phase 5's `tuna run` to load the shims).

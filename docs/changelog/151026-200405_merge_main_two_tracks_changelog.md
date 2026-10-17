# Merge `main` (PR #1) into Phase 1 — two-track consolidation

**Date:** 2026-10-15
**Branch:** `phase-1-core-plumbing`
**Author:** Thomas Tsuma (with Claude)

## Why

`main` moved while this branch was in progress. PR #1 (merged 2026-10-04, built in
a Copilot worktree) shipped an **explicit-workload gRPC/mTLS MVP**: a Windows client
and Qt GUI, plus a Linux server offering `RemoteSumOfSquares` and a CUDA
`RemoteMatrixMultiply`. It also rewrote the guest-agent docs to say that Tuna does
not intercept applications. This branch had meanwhile made the **CUDA-driver-API
remoting** plan canonical and built its Phase 1 core. Neither line knew about the
other: local `main` had not been fetched when Phase 1 started.

The two lines conflicted in docs, CMake, layout and protocol, and the merge would
have produced two `tuna-server`s. The owner chose to **keep both tracks**:

- **Track A — explicit workload API** (PR #1). The working, shippable product.
- **Track B — driver-API remoting** (`docs/plans/2026-09-23-tuna-execution-plan.md`).
  The transparent-acceleration track. This branch's Phase 1 belongs to it.

## Changes

### Merge resolution
- Took PR #1's versions of the guest-agent docs (they describe Track A accurately),
  the GUI rewrite, `LICENSE`, and the removal of `mainwindow.ui`, the scripts and the
  `CMakeLists.txt.user`.
- Kept this branch's real Phase 1 tests (`test_config.cpp` and others). PR #1 had
  deleted the empty originals.
- Accepted PR #1's "no tracked placeholders" rule throughout. Deleted the 0-byte
  `src/{daemon,hooks,tunnel}/*` and `src/core/platform_utils.cpp` stubs from the
  Phase 0 restructure, the `1+1==2` scaffolding tests (`test_hooks`, `test_tunnel`,
  `test_full_agent`, `mock_server.py`, server `test_smoke`), the now-empty pytest
  harness (`pytest.ini`, `requirements.txt`), and the unused `hook_handlers.h`
  (`NtCreateFile` file hooks were dropped by the canonical plan). Later phases create
  real files when they need them.

### One server: `tuna-guest-agent/tuna-server/` → `tuna-server/`
- PR #1's server sources moved to `tuna-server/src/workload/`, with history kept via
  `git mv`.
- Its `main()` became `RunWorkloadServer()` (`workload_server.{h,cpp}`). The single
  `tuna_server` binary dispatches as follows: `--foreground` runs Track B, any other
  command line runs Track A. The systemd unit, the mTLS test and the docs keep working
  unchanged.
- New option `TUNA_SERVER_WORKLOAD_API` (default ON on Linux, OFF on Windows) builds
  Track A. `TUNA_ENABLE_CUDA` now requires it.
- PR #1's `tuna_server_tests` was renamed `tuna_workload_tests`, so it no longer
  clashes with the Track B Catch2 suite.

### Shared protocol and CMake
- `tuna-guest-agent/protocol/tuna/v1/workload.proto` → `proto/tuna/v1/workload.proto`,
  so all wire contracts now live in the root `proto/`.
- New `cmake/TunaProtobuf.cmake`. It defines `tuna_proto` (the frame protocol) for
  both projects and picks **one** protobuf per build: the system/vcpkg protobuf when
  gRPC is present, otherwise a pinned FetchContent v21.12. Linking both into one
  binary would be an ODR violation. The include guard lets the guest-agent superbuild
  reuse the targets inside the server subdirectory.
- The guest-agent `CMakeLists.txt` keeps PR #1's option structure (no hardcoded Qt
  path; GUI, sample client and server are toggles) and always builds the Track B core
  and tests. Catch2 is built only when `BUILD_TESTING` is on.

### CI
- PR #1's workflow lived at `tuna-guest-agent/.github/workflows/`. GitHub only reads
  the repository-root `.github/`, so **that CI had never run**. It is now at
  `.github/workflows/build.yml` with `working-directory: tuna-guest-agent`.
- Added `ctest` to the Windows client job, and a new `windows-server-core` job for
  the gRPC-free Track B server.

### Docs
- New root `README.md`: the two tracks side by side, the layout, and quick-start
  commands.
- `tuna-guest-agent/docs/security_model.md`: added a **Track B draft**. This was the
  last outstanding Phase 1 deliverable. It covers the trust model (authenticated
  clients run arbitrary GPU code by design), mTLS, a per-session worker process, the
  accepted/rejected execution table, memory scrubbing, per-identity quotas, the
  fuzzing attack surface, and release blockers.
- `tuna-server/README.md`: a two-track preamble, PR #1's operator guide (unchanged),
  and a Track B section.
- `tuna-guest-agent/README.md` and the architecture, protocol, daemon and
  implementation-plan docs: paths updated for the moves, plus a Track A/B scoping note.
- Execution plan status: "canonical for Track B", with notes on how it interacts with
  Track A.

## Verification (2026-10-15)

- **Windows, MinGW 13.1 + Ninja, Track B only** (FetchContent protobuf):
  - guest agent with `TUNA_BUILD_SAMPLE_CLIENT=OFF`: `ctest` **13/13**
  - `tuna-server` with `TUNA_SERVER_WORKLOAD_API=OFF`: `ctest` **9/9**
  - Phase 1 had 16 and 10; the difference is exactly the deleted `1+1==2` placeholders.
- **WSL Ubuntu 24.04, full superbuild** (gRPC 1.51, protobuf 3.21; the same
  configuration as CI's Linux job): `ctest` **24/24**, 0 compiler warnings. This
  covers both Catch2 suites, `tuna_server_workload`, and `tuna_mtls_integration`,
  which runs a real client↔server mTLS round trip through the merged, dispatching
  `tuna_server`.
- Dispatch checked by hand: `--foreground --config` → Track B (exit 0); `--help` →
  both usages; no args → PR #1's usage and exit 2, unchanged.
- Not verified locally: the MSVC + vcpkg Windows client job (it first runs in CI on
  this PR) and the CUDA build (`TUNA_ENABLE_CUDA=ON`), which still needs RTX 3060
  hardware, as before.

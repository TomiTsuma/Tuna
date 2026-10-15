# Phase 1 — Core plumbing + plan reconciliation

**Date:** 2026-10-14
**Branch:** `phase-1-core-plumbing`
**Author:** Thomas Tsuma (with Claude)

## Why

Two plans existed in the tree and disagreed on the first vertical slice and the
wire format. This change resolves that and delivers the Phase 1 "core plumbing"
foundation both the transport (Phase 2) and the CUDA slice (Phase 3) build on.

## Decisions recorded

1. **Canonical plan.** `docs/plans/2026-09-23-tuna-execution-plan.md` is promoted
   from *Proposed* to **Approved / canonical**. The CUDA-driver-API-first vertical
   slice and the **v1 wire protocol** (its §4) govern all implementation. The
   2026-07-31 file-first slice is retained for history only.

2. **Build toolchain = MinGW g++ 13.1 + Ninja.** The Qt install here is the
   `mingw_64` build, so MinGW is the only toolchain that can link the GUI target
   (`tuna_gui`) *and* the rest. The prior MSVC build caches only worked because
   they never built the Qt target. New builds live in a gitignored `build-ninja/`.

3. **CI target = mock CUDA driver.** Real-GPU milestones (Phase 3) run later on
   the self-hosted GPU box when it is online. Phases 1–2 need no GPU.

## Changes

### Shared wire protocol — upgraded v0 → v1
- `proto/tuna_frames.proto`: `Frame.payload` (was `encrypted_payload`; TLS owns
  encryption now), 64-bit `frame_id`, added `channel` + `timestamp_ns`, and new
  messages `Hello`, `DeviceInfo`, `RpcRequest`, `RpcResponse`, `RpcBatch`.
- `tuna-guest-agent/tests/unit/test_frame_proto.cpp` and
  `tuna-server/tests/test_frame_proto.cpp`: rewritten for v1 — Frame round-trip,
  >32-bit `frame_id`, `RpcRequest` round-trip, `Hello` device capabilities.

### Guest agent `core/` (was empty stubs)
- `config_manager.{h,cpp}`: JSON config loader (`server_host`, `server_port`,
  `log_level`), `std::optional` on any failure. nlohmann/json.
- `logger.{h,cpp}`: spdlog rotating-file wrapper (5 MB × 3), init/get/shutdown.
- `error_handler.{h,cpp}`: `TunaError` enum + `ToString`/`IsError`.
- Tests: `test_config.cpp`, `test_logger.cpp`, `test_error_handler.cpp`.

### Server `core/` + foreground mode
- `core/config_manager.{h,cpp}`: server config (`listen_port`, `log_level`).
- `core/logger.{h,cpp}`: same spdlog wrapper (independent copy, per design).
- `src/main.cpp`: real `--foreground` / `--config PATH` / `--help`; foreground
  loads config, initializes logging, logs startup, exits cleanly (the listener
  arrives in Phase 2).
- Tests: `test_config.cpp`, `test_logger.cpp`.

### Build
- Both `CMakeLists.txt`: added nlohmann/json 3.11.3 + spdlog 1.14.1 via
  FetchContent; new `tuna_core` / `tuna_server_core` static libraries; new test
  sources wired into the Catch2/CTest targets.
- Added root `.gitignore` (build output, generated `*.pb.*`, Python caches).

### Follow-up fixes (2026-10-15)
- **FetchContent stall fixed.** Configure never finished on a clean `build-ninja/`:
  `nlohmann/json` was fetched with `GIT_REPOSITORY`, which clones the repo's full
  (very large) history. Both `CMakeLists.txt` now fetch the official v3.11.3
  release tarball `json.tar.xz` (~110 KB) pinned with `URL_HASH SHA256`, as the
  upstream README recommends. spdlog gained `GIT_SHALLOW TRUE`. Configure went
  from >30 min (never completed) to ~55 s.
- **Server honours `log_level`.** `RunForeground` previously read `log_level`
  but never applied it. It now calls `set_level(spdlog::level::from_str(...))`;
  an unrecognized name (which spdlog maps to `off`) logs a warning and keeps
  `info` instead of silently disabling logging.

## Exit criteria (Phase 1) — verified 2026-10-15
- `ctest` green in both repos (MinGW 13.1 + Ninja, `build-ninja/`):
  guest agent 16/16, server 10/10.
- `tuna_server --foreground --config <file>` loads config and writes
  `%TEMP%/tuna-server-logs/tuna-server.log`; exits 0. Unknown `log_level` →
  warning + `info`. Missing config file → error message, exit code 1.

## Not included
- `tuna-guest-agent/tests/main.c` and `tests/createfile_hook.c` are local ETW /
  MinHook exploration scratch files (same kind removed in `ebe69d2`) and are
  deliberately left uncommitted.

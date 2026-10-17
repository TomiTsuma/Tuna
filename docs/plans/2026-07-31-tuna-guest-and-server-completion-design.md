# Tuna Guest Agent + Tuna Server — Completion Design

**Date:** 2026-07-31
**Status:** Approved for implementation planning
**Author:** Thomas Tsuma (with Claude)

## 1. Context

`tuna-guest-agent` currently exists but is almost entirely scaffolding:

- Only the Qt GUI has real code (`src/gui/main.cpp`, `src/gui/mainwindow.cpp`, `settings_dialog.cpp`) — and it's a mockup with simulated metrics (`QTimer`-driven fake CPU/GPU/latency values), not wired to any real backend.
- `daemon/`, `syscall_hooks/`, `tunneling/`, `utils/` under `src/gui/` are all empty `.cpp` stub files.
- `CMakeLists.txt` only builds the GUI target; it doesn't reference the daemon/hooks/tunnel sources at all.
- `scripts/build.ps1` and `scripts/start_daemon.ps1` are empty.
- `minhook/` is vendored as a nested `.git` repo instead of a proper submodule.
- `tests/` contains scratch exploration files (`.txt` files, a compiled `main.exe`, ad-hoc `main.c`/`createfile_hook.c`) alongside deletions of the formerly-tracked formal test suite (`test_config.cpp`, `test_hooks.cpp`, `test_tunnel.cpp`, `test_full_agent.cpp`, `mock_server.py`).
- `docs/security_model.md` is empty despite being referenced from the README as a real design doc.
- **`tuna-server` does not exist anywhere.** It is 100% greenfield.

The existing docs (`architecture_overview.md`, `syscall_interception_design.md`, `tunneling_protocol.md`, `daemonization_workflow.md`) describe the intended system well but each proposes its own phase list in isolation, and none of them reflect the actual (mostly-empty) state of the code. This document consolidates them into one dependency-ordered plan and resolves the scope questions the original docs left open.

## 2. Key decisions

These were decided explicitly before writing this plan, since they change the shape of the work significantly:

| Decision | Choice | Why |
|---|---|---|
| Goal for this push | Production-grade foundation, not a quick demo | Prioritize correct architecture (interfaces, testing, security posture) over speed to a flashy demo |
| Guest agent platform scope | Cross-platform **core** (config, logging, daemon interface, tunnel), Windows-first **hook backends** | Avoids a full rewrite when Linux/macOS support is picked up later, without over-building unused abstraction now |
| Tuna Server scope | Compute daemon — executes redirected calls directly on its own OS/hardware | A full per-client hypervisor/VM layer (as loosely mentioned in `architecture_overview.md`) is a separate, much larger project; not justified for the current goal |
| Tuna Server stack | C++ with `msquic` | Shares protobuf frame schemas and tunnel code conventions with the agent; best performance for the GPU/compute execution path |

## 3. Target architecture

### 3.1 Repo/directory layout

**`tuna-guest-agent/src/`** (restructured from the current `src/gui/`-nested layout):

```
src/
├── core/                # OS-agnostic: config, structured logging, error handling, session state
├── daemon/
│   ├── daemon.h          # IDaemon interface (start/stop/status) — platform-agnostic
│   └── win/              # Windows SCM implementation (only real impl for now)
├── tunnel/               # QUIC/TLS client, protobuf frame (de)serialization, multiplexing — OS-agnostic
├── hooks/
│   ├── hook_manager.h    # Platform-agnostic hook registration interface
│   └── win/              # MinHook-based Win32/DirectX/CUDA hook implementations (only real backend)
├── gui/                  # Existing Qt GUI, rewired to talk to the daemon over local IPC instead of QTimer fakes
└── main.cpp              # Entry point: launches GUI or daemon based on args
```

This resolves the current mismatch where the README documents top-level `src/daemon/`, `src/syscall_hooks/`, `src/tunneling/` directories, but the actual (empty) files live nested under `src/gui/`.

**`tuna-server/src/`** (new):

```
src/
├── core/                 # Config, logging (same conventions as agent, not a shared binary)
├── tunnel/               # QUIC/TLS server, same frame schema as the agent
├── session/              # Per-client session manager, auth, resource accounting
├── execution/            # Compute daemon: file I/O proxy, CUDA/OpenCL dispatch, GPU command execution
└── main.cpp
```

**Shared:** a `proto/` directory (new top-level folder, sibling to both repos, or a small shared submodule) holding the `.proto` schemas for tunnel frames, so client and server can't drift out of sync on wire format.

### 3.2 Data flow (steady state, once built)

```
User app (e.g. Photoshop)
  → Win32/DirectX/CUDA call
    → MinHook trampoline (src/hooks/win/)
      → Context Object built + serialized (protobuf)
        → tunnel/ (QUIC/TLS client) ──[TTP frame]──> tuna-server tunnel/ (QUIC/TLS listener)
                                                          → session/ (auth, routing)
                                                            → execution/ (real file op / real CUDA call)
                                                          ←── result ──
        ← tunnel/ receives result frame
      ← result reinjected into process (return value / GetLastError / buffer write-back)
  ← app continues, unaware the call was remote
```

## 4. Phased roadmap

**Phase 0 — Housekeeping** *(do first, low risk, unblocks everything else)*
- Convert `minhook/` from an embedded `.git` folder into a proper git submodule.
- Delete scratch files in `tests/` (`*.txt`, `main.exe`, ad-hoc `main.c`/`createfile_hook.c` — these were exploration, not real tests).
- Restore the deleted formal test scaffolding (`test_config.cpp`, `test_hooks.cpp`, `test_tunnel.cpp`, `test_full_agent.cpp`, `mock_server.py`) as structured (initially minimal) test targets wired into CMake/CTest.
- Restructure `tuna-guest-agent/src/` per §3.1.
- Scaffold `tuna-server` as a new project: CMake, `msquic` dependency, directory layout per §3.1.

**Phase 1 — Core plumbing (both repos)**
- Agent: `core/` — config loading, structured logging (spdlog), centralized error handling; `daemon/` interface + Windows SCM implementation with a real (if minimal) start → idle loop → stop lifecycle.
- Server: `core/` — config loading, logging; minimal process entry point.
- Shared: define `.proto` schemas for tunnel frames (`SYS`, `GPU`, `IO`, `HEARTBEAT`, `CONTROL` per `tunneling_protocol.md`).
- Write `docs/security_model.md` for real (currently empty) — mutual TLS posture, process sandboxing approach, least-privilege daemon account, audit logging policy.

**Phase 2 — Tunnel (both repos)**
- Agent: `msquic` client, mTLS handshake, session negotiation, heartbeat frames.
- Server: `msquic` listener, per-client session object, heartbeat response.
- **Milestone:** agent and server establish a mutually-authenticated QUIC tunnel and exchange heartbeats over localhost. First true end-to-end integration test.

**Phase 3 — Hooking + execution: one vertical slice**
- Agent: MinHook hook for a single target — `CreateFileW` (already prototyped ad hoc in the now-deleted `tests/createfile_hook.c`, lowest risk starting point). Build the Context Object, serialize, forward over the tunnel.
- Server: `execution/` receives `IO` frames, performs the real file operation, returns the result.
- **Milestone:** a real file read/write from a hooked process is transparently executed on the server and reinjected correctly — return value and `GetLastError()` indistinguishable from an unhooked call. This validates the entire hook → serialize → tunnel → execute → reinject loop before expanding API surface.

**Phase 4 — Expand hook coverage**
- Order: remaining file/memory hooks → basic Direct3D/DXGI hooks (`Present`, buffer create) → CUDA hooks (`cudaMalloc`/`cudaMemcpy`/`cudaLaunchKernel`). Each layer reuses the Context Object/serialization pattern validated in Phase 3.
- Server: extend `execution/` with GPU command execution against its own CUDA/GPU stack.

**Phase 5 — GUI rewiring**
- Replace `mainwindow.cpp`'s `QTimer`-driven fake metrics with real IPC calls to the daemon, per the "local loopback API" described in `daemonization_workflow.md`.

**Phase 6 — Hardening & packaging**
- Security audit against the now-real `docs/security_model.md`.
- Watchdog/auto-restart implementation.
- Real `scripts/build.ps1` and NSIS installer (`installer.nsi`) implementations.

## 5. Testing strategy

**Unit tests (Catch2, both repos):**
- Agent: `core/` (config parsing, logging); `tunnel/` frame serialization round-trips; `hooks/` trampoline correctness in isolation (mocked target function — verify install/uninstall is clean and args pass through unchanged).
- Server: `session/` state transitions; `execution/` dispatch logic (given a parsed frame, verify correct handler invocation), mocked at the syscall/CUDA boundary so CI doesn't need real GPU hardware.

**Integration tests, added at each phase milestone (not deferred to the end):**
- Phase 2 milestone: real localhost server + real agent tunnel client, assert handshake completes and heartbeats flow. This is the intended replacement for the deleted `tests/integration/mock_server.py`/`test_tunnel.cpp` — a real server instead of a mock, once Phase 2 exists.
- Phase 3 milestone: launch a small test harness process with the hook DLL injected, perform a `CreateFileW` + write, assert bytes landed server-side and the locally reinjected result is indistinguishable from unhooked behavior.
- Each Phase 4 hook category gets the same shape of test before being marked done.

**GPU/CUDA test lane:** PR-level CI mocks the CUDA call boundary (dispatch/serialization correctness only). Tests against real GPU hardware run in a separate manual/nightly lane on a self-hosted runner, since GitHub-hosted runners don't have CUDA.

**Regression guard:** the "dry-run mode" from `syscall_interception_design.md` (hook but don't forward) doubles as both a manual sanity-check tool and a CI test mode — hooks can be exercised without a live server connection.

**Docs discipline:** each phase gets a changelog entry in `tuna-guest-agent/docs/changelog/` (and a `tuna-server/docs/changelog/` once that repo exists) using the `{ddmmyy-hhmmss}_{feature}_changelog.md` format.

## 6. Open items deferred beyond this plan

- Linux/macOS hook backends (core is designed to allow this later; not built now).
- Full per-client virtualization/hypervisor layer on the server.
- Multi-server load balancing / dynamic GPU allocation across clients.
- CLR profiler / JVMTI integration for .NET/Java high-level hooks.

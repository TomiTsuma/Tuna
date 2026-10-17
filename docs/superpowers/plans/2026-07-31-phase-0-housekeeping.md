# Phase 0 Housekeeping Implementation Plan

> **For agentic workers:** REQUIRED: USE superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up `tuna-guest-agent`'s repo hygiene, restructure `src/` to match the target architecture, stand up a real (if minimal) test harness, and scaffold `tuna-server` as a new buildable project — so Phase 1 (core plumbing) has clean ground to build on.

**Architecture:** No new behavior is introduced in this phase. This is pure structural work: fixing a mis-vendored dependency, removing scratch files, moving existing (currently empty) source files into the directory layout defined in `docs/plans/2026-07-31-tuna-guest-and-server-completion-design.md` §3.1, and wiring up Catch2/CTest + pytest so every later phase has a working test harness from day one.

**Tech Stack:** CMake 3.16+, C++20, Catch2 v3 (via CMake FetchContent), Python 3 + pytest (for integration test scaffolding), git submodules.

**Spec reference:** `docs/plans/2026-07-31-tuna-guest-and-server-completion-design.md`, §3.1 and Phase 0 bullet list.

**Scope note — deviations from the design doc's Phase 0 bullet list:**
1. The design doc says to delete `tests/createfile_hook.c` as scratch. That file is **currently untracked** (never committed) — deleting it would destroy it permanently, with no git history to recover it from. But Phase 3 of the design doc explicitly plans to reuse its logic as the seed for the first real hook. **This plan keeps `createfile_hook.c` in place, untouched**, so it's available when Phase 3 starts. Same reasoning applies to `tests/main.c` (an ETW-based kernel tracing experiment) — it's kept as-is rather than deleted, since it's a real reference artifact, not scratch notes.
2. The design doc's Phase 0 bullet says to add `daemon.h`/`hook_manager.h` interface headers as part of the restructure. This plan **defers writing those interfaces to Phase 1**, because with no real daemon/hook logic to define an interface against yet, those files would only ever contain placeholder/TODO content — which produces a file that has to be rewritten wholesale in Phase 1 anyway. Phase 0 here is scoped strictly to moving/deleting existing files and getting a real, passing test harness in place.
3. The design doc's Phase 0 bullet says to scaffold `tuna-server` with a full `core/`, `tunnel/`, `session/`, `execution/` layout. This plan scaffolds **only** a buildable `main.cpp` + CMake + test harness for `tuna-server` — the subdirectories are created in Phase 1 when they get real content, avoiding empty placeholder directories with no files in them.

---

## Chunk 1: Repo hygiene, restructuring, and test harness

### Task 1: Fix `minhook` vendoring (nested repo → proper git submodule)

**Context:** `tuna-guest-agent/minhook/` currently has its own `.git` directory (confirmed via `cat minhook/.git/config`: `remote "origin" = https://github.com/TsudaKageyu/minhook.git`) and shows up as an untracked directory in `git status` at the parent repo level. This means none of its content is actually tracked by the `tuna` repo — a fresh clone would get an empty `minhook/` folder. It needs to be a real git submodule.

**Files:**
- Remove: `tuna-guest-agent/minhook/` (entire nested repo, as currently vendored)
- Create: `.gitmodules` (at repo root, `c:\Users\tsuma.thomas\Documents\tuna\.gitmodules`)
- Create (via submodule add): `tuna-guest-agent/minhook/` (as a proper gitlink)

- [ ] **Step 1: Confirm current state and remote URL**

Run:
```bash
cd "c:\Users\tsuma.thomas\Documents\tuna"
cat tuna-guest-agent/minhook/.git/config
git status --short tuna-guest-agent/minhook
```
Expected: config shows `url = https://github.com/TsudaKageyu/minhook.git`; status shows `?? tuna-guest-agent/minhook/` (untracked).

- [ ] **Step 2: Remove the nested repo**

```bash
rm -rf tuna-guest-agent/minhook
```
Expected: `ls tuna-guest-agent` no longer lists `minhook`.

- [ ] **Step 3: Add it back as a real submodule**

```bash
git submodule add https://github.com/TsudaKageyu/minhook.git tuna-guest-agent/minhook
```
Expected output ends with `Cloning into '.../tuna-guest-agent/minhook'...` and no errors. A `.gitmodules` file now exists at repo root.

- [ ] **Step 4: Verify the submodule is tracked correctly**

```bash
git submodule status
cat .gitmodules
```
Expected: `git submodule status` prints a line like ` <commit-sha> tuna-guest-agent/minhook (heads/master)` (no leading `-` or `+`, which would mean uninitialized/out-of-sync). `.gitmodules` contains a `[submodule "tuna-guest-agent/minhook"]` block with the correct path and url.

- [ ] **Step 5: Commit**

```bash
git add .gitmodules tuna-guest-agent/minhook
git commit -m "fix: vendor minhook as a proper git submodule instead of a nested repo"
```

---

### Task 2: Remove scratch junk from `tests/`

**Context:** Per the scope note above, only files with zero future value are removed. `createfile_hook.c` and `main.c` are kept.

**Files:**
- Delete: `tuna-guest-agent/tests/New Text Document (2).txt`
- Delete: `tuna-guest-agent/tests/New Text Document.txt`
- Delete: `tuna-guest-agent/tests/asd.txt`
- Delete: `tuna-guest-agent/tests/asdff.txt`
- Delete: `tuna-guest-agent/tests/sa.txt`
- Delete: `tuna-guest-agent/tests/test_file_creation.txt`
- Delete: `tuna-guest-agent/tests/main.exe` (compiled binary, shouldn't be committed)

- [ ] **Step 1: Delete the scratch files**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-guest-agent"
rm "tests/New Text Document (2).txt" "tests/New Text Document.txt" tests/asd.txt tests/asdff.txt tests/sa.txt tests/test_file_creation.txt tests/main.exe
```

- [ ] **Step 2: Verify only the intentionally-kept files remain untracked**

```bash
git status --short tests/
```
Expected: shows `tests/createfile_hook.c` and `tests/main.c` as untracked (`??`), and the deleted formal test files as staged/unstaged deletions (`D`) — those are handled in Task 4, not here.

- [ ] **Step 3: Add a `.gitignore` entry so compiled test binaries don't get re-added by accident**

Add this line to `tuna-guest-agent/.gitignore` (create the section if a build-artifacts section doesn't already exist):
```
tests/*.exe
tests/*.obj
```

- [ ] **Step 4: Commit**

```bash
git add tuna-guest-agent/.gitignore
git commit -m "chore: remove scratch exploration files from tests/"
```

---

### Task 3: Restructure `tuna-guest-agent/src/` to match the target architecture

**Context:** Every file being moved here is currently empty (0 lines) — confirmed via `wc -l` — so these are pure renames, not content-preserving refactors. `hook_handlers.h` is the one exception (29 lines, declares `HookedNtCreateFile`/`handlers_init`/`handlers_uninit`) and must be moved with its content intact. A repo-wide grep confirmed nothing includes any of these files by path (`grep -r "hook_handlers\|syscall_hooks\|tunneling/\|gui/daemon\|gui/utils"` only matched `README.md`), so no include statements need fixing elsewhere.

**Files (moves):**
- Move: `src/gui/utils/config_manager.cpp` → `src/core/config_manager.cpp`
- Move: `src/gui/utils/logger.cpp` → `src/core/logger.cpp`
- Move: `src/gui/utils/error_handler.cpp` → `src/core/error_handler.cpp`
- Move: `src/gui/utils/platform_utils.cpp` → `src/core/platform_utils.cpp`
- Delete: `src/gui/daemon/daemon_logger.cpp` (redundant with `src/core/logger.cpp`; daemon-specific log formatting, if needed, is added inside `daemon_service.cpp` in Phase 1)
- Move: `src/gui/daemon/daemon_main.cpp` → `src/daemon/win/daemon_main.cpp`
- Move: `src/gui/daemon/daemon_service.cpp` → `src/daemon/win/daemon_service.cpp`
- Move: `src/gui/daemon/privilege_utils.cpp` → `src/daemon/win/privilege_utils.cpp`
- Move: `src/gui/daemon/watchdog.cpp` → `src/daemon/win/watchdog.cpp`
- Move: `src/gui/syscall_hooks/hooks_init.cpp` → `src/hooks/win/hooks_init.cpp`
- Move: `src/gui/syscall_hooks/file_hooks.cpp` → `src/hooks/win/file_hooks.cpp`
- Move: `src/gui/syscall_hooks/graphics_hooks.cpp` → `src/hooks/win/graphics_hooks.cpp`
- Move: `src/gui/syscall_hooks/process_hooks.cpp` → `src/hooks/win/process_hooks.cpp`
- Move: `src/gui/syscall_hooks/network_hooks.cpp` → `src/hooks/win/network_hooks.cpp`
- Move: `src/gui/syscall_hooks/memory_hooks.cpp` → `src/hooks/win/memory_hooks.cpp`
- Move: `src/gui/syscall_hooks/hook_handlers.cpp` → `src/hooks/win/hook_handlers.cpp`
- Move: `src/gui/syscall_hooks/hook_handlers.h` → `src/hooks/win/hook_handlers.h`
- Move: `src/gui/tunneling/quic_tunnel.cpp` → `src/tunnel/quic_tunnel.cpp`
- Move: `src/gui/tunneling/multiplexing_layer.cpp` → `src/tunnel/multiplexing_layer.cpp`
- Move: `src/gui/tunneling/data_serializer.cpp` → `src/tunnel/data_serializer.cpp`
- Move: `src/gui/tunneling/handshake_protocol.cpp` → `src/tunnel/handshake_protocol.cpp`
- Move: `src/gui/tunneling/heartbeat.cpp` → `src/tunnel/heartbeat.cpp`
- Unchanged: `src/gui/main.cpp`, `mainwindow.cpp`, `mainwindow.h`, `mainwindow.ui`, `settings_dialog.cpp`, `tray_icon.cpp` (real GUI code, stays in `gui/`)
- Unchanged: `src/main.cpp` (empty unified entry point, filled in during Phase 1)
- Modify: `README.md` (file/folder structure section, to match reality)

- [ ] **Step 1: Move core/ files**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-guest-agent"
mkdir -p src/core src/daemon/win src/hooks/win src/tunnel
git mv src/gui/utils/config_manager.cpp src/core/config_manager.cpp
git mv src/gui/utils/logger.cpp src/core/logger.cpp
git mv src/gui/utils/error_handler.cpp src/core/error_handler.cpp
git mv src/gui/utils/platform_utils.cpp src/core/platform_utils.cpp
```

- [ ] **Step 2: Move daemon/ files, drop the redundant logger**

```bash
git rm src/gui/daemon/daemon_logger.cpp
git mv src/gui/daemon/daemon_main.cpp src/daemon/win/daemon_main.cpp
git mv src/gui/daemon/daemon_service.cpp src/daemon/win/daemon_service.cpp
git mv src/gui/daemon/privilege_utils.cpp src/daemon/win/privilege_utils.cpp
git mv src/gui/daemon/watchdog.cpp src/daemon/win/watchdog.cpp
```

- [ ] **Step 3: Move hooks/ files**

```bash
git mv src/gui/syscall_hooks/hooks_init.cpp src/hooks/win/hooks_init.cpp
git mv src/gui/syscall_hooks/file_hooks.cpp src/hooks/win/file_hooks.cpp
git mv src/gui/syscall_hooks/graphics_hooks.cpp src/hooks/win/graphics_hooks.cpp
git mv src/gui/syscall_hooks/process_hooks.cpp src/hooks/win/process_hooks.cpp
git mv src/gui/syscall_hooks/network_hooks.cpp src/hooks/win/network_hooks.cpp
git mv src/gui/syscall_hooks/memory_hooks.cpp src/hooks/win/memory_hooks.cpp
git mv src/gui/syscall_hooks/hook_handlers.cpp src/hooks/win/hook_handlers.cpp
git mv src/gui/syscall_hooks/hook_handlers.h src/hooks/win/hook_handlers.h
```

- [ ] **Step 4: Move tunnel/ files**

```bash
git mv src/gui/tunneling/quic_tunnel.cpp src/tunnel/quic_tunnel.cpp
git mv src/gui/tunneling/multiplexing_layer.cpp src/tunnel/multiplexing_layer.cpp
git mv src/gui/tunneling/data_serializer.cpp src/tunnel/data_serializer.cpp
git mv src/gui/tunneling/handshake_protocol.cpp src/tunnel/handshake_protocol.cpp
git mv src/gui/tunneling/heartbeat.cpp src/tunnel/heartbeat.cpp
```

- [ ] **Step 5: Clean up now-empty directories and verify `hook_handlers.h` content survived intact**

```bash
rmdir src/gui/utils src/gui/daemon src/gui/syscall_hooks src/gui/tunneling 2>/dev/null
diff <(git show HEAD:tuna-guest-agent/src/gui/syscall_hooks/hook_handlers.h 2>/dev/null || true) src/hooks/win/hook_handlers.h
find src -type f | sort
```
Expected: `diff` prints nothing (files identical) — note `git show HEAD:...` will be empty since `hook_handlers.h` was never committed either (it's part of the working tree only), so this diff is really just confirming the moved file still has its 29 lines; if the first `git show` errors out, just run `wc -l src/hooks/win/hook_handlers.h` instead and confirm it still shows 29 lines. `find src -type f` shows the new layout: `src/core/*.cpp`, `src/daemon/win/*.cpp`, `src/hooks/win/*.cpp(.h)`, `src/tunnel/*.cpp`, `src/gui/{main.cpp,mainwindow.cpp,mainwindow.h,mainwindow.ui,settings_dialog.cpp,tray_icon.cpp}`, `src/main.cpp`.

- [ ] **Step 6: Confirm the existing GUI build target still builds (unaffected by the move)**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-guest-agent"
cmake -B build -S .
cmake --build build --target tuna_gui
```
Expected: configure and build both succeed with no errors (this target's sources — `src/gui/main.cpp`, `mainwindow.cpp`, `mainwindow.h`, `settings_dialog.cpp` — were never touched by the moves above, so this should behave exactly as before). If Qt6 isn't installed in the current environment, this step can be skipped, but note in the commit message that it wasn't verified locally.

- [ ] **Step 7: Update README.md's documented file structure to match reality**

Open `README.md` and replace the "File and Folder Structure" code block (currently lines 22-82) with:

```
tuna-guest-agent/
├── src/
│   ├── core/                       # OS-agnostic config, logging, error handling
│   │   ├── config_manager.cpp
│   │   ├── logger.cpp
│   │   ├── error_handler.cpp
│   │   └── platform_utils.cpp
│   │
│   ├── daemon/                     # Background service (Tuna Daemon)
│   │   └── win/                    # Windows Service implementation
│   │       ├── daemon_main.cpp
│   │       ├── daemon_service.cpp
│   │       ├── privilege_utils.cpp
│   │       └── watchdog.cpp
│   │
│   ├── hooks/                      # Hooking and interception logic
│   │   └── win/                    # MinHook-based Win32/DirectX/CUDA hooks
│   │       ├── hooks_init.cpp
│   │       ├── file_hooks.cpp
│   │       ├── graphics_hooks.cpp
│   │       ├── process_hooks.cpp
│   │       ├── network_hooks.cpp
│   │       ├── memory_hooks.cpp
│   │       ├── hook_handlers.cpp
│   │       └── hook_handlers.h
│   │
│   ├── tunnel/                     # Communication and transport layer
│   │   ├── quic_tunnel.cpp
│   │   ├── multiplexing_layer.cpp
│   │   ├── data_serializer.cpp
│   │   ├── handshake_protocol.cpp
│   │   └── heartbeat.cpp
│   │
│   ├── gui/                        # Qt-based graphical frontend
│   │   ├── main.cpp
│   │   ├── mainwindow.ui
│   │   ├── mainwindow.cpp
│   │   ├── settings_dialog.cpp
│   │   └── tray_icon.cpp
│   │
│   └── main.cpp                    # Unified entry point (launches GUI or daemon)
│
├── docs/
│   ├── architecture_overview.md
│   ├── daemonization_workflow.md
│   ├── syscall_interception_design.md
│   ├── tunneling_protocol.md
│   └── security_model.md
│
├── minhook/                         # Vendored as a git submodule (TsudaKageyu/minhook)
│
├── build/                          # Compiled binaries and build artifacts
│
├── tests/
│   ├── unit/                       # Unit tests (Catch2)
│   ├── integration/                # End-to-end tests (hooks + tunnel)
│   ├── createfile_hook.c           # Reference prototype for the Phase 3 CreateFileW hook
│   └── main.c                      # Reference ETW-based file-monitoring experiment (not part of the build)
│
├── CMakeLists.txt                  # Build configuration
├── LICENSE
└── README.md
```

- [ ] **Step 8: Commit**

```bash
git add -A src README.md
git commit -m "refactor: restructure src/ into core/daemon/hooks/tunnel layout matching the design doc"
```

---

### Task 4: Wire up a real test harness (Catch2 for C++, pytest for Python)

**Context:** `test_config.cpp`, `test_hooks.cpp`, `test_tunnel.cpp`, `test_full_agent.cpp`, and `mock_server.py` were deleted (shown in `git status` as `D`). They were empty when they existed (confirmed via `git show HEAD:tuna-guest-agent/tests/unit/test_config.cpp` etc. returning nothing), so there's no content to restore — only the file harness needs restoring, this time with a real, passing, self-contained smoke test in each, instead of an empty file. Each smoke test's job is to prove the test-running infrastructure (CMake/CTest/Catch2 discovery, and pytest discovery) works end-to-end; each will be superseded by real behavioral tests in the phase that implements the corresponding module (noted inline).

**Files:**
- Create: `tuna-guest-agent/tests/unit/test_config.cpp`
- Create: `tuna-guest-agent/tests/unit/test_hooks.cpp`
- Create: `tuna-guest-agent/tests/unit/test_tunnel.cpp`
- Create: `tuna-guest-agent/tests/integration/test_full_agent.cpp`
- Create: `tuna-guest-agent/tests/integration/mock_server.py`
- Create: `tuna-guest-agent/tests/requirements.txt`
- Modify: `tuna-guest-agent/CMakeLists.txt`

- [ ] **Step 1: Write the four Catch2 smoke test files**

`tuna-guest-agent/tests/unit/test_config.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves CMake/Catch2/CTest discovery works.
// Replaced with real config-loading tests once src/core/config_manager.cpp
// is implemented (Phase 1).
TEST_CASE("test harness is wired up for core/config_manager", "[scaffolding][config]") {
    REQUIRE(1 + 1 == 2);
}
```

`tuna-guest-agent/tests/unit/test_hooks.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves CMake/Catch2/CTest discovery works.
// Replaced with real hook-manager tests once src/hooks/ has an
// implementation (Phase 3 for the first CreateFileW hook).
TEST_CASE("test harness is wired up for hooks/win", "[scaffolding][hooks]") {
    REQUIRE(1 + 1 == 2);
}
```

`tuna-guest-agent/tests/unit/test_tunnel.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves CMake/Catch2/CTest discovery works.
// Replaced with real frame-serialization round-trip tests once
// src/tunnel/ is implemented (Phase 2).
TEST_CASE("test harness is wired up for tunnel/", "[scaffolding][tunnel]") {
    REQUIRE(1 + 1 == 2);
}
```

`tuna-guest-agent/tests/integration/test_full_agent.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves the integration test binary builds and
// CTest discovers it. Replaced with a real localhost agent<->server
// handshake test once Phase 2's tunnel exists.
TEST_CASE("integration test harness is wired up", "[scaffolding][integration]") {
    REQUIRE(1 + 1 == 2);
}
```

- [ ] **Step 2: Write the pytest smoke test**

`tuna-guest-agent/tests/integration/mock_server.py`:
```python
"""Scaffolding smoke test for the Python integration test harness.

Proves pytest discovery and execution works from this directory. Replaced
with a real mock QUIC server (accepting connections, responding to
heartbeat frames) once Phase 2's tunnel protocol exists.
"""


def test_pytest_harness_is_wired_up():
    assert 1 + 1 == 2
```

`tuna-guest-agent/tests/requirements.txt`:
```
pytest==8.3.3
```

- [ ] **Step 3: Add Catch2 and CTest wiring to CMakeLists.txt**

Modify `tuna-guest-agent/CMakeLists.txt` — append this block after the existing `target_link_libraries(tuna_gui ...)` line:

```cmake
# --- Test harness (Catch2 + CTest) ---
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(tuna_tests
    tests/unit/test_config.cpp
    tests/unit/test_hooks.cpp
    tests/unit/test_tunnel.cpp
    tests/integration/test_full_agent.cpp
)
target_link_libraries(tuna_tests PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(tuna_tests)
```

- [ ] **Step 4: Configure, build, and run the C++ test suite**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-guest-agent"
cmake -B build -S .
cmake --build build --target tuna_tests
ctest --test-dir build --output-on-failure
```
Expected: build succeeds, `ctest` reports `100% tests passed, 0 tests failed out of 4`.

- [ ] **Step 5: Run the pytest suite**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-guest-agent"
python -m pip install -r tests/requirements.txt
python -m pytest tests/integration -v
```
Expected: `tests/integration/mock_server.py::test_pytest_harness_is_wired_up PASSED`, 1 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/unit tests/integration tests/requirements.txt CMakeLists.txt
git commit -m "test: restore test scaffolding with real Catch2/pytest smoke tests wired into CMake+CTest"
```

---

### Task 5: Scaffold `tuna-server` as a new, minimal, buildable project

**Context:** `tuna-server` does not exist anywhere yet. Per the scope note, this task creates only a buildable skeleton (project files + a real minimal `main.cpp` + its own test harness) — `core/`, `tunnel/`, `session/`, `execution/` are added in Phase 1 alongside their first real content, not as empty placeholder directories now.

**Files:**
- Create: `tuna-server/CMakeLists.txt`
- Create: `tuna-server/src/main.cpp`
- Create: `tuna-server/tests/test_smoke.cpp`
- Create: `tuna-server/.gitignore`
- Create: `tuna-server/README.md`

- [ ] **Step 1: Create the directory and CMake project file**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna"
mkdir -p tuna-server/src tuna-server/tests
```

`tuna-server/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(TunaServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tuna_server src/main.cpp)

# --- Test harness (Catch2 + CTest) ---
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(tuna_server_tests tests/test_smoke.cpp)
target_link_libraries(tuna_server_tests PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(tuna_server_tests)
```

- [ ] **Step 2: Write a real (not placeholder) minimal entry point**

`tuna-server/src/main.cpp`:
```cpp
#include <cstdio>

int main() {
    std::puts("Tuna Server 0.0.1 (scaffold) — no session/tunnel/execution logic yet.");
    return 0;
}
```

- [ ] **Step 3: Write the smoke test**

`tuna-server/tests/test_smoke.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves CMake/Catch2/CTest discovery works for
// tuna-server. Replaced with real session/tunnel tests starting Phase 1.
TEST_CASE("tuna-server test harness is wired up", "[scaffolding]") {
    REQUIRE(1 + 1 == 2);
}
```

- [ ] **Step 4: Add `.gitignore`**

`tuna-server/.gitignore`:
```
build/
*.obj
*.exe
```

- [ ] **Step 5: Add a minimal README**

`tuna-server/README.md`:
```markdown
# Tuna Server

Remote compute daemon for the Tuna Guest Agent. Receives redirected file
I/O and GPU/CPU compute calls over the Tuna Tunneling Protocol and
executes them on this machine's own hardware.

**Status:** scaffold only — see
`../docs/plans/2026-07-31-tuna-guest-and-server-completion-design.md`
for the full design and phased roadmap.

## Build

\`\`\`bash
cmake -B build -S .
cmake --build build
\`\`\`

## Test

\`\`\`bash
ctest --test-dir build --output-on-failure
\`\`\`
```

- [ ] **Step 6: Configure, build, and test**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-server"
cmake -B build -S .
cmake --build build
ctest --test-dir build --output-on-failure
./build/tuna_server
```
Expected: `ctest` reports `100% tests passed, 0 tests failed out of 1`; running `tuna_server` prints the scaffold message and exits 0.

- [ ] **Step 7: Commit**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna"
git add tuna-server
git commit -m "feat: scaffold tuna-server as a new minimal buildable project"
```

---

## End-of-chunk verification

- [ ] Run `git submodule status` — `minhook` shows as a properly tracked submodule.
- [ ] Run `git status --short` at repo root — no unexpected untracked files remain (only intentional ones: `tests/createfile_hook.c`, `tests/main.c`).
- [ ] Run `find tuna-guest-agent/src -type f | sort` — matches the layout in Task 3 Step 7.
- [ ] Run `ctest --test-dir tuna-guest-agent/build --output-on-failure` — all 4 tests pass.
- [ ] Run `ctest --test-dir tuna-server/build --output-on-failure` — the 1 test passes.
- [ ] Run `python -m pytest tuna-guest-agent/tests/integration -v` — 1 test passes.

**Next:** once this chunk is reviewed and approved, Phase 1 (core plumbing) gets its own plan document — `core/` config+logging implementation, the `IDaemon`/`hook_manager.h` interfaces deferred from this phase, and the `.proto` frame schema definitions.

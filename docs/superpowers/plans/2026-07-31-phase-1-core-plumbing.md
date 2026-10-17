# Phase 1 Core Plumbing Implementation Plan

> **For agentic workers:** REQUIRED: USE superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give both `tuna-guest-agent` and `tuna-server` real (not stubbed) config loading, structured logging, error handling, a shared wire-format schema for tunnel frames, and a testable daemon lifecycle state machine for the agent — the foundation Phase 2 (tunnel) and Phase 3 (hooking) build on.

**Architecture:** `core/` in each repo gets independent (not shared-binary) implementations of config/logging/error-handling, per the design doc. A new top-level `proto/` directory holds the one shared `.proto` schema both repos compile against. The agent's daemon lifecycle is split into a platform-agnostic `DaemonCore` state machine (unit-testable) and a thin Windows SCM wrapper around it (not unit-testable outside a real service context, verified by a build + foreground-mode manual run instead).

**Tech Stack:** CMake FetchContent for `nlohmann/json` (config parsing), `spdlog` (logging), and `protobuf` v21.12 (frame schema codegen) — added to both repos' existing Catch2/CTest harness from Phase 0.

**Spec reference:** `docs/plans/2026-07-31-tuna-guest-and-server-completion-design.md` §4 Phase 1; `tuna-guest-agent/docs/tunneling_protocol.md` §3 (frame structure); `tuna-guest-agent/docs/daemonization_workflow.md` §4, §7 (daemon lifecycle).

**Scope note:** `src/hooks/hook_manager.h` is **not** part of this phase — the design doc places the hook manager interface in Phase 3, alongside the first real hook implementation, so it isn't written against nothing here.

**Build-time warning:** the protobuf FetchContent build compiles a large C++ project from source. Expect the *first* configure+build in Task 1 to take significantly longer (potentially 10+ minutes) than anything in Phase 0. Subsequent builds reuse the cached `_deps` output and are fast.

---

## Chunk 1: Shared proto schema + core/ (config, logging, error handling)

### Task 1: Shared tunnel frame schema (`proto/tuna_frames.proto`) + protobuf wired into both repos

**Files:**
- Create: `proto/tuna_frames.proto` (repo root, sibling to both projects)
- Modify: `tuna-guest-agent/CMakeLists.txt`
- Modify: `tuna-server/CMakeLists.txt`
- Test: `tuna-guest-agent/tests/unit/test_frame_proto.cpp`
- Test: `tuna-server/tests/test_frame_proto.cpp`

- [ ] **Step 1: Write the schema**

`proto/tuna_frames.proto`:
```protobuf
syntax = "proto3";

package tuna;

// Frame types per docs/tunneling_protocol.md section 3.
enum FrameType {
    FRAME_TYPE_UNSPECIFIED = 0;
    FRAME_TYPE_SYS = 1;
    FRAME_TYPE_GPU = 2;
    FRAME_TYPE_IO = 3;
    FRAME_TYPE_HEARTBEAT = 4;
    FRAME_TYPE_CONTROL = 5;
}

message FrameHeader {
    FrameType type = 1;
    uint32 frame_id = 2;
    uint32 payload_length = 3;
    uint64 timestamp_ms = 4;
}

message Frame {
    FrameHeader header = 1;
    bytes encrypted_payload = 2;
}
```

- [ ] **Step 2: Add protobuf + codegen to `tuna-guest-agent/CMakeLists.txt`**

Insert this block into `tuna-guest-agent/CMakeLists.txt`, immediately before the existing `# --- Test harness (Catch2 + CTest) ---` section:

```cmake
# --- Shared tunnel frame schema (protobuf) ---
FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v21.12
)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(protobuf)

add_library(tuna_proto STATIC)
protobuf_generate(
    TARGET tuna_proto
    PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/../proto/tuna_frames.proto
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../proto
    PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_include_directories(tuna_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(tuna_proto PUBLIC protobuf::libprotobuf)
```

Note this must come *before* `include(FetchContent)` is needed by Catch2 — since Catch2's block already has `include(FetchContent)`, move that single `include(FetchContent)` line up so it precedes this new block (it's idempotent, safe to call once at the top).

- [ ] **Step 3: Add `tuna_proto` to the test binary and write the round-trip test**

Modify the existing `add_executable(tuna_tests ...)` block: add `tests/unit/test_frame_proto.cpp` to its sources, and add `tuna_proto` to its `target_link_libraries` line:
```cmake
add_executable(tuna_tests
    tests/unit/test_config.cpp
    tests/unit/test_hooks.cpp
    tests/unit/test_tunnel.cpp
    tests/unit/test_frame_proto.cpp
    tests/integration/test_full_agent.cpp
)
target_link_libraries(tuna_tests PRIVATE Catch2::Catch2WithMain tuna_proto)
```

`tuna-guest-agent/tests/unit/test_frame_proto.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "tuna_frames.pb.h"

TEST_CASE("Frame round-trips through serialize/parse", "[proto][tunnel]") {
    tuna::Frame original;
    original.mutable_header()->set_type(tuna::FRAME_TYPE_HEARTBEAT);
    original.mutable_header()->set_frame_id(42);
    original.mutable_header()->set_timestamp_ms(1234567890ULL);
    original.set_encrypted_payload("test-payload-bytes");
    original.mutable_header()->set_payload_length(
        static_cast<uint32_t>(original.encrypted_payload().size()));

    std::string wire;
    REQUIRE(original.SerializeToString(&wire));

    tuna::Frame parsed;
    REQUIRE(parsed.ParseFromString(wire));

    REQUIRE(parsed.header().type() == tuna::FRAME_TYPE_HEARTBEAT);
    REQUIRE(parsed.header().frame_id() == 42);
    REQUIRE(parsed.header().timestamp_ms() == 1234567890ULL);
    REQUIRE(parsed.encrypted_payload() == "test-payload-bytes");
    REQUIRE(parsed.header().payload_length() == 19);
}
```

- [ ] **Step 4: Configure and build (expect a long first build)**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-guest-agent"
cmake -B build -S .
cmake --build build --target tuna_tests
```
Expected: succeeds, though the protobuf portion alone may take several minutes on first build. If it fails with a CMake error about `protobuf_generate` not being found, the protobuf version's exported CMake API differs — stop and report the exact error rather than guessing at a fix.

- [ ] **Step 5: Run the test**

```bash
ctest --test-dir build --output-on-failure -R "Frame round-trips"
```
Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 6: Repeat steps 2-5 for `tuna-server`**

`tuna-server/CMakeLists.txt` gets the identical protobuf block inserted before its existing `# --- Test harness (Catch2 + CTest) ---` section (paths still resolve to `${CMAKE_CURRENT_SOURCE_DIR}/../proto/tuna_frames.proto` since `tuna-server` is also a sibling of `proto/`). Add `tuna_proto` as a dependency to `tuna_server_tests` the same way, and add `tests/test_frame_proto.cpp` to its sources — same test content as above, saved to `tuna-server/tests/test_frame_proto.cpp`.

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-server"
cmake -B build -S .
cmake --build build --target tuna_server_tests
ctest --test-dir build --output-on-failure -R "Frame round-trips"
```
Expected: same result as Step 5.

- [ ] **Step 7: Commit**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna"
git add proto tuna-guest-agent/CMakeLists.txt tuna-guest-agent/tests/unit/test_frame_proto.cpp tuna-server/CMakeLists.txt tuna-server/tests/test_frame_proto.cpp
git commit -m "feat: add shared protobuf tunnel frame schema, wire into both repos"
```

---

### Task 2: Agent `core/config_manager` — real JSON config loading

**Files:**
- Modify: `tuna-guest-agent/src/core/config_manager.cpp` (currently empty)
- Create: `tuna-guest-agent/src/core/config_manager.h`
- Modify: `tuna-guest-agent/CMakeLists.txt` (add nlohmann/json dependency, add `config_manager.cpp` to a new `tuna_core` library target)
- Test: `tuna-guest-agent/tests/unit/test_config.cpp` (replaces the Phase 0 scaffolding smoke test)

- [ ] **Step 1: Add nlohmann/json via FetchContent**

Add to `tuna-guest-agent/CMakeLists.txt`, next to the protobuf block:
```cmake
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)
```

- [ ] **Step 2: Write the failing test**

Replace the contents of `tuna-guest-agent/tests/unit/test_config.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <filesystem>
#include "core/config_manager.h"

namespace {
std::filesystem::path writeTempConfig(const std::string& json) {
    auto path = std::filesystem::temp_directory_path() / "tuna_test_config.json";
    std::ofstream out(path);
    out << json;
    return path;
}
}

TEST_CASE("ConfigManager loads valid config", "[config]") {
    auto path = writeTempConfig(R"({
        "server_host": "server-us-west-2.tuna.cloud",
        "server_port": 4433,
        "log_level": "debug"
    })");

    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE(result.has_value());
    CHECK(result->server_host == "server-us-west-2.tuna.cloud");
    CHECK(result->server_port == 4433);
    CHECK(result->log_level == "debug");
}

TEST_CASE("ConfigManager rejects missing file", "[config]") {
    auto result = tuna::ConfigManager::LoadFromFile("C:/definitely/not/a/real/path.json");
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("ConfigManager rejects malformed JSON", "[config]") {
    auto path = writeTempConfig("{ not valid json");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("ConfigManager rejects missing required field", "[config]") {
    auto path = writeTempConfig(R"({"log_level": "info"})");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE_FALSE(result.has_value());
}
```

- [ ] **Step 3: Run test to verify it fails (header doesn't exist yet)**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-guest-agent"
cmake -B build -S .
cmake --build build --target tuna_tests
```
Expected: FAIL — compile error, `core/config_manager.h` does not exist.

- [ ] **Step 4: Write `config_manager.h`**

`tuna-guest-agent/src/core/config_manager.h`:
```cpp
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tuna {

struct AgentConfig {
    std::string server_host;
    uint16_t server_port = 0;
    std::string log_level = "info";
};

class ConfigManager {
public:
    // Returns std::nullopt if the file is missing, unreadable, not valid JSON,
    // or missing a required field (server_host, server_port).
    static std::optional<AgentConfig> LoadFromFile(const std::string& path);
};

}  // namespace tuna
```

- [ ] **Step 5: Write `config_manager.cpp`**

`tuna-guest-agent/src/core/config_manager.cpp`:
```cpp
#include "core/config_manager.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace tuna {

std::optional<AgentConfig> ConfigManager::LoadFromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return std::nullopt;
    }

    nlohmann::json parsed;
    try {
        in >> parsed;
    } catch (const nlohmann::json::parse_error&) {
        return std::nullopt;
    }

    if (!parsed.contains("server_host") || !parsed.contains("server_port")) {
        return std::nullopt;
    }

    AgentConfig config;
    try {
        config.server_host = parsed.at("server_host").get<std::string>();
        config.server_port = parsed.at("server_port").get<uint16_t>();
        if (parsed.contains("log_level")) {
            config.log_level = parsed.at("log_level").get<std::string>();
        }
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }

    return config;
}

}  // namespace tuna
```

- [ ] **Step 6: Wire the new source into the build**

Add a `tuna_core` static library to `tuna-guest-agent/CMakeLists.txt` (place near the top, after `project()`):
```cmake
add_library(tuna_core STATIC
    src/core/config_manager.cpp
)
target_include_directories(tuna_core PUBLIC src)
target_link_libraries(tuna_core PUBLIC nlohmann_json::nlohmann_json)
```
Add `tuna_core` to `tuna_tests`'s `target_link_libraries` line (alongside `Catch2::Catch2WithMain` and `tuna_proto`).

- [ ] **Step 7: Run tests to verify they pass**

```bash
cmake -B build -S .
cmake --build build --target tuna_tests
ctest --test-dir build --output-on-failure -R "ConfigManager"
```
Expected: `100% tests passed, 0 tests failed out of 4`.

- [ ] **Step 8: Commit**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna"
git add tuna-guest-agent/CMakeLists.txt tuna-guest-agent/src/core/config_manager.h tuna-guest-agent/src/core/config_manager.cpp tuna-guest-agent/tests/unit/test_config.cpp
git commit -m "feat(agent): implement ConfigManager with real JSON loading"
```

---

### Task 3: Agent `core/logger` — spdlog wrapper

**Files:**
- Modify: `tuna-guest-agent/src/core/logger.cpp` (currently empty)
- Create: `tuna-guest-agent/src/core/logger.h`
- Modify: `tuna-guest-agent/CMakeLists.txt` (add spdlog dependency, add `logger.cpp` to `tuna_core`)
- Test: `tuna-guest-agent/tests/unit/test_logger.cpp` (new)

- [ ] **Step 1: Add spdlog via FetchContent**

Add to `tuna-guest-agent/CMakeLists.txt`, next to the nlohmann_json block:
```cmake
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
)
FetchContent_MakeAvailable(spdlog)
```

- [ ] **Step 2: Write the failing test**

`tuna-guest-agent/tests/unit/test_logger.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include "core/logger.h"

TEST_CASE("Logger initializes and writes to a log file", "[logger]") {
    auto dir = std::filesystem::temp_directory_path() / "tuna_test_logs";
    std::filesystem::remove_all(dir);

    REQUIRE(tuna::Logger::Init(dir.string(), "test_agent"));
    tuna::Logger::Get()->info("hello from test");
    tuna::Logger::Get()->flush();

    auto logFile = dir / "test_agent.log";
    REQUIRE(std::filesystem::exists(logFile));

    std::ifstream in(logFile);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    REQUIRE(content.find("hello from test") != std::string::npos);
}

TEST_CASE("Logger rejects re-initialization without shutdown", "[logger]") {
    auto dir = std::filesystem::temp_directory_path() / "tuna_test_logs_2";
    std::filesystem::remove_all(dir);
    REQUIRE(tuna::Logger::Init(dir.string(), "test_agent_2"));
    REQUIRE_FALSE(tuna::Logger::Init(dir.string(), "test_agent_2"));
    tuna::Logger::Shutdown();
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cmake -B build -S .
cmake --build build --target tuna_tests
```
Expected: FAIL — `core/logger.h` does not exist.

- [ ] **Step 4: Write `logger.h`**

`tuna-guest-agent/src/core/logger.h`:
```cpp
#pragma once

#include <memory>
#include <string>
#include <spdlog/spdlog.h>

namespace tuna {

class Logger {
public:
    // Creates a rotating file logger at <log_dir>/<app_name>.log.
    // Returns false if already initialized (call Shutdown() first to re-init).
    static bool Init(const std::string& log_dir, const std::string& app_name);
    static std::shared_ptr<spdlog::logger> Get();
    static void Shutdown();
};

}  // namespace tuna
```

- [ ] **Step 5: Write `logger.cpp`**

`tuna-guest-agent/src/core/logger.cpp`:
```cpp
#include "core/logger.h"

#include <filesystem>
#include <spdlog/sinks/rotating_file_sink.h>

namespace tuna {

namespace {
std::shared_ptr<spdlog::logger> g_logger;
}

bool Logger::Init(const std::string& log_dir, const std::string& app_name) {
    if (g_logger) {
        return false;
    }

    std::filesystem::create_directories(log_dir);
    auto path = std::filesystem::path(log_dir) / (app_name + ".log");

    // 5 MB per file, keep 3 rotated files.
    g_logger = std::make_shared<spdlog::logger>(
        app_name,
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(path.string(), 5 * 1024 * 1024, 3));
    g_logger->set_level(spdlog::level::info);
    g_logger->flush_on(spdlog::level::warn);
    return true;
}

std::shared_ptr<spdlog::logger> Logger::Get() {
    return g_logger;
}

void Logger::Shutdown() {
    g_logger.reset();
}

}  // namespace tuna
```

- [ ] **Step 6: Wire into the build**

Add `src/core/logger.cpp` to the `tuna_core` target sources in `tuna-guest-agent/CMakeLists.txt`, and add `spdlog::spdlog` to its `target_link_libraries`. Add `tests/unit/test_logger.cpp` to `tuna_tests`'s sources.

- [ ] **Step 7: Run tests to verify they pass**

```bash
cmake -B build -S .
cmake --build build --target tuna_tests
ctest --test-dir build --output-on-failure -R "Logger"
```
Expected: `100% tests passed, 0 tests failed out of 2`.

- [ ] **Step 8: Commit**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna"
git add tuna-guest-agent/CMakeLists.txt tuna-guest-agent/src/core/logger.h tuna-guest-agent/src/core/logger.cpp tuna-guest-agent/tests/unit/test_logger.cpp
git commit -m "feat(agent): implement Logger as a spdlog rotating-file wrapper"
```

---

### Task 4: Agent `core/error_handler` — centralized error codes

**Files:**
- Modify: `tuna-guest-agent/src/core/error_handler.cpp` (currently empty)
- Create: `tuna-guest-agent/src/core/error_handler.h`
- Modify: `tuna-guest-agent/CMakeLists.txt` (add `error_handler.cpp` to `tuna_core`)
- Test: `tuna-guest-agent/tests/unit/test_error_handler.cpp` (new)

- [ ] **Step 1: Write the failing test**

`tuna-guest-agent/tests/unit/test_error_handler.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "core/error_handler.h"

TEST_CASE("ToString covers every TunaError value", "[error_handler]") {
    using tuna::TunaError;
    CHECK(tuna::ToString(TunaError::Ok) == "Ok");
    CHECK(tuna::ToString(TunaError::ConfigNotFound) == "ConfigNotFound");
    CHECK(tuna::ToString(TunaError::ConfigParseError) == "ConfigParseError");
    CHECK(tuna::ToString(TunaError::LoggerInitFailed) == "LoggerInitFailed");
    CHECK(tuna::ToString(TunaError::DaemonAlreadyRunning) == "DaemonAlreadyRunning");
    CHECK(tuna::ToString(TunaError::DaemonNotRunning) == "DaemonNotRunning");
}

TEST_CASE("IsError distinguishes Ok from failure codes", "[error_handler]") {
    CHECK_FALSE(tuna::IsError(tuna::TunaError::Ok));
    CHECK(tuna::IsError(tuna::TunaError::ConfigNotFound));
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake -B build -S .
cmake --build build --target tuna_tests
```
Expected: FAIL — `core/error_handler.h` does not exist.

- [ ] **Step 3: Write `error_handler.h`**

`tuna-guest-agent/src/core/error_handler.h`:
```cpp
#pragma once

#include <string>

namespace tuna {

enum class TunaError {
    Ok,
    ConfigNotFound,
    ConfigParseError,
    LoggerInitFailed,
    DaemonAlreadyRunning,
    DaemonNotRunning,
};

std::string ToString(TunaError error);
bool IsError(TunaError error);

}  // namespace tuna
```

- [ ] **Step 4: Write `error_handler.cpp`**

`tuna-guest-agent/src/core/error_handler.cpp`:
```cpp
#include "core/error_handler.h"

namespace tuna {

std::string ToString(TunaError error) {
    switch (error) {
        case TunaError::Ok: return "Ok";
        case TunaError::ConfigNotFound: return "ConfigNotFound";
        case TunaError::ConfigParseError: return "ConfigParseError";
        case TunaError::LoggerInitFailed: return "LoggerInitFailed";
        case TunaError::DaemonAlreadyRunning: return "DaemonAlreadyRunning";
        case TunaError::DaemonNotRunning: return "DaemonNotRunning";
    }
    return "Unknown";
}

bool IsError(TunaError error) {
    return error != TunaError::Ok;
}

}  // namespace tuna
```

- [ ] **Step 5: Wire into the build**

Add `src/core/error_handler.cpp` to `tuna_core`'s sources, and `tests/unit/test_error_handler.cpp` to `tuna_tests`'s sources.

- [ ] **Step 6: Run tests to verify they pass**

```bash
cmake -B build -S .
cmake --build build --target tuna_tests
ctest --test-dir build --output-on-failure -R "error_handler|ToString|IsError"
```
Expected: `100% tests passed, 0 tests failed out of 2`.

- [ ] **Step 7: Commit**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna"
git add tuna-guest-agent/CMakeLists.txt tuna-guest-agent/src/core/error_handler.h tuna-guest-agent/src/core/error_handler.cpp tuna-guest-agent/tests/unit/test_error_handler.cpp
git commit -m "feat(agent): implement centralized TunaError codes"
```

---

### Task 5: Server `core/` — config loading + logging (independent implementation)

**Context:** Per the design doc, the server's `core/` is *not* a shared binary with the agent's — it gets its own implementation, though it follows the same conventions (nlohmann/json + spdlog). The server's config is simpler: it listens rather than dials out, so there's no `server_host`/`server_port` to connect *to* — instead it has a `listen_port`.

**Files:**
- Create: `tuna-server/src/core/config_manager.h`
- Create: `tuna-server/src/core/config_manager.cpp`
- Create: `tuna-server/src/core/logger.h`
- Create: `tuna-server/src/core/logger.cpp`
- Modify: `tuna-server/CMakeLists.txt`
- Test: `tuna-server/tests/test_config.cpp` (new)
- Test: `tuna-server/tests/test_logger.cpp` (new)

- [ ] **Step 1: Add nlohmann/json and spdlog to `tuna-server/CMakeLists.txt`**

Add next to the protobuf block added in Task 1:
```cmake
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
)
FetchContent_MakeAvailable(spdlog)
```

- [ ] **Step 2: Write the failing tests**

`tuna-server/tests/test_config.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <filesystem>
#include "core/config_manager.h"

namespace {
std::filesystem::path writeTempConfig(const std::string& json) {
    auto path = std::filesystem::temp_directory_path() / "tuna_server_test_config.json";
    std::ofstream out(path);
    out << json;
    return path;
}
}

TEST_CASE("ServerConfigManager loads valid config", "[config]") {
    auto path = writeTempConfig(R"({"listen_port": 4433, "log_level": "debug"})");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE(result.has_value());
    CHECK(result->listen_port == 4433);
    CHECK(result->log_level == "debug");
}

TEST_CASE("ServerConfigManager rejects missing required field", "[config]") {
    auto path = writeTempConfig(R"({"log_level": "info"})");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("ServerConfigManager rejects missing file", "[config]") {
    auto result = tuna::ConfigManager::LoadFromFile("C:/definitely/not/a/real/path.json");
    REQUIRE_FALSE(result.has_value());
}
```

`tuna-server/tests/test_logger.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include "core/logger.h"

TEST_CASE("Server Logger initializes and writes to a log file", "[logger]") {
    auto dir = std::filesystem::temp_directory_path() / "tuna_server_test_logs";
    std::filesystem::remove_all(dir);

    REQUIRE(tuna::Logger::Init(dir.string(), "test_server"));
    tuna::Logger::Get()->info("hello from server test");
    tuna::Logger::Get()->flush();

    auto logFile = dir / "test_server.log";
    REQUIRE(std::filesystem::exists(logFile));

    std::ifstream in(logFile);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    REQUIRE(content.find("hello from server test") != std::string::npos);
    tuna::Logger::Shutdown();
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna\tuna-server"
cmake -B build -S .
cmake --build build --target tuna_server_tests
```
Expected: FAIL — `core/config_manager.h` and `core/logger.h` don't exist yet.

- [ ] **Step 4: Write `core/config_manager.h` and `.cpp`**

`tuna-server/src/core/config_manager.h`:
```cpp
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tuna {

struct ServerConfig {
    uint16_t listen_port = 0;
    std::string log_level = "info";
};

class ConfigManager {
public:
    static std::optional<ServerConfig> LoadFromFile(const std::string& path);
};

}  // namespace tuna
```

`tuna-server/src/core/config_manager.cpp`:
```cpp
#include "core/config_manager.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace tuna {

std::optional<ServerConfig> ConfigManager::LoadFromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return std::nullopt;
    }

    nlohmann::json parsed;
    try {
        in >> parsed;
    } catch (const nlohmann::json::parse_error&) {
        return std::nullopt;
    }

    if (!parsed.contains("listen_port")) {
        return std::nullopt;
    }

    ServerConfig config;
    try {
        config.listen_port = parsed.at("listen_port").get<uint16_t>();
        if (parsed.contains("log_level")) {
            config.log_level = parsed.at("log_level").get<std::string>();
        }
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }

    return config;
}

}  // namespace tuna
```

- [ ] **Step 5: Write `core/logger.h` and `.cpp`**

`tuna-server/src/core/logger.h` — identical in shape to the agent's (separate copy, per the design doc's "not a shared binary" note):
```cpp
#pragma once

#include <memory>
#include <string>
#include <spdlog/spdlog.h>

namespace tuna {

class Logger {
public:
    static bool Init(const std::string& log_dir, const std::string& app_name);
    static std::shared_ptr<spdlog::logger> Get();
    static void Shutdown();
};

}  // namespace tuna
```

`tuna-server/src/core/logger.cpp`:
```cpp
#include "core/logger.h"

#include <filesystem>
#include <spdlog/sinks/rotating_file_sink.h>

namespace tuna {

namespace {
std::shared_ptr<spdlog::logger> g_logger;
}

bool Logger::Init(const std::string& log_dir, const std::string& app_name) {
    if (g_logger) {
        return false;
    }

    std::filesystem::create_directories(log_dir);
    auto path = std::filesystem::path(log_dir) / (app_name + ".log");

    g_logger = std::make_shared<spdlog::logger>(
        app_name,
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(path.string(), 5 * 1024 * 1024, 3));
    g_logger->set_level(spdlog::level::info);
    g_logger->flush_on(spdlog::level::warn);
    return true;
}

std::shared_ptr<spdlog::logger> Logger::Get() {
    return g_logger;
}

void Logger::Shutdown() {
    g_logger.reset();
}

}  // namespace tuna
```

- [ ] **Step 6: Wire into the build**

Add a `tuna_server_core` static library to `tuna-server/CMakeLists.txt` (near the top, after `project()`):
```cmake
add_library(tuna_server_core STATIC
    src/core/config_manager.cpp
    src/core/logger.cpp
)
target_include_directories(tuna_server_core PUBLIC src)
target_link_libraries(tuna_server_core PUBLIC nlohmann_json::nlohmann_json spdlog::spdlog)
```
Add `tests/test_config.cpp` and `tests/test_logger.cpp` to `tuna_server_tests`'s sources, and add `tuna_server_core` to its `target_link_libraries`.

- [ ] **Step 7: Run tests to verify they pass**

```bash
cmake -B build -S .
cmake --build build --target tuna_server_tests
ctest --test-dir build --output-on-failure
```
Expected: `100% tests passed, 0 tests failed out of 6` (1 proto + 3 config + 1 logger + 1 pre-existing scaffolding smoke test).

- [ ] **Step 8: Commit**

```bash
cd "c:\Users\tsuma.thomas\Documents\tuna"
git add tuna-server/CMakeLists.txt tuna-server/src/core tuna-server/tests/test_config.cpp tuna-server/tests/test_logger.cpp
git commit -m "feat(server): implement ConfigManager and Logger"
```

---

## End-of-Chunk-1 verification

- [ ] `ctest --test-dir tuna-guest-agent/build --output-on-failure` — all tests pass (config x4, logger x2, error_handler x2, frame proto x1, plus Phase 0's hooks/tunnel/integration scaffolding smoke tests).
- [ ] `ctest --test-dir tuna-server/build --output-on-failure` — all tests pass (config x3, logger x1, frame proto x1, plus Phase 0's scaffolding smoke test).
- [ ] `git log --oneline -6` shows the 5 commits from this chunk.

**Next:** Chunk 2 covers the `DaemonCore` lifecycle state machine, the Windows SCM wrapper around it, and writing `docs/security_model.md` for real — proceed once Chunk 1 is verified green.

#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves CMake/Catch2/CTest discovery works.
// Replaced with real config-loading tests once src/core/config_manager.cpp
// is implemented (Phase 1).
TEST_CASE("test harness is wired up for core/config_manager", "[scaffolding][config]") {
    REQUIRE(1 + 1 == 2);
}

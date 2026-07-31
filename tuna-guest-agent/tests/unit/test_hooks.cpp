#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves CMake/Catch2/CTest discovery works.
// Replaced with real hook-manager tests once src/hooks/ has an
// implementation (Phase 3 for the first CreateFileW hook).
TEST_CASE("test harness is wired up for hooks/win", "[scaffolding][hooks]") {
    REQUIRE(1 + 1 == 2);
}

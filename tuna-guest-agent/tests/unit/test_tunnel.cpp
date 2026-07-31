#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves CMake/Catch2/CTest discovery works.
// Replaced with real frame-serialization round-trip tests once
// src/tunnel/ is implemented (Phase 2).
TEST_CASE("test harness is wired up for tunnel/", "[scaffolding][tunnel]") {
    REQUIRE(1 + 1 == 2);
}

#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves CMake/Catch2/CTest discovery works for
// tuna-server. Replaced with real session/tunnel tests starting Phase 1.
TEST_CASE("tuna-server test harness is wired up", "[scaffolding]") {
    REQUIRE(1 + 1 == 2);
}

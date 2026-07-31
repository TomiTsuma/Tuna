#include <catch2/catch_test_macros.hpp>

// Scaffolding smoke test: proves the integration test binary builds and
// CTest discovers it. Replaced with a real localhost agent<->server
// handshake test once Phase 2's tunnel exists.
TEST_CASE("integration test harness is wired up", "[scaffolding][integration]") {
    REQUIRE(1 + 1 == 2);
}

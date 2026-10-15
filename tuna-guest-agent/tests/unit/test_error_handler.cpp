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

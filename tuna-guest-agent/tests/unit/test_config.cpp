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
}  // namespace

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

TEST_CASE("ConfigManager defaults log_level to info when absent", "[config]") {
    auto path = writeTempConfig(R"({"server_host": "h", "server_port": 1})");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE(result.has_value());
    CHECK(result->log_level == "info");
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

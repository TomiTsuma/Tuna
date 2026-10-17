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
}  // namespace

TEST_CASE("ServerConfigManager loads valid config", "[config]") {
    auto path = writeTempConfig(R"({"listen_port": 4433, "log_level": "debug"})");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE(result.has_value());
    CHECK(result->listen_port == 4433);
    CHECK(result->log_level == "debug");
}

TEST_CASE("ServerConfigManager defaults log_level to info", "[config]") {
    auto path = writeTempConfig(R"({"listen_port": 4433})");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE(result.has_value());
    CHECK(result->log_level == "info");
    CHECK(result->listen_address == "127.0.0.1");
}

TEST_CASE("ServerConfigManager reads an explicit listen_address", "[config]") {
    auto path = writeTempConfig(R"({"listen_port": 4433, "listen_address": "::1"})");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE(result.has_value());
    CHECK(result->listen_address == "::1");
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

TEST_CASE("ServerConfigManager reads TLS and session settings", "[config]") {
    auto path = writeTempConfig(R"({"listen_port": 4433, "listen_address": "0.0.0.0",
        "max_sessions": 2, "worker_mode": "thread",
        "tls": {"ca_cert": "ca.crt", "cert": "s.crt", "key": "s.key", "allowed_client_names": ["alice", "bob"]}})");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE(result.has_value());
    CHECK(result->max_sessions == 2);
    CHECK(result->worker_mode == "thread");
    REQUIRE(result->tls.has_value());
    CHECK(result->tls->ca_cert == "ca.crt");
    CHECK(result->tls->allowed_client_names == std::vector<std::string>{"alice", "bob"});
}

TEST_CASE("ServerConfigManager defaults to process workers without TLS", "[config]") {
    auto path = writeTempConfig(R"({"listen_port": 4433})");
    auto result = tuna::ConfigManager::LoadFromFile(path.string());
    REQUIRE(result.has_value());
    CHECK(result->worker_mode == "process");
    CHECK(result->max_sessions == 8);
    CHECK_FALSE(result->tls.has_value());
}

TEST_CASE("ServerConfigManager rejects unsafe or invalid session settings", "[config]") {
    for (const char* json : {
             R"({"listen_port": 1, "tls": {"ca_cert": "a", "cert": "b", "key": "c", "allowed_client_names": []}})",
             R"({"listen_port": 1, "tls": {"ca_cert": "a", "cert": "b", "key": "c"}})",
             R"({"listen_port": 1, "worker_mode": "fork"})",
             R"({"listen_port": 1, "max_sessions": 0})",
         }) {
        INFO(json);
        CHECK_FALSE(tuna::ConfigManager::LoadFromFile(writeTempConfig(json).string()).has_value());
    }
}

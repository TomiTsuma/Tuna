#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
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

    tuna::Logger::Shutdown();
}

TEST_CASE("Logger rejects re-initialization without shutdown", "[logger]") {
    auto dir = std::filesystem::temp_directory_path() / "tuna_test_logs_2";
    std::filesystem::remove_all(dir);
    REQUIRE(tuna::Logger::Init(dir.string(), "test_agent_2"));
    REQUIRE_FALSE(tuna::Logger::Init(dir.string(), "test_agent_2"));
    tuna::Logger::Shutdown();
}

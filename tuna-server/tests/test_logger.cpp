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

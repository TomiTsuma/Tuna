#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tuna {

struct AgentConfig {
    std::string server_host;
    uint16_t server_port = 0;
    std::string log_level = "info";
};

class ConfigManager {
public:
    // Returns std::nullopt if the file is missing, unreadable, not valid JSON,
    // or missing a required field (server_host, server_port).
    static std::optional<AgentConfig> LoadFromFile(const std::string& path);
};

}  // namespace tuna

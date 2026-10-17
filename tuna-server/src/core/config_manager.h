#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tuna {

struct ServerConfig {
    uint16_t listen_port = 0;
    std::string log_level = "info";
};

class ConfigManager {
public:
    // Returns std::nullopt if the file is missing, unreadable, not valid JSON,
    // or missing the required field (listen_port).
    static std::optional<ServerConfig> LoadFromFile(const std::string& path);
};

}  // namespace tuna

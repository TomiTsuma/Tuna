#include "core/config_manager.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace tuna {

std::optional<AgentConfig> ConfigManager::LoadFromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return std::nullopt;
    }

    nlohmann::json parsed;
    try {
        in >> parsed;
    } catch (const nlohmann::json::parse_error&) {
        return std::nullopt;
    }

    if (!parsed.contains("server_host") || !parsed.contains("server_port")) {
        return std::nullopt;
    }

    AgentConfig config;
    try {
        config.server_host = parsed.at("server_host").get<std::string>();
        config.server_port = parsed.at("server_port").get<uint16_t>();
        if (parsed.contains("log_level")) {
            config.log_level = parsed.at("log_level").get<std::string>();
        }
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }

    return config;
}

}  // namespace tuna

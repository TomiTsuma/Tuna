#include "core/config_manager.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace tuna {

std::optional<ServerConfig> ConfigManager::LoadFromFile(const std::string& path) {
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

    if (!parsed.contains("listen_port")) {
        return std::nullopt;
    }

    ServerConfig config;
    try {
        config.listen_port = parsed.at("listen_port").get<uint16_t>();
        if (parsed.contains("log_level")) {
            config.log_level = parsed.at("log_level").get<std::string>();
        }
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }

    return config;
}

}  // namespace tuna

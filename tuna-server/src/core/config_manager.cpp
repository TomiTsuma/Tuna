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
        if (parsed.contains("listen_address")) {
            config.listen_address = parsed.at("listen_address").get<std::string>();
        }
        if (parsed.contains("log_level")) {
            config.log_level = parsed.at("log_level").get<std::string>();
        }
        if (parsed.contains("max_sessions")) {
            config.max_sessions = parsed.at("max_sessions").get<uint32_t>();
            if (config.max_sessions == 0) return std::nullopt;
        }
        if (parsed.contains("worker_mode")) {
            config.worker_mode = parsed.at("worker_mode").get<std::string>();
            if (config.worker_mode != "process" && config.worker_mode != "thread") return std::nullopt;
        }
        if (parsed.contains("cuda")) {
            const auto& cuda = parsed.at("cuda");
            config.cuda.enabled = cuda.value("enabled", true);
            config.cuda.runtime_library = cuda.value("runtime_library", std::string());
            config.cuda.driver_library = cuda.value("driver_library", std::string());
            config.cuda.cublas_library = cuda.value("cublas_library", std::string());
            config.cuda.cudnn_library = cuda.value("cudnn_library", std::string());
        }
        if (parsed.contains("tls")) {
            const auto& tls = parsed.at("tls");
            ServerTlsConfig t;
            t.ca_cert = tls.at("ca_cert").get<std::string>();
            t.cert = tls.at("cert").get<std::string>();
            t.key = tls.at("key").get<std::string>();
            t.allowed_client_names = tls.at("allowed_client_names").get<std::vector<std::string>>();
            // An empty allowlist would authorize nobody; treat it as a config error
            // rather than starting a server no client can use.
            if (t.ca_cert.empty() || t.cert.empty() || t.key.empty() || t.allowed_client_names.empty()) {
                return std::nullopt;
            }
            config.tls = std::move(t);
        }
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }

    return config;
}

}  // namespace tuna

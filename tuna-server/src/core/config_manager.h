#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tuna {

// Track B mTLS settings (`"tls": {...}`); paths are PEM files.
struct ServerTlsConfig {
    std::string ca_cert;                        // trust root for client certificates
    std::string cert;                           // server certificate
    std::string key;                            // server private key
    std::vector<std::string> allowed_client_names;  // exact DNS/URI SAN values; required
};

struct ServerConfig {
    uint16_t listen_port = 0;
    // Loopback by default. Without `tls` the server refuses non-loopback
    // addresses: plaintext GPU RPC must never reach a network.
    std::string listen_address = "127.0.0.1";
    std::string log_level = "info";
    std::optional<ServerTlsConfig> tls;
    // Concurrent sessions, including ones still handshaking. Each is a worker
    // process (and, from Phase 3, a CUDA context), so this bounds GPU memory
    // and process count; connections over the limit are closed immediately.
    uint32_t max_sessions = 8;
    // "process" (default): one worker process per session. "thread": sessions
    // run inside the listener, for debugging only - a crashing session then
    // takes the listener down with it.
    std::string worker_mode = "process";
};

class ConfigManager {
public:
    // Returns std::nullopt if the file is missing, unreadable, not valid JSON,
    // or missing the required field (listen_port).
    static std::optional<ServerConfig> LoadFromFile(const std::string& path);
};

}  // namespace tuna

#pragma once

#include <memory>
#include <string>
#include <spdlog/spdlog.h>

namespace tuna {

class Logger {
public:
    // Creates a rotating file logger at <log_dir>/<app_name>.log.
    // Returns false if already initialized (call Shutdown() first to re-init).
    static bool Init(const std::string& log_dir, const std::string& app_name);
    static std::shared_ptr<spdlog::logger> Get();
    static void Shutdown();
};

}  // namespace tuna

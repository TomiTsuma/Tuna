#include "core/logger.h"

#include <filesystem>
#include <spdlog/sinks/rotating_file_sink.h>

namespace tuna {

namespace {
std::shared_ptr<spdlog::logger> g_logger;
}

bool Logger::Init(const std::string& log_dir, const std::string& app_name) {
    if (g_logger) {
        return false;
    }

    std::filesystem::create_directories(log_dir);
    auto path = std::filesystem::path(log_dir) / (app_name + ".log");

    // 5 MB per file, keep 3 rotated files.
    g_logger = std::make_shared<spdlog::logger>(
        app_name,
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(path.string(), 5 * 1024 * 1024, 3));
    g_logger->set_level(spdlog::level::info);
    g_logger->flush_on(spdlog::level::warn);
    return true;
}

std::shared_ptr<spdlog::logger> Logger::Get() {
    return g_logger;
}

void Logger::Shutdown() {
    g_logger.reset();
}

}  // namespace tuna

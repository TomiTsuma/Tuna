#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "core/config_manager.h"
#include "core/logger.h"

#ifdef TUNA_SERVER_WORKLOAD_API
#include "workload/workload_server.h"
#endif

namespace {

constexpr const char* kVersion = "Tuna Server 0.0.1";

// One binary, two tracks (see the repository README):
//   Track A - explicit-workload gRPC/mTLS server (workload/), any non --foreground
//             command line, compiled in when TUNA_SERVER_WORKLOAD_API is on.
//   Track B - driver-API remoting server, `--foreground [--config PATH]`.
void PrintUsage(const char* executable) {
    std::puts("Usage: tuna_server --foreground [--config <path>]");
    std::puts("  --foreground   Remoting track: load config, start logging, run in the foreground.");
    std::puts("  --config PATH  Path to a JSON config file (default: ./tuna-server.json).");
#ifdef TUNA_SERVER_WORKLOAD_API
    std::fflush(stdout);
    PrintWorkloadServerUsage(executable);
#else
    (void)executable;
    std::puts("(workload API not compiled in; rebuild with -DTUNA_SERVER_WORKLOAD_API=ON)");
#endif
}

// Foreground mode: the Phase 1 milestone. Proves config loading and logging
// are wired end-to-end. The accept/listen loop arrives in Phase 2 (transport).
int RunForeground(const std::string& config_path) {
    auto config = tuna::ConfigManager::LoadFromFile(config_path);
    if (!config) {
        std::fprintf(stderr, "Failed to load config from '%s'\n", config_path.c_str());
        return 1;
    }

    const auto log_dir = (std::filesystem::temp_directory_path() / "tuna-server-logs").string();
    if (!tuna::Logger::Init(log_dir, "tuna-server")) {
        std::fprintf(stderr, "Failed to initialize logger\n");
        return 1;
    }

    auto log = tuna::Logger::Get();
    // spdlog::level::from_str returns `off` for unrecognized names; treat that as
    // a config typo and keep the default rather than silently disabling logging.
    const auto level = spdlog::level::from_str(config->log_level);
    if (level == spdlog::level::off && config->log_level != "off") {
        log->warn("unknown log_level '{}', keeping 'info'", config->log_level);
    } else {
        log->set_level(level);
    }
    log->info("{} starting in foreground", kVersion);
    log->info("config: listen_port={}, log_level={}", config->listen_port, config->log_level);
    log->info("transport listener not yet implemented (Phase 2); exiting cleanly");
    log->flush();

    std::printf("%s foreground: config OK (listen_port=%u, log_level=%s), logs at %s\n",
                kVersion, config->listen_port, config->log_level.c_str(), log_dir.c_str());

    tuna::Logger::Shutdown();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool foreground = false;
    std::string config_path = "tuna-server.json";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--foreground") == 0) {
            foreground = true;
        } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    if (foreground) {
        return RunForeground(config_path);
    }

#ifdef TUNA_SERVER_WORKLOAD_API
    return RunWorkloadServer(argc, argv);
#else
    std::printf("%s (scaffold). Run with --foreground to load config and start logging.\n",
                kVersion);
    return 0;
#endif
}

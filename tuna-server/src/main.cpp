#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "core/config_manager.h"
#include "core/logger.h"
#include "remoting/remoting_server.h"
#include "remoting/worker_session.h"
#include "tuna/tunnel/pki.h"

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
    std::puts("  --foreground   Remoting track: load config, start logging, serve sessions until Ctrl-C.");
    std::puts("  --config PATH  Path to a JSON config file (default: ./tuna-server.json).");
    std::puts("");
    std::puts("Certificate tooling (remoting track mTLS; EC P-256; never overwrites files):");
    std::puts("  --gen-ca DIR [--cn NAME] [--days N]          Create DIR/ca.key and DIR/ca.crt.");
    std::puts("  --issue-client-cert NAME --ca-dir DIR --out DIR [--days N]");
    std::puts("                                               Client cert with SAN DNS:NAME (add NAME to");
    std::puts("                                               tls.allowed_client_names to authorize it).");
    std::puts("  --issue-server-cert NAME --ca-dir DIR --out DIR [--san DNS:host,IP:10.0.0.5] [--days N]");
    std::puts("                                               Server cert; SANs default to DNS:NAME.");
#ifdef TUNA_SERVER_WORKLOAD_API
    std::fflush(stdout);
    PrintWorkloadServerUsage(executable);
#else
    (void)executable;
    std::puts("(workload API not compiled in; rebuild with -DTUNA_SERVER_WORKLOAD_API=ON)");
#endif
}

void HandleStopSignal(int) { tuna::RequestRemotingShutdown(); }

// Foreground mode (Track B): load config, start logging, and serve remoting
// sessions until SIGINT/SIGTERM.
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
    log->info("config: listen_address={}, listen_port={}, log_level={}, tls={}, worker_mode={}",
              config->listen_address, config->listen_port, config->log_level, config->tls ? "on" : "off",
              config->worker_mode);

    std::printf("%s foreground: config OK (listen_port=%u, log_level=%s), logs at %s\n",
                kVersion, config->listen_port, config->log_level.c_str(), log_dir.c_str());
    std::fflush(stdout);

    std::signal(SIGINT, HandleStopSignal);
    std::signal(SIGTERM, HandleStopSignal);
#ifdef SIGBREAK
    std::signal(SIGBREAK, HandleStopSignal);  // Windows Ctrl-Break / CTRL_BREAK_EVENT
#endif
    const int exit_code = tuna::ServeRemoting(*config, config_path, log);
    log->info("{} stopped (exit code {})", kVersion, exit_code);
    log->flush();

    tuna::Logger::Shutdown();
    return exit_code;
}

// --gen-ca / --issue-*-cert. Returns the exit code.
int RunCertTool(const std::string& mode, const std::string& subject, const std::string& ca_dir,
                const std::string& out_dir, const std::string& cn, const std::string& sans, int days) {
    std::string error;
    bool ok;
    if (mode == "--gen-ca") {
        ok = tuna::tunnel::GenerateCa(subject, cn.empty() ? "Tuna Remoting CA" : cn, days > 0 ? days : 3650, &error);
        if (ok) std::printf("wrote %s/ca.key and %s/ca.crt (keep ca.key offline)\n", subject.c_str(), subject.c_str());
    } else {
        if (ca_dir.empty() || out_dir.empty()) {
            std::fprintf(stderr, "%s needs --ca-dir DIR and --out DIR\n", mode.c_str());
            return 2;
        }
        tuna::tunnel::IssueOptions options;
        options.name = subject;
        options.server = mode == "--issue-server-cert";
        if (days > 0) options.days = days;
        for (size_t start = 0; !sans.empty() && start <= sans.size();) {
            const size_t comma = sans.find(',', start);
            options.sans.push_back(sans.substr(start, comma - start));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        ok = tuna::tunnel::IssueCertificate(ca_dir, out_dir, options, &error);
        if (ok) {
            std::printf("wrote %s/%s.key and %s/%s.crt\n", out_dir.c_str(), subject.c_str(), out_dir.c_str(),
                        subject.c_str());
        }
    }
    if (!ok) std::fprintf(stderr, "%s failed: %s\n", mode.c_str(), error.c_str());
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    bool foreground = false;
    bool worker = false;
    std::string config_path = "tuna-server.json";
    std::string ack_handle;
    std::string cert_mode, cert_subject, ca_dir, out_dir, cn, sans;
    int days = 0;

    for (int i = 1; i < argc; ++i) {
        const auto value = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (std::strcmp(argv[i], "--foreground") == 0) {
            foreground = true;
        } else if (std::strcmp(argv[i], "--worker") == 0) {
            worker = true;  // internal: started by the listener, one per session
        } else if (std::strcmp(argv[i], "--ack-handle") == 0) {
            ack_handle = value();
        } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--gen-ca") == 0 || std::strcmp(argv[i], "--issue-client-cert") == 0 ||
                   std::strcmp(argv[i], "--issue-server-cert") == 0) {
            cert_mode = argv[i];
            cert_subject = value();
        } else if (std::strcmp(argv[i], "--ca-dir") == 0) {
            ca_dir = value();
        } else if (std::strcmp(argv[i], "--out") == 0) {
            out_dir = value();
        } else if (std::strcmp(argv[i], "--cn") == 0) {
            cn = value();
        } else if (std::strcmp(argv[i], "--san") == 0) {
            sans = value();
        } else if (std::strcmp(argv[i], "--days") == 0) {
            days = std::atoi(value().c_str());
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    if (worker) {
        return tuna::RunWorker(config_path, ack_handle);
    }
    if (!cert_mode.empty()) {
        if (cert_subject.empty()) {
            std::fprintf(stderr, "%s needs an argument\n", cert_mode.c_str());
            return 2;
        }
        return RunCertTool(cert_mode, cert_subject, ca_dir, out_dir, cn, sans, days);
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

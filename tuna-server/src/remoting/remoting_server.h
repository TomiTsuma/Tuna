#pragma once

#include <memory>

#include <spdlog/spdlog.h>

#include "core/config_manager.h"

namespace tuna {

// Track B session host: listens on config.listen_address:listen_port, runs the
// Hello handshake, and serves each session's RPCs on its own thread until
// RequestRemotingShutdown(). Returns the process exit code.
//
// Phase 2a: sessions are threads and only the diagnostic echo API is served.
// Phase 2c moves each session into its own worker process; Phase 3 plugs in
// the generated CUDA dispatchers.
int ServeRemoting(const ServerConfig& config, const std::shared_ptr<spdlog::logger>& log);

// Async-signal-safe: unblocks the accept loop so ServeRemoting() returns.
void RequestRemotingShutdown();

}  // namespace tuna

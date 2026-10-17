#pragma once

#include <memory>
#include <string>

#include <spdlog/spdlog.h>

#include "core/config_manager.h"

namespace tuna {

// Track B listener: accepts connections on config.listen_address:listen_port,
// reads each connection's 24-byte preamble, and routes the socket:
//   new session -> a fresh worker (`tuna_server --worker`, or a thread when
//                  worker_mode is "thread"), which does TLS, mTLS and Hello;
//   attach bulk -> the worker that owns the presented session token.
// The listener itself never runs TLS, never parses RPCs and never touches the
// GPU. Returns the process exit code once RequestRemotingShutdown() is called.
int ServeRemoting(const ServerConfig& config, const std::string& config_path,
                  const std::shared_ptr<spdlog::logger>& log);

// Async-signal-safe: unblocks the accept loop so ServeRemoting() returns.
void RequestRemotingShutdown();

}  // namespace tuna

#pragma once

#include <memory>

#include <spdlog/spdlog.h>

#include "tuna/tunnel/rpc.h"
#include "tuna_frames.pb.h"

namespace tuna {

// The server's half of Hello: protocol version, plus driver version and
// devices once a CUDA backend is present (Phase 3).
Hello ServerHello();

// A fresh dispatcher for one session. Every api_id it does not know returns
// CUDA_ERROR_NOT_SUPPORTED (801).
tunnel::Dispatcher MakeDispatcher(std::shared_ptr<spdlog::logger> log);

}  // namespace tuna

#pragma once

#include <memory>

#include <spdlog/spdlog.h>

#include "core/config_manager.h"
#include "tuna/tunnel/rpc.h"
#include "tuna_frames.pb.h"

namespace tuna {

// The server's half of Hello: protocol version, plus driver version and
// devices once a CUDA backend is present (Phase 3).
Hello ServerHello();

// A fresh dispatcher for one session. With CUDA remoting built in
// (TUNA_SERVER_CUDA) and enabled, CUDA api_ids go to a per-session
// CudaSession over the real CUDA stack, opened on the session's first CUDA
// call; if it cannot be opened they fail with cudaErrorNoDevice (100). Every
// other api_id except the echo returns cudaErrorNotSupported (801).
tunnel::Dispatcher MakeDispatcher(std::shared_ptr<spdlog::logger> log, const ServerCudaConfig& cuda = {});

}  // namespace tuna

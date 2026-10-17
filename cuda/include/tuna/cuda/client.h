#pragma once

#include <string>

#include "tuna/cuda/marshal.h"
#include "tuna/tunnel/rpc.h"

namespace tuna::cudart {

using tunnel::RpcClient;
using tunnel::CallResult;
using tunnel::kResultTransportError;

// Sends one marshalled call. Returns the reply, or transport_error on loss.
Reply Call(uint32_t api_id, const ArgWriter& in, const std::string& bulk_in = {});

// Binds the shim to an established session. Until this is set, every call
// returns transport_error.
void SetRpcClient(RpcClient* client);

// Test seam: route Call() to an in-process function instead of the transport.
// nullptr restores the transport path.
using CallHook = Reply (*)(uint32_t api_id, const std::string& args, const std::string& bulk_in);
void SetCallHook(CallHook hook);

}  // namespace tuna::cudart

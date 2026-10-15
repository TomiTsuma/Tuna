#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include "tuna/tunnel/session.h"
#include "tuna/tunnel/transport.h"

namespace tuna::tunnel {

// Reserved api_id: echoes args back. Used by tuna_ping and health checks;
// generated CUDA api_ids start at 1 and never use it.
constexpr uint32_t kApiEcho = 0;

// RpcRequest.flags
constexpr uint32_t kFlagAsync = 1u << 0;
constexpr uint32_t kFlagHasBulk = 1u << 1;

// RpcResponse.result is a CUresult. The client reports transport loss and
// call timeouts as CUDA_ERROR_UNKNOWN, which every CUDA caller already handles.
constexpr int32_t kResultSuccess = 0;
constexpr int32_t kResultTransportError = 999;

struct HeartbeatOptions {
    std::chrono::milliseconds interval{3000};
    std::chrono::milliseconds timeout{10000};
};

struct RpcClientOptions {
    size_t max_batch_calls = 256;             // flush when this many calls are queued
    size_t max_batch_bytes = 64 * 1024;       // ...or this many bytes of args
    std::chrono::microseconds flush_delay{500};  // ...or this long after the first Post()
    std::chrono::milliseconds call_timeout{30000};
    HeartbeatOptions heartbeat;
};

struct CallResult {
    int32_t result = kResultTransportError;
    std::string out_args;
};

// Client half of an established (post-handshake) session.
class RpcClient {
public:
    explicit RpcClient(Transport& transport, RpcClientOptions options = {});
    ~RpcClient();
    RpcClient(const RpcClient&) = delete;
    RpcClient& operator=(const RpcClient&) = delete;

    // Blocking call. Queued Post()s are sent first, in the same frame.
    // A `synchronizing` call (cuCtxSynchronize, cuStreamSynchronize, blocking
    // memcpy, ...) returns the first deferred async error instead of its own
    // result, if one is latched, and clears it.
    CallResult Call(uint32_t api_id, const std::string& args, bool synchronizing = false);

    // Fire-and-forget. Errors surface on the next synchronizing Call().
    void Post(uint32_t api_id, const std::string& args);

    bool connected() const { return connected_; }

    // Flushes queued posts, then closes. Idempotent; the destructor calls it.
    void Close();

private:
    void ReaderLoop();
    void FlusherLoop();
    bool FlushLocked();  // requires queue_mutex_
    void Enqueue(RpcRequest request, std::unique_lock<std::mutex>& lock);

    Transport& transport_;
    const RpcClientOptions options_;
    std::atomic<bool> connected_{true};
    std::atomic<uint64_t> next_call_id_{1};

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    RpcBatch batch_;            // guarded by queue_mutex_
    size_t batch_bytes_ = 0;    // guarded by queue_mutex_
    bool stopping_ = false;     // guarded by queue_mutex_

    std::mutex calls_mutex_;
    std::unordered_map<uint64_t, std::promise<CallResult>> pending_;  // guarded by calls_mutex_
    std::optional<int32_t> latched_error_;                            // guarded by calls_mutex_

    std::unique_ptr<Heartbeat> heartbeat_;
    std::thread reader_;
    std::thread flusher_;
    std::once_flag close_once_;
};

// Executes one call. The server fills in call_id; the dispatcher sets
// result and out_args.
using Dispatcher = std::function<RpcResponse(const RpcRequest&)>;

// Server half of an established session.
class RpcServer {
public:
    RpcServer(Transport& transport, Dispatcher dispatcher, HeartbeatOptions heartbeat = {});

    // Serves calls in arrival order until the transport closes or the peer
    // violates the protocol. Returns the number of calls executed.
    uint64_t Serve();

private:
    Transport& transport_;
    Dispatcher dispatcher_;
    HeartbeatOptions heartbeat_;
};

}  // namespace tuna::tunnel

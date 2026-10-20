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
#include <string_view>
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

// Bulk channel (Phase 2c). Payloads travel as BulkChunk frames (FrameType IO,
// channel 2), on a separate bulk socket when the session has one, otherwise on
// the main connection. Chunks are always sent before the RpcRequest or
// RpcResponse that references them.
constexpr size_t kBulkChunkBytes = 1024 * 1024;
constexpr uint64_t kMaxBulkTransferBytes = 4ull << 30;  // one payload
constexpr uint64_t kMaxBulkPendingBytes = 4ull << 30;   // incomplete payloads per receiver

// Reassembles bulk payloads from chunks that may arrive on either transport,
// in any order. Thread-safe.
class BulkStore {
public:
    // False on a protocol violation (oversized or inconsistent chunk).
    bool Add(const BulkChunk& chunk);
    // Waits until payload `ref` is complete and removes it. std::nullopt on
    // timeout or after Fail().
    std::optional<std::string> Take(uint64_t ref, std::chrono::milliseconds timeout);
    // The transports are gone; wakes and fails every waiter.
    void Fail();

private:
    struct Pending {
        std::string data;
        uint64_t received = 0;
    };
    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<uint64_t, Pending> pending_;  // guarded by mutex_
    uint64_t pending_bytes_ = 0;                     // guarded by mutex_
    bool failed_ = false;                            // guarded by mutex_
};

// Sends `data` as bulk payload `ref`. False if the transport closed.
bool SendBulk(Transport& transport, uint64_t ref, std::string_view data);

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
    std::string bulk_out;  // the response's bulk payload, if it had one
};

// Client half of an established (post-handshake) session.
class RpcClient {
public:
    // `bulk`, if given, is the session's separate bulk socket (owned by the
    // caller, closed by Close()); otherwise bulk payloads share `transport`.
    explicit RpcClient(Transport& transport, RpcClientOptions options = {}, Transport* bulk = nullptr);
    ~RpcClient();
    RpcClient(const RpcClient&) = delete;
    RpcClient& operator=(const RpcClient&) = delete;

    // Blocking call. Queued Post()s are sent first, in the same frame.
    // A `synchronizing` call (cuCtxSynchronize, cuStreamSynchronize, blocking
    // memcpy, ...) returns the first deferred async error instead of its own
    // result, if one is latched, and clears it.
    // `bulk_in`, if non-empty, is sent on the bulk channel (HAS_BULK).
    CallResult Call(uint32_t api_id, const std::string& args, bool synchronizing = false,
                    std::string_view bulk_in = {});

    // Fire-and-forget. Errors surface on the next synchronizing Call().
    void Post(uint32_t api_id, const std::string& args, std::string_view bulk_in = {});

    // The first failed Post() whose answer has already arrived, without a
    // round trip (and without flushing queued posts); `clear` resets it, as a
    // synchronizing Call() would. A failure still in flight is not seen yet.
    std::optional<int32_t> ArrivedAsyncError(bool clear);

    bool connected() const { return connected_; }

    // Flushes queued posts, then closes. Idempotent; the destructor calls it.
    void Close();

private:
    struct Reply {
        int32_t result = kResultTransportError;
        std::string out_args;
        uint64_t bulk_ref = 0;
    };

    void ReaderLoop(Transport& transport, bool main);
    void FlusherLoop();
    bool FlushLocked();  // requires queue_mutex_
    void Enqueue(RpcRequest request, std::unique_lock<std::mutex>& lock);
    bool AttachBulk(RpcRequest* request, std::string_view bulk_in);

    Transport& transport_;
    Transport* bulk_;  // bulk payloads go here; == &transport_ without a bulk socket
    const RpcClientOptions options_;
    std::atomic<bool> connected_{true};
    std::atomic<uint64_t> next_call_id_{1};

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    RpcBatch batch_;            // guarded by queue_mutex_
    size_t batch_bytes_ = 0;    // guarded by queue_mutex_
    bool stopping_ = false;     // guarded by queue_mutex_

    std::mutex calls_mutex_;
    std::unordered_map<uint64_t, std::promise<Reply>> pending_;  // guarded by calls_mutex_
    std::optional<int32_t> latched_error_;                       // guarded by calls_mutex_
    BulkStore bulk_in_;  // payloads of responses

    std::unique_ptr<Heartbeat> heartbeat_;
    std::thread reader_;
    std::thread bulk_reader_;  // only with a separate bulk socket
    std::thread flusher_;
    std::once_flag close_once_;
};

// One call as seen by a dispatcher.
struct ServerCall {
    const RpcRequest& request;
    std::string bulk_in;   // the request's complete bulk payload (HAS_BULK), else empty
    std::string bulk_out;  // set by the dispatcher to return a bulk payload
};

// Executes one call. The server fills in call_id and bulk_ref; the dispatcher
// sets result and out_args (and optionally call.bulk_out).
using Dispatcher = std::function<RpcResponse(ServerCall&)>;

// Server half of an established session.
class RpcServer {
public:
    RpcServer(Transport& transport, Dispatcher dispatcher, HeartbeatOptions heartbeat = {},
              std::chrono::milliseconds bulk_timeout = std::chrono::seconds(60));
    ~RpcServer();

    // Adds the session's bulk socket; may be called from another thread before
    // or during Serve(). The caller keeps ownership; Serve() closes it and
    // stops using it before returning.
    void AttachBulk(Transport* bulk);

    // Serves calls in arrival order until the transport closes or the peer
    // violates the protocol. Returns the number of calls executed.
    uint64_t Serve();

private:
    void BulkReaderLoop(Transport* bulk);
    void StopBulk();

    Transport& transport_;
    Dispatcher dispatcher_;
    HeartbeatOptions heartbeat_;
    const std::chrono::milliseconds bulk_timeout_;
    BulkStore bulk_in_;  // payloads of requests

    std::mutex bulk_mutex_;
    Transport* bulk_ = nullptr;  // guarded by bulk_mutex_
    bool serving_done_ = false;  // guarded by bulk_mutex_
    std::thread bulk_reader_;    // guarded by bulk_mutex_
};

}  // namespace tuna::tunnel

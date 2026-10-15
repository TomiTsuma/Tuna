#include "tuna/tunnel/rpc.h"

#include <utility>

namespace tuna::tunnel {

namespace {

uint64_t CurrentThreadId() {
    return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

}  // namespace

// ---------------------------------------------------------------- RpcClient

RpcClient::RpcClient(Transport& transport, RpcClientOptions options)
    : transport_(transport), options_(options) {
    heartbeat_ = std::make_unique<Heartbeat>(transport_, options_.heartbeat.interval,
                                             options_.heartbeat.timeout);
    reader_ = std::thread([this] { ReaderLoop(); });
    flusher_ = std::thread([this] { FlusherLoop(); });
}

RpcClient::~RpcClient() { Close(); }

void RpcClient::Close() {
    std::call_once(close_once_, [this] {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            FlushLocked();  // best effort: queued posts were promised to run
            stopping_ = true;
        }
        queue_cv_.notify_all();
        flusher_.join();
        transport_.Close();
        reader_.join();       // the reader uses heartbeat_, so it must stop first
        heartbeat_.reset();
    });
}

void RpcClient::Post(uint32_t api_id, const std::string& args) {
    RpcRequest request;
    request.set_call_id(next_call_id_++);
    request.set_api_id(api_id);
    request.set_flags(kFlagAsync);
    request.set_thread_id(CurrentThreadId());
    request.set_args(args);

    std::unique_lock<std::mutex> lock(queue_mutex_);
    Enqueue(std::move(request), lock);
}

CallResult RpcClient::Call(uint32_t api_id, const std::string& args, bool synchronizing) {
    const uint64_t call_id = next_call_id_++;
    std::future<CallResult> reply;
    {
        std::lock_guard<std::mutex> lock(calls_mutex_);
        if (!connected_) return CallResult{};
        reply = pending_[call_id].get_future();
    }

    RpcRequest request;
    request.set_call_id(call_id);
    request.set_api_id(api_id);
    request.set_thread_id(CurrentThreadId());
    request.set_args(args);
    bool sent;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        batch_.add_calls()->Swap(&request);  // rides behind any queued posts
        sent = FlushLocked();
    }

    CallResult result;
    if (sent && reply.wait_for(options_.call_timeout) == std::future_status::ready) {
        result = reply.get();
    } else {
        // Lost or timed out. The reader may still fulfil the promise later;
        // dropping the map entry makes that a no-op.
        std::lock_guard<std::mutex> lock(calls_mutex_);
        pending_.erase(call_id);
        result = CallResult{};
    }

    if (synchronizing) {
        std::lock_guard<std::mutex> lock(calls_mutex_);
        if (latched_error_) {
            result.result = *latched_error_;
            latched_error_.reset();
        }
    }
    return result;
}

void RpcClient::Enqueue(RpcRequest request, std::unique_lock<std::mutex>& lock) {
    const bool was_empty = batch_.calls_size() == 0;
    batch_bytes_ += request.args().size();
    batch_.add_calls()->Swap(&request);
    if (static_cast<size_t>(batch_.calls_size()) >= options_.max_batch_calls ||
        batch_bytes_ >= options_.max_batch_bytes) {
        FlushLocked();
    } else if (was_empty) {
        lock.unlock();
        queue_cv_.notify_one();  // start the flush-delay timer
    }
}

bool RpcClient::FlushLocked() {
    if (batch_.calls_size() == 0) return true;
    // Sending under queue_mutex_ keeps batches in the order calls were queued.
    const bool ok = transport_.Send(FRAME_TYPE_GPU, kRpcChannel, batch_.SerializeAsString());
    batch_.Clear();
    batch_bytes_ = 0;
    return ok;
}

void RpcClient::FlusherLoop() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    while (true) {
        queue_cv_.wait(lock, [this] { return stopping_ || batch_.calls_size() > 0; });
        if (stopping_) return;
        // Give more Post()s a chance to join this batch, unless a Call() or a
        // size limit flushes it first.
        if (queue_cv_.wait_for(lock, options_.flush_delay, [this] { return stopping_; })) return;
        FlushLocked();
    }
}

void RpcClient::ReaderLoop() {
    while (std::optional<Frame> frame = transport_.Receive()) {
        heartbeat_->NoteActivity();
        if (frame->header().type() != FRAME_TYPE_GPU) continue;  // heartbeats, future control

        RpcResponse response;
        if (!response.ParseFromString(frame->payload())) {
            transport_.Close();  // a peer that sends garbage is not trusted further
            break;
        }
        std::lock_guard<std::mutex> lock(calls_mutex_);
        auto it = pending_.find(response.call_id());
        if (it != pending_.end()) {
            it->second.set_value(CallResult{response.result(), std::move(*response.mutable_out_args())});
            pending_.erase(it);
        } else if (response.result() != kResultSuccess && !latched_error_) {
            latched_error_ = response.result();  // a failed async call; CUDA keeps the first
        }
    }

    // Connection gone: nothing in flight will ever be answered.
    std::lock_guard<std::mutex> lock(calls_mutex_);
    connected_ = false;
    for (auto& [id, promise] : pending_) {
        promise.set_value(CallResult{});
    }
    pending_.clear();
}

// ---------------------------------------------------------------- RpcServer

RpcServer::RpcServer(Transport& transport, Dispatcher dispatcher, HeartbeatOptions heartbeat)
    : transport_(transport), dispatcher_(std::move(dispatcher)), heartbeat_(heartbeat) {}

uint64_t RpcServer::Serve() {
    Heartbeat heartbeat(transport_, heartbeat_.interval, heartbeat_.timeout);
    uint64_t served = 0;
    while (std::optional<Frame> frame = transport_.Receive()) {
        heartbeat.NoteActivity();
        if (frame->header().type() != FRAME_TYPE_GPU) continue;

        RpcBatch batch;
        if (!batch.ParseFromString(frame->payload())) {
            transport_.Close();
            break;
        }
        for (const RpcRequest& request : batch.calls()) {
            RpcResponse response = dispatcher_(request);
            response.set_call_id(request.call_id());
            ++served;
            const bool async = (request.flags() & kFlagAsync) != 0;
            if (!async || response.result() != kResultSuccess) {
                transport_.Send(FRAME_TYPE_GPU, kRpcChannel, response.SerializeAsString());
            }
        }
    }
    return served;
}

}  // namespace tuna::tunnel

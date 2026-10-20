#include "tuna/tunnel/rpc.h"

#include <cstring>
#include <utility>

namespace tuna::tunnel {

namespace {

uint64_t CurrentThreadId() {
    return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

}  // namespace

// ---------------------------------------------------------------- BulkStore

bool BulkStore::Add(const BulkChunk& chunk) {
    const uint64_t total = chunk.total_size();
    const uint64_t size = chunk.data().size();
    if (total > kMaxBulkTransferBytes || chunk.offset() > total || size > total - chunk.offset()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) return false;
    auto [it, inserted] = pending_.try_emplace(chunk.bulk_ref());
    Pending& pending = it->second;
    if (inserted) {
        if (pending_bytes_ + total > kMaxBulkPendingBytes) {
            pending_.erase(it);
            return false;
        }
        pending_bytes_ += total;
        pending.data.resize(static_cast<size_t>(total));
    } else if (pending.data.size() != total || pending.received + size > total) {
        return false;  // the sender changed its mind about the size, or overran it
    }
    if (size > 0) std::memcpy(pending.data.data() + chunk.offset(), chunk.data().data(), size);
    pending.received += size;
    if (pending.received == total) cv_.notify_all();
    return true;
}

std::optional<std::string> BulkStore::Take(uint64_t ref, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool ready = cv_.wait_for(lock, timeout, [&] {
        if (failed_) return true;
        auto it = pending_.find(ref);
        return it != pending_.end() && it->second.received == it->second.data.size();
    });
    auto it = pending_.find(ref);
    if (!ready || failed_ || it == pending_.end() || it->second.received != it->second.data.size()) {
        return std::nullopt;
    }
    std::string data = std::move(it->second.data);
    pending_bytes_ -= data.size();
    pending_.erase(it);
    return data;
}

void BulkStore::Fail() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
        pending_.clear();
        pending_bytes_ = 0;
    }
    cv_.notify_all();
}

bool SendBulk(Transport& transport, uint64_t ref, std::string_view data) {
    BulkChunk chunk;
    chunk.set_bulk_ref(ref);
    chunk.set_total_size(data.size());
    size_t offset = 0;
    do {  // a zero-length payload is still one (empty) chunk
        const size_t size = std::min(kBulkChunkBytes, data.size() - offset);
        chunk.set_offset(offset);
        chunk.set_data(data.data() + offset, size);
        if (!transport.Send(FRAME_TYPE_IO, kBulkChannel, chunk.SerializeAsString())) return false;
        offset += size;
    } while (offset < data.size());
    return true;
}

// ---------------------------------------------------------------- RpcClient

RpcClient::RpcClient(Transport& transport, RpcClientOptions options, Transport* bulk)
    : transport_(transport), bulk_(bulk ? bulk : &transport), options_(options) {
    heartbeat_ = std::make_unique<Heartbeat>(transport_, options_.heartbeat.interval,
                                             options_.heartbeat.timeout);
    reader_ = std::thread([this] { ReaderLoop(transport_, true); });
    if (bulk_ != &transport_) {
        bulk_reader_ = std::thread([this] { ReaderLoop(*bulk_, false); });
    }
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
        bulk_->Close();
        reader_.join();       // the reader uses heartbeat_, so it must stop first
        if (bulk_reader_.joinable()) bulk_reader_.join();
        heartbeat_.reset();
    });
}

bool RpcClient::AttachBulk(RpcRequest* request, std::string_view bulk_in) {
    if (bulk_in.empty()) return true;
    // The call id doubles as the bulk_ref: unique per client, no extra counter.
    request->set_flags(request->flags() | kFlagHasBulk);
    request->set_bulk_ref(request->call_id());
    return SendBulk(*bulk_, request->call_id(), bulk_in);
}

void RpcClient::Post(uint32_t api_id, const std::string& args, std::string_view bulk_in) {
    RpcRequest request;
    request.set_call_id(next_call_id_++);
    request.set_api_id(api_id);
    request.set_flags(kFlagAsync);
    request.set_thread_id(CurrentThreadId());
    request.set_args(args);
    // A failed bulk send means the transport is gone; the request is dropped
    // and the next synchronizing call reports the lost connection.
    if (!AttachBulk(&request, bulk_in)) return;

    std::unique_lock<std::mutex> lock(queue_mutex_);
    Enqueue(std::move(request), lock);
}

CallResult RpcClient::Call(uint32_t api_id, const std::string& args, bool synchronizing,
                           std::string_view bulk_in) {
    const uint64_t call_id = next_call_id_++;
    std::future<Reply> reply_future;
    {
        std::lock_guard<std::mutex> lock(calls_mutex_);
        if (!connected_) return CallResult{};
        reply_future = pending_[call_id].get_future();
    }

    RpcRequest request;
    request.set_call_id(call_id);
    request.set_api_id(api_id);
    request.set_thread_id(CurrentThreadId());
    request.set_args(args);
    bool sent = AttachBulk(&request, bulk_in);
    if (sent) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        batch_.add_calls()->Swap(&request);  // rides behind any queued posts
        sent = FlushLocked();
    }

    CallResult result;
    if (sent && reply_future.wait_for(options_.call_timeout) == std::future_status::ready) {
        Reply reply = reply_future.get();
        result.result = reply.result;
        result.out_args = std::move(reply.out_args);
        if (reply.bulk_ref != 0) {
            // The payload's chunks were sent before the response; on a separate
            // bulk socket they may still be in flight.
            if (std::optional<std::string> data = bulk_in_.Take(reply.bulk_ref, options_.call_timeout)) {
                result.bulk_out = std::move(*data);
            } else {
                result = CallResult{};
            }
        }
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

std::optional<int32_t> RpcClient::ArrivedAsyncError(bool clear) {
    std::lock_guard<std::mutex> lock(calls_mutex_);
    std::optional<int32_t> error = latched_error_;
    if (clear) latched_error_.reset();
    return error;
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

void RpcClient::ReaderLoop(Transport& transport, bool main) {
    while (std::optional<Frame> frame = transport.Receive()) {
        if (main) heartbeat_->NoteActivity();
        const FrameType type = frame->header().type();
        if (type == FRAME_TYPE_IO) {
            BulkChunk chunk;
            if (!chunk.ParseFromString(frame->payload()) || !bulk_in_.Add(chunk)) {
                transport.Close();
                break;
            }
            continue;
        }
        if (!main || type != FRAME_TYPE_GPU) continue;  // heartbeats, future control

        RpcResponse response;
        if (!response.ParseFromString(frame->payload())) {
            transport.Close();  // a peer that sends garbage is not trusted further
            break;
        }
        std::lock_guard<std::mutex> lock(calls_mutex_);
        auto it = pending_.find(response.call_id());
        if (it != pending_.end()) {
            it->second.set_value(Reply{response.result(), std::move(*response.mutable_out_args()),
                                       response.bulk_ref()});
            pending_.erase(it);
        } else if (response.result() != kResultSuccess && !latched_error_) {
            latched_error_ = response.result();  // a failed async call; CUDA keeps the first
        }
    }
    if (!main) return;  // losing only the bulk socket fails the bulk calls (timeout)

    // Connection gone: nothing in flight will ever be answered.
    bulk_in_.Fail();
    std::lock_guard<std::mutex> lock(calls_mutex_);
    connected_ = false;
    for (auto& [id, promise] : pending_) {
        promise.set_value(Reply{});
    }
    pending_.clear();
}

// ---------------------------------------------------------------- RpcServer

RpcServer::RpcServer(Transport& transport, Dispatcher dispatcher, HeartbeatOptions heartbeat,
                     std::chrono::milliseconds bulk_timeout)
    : transport_(transport), dispatcher_(std::move(dispatcher)), heartbeat_(heartbeat),
      bulk_timeout_(bulk_timeout) {}

RpcServer::~RpcServer() { StopBulk(); }

void RpcServer::StopBulk() {
    // After this, the bulk transport is never touched again, so its owner may
    // destroy it as soon as Serve() returns.
    std::thread reader;
    {
        std::lock_guard<std::mutex> lock(bulk_mutex_);
        serving_done_ = true;
        if (bulk_) bulk_->Close();
        bulk_ = nullptr;
        reader = std::move(bulk_reader_);
    }
    if (reader.joinable()) reader.join();
}

void RpcServer::AttachBulk(Transport* bulk) {
    std::lock_guard<std::mutex> lock(bulk_mutex_);
    if (serving_done_ || bulk_) {
        bulk->Close();  // too late, or a second bulk socket: refuse it
        return;
    }
    bulk_ = bulk;
    bulk_reader_ = std::thread([this, bulk] { BulkReaderLoop(bulk); });
}

void RpcServer::BulkReaderLoop(Transport* bulk) {
    while (std::optional<Frame> frame = bulk->Receive()) {
        if (frame->header().type() != FRAME_TYPE_IO) continue;
        BulkChunk chunk;
        if (!chunk.ParseFromString(frame->payload()) || !bulk_in_.Add(chunk)) {
            transport_.Close();  // protocol violation ends the whole session
            break;
        }
    }
}

uint64_t RpcServer::Serve() {
    Heartbeat heartbeat(transport_, heartbeat_.interval, heartbeat_.timeout);
    uint64_t served = 0;
    while (std::optional<Frame> frame = transport_.Receive()) {
        heartbeat.NoteActivity();
        const FrameType type = frame->header().type();
        if (type == FRAME_TYPE_IO) {
            BulkChunk chunk;
            if (!chunk.ParseFromString(frame->payload()) || !bulk_in_.Add(chunk)) {
                transport_.Close();
                break;
            }
            continue;
        }
        if (type != FRAME_TYPE_GPU) continue;

        RpcBatch batch;
        if (!batch.ParseFromString(frame->payload())) {
            transport_.Close();
            break;
        }
        for (const RpcRequest& request : batch.calls()) {
            ServerCall call{request, {}, {}};
            RpcResponse response;
            bool have_input = true;
            if (request.flags() & kFlagHasBulk) {
                std::optional<std::string> data = bulk_in_.Take(request.bulk_ref(), bulk_timeout_);
                have_input = data.has_value();
                if (data) call.bulk_in = std::move(*data);
            }
            if (have_input) {
                response = dispatcher_(call);
            } else {
                response.set_result(kResultTransportError);  // payload never arrived
            }
            response.set_call_id(request.call_id());
            ++served;
            const bool async = (request.flags() & kFlagAsync) != 0;
            // Async calls never return data: nobody would collect the payload.
            if (!async && !call.bulk_out.empty()) {
                Transport* bulk;
                {
                    std::lock_guard<std::mutex> lock(bulk_mutex_);
                    bulk = bulk_ ? bulk_ : &transport_;
                }
                if (SendBulk(*bulk, request.call_id(), call.bulk_out)) {
                    response.set_bulk_ref(request.call_id());
                } else {
                    // Some chunks may have arrived, so resending elsewhere could
                    // duplicate them; fail this call instead.
                    response.set_result(kResultTransportError);
                    response.clear_out_args();
                }
            }
            if (!async || response.result() != kResultSuccess) {
                transport_.Send(FRAME_TYPE_GPU, kRpcChannel, response.SerializeAsString());
            }
        }
    }
    bulk_in_.Fail();
    StopBulk();
    return served;
}

}  // namespace tuna::tunnel

#include "tuna/tunnel/session.h"

namespace tuna::tunnel {

namespace {

void SetError(std::string* error, const std::string& message) {
    if (error) *error = message;
}

}  // namespace

std::optional<Hello> ExchangeHello(Transport& transport, const Hello& local,
                                   std::chrono::milliseconds timeout, std::string* error) {
    if (!transport.Send(FRAME_TYPE_CONTROL, kControlChannel, local.SerializeAsString())) {
        SetError(error, "connection closed before Hello could be sent");
        return std::nullopt;
    }

    // Receive() has no timeout of its own, so a watchdog closes the transport
    // if the peer stays silent; that unblocks Receive() below.
    std::mutex mutex;
    std::condition_variable done_cv;
    bool done = false;
    bool timed_out = false;
    std::thread watchdog([&] {
        std::unique_lock<std::mutex> lock(mutex);
        if (!done_cv.wait_for(lock, timeout, [&] { return done; })) {
            timed_out = true;
            transport.Close();
        }
    });

    std::optional<Frame> frame = transport.Receive();
    {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
    }
    done_cv.notify_one();
    watchdog.join();

    if (!frame) {
        SetError(error, timed_out ? "timed out waiting for peer Hello" : "connection closed during handshake");
        return std::nullopt;
    }
    Hello peer;
    if (frame->header().type() != FRAME_TYPE_CONTROL || frame->header().channel() != kControlChannel ||
        !peer.ParseFromString(frame->payload())) {
        SetError(error, "peer did not start with a Hello frame");
        transport.Close();
        return std::nullopt;
    }
    if (peer.protocol_version() != kProtocolVersion) {
        SetError(error, "protocol version mismatch: local " + std::to_string(kProtocolVersion) +
                            ", peer " + std::to_string(peer.protocol_version()));
        transport.Close();
        return std::nullopt;
    }
    return peer;
}

Heartbeat::Heartbeat(Transport& transport, std::chrono::milliseconds interval,
                     std::chrono::milliseconds timeout)
    : transport_(transport), interval_(interval), timeout_(timeout), last_activity_ns_(NowNs()) {
    thread_ = std::thread([this] { Run(); });
}

Heartbeat::~Heartbeat() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_one();
    thread_.join();
}

void Heartbeat::NoteActivity() { last_activity_ns_ = NowNs(); }

void Heartbeat::Run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!wake_.wait_for(lock, interval_, [this] { return stop_; })) {
        // Read the activity stamp before the clock: the reader thread may bump
        // it concurrently, and unsigned subtraction must never wrap.
        const uint64_t last = last_activity_ns_.load();
        const uint64_t now = NowNs();
        const uint64_t silent_ns = now > last ? now - last : 0;
        if (silent_ns > static_cast<uint64_t>(std::chrono::nanoseconds(timeout_).count())) {
            expired_ = true;
            transport_.Close();
            return;
        }
        if (!transport_.Send(FRAME_TYPE_HEARTBEAT, kControlChannel, std::string())) {
            return;  // transport already closed; the reader thread handles it
        }
    }
}

}  // namespace tuna::tunnel

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include "tuna/tunnel/frame_codec.h"
#include "tuna/tunnel/tcp_socket.h"

namespace tuna::tunnel {

// Logical channels carried in FrameHeader.channel.
enum Channel : uint32_t { kControlChannel = 0, kRpcChannel = 1, kBulkChannel = 2 };

// A bidirectional, ordered, framed connection. Send() may be called from any
// thread; Receive() is meant for a single reader thread. TLS (Phase 2b) and
// QUIC (Phase 4) implement this same interface.
class Transport {
public:
    virtual ~Transport() = default;

    // Assigns frame_id/timestamp and sends. False once the transport is closed.
    virtual bool Send(FrameType type, uint32_t channel, const std::string& payload) = 0;

    // Blocks for the next frame. std::nullopt when the peer closed, the
    // connection failed, the peer violated framing, or Close() was called.
    virtual std::optional<Frame> Receive() = 0;

    // Idempotent; unblocks a concurrent Receive().
    virtual void Close() = 0;
};

class TcpTransport final : public Transport {
public:
    explicit TcpTransport(TcpStream stream);

    bool Send(FrameType type, uint32_t channel, const std::string& payload) override;
    std::optional<Frame> Receive() override;
    void Close() override;

private:
    TcpStream stream_;
    std::mutex send_mutex_;           // one writer at a time keeps frames intact
    uint64_t next_frame_id_ = 1;      // guarded by send_mutex_
    FrameDecoder decoder_;            // reader thread only
    std::atomic<bool> closed_{false};
};

// Monotonic nanoseconds for FrameHeader.timestamp_ns.
uint64_t NowNs();

}  // namespace tuna::tunnel

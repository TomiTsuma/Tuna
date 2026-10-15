#include "tuna/tunnel/transport.h"

#include <array>
#include <chrono>
#include <utility>

namespace tuna::tunnel {

uint64_t NowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

TcpTransport::TcpTransport(TcpStream stream) : stream_(std::move(stream)) {}

bool TcpTransport::Send(FrameType type, uint32_t channel, const std::string& payload) {
    if (closed_) return false;

    Frame frame;
    frame.mutable_header()->set_type(type);
    frame.mutable_header()->set_channel(channel);
    frame.set_payload(payload);

    std::lock_guard<std::mutex> lock(send_mutex_);
    frame.mutable_header()->set_frame_id(next_frame_id_++);
    frame.mutable_header()->set_timestamp_ns(NowNs());
    std::string wire;
    if (!EncodeFrame(frame, &wire)) return false;
    if (!stream_.SendAll(wire.data(), wire.size())) {
        Close();
        return false;
    }
    return true;
}

std::optional<Frame> TcpTransport::Receive() {
    Frame frame;
    std::array<char, 64 * 1024> buffer;
    while (!decoder_.Next(&frame)) {
        if (closed_) return std::nullopt;
        const size_t n = stream_.Receive(buffer.data(), buffer.size());
        if (n == 0 || !decoder_.Feed(std::string_view(buffer.data(), n))) {
            Close();
            return std::nullopt;
        }
    }
    return frame;
}

void TcpTransport::Close() {
    if (!closed_.exchange(true)) {
        stream_.Shutdown();
    }
}

}  // namespace tuna::tunnel

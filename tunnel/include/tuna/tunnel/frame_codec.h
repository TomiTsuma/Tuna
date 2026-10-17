#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>

#include "tuna_frames.pb.h"

namespace tuna::tunnel {

// Wire format: 4-byte little-endian length, then a serialized tuna::Frame.
// Bulk payloads above this limit belong on the bulk channel (Phase 2c).
constexpr uint32_t kMaxFrameBytes = 16u * 1024u * 1024u;
constexpr size_t kLengthPrefixBytes = 4;

// Appends the length-prefixed encoding of `frame` to `out`.
// Returns false (and leaves `out` unchanged) if the frame exceeds kMaxFrameBytes.
bool EncodeFrame(const Frame& frame, std::string* out);

// Incremental decoder for a byte stream that may split or coalesce frames.
class FrameDecoder {
public:
    // Feeds received bytes. Returns false on a protocol error (oversized length
    // prefix or unparseable frame); the decoder is then unusable and the
    // connection must be closed.
    bool Feed(std::string_view bytes);

    // Pops the next complete frame, if any.
    bool Next(Frame* frame);

private:
    std::string buffer_;
    std::deque<Frame> ready_;
    bool failed_ = false;
};

}  // namespace tuna::tunnel

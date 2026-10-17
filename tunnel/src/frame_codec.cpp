#include "tuna/tunnel/frame_codec.h"

namespace tuna::tunnel {

namespace {

void PutLength(uint32_t length, std::string* out) {
    for (int i = 0; i < 4; ++i) {
        out->push_back(static_cast<char>((length >> (8 * i)) & 0xFFu));
    }
}

uint32_t GetLength(const char* p) {
    uint32_t length = 0;
    for (int i = 0; i < 4; ++i) {
        length |= static_cast<uint32_t>(static_cast<unsigned char>(p[i])) << (8 * i);
    }
    return length;
}

}  // namespace

bool EncodeFrame(const Frame& frame, std::string* out) {
    const size_t size = frame.ByteSizeLong();
    if (size > kMaxFrameBytes) {
        return false;
    }
    const size_t start = out->size();
    PutLength(static_cast<uint32_t>(size), out);
    out->resize(start + kLengthPrefixBytes + size);
    if (!frame.SerializeToArray(out->data() + start + kLengthPrefixBytes, static_cast<int>(size))) {
        out->resize(start);
        return false;
    }
    return true;
}

bool FrameDecoder::Feed(std::string_view bytes) {
    if (failed_) {
        return false;
    }
    buffer_.append(bytes.data(), bytes.size());

    size_t offset = 0;
    while (buffer_.size() - offset >= kLengthPrefixBytes) {
        const uint32_t length = GetLength(buffer_.data() + offset);
        if (length > kMaxFrameBytes) {
            failed_ = true;
            return false;
        }
        if (buffer_.size() - offset - kLengthPrefixBytes < length) {
            break;  // wait for the rest of this frame
        }
        Frame frame;
        if (!frame.ParseFromArray(buffer_.data() + offset + kLengthPrefixBytes,
                                  static_cast<int>(length))) {
            failed_ = true;
            return false;
        }
        ready_.push_back(std::move(frame));
        offset += kLengthPrefixBytes + length;
    }
    buffer_.erase(0, offset);
    return true;
}

bool FrameDecoder::Next(Frame* frame) {
    if (ready_.empty()) {
        return false;
    }
    *frame = std::move(ready_.front());
    ready_.pop_front();
    return true;
}

}  // namespace tuna::tunnel

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

// Flat little-endian argument encoding for remoted CUDA calls (wire plan §4).
// Protobuf-per-call is too slow on the hot path, so args are a packed byte
// string: scalars in declaration order, device pointers as uint64 handles the
// server owns. The decoder fails closed on a short or malformed buffer.

namespace tuna::cudart {

class ArgWriter {
public:
    void put(const void* data, size_t size) { buffer_.append(static_cast<const char*>(data), size); }
    void put_u64(uint64_t value) {
        char bytes[8];
        for (int i = 0; i < 8; ++i) bytes[i] = static_cast<char>((value >> (8 * i)) & 0xff);
        buffer_.append(bytes, 8);
    }
    const std::string& str() const { return buffer_; }

private:
    std::string buffer_;
};

class ArgReader {
public:
    explicit ArgReader(const std::string& data) : data_(data) {}

    // Copies `size` bytes into `out`. Throws if the buffer is short: a
    // truncated request from a peer is a protocol error, never a silent
    // read of adjacent memory.
    void get(void* out, size_t size) {
        if (offset_ + size > data_.size()) throw std::runtime_error("cuda arg buffer underrun");
        std::memcpy(out, data_.data() + offset_, size);
        offset_ += size;
    }
    uint64_t get_u64() {
        unsigned char bytes[8];
        get(bytes, 8);
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
        return value;
    }
    size_t remaining() const { return data_.size() - offset_; }

private:
    const std::string& data_;
    size_t offset_ = 0;
};

// Result of a remoted call as the client sees it.
struct Reply {
    int32_t result = 0;          // cudaError_t
    std::string out_args;        // encoded OUT params
    std::string bulk_out;        // device->host payload (cudaMemcpy D2H)
    bool transport_error = false;
};

}  // namespace tuna::cudart

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
    void put_u32(uint32_t value) {
        char bytes[4];
        for (int i = 0; i < 4; ++i) bytes[i] = static_cast<char>((value >> (8 * i)) & 0xff);
        buffer_.append(bytes, 4);
    }
    // Length-prefixed (u32) bytes, e.g. a kernel or symbol name.
    void put_string(const std::string& value) {
        put_u32(static_cast<uint32_t>(value.size()));
        buffer_.append(value);
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
    uint32_t get_u32() {
        unsigned char bytes[4];
        get(bytes, 4);
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) value |= static_cast<uint32_t>(bytes[i]) << (8 * i);
        return value;
    }
    std::string get_string() {
        const uint32_t size = get_u32();
        if (size > remaining()) throw std::runtime_error("cuda arg buffer underrun");
        std::string value = data_.substr(offset_, size);
        offset_ += size;
        return value;
    }
    // The unread tail (e.g. a kernel's packed parameters).
    std::string rest() {
        std::string value = data_.substr(offset_);
        offset_ = data_.size();
        return value;
    }
    void skip(size_t size) {
        if (offset_ + size > data_.size()) throw std::runtime_error("cuda arg buffer underrun");
        offset_ += size;
    }
    size_t remaining() const { return data_.size() - offset_; }

private:
    const std::string& data_;
    size_t offset_ = 0;
};

// A library scalar passed by pointer (cuBLAS alpha/beta, a dot product's
// result). Its pointer mode decides what crosses the wire: in host mode the
// value itself (tag 0 + length-prefixed bytes), in device mode the server
// address (tag 1 + u64).
constexpr uint32_t kScalarOnHost = 0;
constexpr uint32_t kScalarOnDevice = 1;
constexpr size_t kMaxScalarBytes = 16;  // cuDoubleComplex, the widest BLAS scalar

inline void PutHostScalar(ArgWriter& out, const void* value, size_t size) {
    out.put_u32(kScalarOnHost);
    out.put_string(std::string(static_cast<const char*>(value), size));
}

inline void PutDeviceScalar(ArgWriter& out, const void* address) {
    out.put_u32(kScalarOnDevice);
    out.put_u64(reinterpret_cast<uint64_t>(address));
}

// Server side: decodes one scalar reference. ptr() is what the real library
// gets: the device address, or local storage holding the value (zero padded
// to kMaxScalarBytes, so the library can never read past what the client
// sent). For an OUT scalar in host mode the library writes that storage and
// the dispatcher returns it.
class ScalarRef {
public:
    explicit ScalarRef(ArgReader& in) {
        const uint32_t tag = in.get_u32();
        if (tag == kScalarOnDevice) {
            device_ = reinterpret_cast<void*>(in.get_u64());
            on_device_ = true;
        } else if (tag == kScalarOnHost) {
            const std::string value = in.get_string();
            if (value.size() > kMaxScalarBytes) throw std::runtime_error("scalar too wide");
            std::memcpy(storage_, value.data(), value.size());
        } else {
            throw std::runtime_error("bad scalar tag");
        }
    }
    ScalarRef(const ScalarRef&) = delete;
    ScalarRef& operator=(const ScalarRef&) = delete;

    void* ptr() { return on_device_ ? device_ : storage_; }
    bool on_device() const { return on_device_; }

private:
    alignas(16) unsigned char storage_[kMaxScalarBytes] = {};
    void* device_ = nullptr;
    bool on_device_ = false;
};

// Result of a remoted call as the client sees it.
struct Reply {
    int32_t result = 0;          // cudaError_t
    std::string out_args;        // encoded OUT params
    std::string bulk_out;        // device->host payload (cudaMemcpy D2H)
    bool transport_error = false;
};

}  // namespace tuna::cudart

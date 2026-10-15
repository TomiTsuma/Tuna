#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace tuna::tunnel {

// Native socket handle, wide enough for both Winsock SOCKET and a POSIX fd.
using NativeSocket = std::intptr_t;
constexpr NativeSocket kInvalidSocket = -1;

// A connected TCP stream (move-only, closes on destruction).
// TCP_NODELAY is always on: RPC latency matters more than packet count.
class TcpStream {
public:
    TcpStream() = default;
    explicit TcpStream(NativeSocket socket);
    ~TcpStream();
    TcpStream(TcpStream&& other) noexcept;
    TcpStream& operator=(TcpStream&& other) noexcept;
    TcpStream(const TcpStream&) = delete;
    TcpStream& operator=(const TcpStream&) = delete;

    // Resolves host and connects. std::nullopt on failure (error in *error).
    static std::optional<TcpStream> Connect(const std::string& host, uint16_t port,
                                            std::string* error = nullptr);

    bool valid() const { return socket_.load() != kInvalidSocket; }

    // Sends every byte or returns false (peer gone / socket shut down).
    bool SendAll(const char* data, size_t size);

    // Blocks for at least one byte. Returns bytes read; 0 on orderly close or
    // error (callers treat both as end of stream).
    size_t Receive(char* buffer, size_t capacity);

    // Unblocks any thread in Receive()/SendAll(); safe to call from any thread.
    // On Windows this also closes the handle: Winsock's shutdown() does not
    // cancel a blocking recv(), only closesocket() does.
    void Shutdown();

private:
    void Close();
    // Atomic: Shutdown() may run on another thread while Receive() blocks.
    std::atomic<NativeSocket> socket_{kInvalidSocket};
};

class TcpListener {
public:
    TcpListener() = default;
    ~TcpListener();
    TcpListener(TcpListener&& other) noexcept;
    TcpListener& operator=(TcpListener&& other) noexcept;
    TcpListener(const TcpListener&) = delete;
    TcpListener& operator=(const TcpListener&) = delete;

    // Binds and listens. Port 0 picks an ephemeral port (see port()).
    static std::optional<TcpListener> Bind(const std::string& host, uint16_t port,
                                           std::string* error = nullptr);

    uint16_t port() const { return port_; }

    // Blocks until a client connects. std::nullopt once Shutdown() was called.
    std::optional<TcpStream> Accept();

    // Unblocks Accept(); safe to call from any thread.
    void Shutdown();

private:
    // Atomic: Shutdown() may run on another thread while Accept() blocks.
    std::atomic<NativeSocket> socket_{kInvalidSocket};
    uint16_t port_ = 0;
};

// True for 127.0.0.0/8, ::1 and "localhost". Used to refuse network-exposed
// binds until the TLS transport exists (Phase 2b).
bool IsLoopbackHost(const std::string& host);

}  // namespace tuna::tunnel

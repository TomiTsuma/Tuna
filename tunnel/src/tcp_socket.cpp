#include "tuna/tunnel/tcp_socket.h"

#include <cerrno>
#include <cstring>
#include <memory>
#include <utility>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace tuna::tunnel {

namespace {

#ifdef _WIN32
using Socklen = int;
constexpr int kShutdownBoth = SD_BOTH;
constexpr int kSendFlags = 0;

// Winsock needs one WSAStartup per process before any socket call.
struct WinsockInit {
    WinsockInit() {
        WSADATA data;
        WSAStartup(MAKEWORD(2, 2), &data);
    }
    ~WinsockInit() { WSACleanup(); }
};
void EnsureSocketsReady() { static WinsockInit init; }

void CloseNative(NativeSocket s) { closesocket(static_cast<SOCKET>(s)); }
std::string LastError() { return "winsock error " + std::to_string(WSAGetLastError()); }
#else
using Socklen = socklen_t;
constexpr int kShutdownBoth = SHUT_RDWR;
constexpr int kSendFlags = MSG_NOSIGNAL;  // a dead peer must not SIGPIPE the process

void EnsureSocketsReady() {}
void CloseNative(NativeSocket s) { ::close(static_cast<int>(s)); }
std::string LastError() { return std::strerror(errno); }
#endif

auto Raw(NativeSocket s) {
#ifdef _WIN32
    return static_cast<SOCKET>(s);
#else
    return static_cast<int>(s);
#endif
}

bool IsInvalid(decltype(Raw(0)) s) {
#ifdef _WIN32
    return s == INVALID_SOCKET;
#else
    return s < 0;
#endif
}

void SetError(std::string* error, const std::string& message) {
    if (error) *error = message;
}

struct AddrInfoDeleter {
    void operator()(addrinfo* info) const { freeaddrinfo(info); }
};

addrinfo* Resolve(const std::string& host, uint16_t port, bool passive, std::string* error) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (passive) hints.ai_flags = AI_PASSIVE;
    addrinfo* result = nullptr;
    const std::string service = std::to_string(port);
    const int rc = getaddrinfo(host.empty() ? nullptr : host.c_str(), service.c_str(), &hints, &result);
    if (rc != 0) {
        SetError(error, "cannot resolve '" + host + "'");
        return nullptr;
    }
    return result;
}

}  // namespace

// ---------------------------------------------------------------- TcpStream

TcpStream::TcpStream(NativeSocket socket) : socket_(socket) {
    int one = 1;
    setsockopt(Raw(socket), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
}

TcpStream::~TcpStream() { Close(); }

TcpStream::TcpStream(TcpStream&& other) noexcept : socket_(other.socket_.exchange(kInvalidSocket)) {}

TcpStream& TcpStream::operator=(TcpStream&& other) noexcept {
    if (this != &other) {
        Close();
        socket_ = other.socket_.exchange(kInvalidSocket);
    }
    return *this;
}

std::optional<TcpStream> TcpStream::Connect(const std::string& host, uint16_t port, std::string* error) {
    EnsureSocketsReady();
    std::unique_ptr<addrinfo, AddrInfoDeleter> list(Resolve(host, port, false, error));
    if (!list) return std::nullopt;

    for (addrinfo* ai = list.get(); ai != nullptr; ai = ai->ai_next) {
        auto s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (IsInvalid(s)) continue;
        if (connect(s, ai->ai_addr, static_cast<Socklen>(ai->ai_addrlen)) == 0) {
            return TcpStream(static_cast<NativeSocket>(s));
        }
        SetError(error, "connect to " + host + ":" + std::to_string(port) + " failed: " + LastError());
        CloseNative(static_cast<NativeSocket>(s));
    }
    return std::nullopt;
}

bool TcpStream::SendAll(const char* data, size_t size) {
    while (size > 0) {
        const int chunk = size > (1u << 30) ? (1 << 30) : static_cast<int>(size);
        const NativeSocket s = socket_.load();
        if (s == kInvalidSocket) return false;
        const auto sent = send(Raw(s), data, chunk, kSendFlags);
        if (sent <= 0) return false;
        data += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

size_t TcpStream::Receive(char* buffer, size_t capacity) {
    const int max = capacity > (1u << 30) ? (1 << 30) : static_cast<int>(capacity);
    const NativeSocket s = socket_.load();
    if (s == kInvalidSocket) return 0;
    const auto received = recv(Raw(s), buffer, max, 0);
    return received > 0 ? static_cast<size_t>(received) : 0;
}

void TcpStream::Shutdown() {
#ifdef _WIN32
    const NativeSocket s = socket_.exchange(kInvalidSocket);
    if (s != kInvalidSocket) {
        shutdown(Raw(s), kShutdownBoth);  // send FIN so the peer sees an orderly close
        CloseNative(s);                   // ...and cancel our own blocking recv()
    }
#else
    // POSIX shutdown() wakes recv(); closing an fd another thread is blocked on
    // is unsafe (the number can be reused), so the destructor closes it.
    const NativeSocket s = socket_.load();
    if (s != kInvalidSocket) shutdown(Raw(s), kShutdownBoth);
#endif
}

void TcpStream::Close() {
    const NativeSocket s = socket_.exchange(kInvalidSocket);
    if (s != kInvalidSocket) CloseNative(s);
}

// -------------------------------------------------------------- TcpListener

TcpListener::~TcpListener() {
    const NativeSocket s = socket_.exchange(kInvalidSocket);
    if (s != kInvalidSocket) CloseNative(s);
}

TcpListener::TcpListener(TcpListener&& other) noexcept
    : socket_(other.socket_.exchange(kInvalidSocket)), port_(other.port_) {}

TcpListener& TcpListener::operator=(TcpListener&& other) noexcept {
    if (this != &other) {
        const NativeSocket old = socket_.exchange(other.socket_.exchange(kInvalidSocket));
        if (old != kInvalidSocket) CloseNative(old);
        port_ = other.port_;
    }
    return *this;
}

std::optional<TcpListener> TcpListener::Bind(const std::string& host, uint16_t port, std::string* error) {
    EnsureSocketsReady();
    std::unique_ptr<addrinfo, AddrInfoDeleter> list(Resolve(host, port, true, error));
    if (!list) return std::nullopt;

    for (addrinfo* ai = list.get(); ai != nullptr; ai = ai->ai_next) {
        auto s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (IsInvalid(s)) continue;
#ifndef _WIN32
        // Allow quick restarts; on Windows SO_REUSEADDR would allow port hijacking.
        int one = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
        if (bind(s, ai->ai_addr, static_cast<Socklen>(ai->ai_addrlen)) != 0 || listen(s, SOMAXCONN) != 0) {
            SetError(error, "bind " + host + ":" + std::to_string(port) + " failed: " + LastError());
            CloseNative(static_cast<NativeSocket>(s));
            continue;
        }
        sockaddr_storage bound{};
        Socklen length = sizeof(bound);
        getsockname(s, reinterpret_cast<sockaddr*>(&bound), &length);
        TcpListener listener;
        listener.socket_ = static_cast<NativeSocket>(s);  // atomic store
        listener.port_ = ntohs(bound.ss_family == AF_INET6
                                   ? reinterpret_cast<sockaddr_in6*>(&bound)->sin6_port
                                   : reinterpret_cast<sockaddr_in*>(&bound)->sin_port);
        return listener;
    }
    return std::nullopt;
}

std::optional<TcpStream> TcpListener::Accept() {
    const NativeSocket listening = socket_.load();
    if (listening == kInvalidSocket) return std::nullopt;
    const auto s = accept(Raw(listening), nullptr, nullptr);
    if (IsInvalid(s)) return std::nullopt;
    return TcpStream(static_cast<NativeSocket>(s));
}

void TcpListener::Shutdown() {
#ifdef _WIN32
    // shutdown() does not wake accept() on Winsock; closing the handle does.
    const NativeSocket s = socket_.exchange(kInvalidSocket);
    if (s != kInvalidSocket) CloseNative(s);
#else
    // Closing an fd another thread is blocked on is unsafe on POSIX (the number
    // can be reused); shutdown() wakes accept() and the destructor closes it.
    const NativeSocket s = socket_.load();
    if (s != kInvalidSocket) shutdown(Raw(s), SHUT_RDWR);
#endif
}

bool IsLoopbackHost(const std::string& host) {
    if (host == "localhost") return true;
    in_addr v4{};
    if (inet_pton(AF_INET, host.c_str(), &v4) == 1) {
        return (ntohl(v4.s_addr) >> 24) == 127;
    }
    in6_addr v6{};
    if (inet_pton(AF_INET6, host.c_str(), &v6) == 1) {
        static const in6_addr loopback = IN6ADDR_LOOPBACK_INIT;
        return std::memcmp(&v6, &loopback, sizeof(v6)) == 0;
    }
    return false;
}

}  // namespace tuna::tunnel

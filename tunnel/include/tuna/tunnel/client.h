#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "tuna/tunnel/tcp_socket.h"
#include "tuna/tunnel/tls.h"
#include "tuna/tunnel/transport.h"

namespace tuna::tunnel {

// ------------------------------------------------------- connection preamble
//
// Every TCP connection to a tuna_server listener starts with 24 plaintext
// bytes, before TLS: "TUNA", a little-endian u32 kind, and a 16-byte session
// token (zero for a new session). The listener reads exactly these bytes to
// decide which worker gets the socket, and parses nothing else.
constexpr size_t kPreambleBytes = 24;
using SessionToken = std::array<uint8_t, 16>;

enum class PreambleKind : uint32_t {
    kNewSession = 1,  // hand the socket to a new worker
    kAttachBulk = 2,  // hand the socket to the worker holding `token`, as its bulk socket
};

struct Preamble {
    PreambleKind kind = PreambleKind::kNewSession;
    SessionToken token{};
};

std::string EncodePreamble(const Preamble& preamble);
// std::nullopt unless `bytes` is exactly a well-formed preamble.
std::optional<Preamble> DecodePreamble(std::string_view bytes);

// --------------------------------------------------------------- client side

struct ClientOptions {
    std::string host = "127.0.0.1";
    uint16_t port = 0;
    // mTLS settings; std::nullopt speaks plaintext, which tuna_server only
    // accepts on loopback.
    std::optional<TlsConfig> tls;
    std::string server_name;  // identity to verify; empty = `host`
    std::string client_os;    // reported in Hello; empty = this platform
    bool open_bulk_socket = true;
    // Connection attempts are retried with exponential backoff (100 ms doubling
    // to 5 s) until this much time has passed; 0 makes a single attempt.
    // Authentication and protocol failures are never retried.
    std::chrono::milliseconds connect_deadline{0};
    std::chrono::milliseconds handshake_timeout{5000};
};

struct ClientSession {
    std::unique_ptr<Transport> transport;  // control + RPC
    std::unique_ptr<Transport> bulk;       // null if the server offers none or attaching failed
    Hello server_hello;
};

// Connects, authenticates, exchanges Hello and, when the server offers it,
// attaches the bulk socket. A failed bulk attach is not fatal: bulk payloads
// then share the main connection.
std::optional<ClientSession> ConnectSession(const ClientOptions& options, std::string* error = nullptr);

}  // namespace tuna::tunnel

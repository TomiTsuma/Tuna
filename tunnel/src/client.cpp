#include "tuna/tunnel/client.h"

#include <algorithm>
#include <cstring>
#include <thread>

#include "tuna/tunnel/session.h"

namespace tuna::tunnel {

namespace {

constexpr char kMagic[4] = {'T', 'U', 'N', 'A'};

void SetError(std::string* error, const std::string& message) {
    if (error) *error = message;
}

std::string ThisPlatform() {
#ifdef _WIN32
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#else
    return "linux";
#endif
}

// Sends the preamble and runs TLS (if configured) on a fresh connection.
std::unique_ptr<Transport> OpenTransport(TcpStream stream, const Preamble& preamble,
                                         const std::shared_ptr<TlsContext>& tls,
                                         const ClientOptions& options, std::string* error) {
    const std::string bytes = EncodePreamble(preamble);
    if (!stream.SendAll(bytes.data(), bytes.size())) {
        SetError(error, "connection closed before the preamble was sent");
        return nullptr;
    }
    if (!tls) return std::make_unique<TcpTransport>(std::move(stream));
    const std::string& name = options.server_name.empty() ? options.host : options.server_name;
    return TlsHandshake(std::move(stream), tls, name, options.handshake_timeout, error);
}

}  // namespace

std::string EncodePreamble(const Preamble& preamble) {
    std::string bytes(kPreambleBytes, '\0');
    std::memcpy(bytes.data(), kMagic, 4);
    const auto kind = static_cast<uint32_t>(preamble.kind);
    for (int i = 0; i < 4; ++i) bytes[4 + i] = static_cast<char>((kind >> (8 * i)) & 0xff);
    std::memcpy(bytes.data() + 8, preamble.token.data(), preamble.token.size());
    return bytes;
}

std::optional<Preamble> DecodePreamble(std::string_view bytes) {
    if (bytes.size() != kPreambleBytes || std::memcmp(bytes.data(), kMagic, 4) != 0) return std::nullopt;
    uint32_t kind = 0;
    for (int i = 0; i < 4; ++i) kind |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[4 + i])) << (8 * i);
    if (kind != static_cast<uint32_t>(PreambleKind::kNewSession) &&
        kind != static_cast<uint32_t>(PreambleKind::kAttachBulk)) {
        return std::nullopt;
    }
    Preamble preamble;
    preamble.kind = static_cast<PreambleKind>(kind);
    std::memcpy(preamble.token.data(), bytes.data() + 8, preamble.token.size());
    return preamble;
}

std::optional<ClientSession> ConnectSession(const ClientOptions& options, std::string* error) {
    std::shared_ptr<TlsContext> tls;
    if (options.tls) {
        tls = TlsContext::Create(TlsRole::kClient, *options.tls, error);
        if (!tls) return std::nullopt;
    }

    // Retry only the TCP connect: a refused connection usually means the
    // server is starting or restarting. A failed handshake is a real answer.
    const auto deadline = std::chrono::steady_clock::now() + options.connect_deadline;
    auto backoff = std::chrono::milliseconds(100);
    std::optional<TcpStream> stream;
    while (!(stream = TcpStream::Connect(options.host, options.port, error))) {
        if (std::chrono::steady_clock::now() + backoff > deadline) return std::nullopt;
        std::this_thread::sleep_for(backoff);
        backoff = std::min(backoff * 2, std::chrono::milliseconds(5000));
    }

    ClientSession session;
    session.transport = OpenTransport(std::move(*stream), Preamble{}, tls, options, error);
    if (!session.transport) return std::nullopt;

    Hello hello;
    hello.set_protocol_version(kProtocolVersion);
    hello.set_client_os(options.client_os.empty() ? ThisPlatform() : options.client_os);
    std::optional<Hello> peer = ExchangeHello(*session.transport, hello, options.handshake_timeout, error);
    if (!peer) return std::nullopt;
    session.server_hello = std::move(*peer);

    const std::string& token = session.server_hello.session_token();
    if (options.open_bulk_socket && session.server_hello.bulk_socket_available() &&
        token.size() == sizeof(SessionToken)) {
        Preamble attach;
        attach.kind = PreambleKind::kAttachBulk;
        std::memcpy(attach.token.data(), token.data(), attach.token.size());
        if (auto bulk_stream = TcpStream::Connect(options.host, options.port)) {
            std::string ignored;
            auto bulk = OpenTransport(std::move(*bulk_stream), attach, tls, options, &ignored);
            // The worker answers with a Hello once it has accepted the socket.
            if (bulk && ExchangeHello(*bulk, hello, options.handshake_timeout, &ignored)) {
                session.bulk = std::move(bulk);
            }
        }
    }
    return session;
}

}  // namespace tuna::tunnel

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "tuna/tunnel/frame_codec.h"
#include "tuna/tunnel/tcp_socket.h"
#include "tuna/tunnel/transport.h"

// OpenSSL types, forward-declared so including this header does not pull in
// OpenSSL (the CUDA shim and tests only need the Transport interface).
struct ssl_st;
struct ssl_ctx_st;
struct bio_st;

namespace tuna::tunnel {

enum class TlsRole { kClient, kServer };

// PEM file paths for one endpoint. Both sides always present a certificate:
// Track B is mutual TLS only.
struct TlsConfig {
    std::string ca_file;    // trust root that the peer's chain must verify against
    std::string cert_file;  // this endpoint's certificate (PEM, may include intermediates)
    std::string key_file;   // this endpoint's private key (PEM)
    // Server role only, required: client identities allowed to connect. A
    // client is accepted when one of its DNS or URI subjectAltName values
    // equals an entry exactly (no wildcards, no prefix matching), the same rule
    // as Track A's --allowed-client-san.
    std::vector<std::string> allowed_peer_names;
};

// A configured SSL_CTX: TLS 1.3 minimum, peer verification required, no
// session tickets. Shared by every connection of one endpoint.
class TlsContext {
public:
    static std::shared_ptr<TlsContext> Create(TlsRole role, const TlsConfig& config,
                                              std::string* error = nullptr);
    ~TlsContext();
    TlsContext(const TlsContext&) = delete;
    TlsContext& operator=(const TlsContext&) = delete;

    TlsRole role() const { return role_; }
    ssl_ctx_st* native() const { return ctx_; }
    const std::vector<std::string>& allowed_peer_names() const { return allowed_; }

private:
    TlsContext(TlsRole role, ssl_ctx_st* ctx, std::vector<std::string> allowed)
        : role_(role), ctx_(ctx), allowed_(std::move(allowed)) {}

    TlsRole role_;
    ssl_ctx_st* ctx_;
    std::vector<std::string> allowed_;
};

// TLS over a TcpStream, doing its own socket I/O through memory BIOs so one
// reader thread and any number of senders can share the connection (OpenSSL
// forbids concurrent SSL_read/SSL_write on one SSL object).
//
// Locking: ssl_mutex_ guards the SSL object and is never held across socket
// I/O. write_mutex_ orders ciphertext: whoever drains the outgoing BIO holds
// it until those bytes are sent, so TLS records leave in sequence order.
class TlsTransport final : public Transport {
public:
    ~TlsTransport() override;

    bool Send(FrameType type, uint32_t channel, const std::string& payload) override;
    std::optional<Frame> Receive() override;
    void Close() override;

    // Server role: the allowlisted SAN value the client matched.
    // Client role: the server name that was verified.
    const std::string& peer_identity() const { return peer_identity_; }

private:
    friend std::unique_ptr<TlsTransport> TlsHandshake(TcpStream, std::shared_ptr<TlsContext>,
                                                      const std::string&, std::chrono::milliseconds,
                                                      std::string*);
    TlsTransport(TcpStream stream, std::shared_ptr<TlsContext> context);

    bool Handshake(const std::string& server_name, std::string* error);
    bool VerifyPeer(const std::string& server_name, std::string* error);
    bool FlushLocked();  // requires write_mutex_

    TcpStream stream_;
    std::shared_ptr<TlsContext> context_;
    ssl_st* ssl_ = nullptr;
    bio_st* in_bio_ = nullptr;   // ciphertext received from the socket (owned by ssl_)
    bio_st* out_bio_ = nullptr;  // ciphertext waiting to be sent (owned by ssl_)
    std::mutex ssl_mutex_;
    std::mutex write_mutex_;
    uint64_t next_frame_id_ = 1;  // guarded by write_mutex_
    FrameDecoder decoder_;        // reader thread only
    std::atomic<bool> closed_{false};
    std::string peer_identity_;
};

// Runs the TLS handshake on `stream` and verifies the peer:
//   client role: the server chain verifies against ca_file and the certificate
//                matches `server_name` (DNS name or IP literal);
//   server role: the client chain verifies against ca_file and one of its
//                SANs is in allowed_peer_names (`server_name` is ignored).
// Returns nullptr (stream closed, reason in *error) on any failure, including
// a peer that is silent past `timeout`.
std::unique_ptr<TlsTransport> TlsHandshake(TcpStream stream, std::shared_ptr<TlsContext> context,
                                           const std::string& server_name,
                                           std::chrono::milliseconds timeout,
                                           std::string* error = nullptr);

}  // namespace tuna::tunnel

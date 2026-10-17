#include "tuna/tunnel/tls.h"

#include <array>
#include <condition_variable>
#include <cstring>
#include <thread>
#include <utility>

#ifdef _WIN32
// Before OpenSSL, and lean: wincrypt.h's X509_NAME macro clashes with OpenSSL.
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#endif

#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

namespace tuna::tunnel {

namespace {

void SetError(std::string* error, const std::string& message) {
    if (error) *error = message;
}

// Drains OpenSSL's thread-local error queue into one readable string.
std::string OpenSslErrors(const std::string& context) {
    std::string text = context;
    while (const unsigned long code = ERR_get_error()) {
        char buffer[256];
        ERR_error_string_n(code, buffer, sizeof(buffer));
        text += "; ";
        text += buffer;
    }
    return text;
}

bool IsIpLiteral(const std::string& host) {
    in_addr v4{};
    in6_addr v6{};
    return inet_pton(AF_INET, host.c_str(), &v4) == 1 || inet_pton(AF_INET6, host.c_str(), &v6) == 1;
}

// DNS and URI subjectAltName values of `cert`. Values with embedded NULs are
// skipped: they are a classic way to make "evil.com\0.good" match "good".
std::vector<std::string> SubjectAltNames(X509* cert) {
    std::vector<std::string> names;
    auto* sans = static_cast<GENERAL_NAMES*>(X509_get_ext_d2i(cert, NID_subject_alt_name, nullptr, nullptr));
    if (!sans) return names;
    for (int i = 0; i < sk_GENERAL_NAME_num(sans); ++i) {
        const GENERAL_NAME* name = sk_GENERAL_NAME_value(sans, i);
        const ASN1_IA5STRING* value = nullptr;
        if (name->type == GEN_DNS) value = name->d.dNSName;
        else if (name->type == GEN_URI) value = name->d.uniformResourceIdentifier;
        if (!value) continue;
        const auto* data = reinterpret_cast<const char*>(ASN1_STRING_get0_data(value));
        const int length = ASN1_STRING_length(value);
        if (length <= 0 || std::memchr(data, '\0', static_cast<size_t>(length)) != nullptr) continue;
        names.emplace_back(data, static_cast<size_t>(length));
    }
    GENERAL_NAMES_free(sans);
    return names;
}

}  // namespace

// --------------------------------------------------------------- TlsContext

std::shared_ptr<TlsContext> TlsContext::Create(TlsRole role, const TlsConfig& config, std::string* error) {
    if (config.ca_file.empty() || config.cert_file.empty() || config.key_file.empty()) {
        SetError(error, "TLS needs ca_file, cert_file and key_file (Track B is mutual TLS only)");
        return nullptr;
    }
    if (role == TlsRole::kServer && config.allowed_peer_names.empty()) {
        SetError(error, "TLS server needs at least one allowed client name; a certificate that "
                        "chains to the CA is not by itself authorization");
        return nullptr;
    }

    ERR_clear_error();
    SSL_CTX* ctx = SSL_CTX_new(role == TlsRole::kServer ? TLS_server_method() : TLS_client_method());
    if (!ctx) {
        SetError(error, OpenSslErrors("SSL_CTX_new failed"));
        return nullptr;
    }
    // Wrap immediately so every early return frees the context.
    std::shared_ptr<TlsContext> context(new TlsContext(role, ctx, config.allowed_peer_names));

    // Release gate shared with Track A: never negotiate below TLS 1.3.
    if (SSL_CTX_set_min_proto_version(ctx, TLS1_3_VERSION) != 1) {
        SetError(error, OpenSslErrors("cannot require TLS 1.3"));
        return nullptr;
    }
    // Sessions are short-lived and per process; tickets would be the only
    // post-handshake traffic, so turn them off and keep the data phase simple.
    SSL_CTX_set_num_tickets(ctx, 0);
    SSL_CTX_set_options(ctx, SSL_OP_NO_TICKET);

    if (SSL_CTX_use_certificate_chain_file(ctx, config.cert_file.c_str()) != 1) {
        SetError(error, OpenSslErrors("cannot load certificate '" + config.cert_file + "'"));
        return nullptr;
    }
    if (SSL_CTX_use_PrivateKey_file(ctx, config.key_file.c_str(), SSL_FILETYPE_PEM) != 1 ||
        SSL_CTX_check_private_key(ctx) != 1) {
        SetError(error, OpenSslErrors("cannot load private key '" + config.key_file +
                                      "' (or it does not match the certificate)"));
        return nullptr;
    }
    if (SSL_CTX_load_verify_locations(ctx, config.ca_file.c_str(), nullptr) != 1) {
        SetError(error, OpenSslErrors("cannot load CA '" + config.ca_file + "'"));
        return nullptr;
    }

    // The default verify purpose already enforces EKU: a server checks the
    // client cert for clientAuth, a client checks the server cert for serverAuth.
    if (role == TlsRole::kServer) {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, nullptr);
        if (STACK_OF(X509_NAME)* names = SSL_load_client_CA_file(config.ca_file.c_str())) {
            SSL_CTX_set_client_CA_list(ctx, names);  // hint for which cert to present
        }
    } else {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
    }
    return context;
}

TlsContext::~TlsContext() { SSL_CTX_free(ctx_); }

// -------------------------------------------------------------- TlsTransport

TlsTransport::TlsTransport(TcpStream stream, std::shared_ptr<TlsContext> context)
    : stream_(std::move(stream)), context_(std::move(context)) {
    ssl_ = SSL_new(context_->native());
    in_bio_ = BIO_new(BIO_s_mem());
    out_bio_ = BIO_new(BIO_s_mem());
    if (ssl_ && in_bio_ && out_bio_) {
        SSL_set_bio(ssl_, in_bio_, out_bio_);  // ssl_ owns both BIOs from here on
    } else {
        BIO_free(in_bio_);
        BIO_free(out_bio_);
        in_bio_ = out_bio_ = nullptr;
    }
}

TlsTransport::~TlsTransport() {
    Close();
    SSL_free(ssl_);
}

bool TlsTransport::FlushLocked() {
    std::string ciphertext;
    {
        std::lock_guard<std::mutex> lock(ssl_mutex_);
        char* data = nullptr;
        const long pending = BIO_get_mem_data(out_bio_, &data);
        if (pending > 0) {
            ciphertext.assign(data, static_cast<size_t>(pending));
            (void)BIO_reset(out_bio_);
        }
    }
    if (ciphertext.empty()) return true;
    return stream_.SendAll(ciphertext.data(), ciphertext.size());
}

bool TlsTransport::Handshake(const std::string& server_name, std::string* error) {
    if (!ssl_ || !in_bio_) {
        SetError(error, OpenSslErrors("cannot allocate TLS session"));
        return false;
    }
    if (context_->role() == TlsRole::kServer) {
        SSL_set_accept_state(ssl_);
    } else {
        SSL_set_connect_state(ssl_);
        X509_VERIFY_PARAM* param = SSL_get0_param(ssl_);
        X509_VERIFY_PARAM_set_hostflags(param, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
        const bool ip = IsIpLiteral(server_name);
        const int ok = ip ? X509_VERIFY_PARAM_set1_ip_asc(param, server_name.c_str())
                          : X509_VERIFY_PARAM_set1_host(param, server_name.c_str(), 0);
        if (ok != 1 || (!ip && SSL_set_tlsext_host_name(ssl_, server_name.c_str()) != 1)) {
            SetError(error, OpenSslErrors("invalid server name '" + server_name + "'"));
            return false;
        }
    }

    // No other thread uses this transport yet, so the locks are uncontended;
    // they are taken anyway to keep the invariants simple.
    std::array<char, 16 * 1024> buffer;
    while (true) {
        int rc;
        int status;
        {
            std::lock_guard<std::mutex> lock(ssl_mutex_);
            ERR_clear_error();
            rc = SSL_do_handshake(ssl_);
            status = SSL_get_error(ssl_, rc);
        }
        {
            std::lock_guard<std::mutex> lock(write_mutex_);
            if (!FlushLocked()) {
                SetError(error, "connection lost during TLS handshake");
                return false;
            }
        }
        if (rc == 1) break;
        if (status != SSL_ERROR_WANT_READ) {
            std::string reason = OpenSslErrors("TLS handshake failed");
            const long verify = SSL_get_verify_result(ssl_);
            if (verify != X509_V_OK) {
                reason += std::string("; certificate: ") + X509_verify_cert_error_string(verify);
            }
            SetError(error, reason);
            return false;
        }
        const size_t n = stream_.Receive(buffer.data(), buffer.size());
        if (n == 0) {
            SetError(error, "peer closed the connection during the TLS handshake "
                            "(it may have rejected our certificate or protocol version)");
            return false;
        }
        std::lock_guard<std::mutex> lock(ssl_mutex_);
        BIO_write(in_bio_, buffer.data(), static_cast<int>(n));
    }
    return VerifyPeer(server_name, error);
}

bool TlsTransport::VerifyPeer(const std::string& server_name, std::string* error) {
    // SSL_VERIFY_PEER already failed the handshake on a bad chain; check again
    // so a future change to the verify callback cannot silently weaken this.
    if (SSL_get_verify_result(ssl_) != X509_V_OK) {
        SetError(error, "peer certificate did not verify");
        return false;
    }
    X509* peer = SSL_get1_peer_certificate(ssl_);
    if (!peer) {
        SetError(error, "peer presented no certificate");
        return false;
    }
    bool ok = false;
    if (context_->role() == TlsRole::kClient) {
        peer_identity_ = server_name;  // matched by OpenSSL's host/IP check
        ok = true;
    } else {
        for (const std::string& san : SubjectAltNames(peer)) {
            for (const std::string& allowed : context_->allowed_peer_names()) {
                if (san == allowed) {
                    peer_identity_ = san;
                    ok = true;
                    break;
                }
            }
            if (ok) break;
        }
        if (!ok) SetError(error, "client certificate identity is not in allowed_client_names");
    }
    X509_free(peer);
    return ok;
}

bool TlsTransport::Send(FrameType type, uint32_t channel, const std::string& payload) {
    if (closed_) return false;

    Frame frame;
    frame.mutable_header()->set_type(type);
    frame.mutable_header()->set_channel(channel);
    frame.set_payload(payload);

    std::lock_guard<std::mutex> write_lock(write_mutex_);
    frame.mutable_header()->set_frame_id(next_frame_id_++);
    frame.mutable_header()->set_timestamp_ns(NowNs());
    std::string wire;
    if (!EncodeFrame(frame, &wire)) return false;
    {
        std::lock_guard<std::mutex> lock(ssl_mutex_);
        ERR_clear_error();
        // Frames are capped at kMaxFrameBytes (16 MiB), well inside int range;
        // with a memory BIO SSL_write consumes everything or fails fatally.
        if (SSL_write(ssl_, wire.data(), static_cast<int>(wire.size())) <= 0) {
            closed_ = true;
            stream_.Shutdown();
            return false;
        }
    }
    if (!FlushLocked()) {
        Close();
        return false;
    }
    return true;
}

std::optional<Frame> TlsTransport::Receive() {
    Frame frame;
    std::array<char, 64 * 1024> buffer;
    std::string plaintext;
    // Decrypt what is already buffered before blocking on the socket: the
    // last handshake recv() often also carried the peer's first records
    // (TLS 1.3 clients send data right behind their Finished message).
    bool need_bytes = false;
    while (!decoder_.Next(&frame)) {
        if (closed_) return std::nullopt;
        size_t n = 0;
        if (need_bytes) {
            n = stream_.Receive(buffer.data(), buffer.size());  // no lock held
            if (n == 0) {
                Close();
                return std::nullopt;
            }
        }
        need_bytes = true;  // each pass below decrypts until OpenSSL wants more input

        bool failed = false;
        bool has_output = false;
        plaintext.clear();
        {
            std::lock_guard<std::mutex> lock(ssl_mutex_);
            if (n > 0) BIO_write(in_bio_, buffer.data(), static_cast<int>(n));
            while (true) {
                ERR_clear_error();
                const int got = SSL_read(ssl_, buffer.data(), static_cast<int>(buffer.size()));
                if (got > 0) {
                    plaintext.append(buffer.data(), static_cast<size_t>(got));
                    continue;
                }
                const int status = SSL_get_error(ssl_, got);
                // WANT_READ: record incomplete, wait for more bytes. ZERO_RETURN
                // (close_notify) or anything else ends the connection.
                failed = status != SSL_ERROR_WANT_READ;
                break;
            }
            has_output = BIO_ctrl_pending(out_bio_) > 0;  // e.g. an alert
        }
        if (has_output) {
            std::lock_guard<std::mutex> lock(write_mutex_);
            FlushLocked();
        }
        if (!plaintext.empty() && !decoder_.Feed(plaintext)) failed = true;
        if (failed) {
            // Deliver frames that were complete before the failure first.
            if (decoder_.Next(&frame)) {
                Close();
                return frame;
            }
            Close();
            return std::nullopt;
        }
    }
    return frame;
}

void TlsTransport::Close() {
    // No close_notify: a sender may be blocked in SendAll() holding
    // write_mutex_, and shutting the socket down is what unblocks it. Frames
    // are length-prefixed, so truncation is still detected.
    if (!closed_.exchange(true)) {
        stream_.Shutdown();
    }
}

std::unique_ptr<TlsTransport> TlsHandshake(TcpStream stream, std::shared_ptr<TlsContext> context,
                                           const std::string& server_name,
                                           std::chrono::milliseconds timeout, std::string* error) {
    std::unique_ptr<TlsTransport> transport(new TlsTransport(std::move(stream), std::move(context)));

    // The handshake blocks in recv(); a watchdog shuts the socket down if the
    // peer stays silent, the same pattern as ExchangeHello().
    std::mutex mutex;
    std::condition_variable done_cv;
    bool done = false;
    bool timed_out = false;
    std::thread watchdog([&] {
        std::unique_lock<std::mutex> lock(mutex);
        if (!done_cv.wait_for(lock, timeout, [&] { return done; })) {
            timed_out = true;
            transport->Close();
        }
    });

    std::string reason;
    const bool ok = transport->Handshake(server_name, &reason);
    {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
    }
    done_cv.notify_one();
    watchdog.join();

    if (!ok || timed_out) {
        SetError(error, timed_out ? "timed out during TLS handshake" : reason);
        transport->Close();
        return nullptr;
    }
    return transport;
}

}  // namespace tuna::tunnel

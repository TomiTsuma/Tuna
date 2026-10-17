#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#include <openssl/ssl.h>

#include "tuna/tunnel/pki.h"
#include "tuna/tunnel/rpc.h"
#include "tuna/tunnel/session.h"
#include "tuna/tunnel/tcp_socket.h"
#include "tuna/tunnel/tls.h"

using namespace std::chrono_literals;
using namespace tuna;
using namespace tuna::tunnel;
namespace fs = std::filesystem;

namespace {

// A throwaway PKI, generated once per test binary run:
//   ca/       trusted CA; server "localhost", clients "alice" and "mallory"
//   evil-ca/  untrusted CA; a client also named "alice"
struct TestPki {
    fs::path root;

    TestPki() {
        std::random_device rd;
        root = fs::temp_directory_path() / ("tuna-tls-test-" + std::to_string(rd()));
        std::string error;
        REQUIRE(GenerateCa(root / "ca", "Tuna Test CA", 2, &error));
        REQUIRE(GenerateCa(root / "evil-ca", "Untrusted CA", 2, &error));
        REQUIRE(IssueCertificate(root / "ca", root / "ca",
                                 {"localhost", true, {"DNS:localhost", "IP:127.0.0.1"}, 2}, &error));
        REQUIRE(IssueCertificate(root / "ca", root / "ca", {"alice", false, {}, 2}, &error));
        REQUIRE(IssueCertificate(root / "ca", root / "ca", {"mallory", false, {}, 2}, &error));
        REQUIRE(IssueCertificate(root / "evil-ca", root / "evil-ca", {"alice", false, {}, 2}, &error));
    }
    ~TestPki() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    TlsConfig Server() const {
        return {(root / "ca/ca.crt").string(), (root / "ca/localhost.crt").string(),
                (root / "ca/localhost.key").string(), {"alice"}};
    }
    TlsConfig Client(const std::string& ca_dir, const std::string& name) const {
        return {(root / "ca/ca.crt").string(), (root / ca_dir / (name + ".crt")).string(),
                (root / ca_dir / (name + ".key")).string(), {}};
    }
};

const TestPki& Pki() {
    static TestPki pki;
    return pki;
}

// Accepts one TCP connection and runs the server side of the TLS handshake.
class TlsServer {
public:
    explicit TlsServer(std::chrono::milliseconds timeout = 5s, bool serve_echo = true) {
        context_ = TlsContext::Create(TlsRole::kServer, Pki().Server(), &error);
        REQUIRE(context_);
        auto listener = TcpListener::Bind("127.0.0.1", 0);
        REQUIRE(listener);
        listener_ = std::make_unique<TcpListener>(std::move(*listener));
        thread_ = std::thread([this, timeout, serve_echo] {
            auto stream = listener_->Accept();
            if (!stream) return;
            std::shared_ptr<TlsTransport> tls = TlsHandshake(std::move(*stream), context_, "", timeout, &error);
            if (!tls) {
                done = true;
                return;
            }
            identity = tls->peer_identity();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                transport_ = tls;
            }
            if (serve_echo) {
                Hello hello;
                hello.set_protocol_version(kProtocolVersion);
                if (ExchangeHello(*tls, hello, 2s)) {
                    RpcServer server(*tls, [](ServerCall& call) {
                        RpcResponse response;
                        response.set_out_args(call.request.args());
                        call.bulk_out = call.bulk_in;
                        return response;
                    });
                    server.Serve();
                }
            }
            handshake_ok = true;
            done = true;
        });
    }
    ~TlsServer() {
        listener_->Shutdown();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (transport_) transport_->Close();
        }
        thread_.join();
    }
    uint16_t port() const { return listener_->port(); }

    std::string error;
    std::string identity;
    std::atomic<bool> handshake_ok{false};
    std::atomic<bool> done{false};

private:
    std::shared_ptr<TlsContext> context_;
    std::unique_ptr<TcpListener> listener_;
    std::mutex mutex_;
    std::shared_ptr<TlsTransport> transport_;  // guarded by mutex_
    std::thread thread_;
};

std::unique_ptr<TlsTransport> Dial(uint16_t port, std::shared_ptr<TlsContext> context,
                                   const std::string& server_name = "localhost", std::string* error = nullptr) {
    auto stream = TcpStream::Connect("127.0.0.1", port);
    REQUIRE(stream);
    return TlsHandshake(std::move(*stream), std::move(context), server_name, 5s, error);
}

std::shared_ptr<TlsContext> ClientContext(const std::string& ca_dir, const std::string& name) {
    std::string error;
    auto context = TlsContext::Create(TlsRole::kClient, Pki().Client(ca_dir, name), &error);
    INFO(error);
    REQUIRE(context);
    return context;
}

std::optional<Hello> Handshake(Transport& transport) {
    Hello hello;
    hello.set_protocol_version(kProtocolVersion);
    return ExchangeHello(transport, hello, 2s);
}

}  // namespace

TEST_CASE("mTLS session with an allowlisted client serves RPCs", "[tls]") {
    TlsServer server;
    std::string error;
    auto client = Dial(server.port(), ClientContext("ca", "alice"), "localhost", &error);
    INFO(error);
    REQUIRE(client);
    REQUIRE(Handshake(*client));
    {
        RpcClient rpc(*client);
        const CallResult r = rpc.Call(kApiEcho, "over tls");
        CHECK(r.result == kResultSuccess);
        CHECK(r.out_args == "over tls");
    }
    client->Close();
    while (!server.done) std::this_thread::sleep_for(1ms);
    CHECK(server.identity == "alice");
}

TEST_CASE("The server's IP SAN satisfies a client dialling by address", "[tls]") {
    TlsServer server;
    std::string error;
    auto client = Dial(server.port(), ClientContext("ca", "alice"), "127.0.0.1", &error);
    INFO(error);
    REQUIRE(client);
    CHECK(client->peer_identity() == "127.0.0.1");
}

TEST_CASE("Large frames flow both ways at once over TLS", "[tls]") {
    TlsServer server;
    auto client = Dial(server.port(), ClientContext("ca", "alice"));
    REQUIRE(client);
    REQUIRE(Handshake(*client));
    RpcClient rpc(*client);
    // 8 MiB args echoed back while more big posts are in flight: both peers
    // send large ciphertext concurrently, which deadlocks if decryption ever
    // waits behind a blocked socket write.
    const std::string big(8 * 1024 * 1024, 'x');
    for (int i = 0; i < 4; ++i) rpc.Post(kApiEcho, big);
    const CallResult r = rpc.Call(kApiEcho, big, true);
    CHECK(r.result == kResultSuccess);
    CHECK(r.out_args.size() == big.size());
}

TEST_CASE("Data sent right behind the handshake is not stranded in TLS buffers", "[tls]") {
    // A TLS 1.3 client's first records usually ride in the same TCP segment
    // as its Finished message, so the server's last handshake recv() holds
    // application data too. That data must be delivered without waiting for
    // more bytes (a regression: it once sat in the BIO until the peer sent
    // again, stalling the Hello exchange).
    auto server_context = TlsContext::Create(TlsRole::kServer, Pki().Server());
    auto client_context = ClientContext("ca", "alice");
    REQUIRE(server_context);
    for (int i = 0; i < 40; ++i) {
        auto listener = TcpListener::Bind("127.0.0.1", 0);
        REQUIRE(listener);
        std::atomic<bool> got_frame{false};
        std::thread server([&] {
            auto stream = listener->Accept();
            if (!stream) return;
            auto tls = TlsHandshake(std::move(*stream), server_context, "", 5s);
            if (!tls) return;
            got_frame = Handshake(*tls).has_value();  // 2 s deadline
            tls->Close();
        });
        auto stream = TcpStream::Connect("127.0.0.1", listener->port());
        REQUIRE(stream);
        auto client = TlsHandshake(std::move(*stream), client_context, "localhost", 5s);
        REQUIRE(client);
        Hello hello;
        hello.set_protocol_version(kProtocolVersion);
        client->Send(FRAME_TYPE_CONTROL, kControlChannel, hello.SerializeAsString());  // immediately
        server.join();
        INFO("iteration " << i);
        REQUIRE(got_frame);
        client->Close();
    }
}

TEST_CASE("A client from an untrusted CA is rejected", "[tls]") {
    TlsServer server;
    auto context = TlsContext::Create(TlsRole::kClient, {(Pki().root / "ca/ca.crt").string(),
                                                         (Pki().root / "evil-ca/alice.crt").string(),
                                                         (Pki().root / "evil-ca/alice.key").string(), {}});
    REQUIRE(context);
    auto client = Dial(server.port(), context);
    if (client) {
        // TLS 1.3 clients finish before the server checks their certificate;
        // the server's rejection then shows up as a closed connection.
        CHECK_FALSE(Handshake(*client));
    }
    while (!server.done) std::this_thread::sleep_for(1ms);
    CHECK_FALSE(server.handshake_ok);
    CHECK(server.error.find("certificate") != std::string::npos);
}

TEST_CASE("A trusted client that is not allowlisted is rejected", "[tls]") {
    TlsServer server;
    auto client = Dial(server.port(), ClientContext("ca", "mallory"));
    if (client) CHECK_FALSE(Handshake(*client));
    while (!server.done) std::this_thread::sleep_for(1ms);
    CHECK_FALSE(server.handshake_ok);
    CHECK(server.error.find("allowed_client_names") != std::string::npos);
}

TEST_CASE("A client that presents no certificate is rejected", "[tls]") {
    TlsServer server;
    auto context = ClientContext("ca", "alice");
    // Withhold the certificate when the server asks for it.
    SSL_CTX_set_cert_cb(context->native(), [](SSL* ssl, void*) {
        SSL_certs_clear(ssl);
        return 1;
    }, nullptr);
    auto client = Dial(server.port(), context);
    if (client) CHECK_FALSE(Handshake(*client));
    while (!server.done) std::this_thread::sleep_for(1ms);
    CHECK_FALSE(server.handshake_ok);
}

TEST_CASE("A TLS 1.2-only client cannot connect", "[tls]") {
    TlsServer server;
    auto context = ClientContext("ca", "alice");
    SSL_CTX_set_min_proto_version(context->native(), TLS1_2_VERSION);
    SSL_CTX_set_max_proto_version(context->native(), TLS1_2_VERSION);
    std::string error;
    CHECK_FALSE(Dial(server.port(), context, "localhost", &error));
    while (!server.done) std::this_thread::sleep_for(1ms);
    CHECK_FALSE(server.handshake_ok);
}

TEST_CASE("The client rejects a server whose name does not match", "[tls]") {
    TlsServer server;
    std::string error;
    CHECK_FALSE(Dial(server.port(), ClientContext("ca", "alice"), "gpu.example.com", &error));
    CHECK(error.find("certificate") != std::string::npos);
}

TEST_CASE("A server context without an allowlist is refused", "[tls]") {
    TlsConfig config = Pki().Server();
    config.allowed_peer_names.clear();
    std::string error;
    CHECK_FALSE(TlsContext::Create(TlsRole::kServer, config, &error));
    CHECK(error.find("allowed client") != std::string::npos);
}

TEST_CASE("The TLS handshake times out on a silent peer", "[tls]") {
    TlsServer server(200ms, false);
    auto silent = TcpStream::Connect("127.0.0.1", server.port());
    REQUIRE(silent);
    const auto start = std::chrono::steady_clock::now();
    while (!server.done && std::chrono::steady_clock::now() - start < 5s) std::this_thread::sleep_for(5ms);
    CHECK(server.done);
    CHECK(server.error.find("timed out") != std::string::npos);
}

TEST_CASE("Certificate tooling refuses to overwrite existing files", "[tls][pki]") {
    std::string error;
    CHECK_FALSE(GenerateCa(Pki().root / "ca", "again", 1, &error));
    CHECK(error.find("refusing to overwrite") != std::string::npos);
    CHECK_FALSE(IssueCertificate(Pki().root / "ca", Pki().root / "ca", {"alice", false, {}, 1}, &error));
    CHECK_FALSE(IssueCertificate(Pki().root / "ca", Pki().root / "ca", {"../escape", false, {}, 1}, &error));
}

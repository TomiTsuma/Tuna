#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <memory>
#include <thread>

#include "tuna/tunnel/rpc.h"
#include "tuna/tunnel/session.h"
#include "tuna/tunnel/tcp_socket.h"
#include "tuna/tunnel/transport.h"

using namespace std::chrono_literals;
using namespace tuna;
using namespace tuna::tunnel;

namespace {

// Test API ids served by the loopback server.
constexpr uint32_t kEcho = 1;       // out_args = args
constexpr uint32_t kFail = 2;       // result = 700 (CUDA_ERROR_ILLEGAL_ADDRESS)
constexpr uint32_t kIncrement = 3;  // counter += 1
constexpr uint32_t kRead = 4;       // out_args = counter
constexpr uint32_t kBulkEcho = 5;   // bulk_out = bulk_in
constexpr uint32_t kBulkCount = 6;  // counter += bulk_in.size()
constexpr int32_t kIllegalAddress = 700;

Hello MakeHello(const std::string& os) {
    Hello hello;
    hello.set_protocol_version(kProtocolVersion);
    hello.set_client_os(os);
    return hello;
}

// One loopback session: a server thread that handshakes and serves the test
// API, and a client transport connected to it (handshake done by the test).
class Loopback {
public:
    explicit Loopback(HeartbeatOptions server_heartbeat = {}) {
        auto listener = TcpListener::Bind("127.0.0.1", 0);
        REQUIRE(listener);
        listener_ = std::make_unique<TcpListener>(std::move(*listener));
        server_ = std::thread([this, server_heartbeat] {
            auto stream = listener_->Accept();
            if (!stream) return;
            server_transport_ = std::make_unique<TcpTransport>(std::move(*stream));
            server_transport_ptr_ = server_transport_.get();
            Hello local = MakeHello("server");
            local.set_cuda_driver_version(12080);
            DeviceInfo* device = local.add_devices();
            device->set_name("Mock GPU");
            device->set_cc_major(8);
            device->set_cc_minor(6);
            if (!ExchangeHello(*server_transport_, local, 2s)) return;
            RpcServer server(*server_transport_, [this](ServerCall& c) { return Dispatch(c); },
                             server_heartbeat);
            served_ = server.Serve();
        });
        auto stream = TcpStream::Connect("127.0.0.1", listener_->port());
        REQUIRE(stream);
        client = std::make_unique<TcpTransport>(std::move(*stream));
    }

    ~Loopback() {
        client->Close();
        server_.join();
    }

    void KillServer() {
        while (!server_transport_ptr_) std::this_thread::sleep_for(1ms);
        server_transport_ptr_.load()->Close();
    }

    uint64_t counter() const { return counter_; }

    std::unique_ptr<TcpTransport> client;
    std::atomic<uint64_t> served_{0};

private:
    RpcResponse Dispatch(ServerCall& call) {
        const RpcRequest& request = call.request;
        RpcResponse response;
        switch (request.api_id()) {
            case kBulkEcho: call.bulk_out = call.bulk_in; break;
            case kBulkCount: counter_ += call.bulk_in.size(); break;
            case kEcho: response.set_out_args(request.args()); break;
            case kFail: response.set_result(kIllegalAddress); break;
            case kIncrement: ++counter_; break;
            case kRead: response.set_out_args(std::to_string(counter_.load())); break;
            default: response.set_result(801);  // CUDA_ERROR_NOT_SUPPORTED
        }
        return response;
    }

    std::unique_ptr<TcpListener> listener_;
    std::unique_ptr<TcpTransport> server_transport_;  // owned by the server thread
    std::atomic<TcpTransport*> server_transport_ptr_{nullptr};  // published for KillServer()
    std::atomic<uint64_t> counter_{0};
    std::thread server_;
};

}  // namespace

TEST_CASE("Hello exchange carries server capabilities", "[tunnel][rpc]") {
    Loopback loop;
    std::string error;
    auto peer = ExchangeHello(*loop.client, MakeHello("windows"), 2s, &error);
    REQUIRE(peer);
    CHECK(peer->cuda_driver_version() == 12080);
    REQUIRE(peer->devices_size() == 1);
    CHECK(peer->devices(0).name() == "Mock GPU");
}

TEST_CASE("Hello exchange rejects a protocol version mismatch", "[tunnel][rpc]") {
    Loopback loop;
    Hello wrong = MakeHello("windows");
    wrong.set_protocol_version(kProtocolVersion + 1);
    // The server rejects us and closes, so our side sees a closed connection or
    // (if its Hello arrived first) accepts and then fails on the first call.
    std::string error;
    auto peer = ExchangeHello(*loop.client, wrong, 2s, &error);
    if (peer) {
        RpcClient rpc(*loop.client);
        CHECK(rpc.Call(kEcho, "x").result == kResultTransportError);
    }
}

TEST_CASE("Hello exchange times out on a silent peer", "[tunnel][rpc]") {
    auto listener = TcpListener::Bind("127.0.0.1", 0);
    REQUIRE(listener);
    auto stream = TcpStream::Connect("127.0.0.1", listener->port());
    REQUIRE(stream);
    auto accepted = listener->Accept();  // accepted but never speaks
    REQUIRE(accepted);

    TcpTransport transport(std::move(*stream));
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    CHECK_FALSE(ExchangeHello(transport, MakeHello("windows"), 200ms, &error));
    CHECK(error == "timed out waiting for peer Hello");
    CHECK(std::chrono::steady_clock::now() - start < 2s);
}

TEST_CASE("Call returns the server's reply", "[tunnel][rpc]") {
    Loopback loop;
    REQUIRE(ExchangeHello(*loop.client, MakeHello("windows"), 2s));
    RpcClient rpc(*loop.client);

    auto echoed = rpc.Call(kEcho, std::string("payload\0with-nul", 16));
    CHECK(echoed.result == kResultSuccess);
    CHECK(echoed.out_args == std::string("payload\0with-nul", 16));
    CHECK(rpc.Call(999, "").result == 801);
}

TEST_CASE("Posts execute in order before a following Call", "[tunnel][rpc]") {
    Loopback loop;
    REQUIRE(ExchangeHello(*loop.client, MakeHello("windows"), 2s));
    RpcClient rpc(*loop.client);

    for (int i = 0; i < 1000; ++i) rpc.Post(kIncrement, "");
    auto read = rpc.Call(kRead, "");
    REQUIRE(read.result == kResultSuccess);
    CHECK(read.out_args == "1000");
}

TEST_CASE("Posts are flushed by the timer without a Call", "[tunnel][rpc]") {
    Loopback loop;
    REQUIRE(ExchangeHello(*loop.client, MakeHello("windows"), 2s));
    RpcClientOptions options;
    options.flush_delay = 1ms;
    RpcClient rpc(*loop.client, options);

    rpc.Post(kIncrement, "");
    // No Call() follows, so only the flush timer can deliver the post.
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (loop.counter() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    CHECK(loop.counter() == 1);
}

TEST_CASE("A failed async call surfaces on the next synchronizing call only", "[tunnel][rpc]") {
    Loopback loop;
    REQUIRE(ExchangeHello(*loop.client, MakeHello("windows"), 2s));
    RpcClient rpc(*loop.client);

    rpc.Post(kFail, "");
    rpc.Post(kIncrement, "");
    // A non-synchronizing call reports its own result...
    CHECK(rpc.Call(kEcho, "a").result == kResultSuccess);
    // ...the next synchronizing call reports the deferred error, once.
    CHECK(rpc.Call(kEcho, "b", /*synchronizing=*/true).result == kIllegalAddress);
    CHECK(rpc.Call(kEcho, "c", /*synchronizing=*/true).result == kResultSuccess);
    // The work after the failure still ran (CUDA does not cancel the stream here).
    CHECK(rpc.Call(kRead, "").out_args == "1");
}

TEST_CASE("Losing the server fails in-flight and later calls", "[tunnel][rpc]") {
    Loopback loop;
    REQUIRE(ExchangeHello(*loop.client, MakeHello("windows"), 2s));
    RpcClient rpc(*loop.client);
    REQUIRE(rpc.Call(kEcho, "up").result == kResultSuccess);

    loop.KillServer();
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (rpc.connected() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    CHECK_FALSE(rpc.connected());
    CHECK(rpc.Call(kEcho, "down").result == kResultTransportError);
}

TEST_CASE("Heartbeat closes a session whose peer goes silent", "[tunnel][rpc]") {
    auto listener = TcpListener::Bind("127.0.0.1", 0);
    REQUIRE(listener);
    std::thread silent_peer([&] {
        // Plain checks: Catch2 assertions are not thread-safe off the main thread.
        auto stream = listener->Accept();
        if (!stream) return;
        TcpTransport transport(std::move(*stream));
        if (!ExchangeHello(transport, MakeHello("server"), 2s)) return;
        // ...then never sends again; drain until the client hangs up.
        while (transport.Receive()) {
        }
    });

    auto stream = TcpStream::Connect("127.0.0.1", listener->port());
    REQUIRE(stream);
    TcpTransport transport(std::move(*stream));
    REQUIRE(ExchangeHello(transport, MakeHello("windows"), 2s));
    RpcClientOptions options;
    options.heartbeat.interval = 20ms;
    options.heartbeat.timeout = 150ms;
    {
        RpcClient rpc(transport, options);
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (rpc.connected() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(5ms);
        }
        CHECK_FALSE(rpc.connected());
    }
    silent_peer.join();
}

TEST_CASE("Loopback detection", "[tunnel]") {
    CHECK(IsLoopbackHost("127.0.0.1"));
    CHECK(IsLoopbackHost("127.10.0.3"));
    CHECK(IsLoopbackHost("::1"));
    CHECK(IsLoopbackHost("localhost"));
    CHECK_FALSE(IsLoopbackHost("0.0.0.0"));
    CHECK_FALSE(IsLoopbackHost("192.168.1.10"));
    CHECK_FALSE(IsLoopbackHost("::"));
    CHECK_FALSE(IsLoopbackHost("example.com"));
}

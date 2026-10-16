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

constexpr uint32_t kBulkEcho = 1;   // bulk_out = bulk_in, out_args = size
constexpr uint32_t kBulkCount = 2;  // bytes += bulk_in.size()
constexpr uint32_t kReadCount = 3;  // out_args = bytes

std::string Pattern(size_t size) {
    std::string data(size, '\0');
    for (size_t i = 0; i < size; ++i) data[i] = static_cast<char>((i * 131) ^ (i >> 11));
    return data;
}

BulkChunk Chunk(uint64_t ref, uint64_t offset, uint64_t total, const std::string& data) {
    BulkChunk chunk;
    chunk.set_bulk_ref(ref);
    chunk.set_offset(offset);
    chunk.set_total_size(total);
    chunk.set_data(data);
    return chunk;
}

Hello MakeHello() {
    Hello hello;
    hello.set_protocol_version(kProtocolVersion);
    return hello;
}

// A session whose server optionally gets a second, bulk-only connection that
// is attached after Serve() has started (as a worker receives it later from
// the listener).
class BulkSession {
public:
    explicit BulkSession(bool separate_bulk) {
        auto listener = TcpListener::Bind("127.0.0.1", 0);
        REQUIRE(listener);
        listener_ = std::make_unique<TcpListener>(std::move(*listener));
        server_ = std::thread([this, separate_bulk] {
            auto stream = listener_->Accept();
            if (!stream) return;
            TcpTransport main(std::move(*stream));
            server_main_ = &main;
            if (!ExchangeHello(main, MakeHello(), 2s)) return;
            RpcServer server(main, [this](ServerCall& call) {
                RpcResponse response;
                switch (call.request.api_id()) {
                    case kBulkEcho:
                        call.bulk_out = call.bulk_in;
                        response.set_out_args(std::to_string(call.bulk_in.size()));
                        break;
                    case kBulkCount: bytes_ += call.bulk_in.size(); break;
                    case kReadCount: response.set_out_args(std::to_string(bytes_.load())); break;
                }
                return response;
            });
            std::unique_ptr<TcpTransport> bulk;
            std::thread attach;
            if (separate_bulk) {
                attach = std::thread([&] {
                    if (auto s = listener_->Accept()) {
                        bulk = std::make_unique<TcpTransport>(std::move(*s));
                        server.AttachBulk(bulk.get());
                    }
                });
            }
            server.Serve();
            listener_->Shutdown();
            if (attach.joinable()) attach.join();
        });

        auto stream = TcpStream::Connect("127.0.0.1", listener_->port());
        REQUIRE(stream);
        main = std::make_unique<TcpTransport>(std::move(*stream));
        REQUIRE(ExchangeHello(*main, MakeHello(), 2s));
        if (separate_bulk) {
            auto bulk_stream = TcpStream::Connect("127.0.0.1", listener_->port());
            REQUIRE(bulk_stream);
            bulk = std::make_unique<TcpTransport>(std::move(*bulk_stream));
        }
    }
    ~BulkSession() {
        main->Close();
        if (bulk) bulk->Close();
        server_.join();
    }

    std::unique_ptr<TcpTransport> main;
    std::unique_ptr<TcpTransport> bulk;

private:
    std::unique_ptr<TcpListener> listener_;
    std::atomic<Transport*> server_main_{nullptr};
    std::atomic<uint64_t> bytes_{0};
    std::thread server_;
};

}  // namespace

TEST_CASE("BulkStore reassembles chunks in any order", "[tunnel][bulk]") {
    BulkStore store;
    REQUIRE(store.Add(Chunk(7, 4, 8, "5678")));
    CHECK_FALSE(store.Take(7, 10ms));  // incomplete
    REQUIRE(store.Add(Chunk(7, 0, 8, "1234")));
    CHECK(store.Take(7, 10ms) == "12345678");
    CHECK_FALSE(store.Take(7, 10ms));  // taken once
}

TEST_CASE("BulkStore completes zero-length payloads", "[tunnel][bulk]") {
    BulkStore store;
    REQUIRE(store.Add(Chunk(1, 0, 0, "")));
    CHECK(store.Take(1, 10ms) == "");
}

TEST_CASE("BulkStore rejects inconsistent or oversized chunks", "[tunnel][bulk]") {
    BulkStore store;
    CHECK_FALSE(store.Add(Chunk(1, 6, 8, "abc")));                  // runs past total
    CHECK_FALSE(store.Add(Chunk(2, 0, kMaxBulkTransferBytes + 1, "x")));
    REQUIRE(store.Add(Chunk(3, 0, 8, "ab")));
    CHECK_FALSE(store.Add(Chunk(3, 2, 9, "cd")));                   // total changed
    REQUIRE(store.Add(Chunk(3, 2, 8, "cdef")));
    CHECK_FALSE(store.Add(Chunk(3, 0, 8, "abc")));                  // overruns total
}

TEST_CASE("BulkStore::Fail wakes waiters", "[tunnel][bulk]") {
    BulkStore store;
    std::thread failer([&] {
        std::this_thread::sleep_for(20ms);
        store.Fail();
    });
    const auto start = std::chrono::steady_clock::now();
    CHECK_FALSE(store.Take(1, 5s));
    CHECK(std::chrono::steady_clock::now() - start < 2s);
    failer.join();
}

TEST_CASE("Payloads larger than a frame round-trip on the main connection", "[tunnel][bulk]") {
    BulkSession session(false);
    RpcClient rpc(*session.main);
    const std::string data = Pattern(40 * 1024 * 1024);  // > kMaxFrameBytes (16 MiB)
    const CallResult r = rpc.Call(kBulkEcho, "", false, data);
    REQUIRE(r.result == kResultSuccess);
    CHECK(r.out_args == std::to_string(data.size()));
    CHECK(r.bulk_out == data);
}

TEST_CASE("Bulk payloads use a separate socket attached after Serve() starts", "[tunnel][bulk]") {
    BulkSession session(true);
    RpcClient rpc(*session.main, {}, session.bulk.get());
    const std::string data = Pattern(20 * 1024 * 1024 + 17);
    const CallResult r = rpc.Call(kBulkEcho, "", false, data);
    REQUIRE(r.result == kResultSuccess);
    CHECK(r.bulk_out == data);

    // Posts with bulk payloads, then a call that observes all of them.
    for (int i = 0; i < 10; ++i) rpc.Post(kBulkCount, "", Pattern(100000));
    const CallResult count = rpc.Call(kReadCount, "", true);
    CHECK(count.out_args == "1000000");
}

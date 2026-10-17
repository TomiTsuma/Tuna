// Phase 2 exit benchmark: RPC round-trip latency, batched throughput and bulk
// throughput over loopback TCP, optionally over mTLS (with a throwaway PKI).
// Usage: tuna_rpc_bench [--calls N] [--posts N] [--args-bytes N] [--bulk-mb N] [--tls]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "tuna/tunnel/pki.h"
#include "tuna/tunnel/rpc.h"
#include "tuna/tunnel/session.h"
#include "tuna/tunnel/tcp_socket.h"
#include "tuna/tunnel/tls.h"
#include "tuna/tunnel/transport.h"

using namespace std::chrono;
using namespace tuna;
using namespace tuna::tunnel;

namespace {

Hello MakeHello() {
    Hello hello;
    hello.set_protocol_version(kProtocolVersion);
    return hello;
}

double Percentile(std::vector<double>& sorted_us, double p) {
    const size_t index = static_cast<size_t>(p * static_cast<double>(sorted_us.size() - 1));
    return sorted_us[index];
}

// A CA, a server cert for "localhost" and a client cert "bench".
bool MakePki(const std::filesystem::path& dir) {
    return GenerateCa(dir, "Bench CA", 1) && IssueCertificate(dir, dir, {"localhost", true, {}, 1}) &&
           IssueCertificate(dir, dir, {"bench", false, {}, 1});
}

}  // namespace

int main(int argc, char** argv) {
    size_t calls = 10000, posts = 100000, args_bytes = 64, bulk_mb = 64;
    bool use_tls = false;
    for (int i = 1; i < argc; ++i) {
        const auto next = [&] { return i + 1 < argc ? std::stoul(argv[++i]) : 0ul; };
        if (std::strcmp(argv[i], "--calls") == 0) calls = next();
        else if (std::strcmp(argv[i], "--posts") == 0) posts = next();
        else if (std::strcmp(argv[i], "--args-bytes") == 0) args_bytes = next();
        else if (std::strcmp(argv[i], "--bulk-mb") == 0) bulk_mb = next();
        else if (std::strcmp(argv[i], "--tls") == 0) use_tls = true;
    }
    if (calls == 0) calls = 1;

    std::shared_ptr<TlsContext> server_tls, client_tls;
    const auto pki = std::filesystem::temp_directory_path() /
                     ("tuna-bench-pki-" + std::to_string(std::random_device{}()));
    if (use_tls) {
        const auto file = [&](const char* name) { return (pki / name).string(); };
        if (!MakePki(pki) ||
            !(server_tls = TlsContext::Create(TlsRole::kServer, {file("ca.crt"), file("localhost.crt"),
                                                                 file("localhost.key"), {"bench"}})) ||
            !(client_tls = TlsContext::Create(TlsRole::kClient,
                                              {file("ca.crt"), file("bench.crt"), file("bench.key"), {}}))) {
            std::fprintf(stderr, "cannot set up the benchmark PKI\n");
            return 1;
        }
    }
    // Plain TCP, or mTLS over the accepted/connected stream.
    const auto secure = [&](TcpStream stream, bool server) -> std::unique_ptr<Transport> {
        if (!use_tls) return std::make_unique<TcpTransport>(std::move(stream));
        return TlsHandshake(std::move(stream), server ? server_tls : client_tls, "localhost", seconds(5));
    };

    auto listener = TcpListener::Bind("127.0.0.1", 0);
    if (!listener) {
        std::fprintf(stderr, "bind failed\n");
        return 1;
    }
    std::thread server([&] {
        auto stream = listener->Accept();
        if (!stream) return;
        auto transport = secure(std::move(*stream), true);
        if (!transport || !ExchangeHello(*transport, MakeHello(), seconds(5))) return;
        RpcServer rpc(*transport, [](ServerCall& call) {
            RpcResponse response;
            response.set_out_args(call.request.args());
            call.bulk_out = std::move(call.bulk_in);
            return response;
        });
        rpc.Serve();
    });

    auto stream = TcpStream::Connect("127.0.0.1", listener->port());
    if (!stream) {
        std::fprintf(stderr, "connect failed\n");
        return 1;
    }
    auto transport = secure(std::move(*stream), false);
    if (!transport || !ExchangeHello(*transport, MakeHello(), seconds(5))) {
        std::fprintf(stderr, "handshake failed\n");
        return 1;
    }

    int exit_code = 0;
    {
        RpcClient rpc(*transport);
        const std::string args(args_bytes, 'a');

        // Warm up connections, allocators and caches.
        for (int i = 0; i < 200; ++i) rpc.Call(1, args);

        std::vector<double> rtt_us;
        rtt_us.reserve(calls);
        const auto sync_start = steady_clock::now();
        for (size_t i = 0; i < calls; ++i) {
            const auto t0 = steady_clock::now();
            const CallResult r = rpc.Call(1, args);
            rtt_us.push_back(duration<double, std::micro>(steady_clock::now() - t0).count());
            if (r.result != kResultSuccess || r.out_args != args) {
                std::fprintf(stderr, "call %zu returned a wrong result\n", i);
                exit_code = 1;
                break;
            }
        }
        const double sync_s = duration<double>(steady_clock::now() - sync_start).count();
        std::sort(rtt_us.begin(), rtt_us.end());

        const auto async_start = steady_clock::now();
        for (size_t i = 0; i < posts; ++i) rpc.Post(1, args);
        const CallResult fence = rpc.Call(1, args, /*synchronizing=*/true);
        const double async_s = duration<double>(steady_clock::now() - async_start).count();
        if (fence.result != kResultSuccess) exit_code = 1;

        double bulk_s = 0;
        if (bulk_mb > 0) {
            const std::string data(bulk_mb * 1024 * 1024, 'b');
            const auto bulk_start = steady_clock::now();
            const CallResult r = rpc.Call(1, "", false, data);
            bulk_s = duration<double>(steady_clock::now() - bulk_start).count();
            if (r.result != kResultSuccess || r.bulk_out.size() != data.size()) exit_code = 1;
        }

        std::printf("tuna_rpc_bench (loopback %s, %zu-byte args)\n", use_tls ? "TCP + mTLS 1.3" : "TCP",
                    args_bytes);
        std::printf("  sync calls     : %zu in %.3f s -> %.0f calls/s\n", rtt_us.size(), sync_s,
                    static_cast<double>(rtt_us.size()) / sync_s);
        std::printf("  RTT p50 / p99  : %.1f us / %.1f us (min %.1f, max %.1f)\n", Percentile(rtt_us, 0.50),
                    Percentile(rtt_us, 0.99), rtt_us.front(), rtt_us.back());
        std::printf("  batched posts  : %zu + 1 fence in %.3f s -> %.0f calls/s\n", posts, async_s,
                    static_cast<double>(posts + 1) / async_s);
        if (bulk_mb > 0) {
            std::printf("  bulk echo      : %zu MiB each way in %.3f s -> %.0f MiB/s per direction\n", bulk_mb,
                        bulk_s, static_cast<double>(bulk_mb) / bulk_s);
        }
    }
    transport->Close();
    server.join();
    std::error_code ec;
    std::filesystem::remove_all(pki, ec);
    return exit_code;
}

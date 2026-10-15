// Phase 2 exit benchmark: RPC round-trip latency and batched throughput over
// loopback TCP. Usage: tuna_rpc_bench [--calls N] [--posts N] [--args-bytes N]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "tuna/tunnel/rpc.h"
#include "tuna/tunnel/session.h"
#include "tuna/tunnel/tcp_socket.h"
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

}  // namespace

int main(int argc, char** argv) {
    size_t calls = 10000, posts = 100000, args_bytes = 64;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--calls") == 0) calls = std::stoul(argv[i + 1]);
        else if (std::strcmp(argv[i], "--posts") == 0) posts = std::stoul(argv[i + 1]);
        else if (std::strcmp(argv[i], "--args-bytes") == 0) args_bytes = std::stoul(argv[i + 1]);
    }
    if (calls == 0) calls = 1;

    auto listener = TcpListener::Bind("127.0.0.1", 0);
    if (!listener) {
        std::fprintf(stderr, "bind failed\n");
        return 1;
    }
    std::thread server([&] {
        auto stream = listener->Accept();
        if (!stream) return;
        TcpTransport transport(std::move(*stream));
        if (!ExchangeHello(transport, MakeHello(), seconds(5))) return;
        RpcServer rpc(transport, [](const RpcRequest& request) {
            RpcResponse response;
            response.set_out_args(request.args());
            return response;
        });
        rpc.Serve();
    });

    auto stream = TcpStream::Connect("127.0.0.1", listener->port());
    if (!stream) {
        std::fprintf(stderr, "connect failed\n");
        return 1;
    }
    TcpTransport transport(std::move(*stream));
    if (!ExchangeHello(transport, MakeHello(), seconds(5))) {
        std::fprintf(stderr, "handshake failed\n");
        return 1;
    }

    int exit_code = 0;
    {
        RpcClient rpc(transport);
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

        std::printf("tuna_rpc_bench (loopback TCP, %zu-byte args)\n", args_bytes);
        std::printf("  sync calls     : %zu in %.3f s -> %.0f calls/s\n", rtt_us.size(), sync_s,
                    static_cast<double>(rtt_us.size()) / sync_s);
        std::printf("  RTT p50 / p99  : %.1f us / %.1f us (min %.1f, max %.1f)\n", Percentile(rtt_us, 0.50),
                    Percentile(rtt_us, 0.99), rtt_us.front(), rtt_us.back());
        std::printf("  batched posts  : %zu + 1 fence in %.3f s -> %.0f calls/s\n", posts, async_s,
                    static_cast<double>(posts + 1) / async_s);
    }
    transport.Close();
    server.join();
    return exit_code;
}

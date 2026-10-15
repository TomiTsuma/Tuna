// Track B connectivity check: handshake with a tuna_server --foreground and
// time echo RPCs. Usage: tuna_ping [--host H] --port P [--count N]
// Exit code 0 only if the handshake and every echo succeed.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tuna/tunnel/rpc.h"
#include "tuna/tunnel/session.h"
#include "tuna/tunnel/tcp_socket.h"
#include "tuna/tunnel/transport.h"

using namespace std::chrono;
using namespace tuna;
using namespace tuna::tunnel;

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    unsigned long port = 0, count = 5;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--host") == 0) host = argv[i + 1];
        else if (std::strcmp(argv[i], "--port") == 0) port = std::stoul(argv[i + 1]);
        else if (std::strcmp(argv[i], "--count") == 0) count = std::stoul(argv[i + 1]);
    }
    if (port == 0 || port > 65535) {
        std::fprintf(stderr, "usage: tuna_ping [--host H] --port P [--count N]\n");
        return 2;
    }

    std::string error;
    auto stream = TcpStream::Connect(host, static_cast<uint16_t>(port), &error);
    if (!stream) {
        std::fprintf(stderr, "connect failed: %s\n", error.c_str());
        return 1;
    }
    TcpTransport transport(std::move(*stream));

    Hello local;
    local.set_protocol_version(kProtocolVersion);
#ifdef _WIN32
    local.set_client_os("windows");
#else
    local.set_client_os("posix");
#endif
    auto peer = ExchangeHello(transport, local, seconds(5), &error);
    if (!peer) {
        std::fprintf(stderr, "handshake failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("connected to %s:%lu (protocol v%u, driver %u, %d device(s))\n", host.c_str(), port,
                peer->protocol_version(), peer->cuda_driver_version(), peer->devices_size());

    int exit_code = 0;
    {
        RpcClient rpc(transport);
        std::vector<double> rtt_us;
        for (unsigned long i = 0; i < count; ++i) {
            const std::string payload = "ping " + std::to_string(i);
            const auto t0 = steady_clock::now();
            const CallResult r = rpc.Call(kApiEcho, payload);
            const double us = duration<double, std::micro>(steady_clock::now() - t0).count();
            if (r.result != kResultSuccess || r.out_args != payload) {
                std::fprintf(stderr, "echo %lu failed (result %d)\n", i, r.result);
                exit_code = 1;
                break;
            }
            rtt_us.push_back(us);
        }
        if (!rtt_us.empty()) {
            double sum = 0;
            for (double v : rtt_us) sum += v;
            std::printf("%zu echo(s): min %.1f us, avg %.1f us, max %.1f us\n", rtt_us.size(),
                        *std::min_element(rtt_us.begin(), rtt_us.end()), sum / static_cast<double>(rtt_us.size()),
                        *std::max_element(rtt_us.begin(), rtt_us.end()));
        }
    }
    transport.Close();
    return exit_code;
}

// Track B connectivity check: connects to a tuna_server --foreground listener,
// handshakes, and times echo RPCs (optionally over mTLS and with a bulk
// payload). Exit code 0 only if the handshake and every echo succeed.
//
//   tuna_ping [--host H] --port P [--count N]
//             [--ca CA.crt --cert ME.crt --key ME.key [--server-name NAME]]
//             [--bulk-mb N] [--no-bulk-socket] [--hold-ms MS] [--connect-deadline-ms MS]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "tuna/tunnel/client.h"
#include "tuna/tunnel/rpc.h"

using namespace std::chrono;
using namespace tuna;
using namespace tuna::tunnel;

int main(int argc, char** argv) {
    ClientOptions options;
    TlsConfig tls;
    unsigned long count = 5, bulk_mb = 0, hold_ms = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (arg == "--host") options.host = value();
        else if (arg == "--port") options.port = static_cast<uint16_t>(std::stoul(value()));
        else if (arg == "--count") count = std::stoul(value());
        else if (arg == "--ca") tls.ca_file = value();
        else if (arg == "--cert") tls.cert_file = value();
        else if (arg == "--key") tls.key_file = value();
        else if (arg == "--server-name") options.server_name = value();
        else if (arg == "--bulk-mb") bulk_mb = std::stoul(value());
        else if (arg == "--no-bulk-socket") options.open_bulk_socket = false;
        else if (arg == "--hold-ms") hold_ms = std::stoul(value());
        else if (arg == "--connect-deadline-ms") options.connect_deadline = milliseconds(std::stoul(value()));
    }
    if (options.port == 0) {
        std::fprintf(stderr, "usage: tuna_ping [--host H] --port P [--count N] [--ca F --cert F --key F]\n");
        return 2;
    }
    if (!tls.ca_file.empty()) options.tls = tls;

    std::string error;
    std::optional<ClientSession> session = ConnectSession(options, &error);
    if (!session) {
        std::fprintf(stderr, "connect failed: %s\n", error.c_str());
        return 1;
    }
    const Hello& hello = session->server_hello;
    std::printf("connected to %s:%u (%s, protocol v%u, driver %u, %d device(s), worker %u, bulk socket %s)\n",
                options.host.c_str(), static_cast<unsigned>(options.port), options.tls ? "mTLS" : "plaintext",
                hello.protocol_version(), hello.cuda_driver_version(), hello.devices_size(), hello.worker_pid(),
                session->bulk ? "yes" : "no");
    std::fflush(stdout);

    int exit_code = 0;
    {
        RpcClient rpc(*session->transport, {}, session->bulk.get());
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
        if (exit_code == 0 && bulk_mb > 0) {
            std::string data(bulk_mb * 1024 * 1024, '\0');
            for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<char>(i * 7 + (i >> 13));
            const auto t0 = steady_clock::now();
            const CallResult r = rpc.Call(kApiEcho, "bulk", false, data);
            const double s = duration<double>(steady_clock::now() - t0).count();
            if (r.result != kResultSuccess || r.bulk_out != data) {
                std::fprintf(stderr, "bulk echo of %lu MiB failed (result %d)\n", bulk_mb, r.result);
                exit_code = 1;
            } else {
                std::printf("bulk echo %lu MiB: %.1f ms, %.0f MiB/s each way\n", bulk_mb, s * 1000,
                            static_cast<double>(bulk_mb) / s);
            }
        }
        if (exit_code == 0 && hold_ms > 0) {
            // Keep the session open (for tests that act on its worker), then
            // check it is still usable.
            std::this_thread::sleep_for(milliseconds(hold_ms));
            const CallResult r = rpc.Call(kApiEcho, "after hold");
            if (r.result != kResultSuccess) {
                std::fprintf(stderr, "session lost during hold (result %d)\n", r.result);
                exit_code = 1;
            }
        }
    }
    session->transport->Close();
    if (session->bulk) session->bulk->Close();
    return exit_code;
}

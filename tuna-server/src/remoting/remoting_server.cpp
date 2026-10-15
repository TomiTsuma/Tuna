#include "remoting/remoting_server.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <list>
#include <string>
#include <thread>

#include "tuna/tunnel/rpc.h"
#include "tuna/tunnel/session.h"
#include "tuna/tunnel/tcp_socket.h"
#include "tuna/tunnel/transport.h"

namespace tuna {

namespace {

constexpr int32_t kCudaErrorNotSupported = 801;
constexpr auto kHandshakeTimeout = std::chrono::seconds(5);

std::atomic<tunnel::TcpListener*> g_listener{nullptr};

Hello ServerHello() {
    Hello hello;
    hello.set_protocol_version(tunnel::kProtocolVersion);
    // cuda_driver_version and devices are filled from the real driver in Phase 3.
    return hello;
}

RpcResponse Dispatch(const RpcRequest& request) {
    RpcResponse response;
    if (request.api_id() == tunnel::kApiEcho) {
        response.set_out_args(request.args());
    } else {
        response.set_result(kCudaErrorNotSupported);
    }
    return response;
}

struct Session {
    uint64_t id = 0;
    std::shared_ptr<tunnel::TcpTransport> transport;
    std::shared_ptr<std::atomic<bool>> finished = std::make_shared<std::atomic<bool>>(false);
    std::thread thread;
};

void RunSession(uint64_t id, tunnel::Transport& transport, const std::shared_ptr<spdlog::logger>& log) {
    std::string error;
    auto peer = tunnel::ExchangeHello(transport, ServerHello(), kHandshakeTimeout, &error);
    if (!peer) {
        log->warn("session {}: handshake failed: {}", id, error);
        return;
    }
    log->info("session {}: connected (client_os='{}')", id, peer->client_os());
    tunnel::RpcServer server(transport, Dispatch);
    const uint64_t calls = server.Serve();
    log->info("session {}: ended after {} calls", id, calls);
}

}  // namespace

void RequestRemotingShutdown() {
    if (tunnel::TcpListener* listener = g_listener.load()) {
        listener->Shutdown();
    }
}

int ServeRemoting(const ServerConfig& config, const std::shared_ptr<spdlog::logger>& log) {
    if (!tunnel::IsLoopbackHost(config.listen_address)) {
        log->error("refusing to listen on non-loopback address '{}': the TLS transport "
                   "(Phase 2b) is not implemented, and plaintext GPU RPC must not be "
                   "exposed to a network", config.listen_address);
        std::fprintf(stderr, "Refusing non-loopback listen_address '%s' until TLS exists (Phase 2b)\n",
                     config.listen_address.c_str());
        return 1;
    }

    std::string error;
    auto listener = tunnel::TcpListener::Bind(config.listen_address, config.listen_port, &error);
    if (!listener) {
        log->error("cannot listen: {}", error);
        std::fprintf(stderr, "Cannot listen: %s\n", error.c_str());
        return 1;
    }
    g_listener = &*listener;
    log->info("remoting listener on {}:{}", config.listen_address, listener->port());
    // Machine-readable for scripts and tests (listen_port 0 picks a free port).
    std::printf("listening on %s:%u\n", config.listen_address.c_str(), static_cast<unsigned>(listener->port()));
    std::fflush(stdout);

    std::list<Session> sessions;
    uint64_t next_id = 1;
    while (auto stream = listener->Accept()) {
        // Reap sessions that already ended so a long-lived server does not
        // accumulate finished threads.
        for (auto it = sessions.begin(); it != sessions.end();) {
            if (*it->finished) {
                it->thread.join();
                it = sessions.erase(it);
            } else {
                ++it;
            }
        }

        Session& session = sessions.emplace_back();
        session.id = next_id++;
        session.transport = std::make_shared<tunnel::TcpTransport>(std::move(*stream));
        session.thread = std::thread([id = session.id, transport = session.transport,
                                      finished = session.finished, log] {
            RunSession(id, *transport, log);
            *finished = true;
        });
    }

    g_listener = nullptr;
    log->info("shutting down: closing {} session(s)", sessions.size());
    for (Session& session : sessions) session.transport->Close();
    for (Session& session : sessions) session.thread.join();
    return 0;
}

}  // namespace tuna

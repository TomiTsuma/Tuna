#include "remoting/worker_session.h"

#include <cstdio>
#include <filesystem>
#include <thread>

#include "core/config_manager.h"
#include "core/logger.h"
#include "remoting/dispatch.h"
#include "remoting/worker_process.h"
#include "tuna/tunnel/session.h"

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace tuna {

namespace {

constexpr auto kHandshakeTimeout = std::chrono::seconds(5);

}  // namespace

WorkerSession::WorkerSession(uint64_t id, std::shared_ptr<tunnel::TlsContext> tls, tunnel::SessionToken token,
                             std::shared_ptr<spdlog::logger> log, ServerCudaConfig cuda)
    : id_(id), tls_(std::move(tls)), token_(token), log_(std::move(log)), cuda_(std::move(cuda)) {}

std::unique_ptr<tunnel::Transport> WorkerSession::Secure(tunnel::TcpStream stream, std::string* identity,
                                                         std::string* error) {
    if (!tls_) return std::make_unique<tunnel::TcpTransport>(std::move(stream));
    auto transport = tunnel::TlsHandshake(std::move(stream), tls_, "", kHandshakeTimeout, error);
    if (transport) *identity = transport->peer_identity();
    return transport;
}

void WorkerSession::RunMain(tunnel::TcpStream stream) {
    std::string identity;
    std::string error;
    std::unique_ptr<tunnel::Transport> transport = Secure(std::move(stream), &identity, &error);
    if (!transport) {
        log_->warn("session {}: TLS handshake failed: {}", id_, error);
        EndSession();
        return;
    }

    Hello hello = ServerHello();
    hello.set_session_token(std::string(reinterpret_cast<const char*>(token_.data()), token_.size()));
    hello.set_bulk_socket_available(true);
    hello.set_worker_pid(static_cast<uint32_t>(getpid()));
    auto peer = tunnel::ExchangeHello(*transport, hello, kHandshakeTimeout, &error);
    if (!peer) {
        log_->warn("session {}: handshake failed: {}", id_, error);
        EndSession();
        return;
    }
    log_->info("session {}: connected (identity='{}', client_os='{}')", id_,
               identity.empty() ? "<plaintext>" : identity, peer->client_os());

    tunnel::RpcServer server(*transport, MakeDispatcher(log_, cuda_));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            ended_ = true;
            serving_cv_.notify_all();
            return;
        }
        main_ = transport.get();
        server_ = &server;
        identity_ = identity;
    }
    serving_cv_.notify_all();
    const uint64_t calls = server.Serve();
    EndSession();  // after this, AttachBulk() can no longer reach `server`
    log_->info("session {}: ended after {} calls", id_, calls);
}

void WorkerSession::AttachBulk(tunnel::TcpStream stream) {
    std::string identity;
    std::string error;
    std::unique_ptr<tunnel::Transport> bulk = Secure(std::move(stream), &identity, &error);
    if (!bulk) {
        log_->warn("session {}: bulk socket TLS handshake failed: {}", id_, error);
        return;
    }
    {
        std::unique_lock<std::mutex> lock(mutex_);
        // The client opens its bulk socket as soon as it has the server's
        // Hello, which can be before RunMain starts serving; wait for that.
        serving_cv_.wait_for(lock, kHandshakeTimeout, [this] { return server_ || ended_; });
        // The session token routed this socket here; the identity check is
        // what authorizes it. An eavesdropper who saw the token cannot pass.
        if (!server_ || identity != identity_ || bulk_) {
            log_->warn("session {}: refusing bulk socket (identity '{}', session identity '{}')", id_, identity,
                       identity_);
            return;
        }
    }
    Hello hello;
    hello.set_protocol_version(tunnel::kProtocolVersion);
    if (!tunnel::ExchangeHello(*bulk, hello, kHandshakeTimeout, &error)) {
        log_->warn("session {}: bulk socket handshake failed: {}", id_, error);
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!server_ || bulk_) return;  // the session ended meanwhile
    bulk_ = std::move(bulk);
    server_->AttachBulk(bulk_.get());
    log_->info("session {}: bulk socket attached", id_);
}

void WorkerSession::EndSession() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        main_ = nullptr;
        server_ = nullptr;
        ended_ = true;
    }
    serving_cv_.notify_all();
}

void WorkerSession::Stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    if (main_) main_->Close();
}

int RunWorker(const std::string& config_path, const std::string& ack_handle) {
    auto config = ConfigManager::LoadFromFile(config_path);
    if (!config) return 1;

    const auto log_dir = (std::filesystem::temp_directory_path() / "tuna-server-logs").string();
    if (!Logger::Init(log_dir, "tuna-worker-" + std::to_string(getpid()))) return 1;
    auto log = Logger::Get();
    const auto level = spdlog::level::from_str(config->log_level);
    if (level != spdlog::level::off || config->log_level == "off") log->set_level(level);

    std::string error;
    auto channel = WorkerChannel::FromInherited(ack_handle, &error);
    if (!channel) {
        log->error("worker: {}", error);
        std::fprintf(stderr, "tuna_server --worker: %s\n", error.c_str());
        return 1;
    }

    std::shared_ptr<tunnel::TlsContext> tls;
    if (config->tls) {
        tls = tunnel::TlsContext::Create(tunnel::TlsRole::kServer,
                                         {config->tls->ca_cert, config->tls->cert, config->tls->key,
                                          config->tls->allowed_client_names},
                                         &error);
        if (!tls) {
            log->error("worker: TLS configuration: {}", error);
            return 1;
        }
    }

    auto first = channel->Receive();
    if (!first || first->first.kind != static_cast<uint32_t>(HandoffKind::kSession)) {
        log->error("worker: no session socket received");
        return 1;
    }
    WorkerSession session(getpid(), tls, first->first.token, log, config->cuda);

    // Later handoffs are bulk sockets. EOF means the listener stopped us or
    // died: end the session, and if it cannot end (a thread stuck inside a
    // driver call), exit anyway so no orphaned worker holds the GPU.
    std::thread ipc([&] {
        while (auto next = channel->Receive()) {
            if (next->first.kind == static_cast<uint32_t>(HandoffKind::kBulk)) {
                session.AttachBulk(std::move(next->second));
            }
        }
        session.Stop();
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::_Exit(3);
    });

    session.RunMain(std::move(first->second));
    log->flush();
    Logger::Shutdown();
    // The IPC thread may be blocked reading the channel; the session is over,
    // so exit without waiting for it.
    ipc.detach();
    std::fflush(nullptr);
    std::_Exit(0);
}

}  // namespace tuna

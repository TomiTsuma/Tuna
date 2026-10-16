#include "remoting/remoting_server.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <openssl/rand.h>

#include "remoting/worker_process.h"
#include "remoting/worker_session.h"
#include "tuna/tunnel/client.h"
#include "tuna/tunnel/tcp_socket.h"
#include "tuna/tunnel/tls.h"

namespace tuna {

namespace {

constexpr auto kPreambleTimeout = std::chrono::seconds(5);
constexpr auto kHandoffTimeout = std::chrono::seconds(5);
constexpr auto kStopTimeout = std::chrono::seconds(5);
// Connections still sending their preamble. Bounds the threads a flood of
// idle connections can tie up; beyond it new connections are dropped.
constexpr int kMaxPendingConnections = 64;

std::atomic<tunnel::TcpListener*> g_listener{nullptr};

// Prints a machine-readable line (scripts and tests parse stdout).
void Announce(const std::string& line) {
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}

// A live session as the listener sees it.
class SessionHost {
public:
    virtual ~SessionHost() = default;
    virtual bool Start(tunnel::TcpStream stream) = 0;
    virtual void AttachBulk(tunnel::TcpStream stream) = 0;
    virtual bool Finished() = 0;
    virtual void RequestStop() = 0;
    virtual void Join() = 0;
};

// worker_mode "process": the session runs in its own `tuna_server --worker`.
class ProcessSessionHost final : public SessionHost {
public:
    ProcessSessionHost(uint64_t id, tunnel::SessionToken token, std::string executable, std::string config_path,
                       std::shared_ptr<spdlog::logger> log)
        : id_(id), token_(token), executable_(std::move(executable)), config_path_(std::move(config_path)),
          log_(std::move(log)) {}

    bool Start(tunnel::TcpStream stream) override {
        std::string error;
        auto worker = WorkerProcess::Spawn(executable_, {"--worker", "--config", config_path_}, &error);
        if (!worker) {
            log_->error("session {}: cannot start worker: {}", id_, error);
            return false;
        }
        const uint32_t pid = worker->pid();
        std::lock_guard<std::mutex> lock(mutex_);
        worker_ = std::move(worker);
        if (!worker_->HandOff({static_cast<uint32_t>(HandoffKind::kSession), token_}, std::move(stream),
                              kHandoffTimeout)) {
            log_->error("session {}: worker {} did not accept the session socket", id_, pid);
            return false;
        }
        log_->info("session {}: worker {} started", id_, pid);
        Announce("session " + std::to_string(id_) + " worker " + std::to_string(pid));
        return true;
    }

    void AttachBulk(tunnel::TcpStream stream) override {
        // One handoff at a time: two interleaved messages would corrupt the channel.
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_ && !worker_->HandOff({static_cast<uint32_t>(HandoffKind::kBulk), token_}, std::move(stream),
                                         kHandoffTimeout)) {
            log_->warn("session {}: worker did not accept the bulk socket", id_);
        }
    }

    bool Finished() override {
        // Called with the listener's session table locked; a host that is
        // still starting (holding mutex_ during the handoff) is not finished,
        // and waiting for it would stall every other connection.
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return false;
        int code = 0;
        if (!worker_ || !worker_->Exited(&code)) return false;
        if (!reported_) {
            reported_ = true;
            log_->info("session {}: worker {} exited with code {}", id_, worker_->pid(), code);
        }
        return true;
    }

    void RequestStop() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_) worker_->CloseChannel();
    }

    void Join() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_) worker_->StopAndWait(kStopTimeout);
    }

private:
    const uint64_t id_;
    const tunnel::SessionToken token_;
    const std::string executable_;
    const std::string config_path_;
    const std::shared_ptr<spdlog::logger> log_;
    std::mutex mutex_;
    std::unique_ptr<WorkerProcess> worker_;  // guarded by mutex_
    bool reported_ = false;                  // guarded by mutex_
};

// worker_mode "thread": the session runs on listener threads (debugging only).
class ThreadSessionHost final : public SessionHost {
public:
    ThreadSessionHost(uint64_t id, std::shared_ptr<tunnel::TlsContext> tls, tunnel::SessionToken token,
                      std::shared_ptr<spdlog::logger> log)
        : session_(id, std::move(tls), token, std::move(log)) {}
    ~ThreadSessionHost() override { Join(); }

    bool Start(tunnel::TcpStream stream) override {
        main_ = std::thread([this, s = std::move(stream)]() mutable {
            session_.RunMain(std::move(s));
            finished_ = true;
        });
        return true;
    }
    void AttachBulk(tunnel::TcpStream stream) override {
        std::lock_guard<std::mutex> lock(mutex_);
        bulk_threads_.emplace_back([this, s = std::move(stream)]() mutable { session_.AttachBulk(std::move(s)); });
    }
    bool Finished() override { return finished_; }
    void RequestStop() override { session_.Stop(); }
    void Join() override {
        session_.Stop();
        if (main_.joinable()) main_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::thread& t : bulk_threads_) t.join();
        bulk_threads_.clear();
    }

private:
    WorkerSession session_;
    std::atomic<bool> finished_{false};
    std::thread main_;
    std::mutex mutex_;
    std::vector<std::thread> bulk_threads_;  // guarded by mutex_
};

class Listener {
public:
    Listener(const ServerConfig& config, std::string config_path, std::shared_ptr<tunnel::TlsContext> tls,
             std::shared_ptr<spdlog::logger> log)
        : config_(config), config_path_(std::move(config_path)), tls_(std::move(tls)), log_(std::move(log)),
          executable_(CurrentExecutablePath()) {}

    // Runs one accepted connection's routing on its own short-lived thread.
    void Dispatch(tunnel::TcpStream stream) {
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            if (pending_ >= kMaxPendingConnections) {
                log_->warn("dropping connection: {} connections already pending", pending_);
                return;
            }
            ++pending_;
        }
        std::thread([this, s = std::move(stream)]() mutable {
            Route(std::move(s));
            std::lock_guard<std::mutex> lock(pending_mutex_);
            --pending_;
            pending_cv_.notify_all();
        }).detach();
    }

    void Shutdown() {
        {
            // Routing threads finish within the preamble + handoff timeouts.
            std::unique_lock<std::mutex> lock(pending_mutex_);
            pending_cv_.wait(lock, [this] { return pending_ == 0; });
        }
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        log_->info("shutting down: stopping {} session(s)", sessions_.size());
        for (auto& [token, host] : sessions_) host->RequestStop();  // all at once, then wait
        for (auto& [token, host] : sessions_) host->Join();
        sessions_.clear();
    }

private:
    void Route(tunnel::TcpStream stream) {
        std::string bytes(tunnel::kPreambleBytes, '\0');
        if (!stream.ReceiveExact(bytes.data(), bytes.size(), kPreambleTimeout)) {
            log_->debug("dropping connection: no preamble within {} s", kPreambleTimeout.count());
            return;
        }
        const std::optional<tunnel::Preamble> preamble = tunnel::DecodePreamble(bytes);
        if (!preamble) {
            log_->warn("dropping connection: malformed preamble (not a Tuna client?)");
            return;
        }
        if (preamble->kind == tunnel::PreambleKind::kAttachBulk) {
            std::shared_ptr<SessionHost> host;
            {
                std::lock_guard<std::mutex> lock(sessions_mutex_);
                auto it = sessions_.find(Key(preamble->token));
                if (it != sessions_.end()) host = it->second;
            }
            if (!host) {
                log_->warn("dropping bulk connection: unknown session token");
                return;
            }
            host->AttachBulk(std::move(stream));
            return;
        }
        StartSession(std::move(stream));
    }

    void StartSession(tunnel::TcpStream stream) {
        tunnel::SessionToken token;
        if (RAND_bytes(token.data(), static_cast<int>(token.size())) != 1) {
            log_->error("dropping connection: no randomness for a session token");
            return;
        }
        std::shared_ptr<SessionHost> host;
        uint64_t id;
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            for (auto it = sessions_.begin(); it != sessions_.end();) {
                it = it->second->Finished() ? sessions_.erase(it) : std::next(it);
            }
            if (sessions_.size() >= config_.max_sessions) {
                log_->warn("dropping connection: max_sessions ({}) reached", config_.max_sessions);
                return;
            }
            id = next_id_++;
            if (config_.worker_mode == "thread") {
                host = std::make_shared<ThreadSessionHost>(id, tls_, token, log_);
            } else {
                host = std::make_shared<ProcessSessionHost>(id, token, executable_, config_path_, log_);
            }
            sessions_.emplace(Key(token), host);  // reserves the slot while the worker starts
        }
        if (!host->Start(std::move(stream))) {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            sessions_.erase(Key(token));
        }
    }

    static std::string Key(const tunnel::SessionToken& token) {
        return std::string(reinterpret_cast<const char*>(token.data()), token.size());
    }

    const ServerConfig config_;
    const std::string config_path_;
    const std::shared_ptr<tunnel::TlsContext> tls_;
    const std::shared_ptr<spdlog::logger> log_;
    const std::string executable_;

    std::mutex pending_mutex_;
    std::condition_variable pending_cv_;
    int pending_ = 0;  // guarded by pending_mutex_

    std::mutex sessions_mutex_;
    std::map<std::string, std::shared_ptr<SessionHost>> sessions_;  // guarded by sessions_mutex_
    uint64_t next_id_ = 1;                                         // guarded by sessions_mutex_
};

}  // namespace

void RequestRemotingShutdown() {
    if (tunnel::TcpListener* listener = g_listener.load()) {
        listener->Shutdown();
    }
}

int ServeRemoting(const ServerConfig& config, const std::string& config_path,
                  const std::shared_ptr<spdlog::logger>& log) {
    if (!config.tls && !tunnel::IsLoopbackHost(config.listen_address)) {
        log->error("refusing to listen on non-loopback address '{}' without TLS: plaintext GPU RPC "
                   "must not be exposed to a network", config.listen_address);
        std::fprintf(stderr, "Refusing non-loopback listen_address '%s' without a \"tls\" config section\n",
                     config.listen_address.c_str());
        return 1;
    }

    std::string error;
    std::shared_ptr<tunnel::TlsContext> tls;
    if (config.tls) {
        // Validated here even in process mode, so a bad path fails at startup
        // rather than in every worker.
        tls = tunnel::TlsContext::Create(tunnel::TlsRole::kServer,
                                         {config.tls->ca_cert, config.tls->cert, config.tls->key,
                                          config.tls->allowed_client_names},
                                         &error);
        if (!tls) {
            log->error("TLS configuration: {}", error);
            std::fprintf(stderr, "TLS configuration error: %s\n", error.c_str());
            return 1;
        }
    }

    auto listener = tunnel::TcpListener::Bind(config.listen_address, config.listen_port, &error);
    if (!listener) {
        log->error("cannot listen: {}", error);
        std::fprintf(stderr, "Cannot listen: %s\n", error.c_str());
        return 1;
    }
    g_listener = &*listener;
    log->info("remoting listener on {}:{} ({}, {} workers, max_sessions {})", config.listen_address,
              listener->port(), tls ? "mTLS" : "plaintext loopback", config.worker_mode, config.max_sessions);
    // Machine-readable for scripts and tests (listen_port 0 picks a free port).
    Announce("listening on " + config.listen_address + ":" + std::to_string(listener->port()));

    std::error_code ec;
    const std::string absolute_config = std::filesystem::absolute(config_path, ec).string();
    Listener sessions(config, absolute_config, tls, log);
    while (auto stream = listener->Accept()) {
        sessions.Dispatch(std::move(*stream));
    }
    g_listener = nullptr;
    sessions.Shutdown();
    return 0;
}

}  // namespace tuna

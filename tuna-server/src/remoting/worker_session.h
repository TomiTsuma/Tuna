#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>

#include <spdlog/spdlog.h>

#include "core/config_manager.h"
#include "tuna/tunnel/client.h"
#include "tuna/tunnel/rpc.h"
#include "tuna/tunnel/tcp_socket.h"
#include "tuna/tunnel/tls.h"

namespace tuna {

// One client session: everything that happens after the listener has routed
// a connection. Runs inside a worker process (worker_mode "process") or on a
// listener thread (worker_mode "thread"). The listener never parses anything
// past the preamble; TLS, mTLS, Hello and every RPC are handled here.
class WorkerSession {
public:
    // `tls` is null for a plaintext (loopback-only) server.
    WorkerSession(uint64_t id, std::shared_ptr<tunnel::TlsContext> tls, tunnel::SessionToken token,
                  std::shared_ptr<spdlog::logger> log, ServerCudaConfig cuda = {});

    // Handshakes on the session socket and serves RPCs until the client
    // disconnects, the session is stopped, or the peer violates the protocol.
    void RunMain(tunnel::TcpStream stream);

    // Adds a bulk socket. It must pass mTLS as the same client identity as
    // the session socket; anything else is closed. Blocks for its handshake.
    void AttachBulk(tunnel::TcpStream stream);

    // Ends the session from another thread (listener shutdown).
    void Stop();

private:
    void EndSession();
    std::unique_ptr<tunnel::Transport> Secure(tunnel::TcpStream stream, std::string* identity, std::string* error);

    const uint64_t id_;
    const std::shared_ptr<tunnel::TlsContext> tls_;
    const tunnel::SessionToken token_;
    const std::shared_ptr<spdlog::logger> log_;
    const ServerCudaConfig cuda_;

    std::mutex mutex_;
    std::condition_variable serving_cv_;  // signalled when server_ is set or the session ends
    bool ended_ = false;                  // guarded by mutex_
    tunnel::Transport* main_ = nullptr;   // guarded by mutex_; set while RunMain serves
    tunnel::RpcServer* server_ = nullptr;  // guarded by mutex_
    std::string identity_;                 // guarded by mutex_
    std::unique_ptr<tunnel::Transport> bulk_;  // guarded by mutex_; outlives server_
    bool stopped_ = false;                 // guarded by mutex_
};

// Process entry point for `tuna_server --worker`: adopts the IPC channel,
// receives the session socket and later bulk sockets, and exits when the
// session ends or the listener closes the channel.
int RunWorker(const std::string& config_path, const std::string& ack_handle);

}  // namespace tuna

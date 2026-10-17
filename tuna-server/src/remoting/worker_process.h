#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tuna/tunnel/client.h"
#include "tuna/tunnel/tcp_socket.h"

namespace tuna {

// Listener <-> worker IPC (Phase 2c). One channel per worker, open for the
// worker's whole life:
//   POSIX:   one end of a socketpair on the worker's fd 3; each handoff is one
//            message carrying the socket via SCM_RIGHTS.
//   Windows: the worker's stdin pipe carries the WSAPROTOCOL_INFOW from
//            WSADuplicateSocketW; a second pipe (handle passed on the command
//            line) carries the worker's acknowledgement.
// The worker acknowledges every handoff with one byte. Closing the channel
// (EOF) tells the worker to end its session and exit.

enum class HandoffKind : uint32_t { kSession = 1, kBulk = 2 };

struct HandoffHeader {
    uint32_t kind = 0;               // HandoffKind
    tunnel::SessionToken token{};
};

// Listener side: one spawned worker process.
class WorkerProcess {
public:
    // Starts `executable` with `args` and the IPC channel attached. Workers
    // never outlive a crashed listener: the kernel closes the listener's end
    // of the channel, the worker sees EOF and exits (RunWorker), and on
    // Windows the worker is also in a kill-on-close Job object.
    static std::unique_ptr<WorkerProcess> Spawn(const std::string& executable,
                                                const std::vector<std::string>& args, std::string* error);
    ~WorkerProcess();
    WorkerProcess(const WorkerProcess&) = delete;
    WorkerProcess& operator=(const WorkerProcess&) = delete;

    uint32_t pid() const { return pid_; }

    // Transfers `stream` to the worker and waits for its acknowledgement.
    // The stream is closed in this process either way.
    bool HandOff(const HandoffHeader& header, tunnel::TcpStream stream, std::chrono::milliseconds timeout);

    // Non-blocking: true once the worker has exited (exit code in *code).
    bool Exited(int* code = nullptr);
    // Asks the worker to stop by closing the channel.
    void CloseChannel();
    // Waits up to `timeout` for exit, then kills the worker.
    int StopAndWait(std::chrono::milliseconds timeout);

private:
    WorkerProcess() = default;
    bool ReadAck(std::chrono::milliseconds timeout);

    uint32_t pid_ = 0;
    std::optional<int> exit_code_;
#ifdef _WIN32
    void* process_ = nullptr;   // HANDLE
    void* to_worker_ = nullptr; // HANDLE: write end of the worker's stdin
    void* from_worker_ = nullptr;  // HANDLE: read end of the ack pipe
#else
    int channel_ = -1;
#endif
};

// Worker side: the inherited end of the channel.
class WorkerChannel {
public:
    // `ack_handle` is the --ack-handle argument (Windows only; ignored elsewhere).
    static std::unique_ptr<WorkerChannel> FromInherited(const std::string& ack_handle, std::string* error);
    ~WorkerChannel();

    // Blocks for the next handoff and acknowledges it. std::nullopt on EOF
    // (the listener asked this worker to stop) or a broken channel.
    std::optional<std::pair<HandoffHeader, tunnel::TcpStream>> Receive();

private:
    WorkerChannel() = default;
#ifdef _WIN32
    void* from_listener_ = nullptr;
    void* to_listener_ = nullptr;
#else
    int channel_ = -1;
#endif
};

// Absolute path of the running executable (workers are the same binary).
std::string CurrentExecutablePath();

}  // namespace tuna

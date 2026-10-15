#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "tuna/tunnel/transport.h"

namespace tuna::tunnel {

constexpr uint32_t kProtocolVersion = 1;

// Sends `local` and waits for the peer's Hello (both sides call this; it is
// symmetric). Fails, and closes the transport, if the peer sends anything
// else first, speaks another protocol_version, or is silent past `timeout`.
std::optional<Hello> ExchangeHello(Transport& transport, const Hello& local,
                                   std::chrono::milliseconds timeout, std::string* error = nullptr);

// Liveness for an established session: sends a HEARTBEAT frame every
// `interval` and closes the transport if no frame of any kind was received for
// `timeout`. The endpoint's reader thread calls NoteActivity() per frame.
class Heartbeat {
public:
    Heartbeat(Transport& transport, std::chrono::milliseconds interval,
              std::chrono::milliseconds timeout);
    ~Heartbeat();
    Heartbeat(const Heartbeat&) = delete;
    Heartbeat& operator=(const Heartbeat&) = delete;

    void NoteActivity();
    // True if the session was closed because the peer went silent.
    bool expired() const { return expired_; }

private:
    void Run();

    Transport& transport_;
    const std::chrono::milliseconds interval_;
    const std::chrono::milliseconds timeout_;
    std::atomic<uint64_t> last_activity_ns_;
    std::atomic<bool> expired_{false};
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_ = false;  // guarded by mutex_
    std::thread thread_;
};

}  // namespace tuna::tunnel

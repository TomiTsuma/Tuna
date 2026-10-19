// Client-side core of the replacement libcudart: the session (connected
// lazily from environment variables), call/post plumbing, and the per-thread
// last-error state the runtime API exposes through cudaGetLastError.
//
// The exported runtime functions live in client_runtime.cpp (hand-written),
// client_launch.cpp (kernel registration and launch), cuda_client_stubs.inc
// (generated) and client_unsupported.cpp (generated not-supported stubs).
//
// A test seam (SetCallHook) routes calls to an in-process dispatcher with a
// mock backend, so the marshalling is tested with no GPU and no transport.

#include "tuna/cuda/client.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <vector>

#include "cuda_api_ids.h"
#include "tuna/cuda/bridge.h"
#include "tuna/tunnel/client.h"

namespace tuna::cudart {

namespace {

#include "cuda_coverage.inc"

// TUNA_STATS=1: per-api counts of blocking and posted calls, and the time
// blocking calls waited, printed at exit. Every blocking call costs a round
// trip, so this is where a workload's remoting overhead comes from.
struct ApiStats {
    std::atomic<uint64_t> blocking{0};
    std::atomic<uint64_t> posted{0};
    std::atomic<uint64_t> wait_ns{0};
};
std::array<ApiStats, kLastCudaApi + 1> g_stats;

bool StatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("TUNA_STATS");
        return value && *value;
    }();
    return enabled;
}

void PrintStats() {
    struct Row {
        const char* name;
        uint64_t blocking, posted, wait_ns;
    };
    std::vector<Row> rows;
    uint64_t blocking = 0, posted = 0, wait_ns = 0;
    for (const auto& entry : kCudaApiNames) {
        const ApiStats& s = g_stats[entry.id];
        Row row{entry.name, s.blocking.load(), s.posted.load(), s.wait_ns.load()};
        if (row.blocking + row.posted == 0) continue;
        blocking += row.blocking;
        posted += row.posted;
        wait_ns += row.wait_ns;
        rows.push_back(row);
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.wait_ns > b.wait_ns; });
    std::fprintf(stderr, "tuna: call statistics (blocking calls wait for a round trip; posted calls are batched)\n");
    std::fprintf(stderr, "tuna:   %-44s %10s %10s %10s\n", "api", "blocking", "posted", "wait ms");
    for (const Row& r : rows) {
        std::fprintf(stderr, "tuna:   %-44s %10llu %10llu %10.1f\n", r.name, static_cast<unsigned long long>(r.blocking),
                     static_cast<unsigned long long>(r.posted), r.wait_ns / 1e6);
    }
    std::fprintf(stderr, "tuna:   %-44s %10llu %10llu %10.1f\n", "total", static_cast<unsigned long long>(blocking),
                 static_cast<unsigned long long>(posted), wait_ns / 1e6);
}

void CountCall(uint32_t api_id, bool posted, uint64_t wait_ns) {
    if (api_id > kLastCudaApi) return;
    static const bool registered = [] { return std::atexit(PrintStats) == 0; }();
    (void)registered;
    ApiStats& s = g_stats[api_id];
    if (posted) {
        ++s.posted;
    } else {
        ++s.blocking;
        s.wait_ns += wait_ns;
    }
}

std::atomic<CallHook> g_hook{nullptr};
std::mutex g_hook_mutex;
std::optional<int32_t> g_hook_latched;  // guarded by g_hook_mutex: emulates RpcClient's latch

std::mutex g_session_mutex;
RpcClient* g_rpc = nullptr;  // guarded by g_session_mutex; never freed (see Connect)
bool g_connect_attempted = false;  // guarded by g_session_mutex
std::string g_connect_error;       // guarded by g_session_mutex

std::atomic<uint64_t> g_generation{1};
thread_local int32_t t_last_error = 0;

const char* Env(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? value : nullptr;
}

// Owns the connection objects for the life of the process. Deliberately
// leaked: CUDA programs call the runtime from static destructors
// (__cudaUnregisterFatBinary, frameworks freeing memory), so the session must
// outlive every one of them. The OS closes the sockets at exit.
struct Session {
    tunnel::ClientSession connection;
    std::unique_ptr<RpcClient> rpc;
};

bool ConnectLocked(std::string* error) {
    const char* server = Env("TUNA_SERVER");
    if (!server) {
        *error = "TUNA_SERVER is not set (host:port of a tuna_server)";
        return false;
    }
    const std::string address = server;
    const size_t colon = address.rfind(':');
    if (colon == std::string::npos || colon == 0) {
        *error = "TUNA_SERVER must be host:port, got '" + address + "'";
        return false;
    }
    tunnel::ClientOptions options;
    options.host = address.substr(0, colon);
    try {
        const int port = std::stoi(address.substr(colon + 1));
        if (port <= 0 || port > 65535) throw std::out_of_range("port");
        options.port = static_cast<uint16_t>(port);
    } catch (const std::exception&) {
        *error = "TUNA_SERVER has an invalid port: '" + address + "'";
        return false;
    }

    const char* ca = Env("TUNA_CA_CERT");
    const char* cert = Env("TUNA_CLIENT_CERT");
    const char* key = Env("TUNA_CLIENT_KEY");
    if (ca || cert || key) {
        if (!ca || !cert || !key) {
            *error = "set all of TUNA_CA_CERT, TUNA_CLIENT_CERT and TUNA_CLIENT_KEY, or none";
            return false;
        }
        options.tls = tunnel::TlsConfig{ca, cert, key, {}};
    }
    if (const char* name = Env("TUNA_SERVER_NAME")) options.server_name = name;
    options.connect_deadline = std::chrono::milliseconds(5000);
    if (const char* timeout = Env("TUNA_CONNECT_TIMEOUT_MS")) {
        options.connect_deadline = std::chrono::milliseconds(std::atoll(timeout));
    }

    std::optional<tunnel::ClientSession> connection = tunnel::ConnectSession(options, error);
    if (!connection) return false;

    auto* session = new Session{std::move(*connection), nullptr};
    tunnel::RpcClientOptions rpc_options;
    // A synchronizing call legitimately waits as long as the GPU work it
    // follows; the heartbeat (10 s) is what detects a dead server.
    rpc_options.call_timeout = std::chrono::hours(24 * 365);
    session->rpc = std::make_unique<RpcClient>(*session->connection.transport, rpc_options,
                                               session->connection.bulk.get());
    g_rpc = session->rpc.get();
    if (LogEnabled()) {
        Log("connected to " + address + (options.tls ? " (mTLS)" : " (plaintext)") + ", worker pid " +
            std::to_string(session->connection.server_hello.worker_pid()));
    }
    return true;
}

// The session's RpcClient, connecting on first use; nullptr if unavailable.
RpcClient* Rpc() {
    std::lock_guard<std::mutex> lock(g_session_mutex);
    if (!g_rpc && !g_connect_attempted) {
        g_connect_attempted = true;
        if (!ConnectLocked(&g_connect_error)) {
            std::fprintf(stderr, "tuna: no remote GPU: %s\n", g_connect_error.c_str());
        }
    }
    return g_rpc;
}

}  // namespace

bool LogEnabled() {
    static const bool enabled = Env("TUNA_LOG") != nullptr;
    return enabled;
}

void Log(const std::string& message) { std::fprintf(stderr, "tuna: %s\n", message.c_str()); }

bool Connect(std::string* error) {
    if (Rpc()) return true;
    std::lock_guard<std::mutex> lock(g_session_mutex);
    if (error) *error = g_connect_error;
    return false;
}

void SetCallHook(CallHook hook) {
    g_hook = hook;
    std::lock_guard<std::mutex> lock(g_hook_mutex);
    g_hook_latched.reset();
}

void SetRpcClient(RpcClient* client) {
    std::lock_guard<std::mutex> lock(g_session_mutex);
    g_rpc = client;
    g_connect_attempted = true;  // an explicit binding replaces the env-driven connect
}

Reply Call(uint32_t api_id, const ArgWriter& in, const std::string& bulk_in, bool synchronizing) {
    if (CallHook hook = g_hook.load()) {
        Reply reply = hook(api_id, in.str(), bulk_in, /*async=*/false);
        if (synchronizing) {
            std::lock_guard<std::mutex> lock(g_hook_mutex);
            if (g_hook_latched) {
                reply.result = *g_hook_latched;
                g_hook_latched.reset();
            }
        }
        return reply;
    }

    Reply reply;
    RpcClient* rpc = Rpc();
    if (!rpc) {
        reply.result = kErrNoDevice;
        return reply;
    }
    const auto start = std::chrono::steady_clock::now();
    CallResult r = rpc->Call(api_id, in.str(), synchronizing, bulk_in);
    if (StatsEnabled()) {
        const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
        CountCall(api_id, /*posted=*/false, static_cast<uint64_t>(waited.count()));
    }
    if (r.result == kResultTransportError) {
        reply.transport_error = true;
        return reply;
    }
    reply.result = r.result;
    reply.out_args = std::move(r.out_args);
    reply.bulk_out = std::move(r.bulk_out);
    return reply;
}

int32_t Post(uint32_t api_id, const ArgWriter& in, const std::string& bulk_in) {
    if (CallHook hook = g_hook.load()) {
        Reply reply = hook(api_id, in.str(), bulk_in, /*async=*/true);
        std::lock_guard<std::mutex> lock(g_hook_mutex);
        if (reply.result != 0 && !g_hook_latched) g_hook_latched = reply.result;
        return 0;
    }
    RpcClient* rpc = Rpc();
    if (!rpc) return Fail(kErrNoDevice);
    if (!rpc->connected()) return Fail(kErrUnknown);
    rpc->Post(api_id, in.str(), bulk_in);
    if (StatsEnabled()) CountCall(api_id, /*posted=*/true, 0);
    return 0;
}

int32_t Fail(int32_t error) {
    if (error != 0) t_last_error = error;
    return error;
}

int32_t Finish(const Reply& reply) { return Fail(reply.transport_error ? kErrUnknown : reply.result); }

int32_t LastError(bool clear) {
    const int32_t error = t_last_error;
    if (clear) t_last_error = 0;
    return error;
}

int32_t ArrivedAsyncError(bool clear) {
    if (g_hook.load()) {
        std::lock_guard<std::mutex> lock(g_hook_mutex);
        const int32_t error = g_hook_latched.value_or(0);
        if (clear) g_hook_latched.reset();
        return error;
    }
    RpcClient* rpc = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_session_mutex);
        rpc = g_rpc;
    }
    if (!rpc) return 0;
    return rpc->ArrivedAsyncError(clear).value_or(0);
}

bool LaunchBlocking() {
    static const bool enabled = [] {
        const char* value = std::getenv("CUDA_LAUNCH_BLOCKING");
        return value && *value && std::string(value) != "0";
    }();
    return enabled;
}

int Unsupported(const char* name) {
    if (LogEnabled()) {
        static std::mutex mutex;
        static std::set<std::string> reported;
        std::lock_guard<std::mutex> lock(mutex);
        if (reported.insert(name).second) Log(std::string(name) + " is not supported by Tuna yet");
    }
    return Fail(kErrNotSupported);
}

uint64_t ContextGeneration() { return g_generation.load(); }
void BumpContextGeneration() { ++g_generation; }

}  // namespace tuna::cudart

// ------------------------------------------------------ library bridge

// Tuna's other library shims (libcublas, ...) send their calls through here,
// so they share this process's session (see tuna/cuda/bridge.h). Unlike the
// runtime's own calls, nothing here touches the CUDA last-error state: a
// library reports failures in its own status codes.
TUNA_CUDART_EXPORT int32_t tuna_cudart_call(uint32_t api_id, const char* args, size_t args_size, const char* bulk_in,
                                            size_t bulk_in_size, uint32_t mode, TunaCallReply* reply) {
    using namespace tuna::cudart;
    thread_local Reply last;  // backs the pointers handed out in *reply
    ArgWriter in;
    if (args_size) in.put(args, args_size);
    const std::string bulk = bulk_in_size ? std::string(bulk_in, bulk_in_size) : std::string();

    if (mode == kTunaPost) {
        // Post records a connection failure as the runtime's last error; keep
        // the runtime's state as it was.
        const int32_t before = LastError(/*clear=*/false);
        const int32_t queued = Post(api_id, in, bulk);
        if (queued != 0) {
            LastError(/*clear=*/true);
            Fail(before);
        }
        last = Reply{};
        last.result = queued;
    } else {
        last = Call(api_id, in, bulk, /*synchronizing=*/mode == kTunaCallSynchronize);
    }
    if (reply) {
        reply->result = last.result;
        reply->transport_error = last.transport_error ? 1 : 0;
        reply->out_args = last.out_args.data();
        reply->out_args_size = last.out_args.size();
        reply->bulk_out = last.bulk_out.data();
        reply->bulk_out_size = last.bulk_out.size();
    }
    return last.result;
}

TUNA_CUDART_EXPORT uint64_t tuna_cudart_context_generation(void) { return tuna::cudart::ContextGeneration(); }

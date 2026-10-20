#pragma once

#include <cstdint>
#include <string>

#include "tuna/cuda/marshal.h"
#include "tuna/tunnel/rpc.h"

// Symbols the replacement libcudart exports. The Linux version script
// (cuda/libcudart.map) hides everything else, including the statically linked
// tunnel, protobuf and C++ runtime pieces, so they cannot clash with an app's.
#if defined(_WIN32)
#define TUNA_CUDART_EXPORT extern "C" __declspec(dllexport)
#else
#define TUNA_CUDART_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace tuna::cudart {

using tunnel::CallResult;
using tunnel::kResultTransportError;
using tunnel::RpcClient;

// cudaError_t values the shim produces itself (no CUDA header needed here).
constexpr int32_t kErrInvalidValue = 1;
constexpr int32_t kErrMemoryAllocation = 2;
constexpr int32_t kErrNoDevice = 100;           // no server configured / reachable
constexpr int32_t kErrInvalidDeviceFunction = 98;
constexpr int32_t kErrInvalidSymbol = 13;
constexpr int32_t kErrNotSupported = 801;
constexpr int32_t kErrUnknown = 999;            // connection lost mid-session

// Sends one marshalled call and waits for its reply. Connects on first use
// (see Connect()). A `synchronizing` call reports a latched async error in
// place of its own result, the way cudaDeviceSynchronize reports a failed
// kernel launch.
Reply Call(uint32_t api_id, const ArgWriter& in, const std::string& bulk_in = {}, bool synchronizing = false);

// Fire-and-forget: queued, batched, and executed in order with every other
// call. Returns 0, or the connection error if there is no session. A failure
// on the server surfaces on the next synchronizing Call().
int32_t Post(uint32_t api_id, const ArgWriter& in, const std::string& bulk_in = {});

// Turns a reply into the cudaError_t to return, and records a failure as this
// thread's last error (what cudaGetLastError reports).
int32_t Finish(const Reply& reply);

// Records `error` as this thread's last error and returns it.
int32_t Fail(int32_t error);

// The client's own last error for this thread. `clear` resets it.
int32_t LastError(bool clear);

// The failure of an earlier fire-and-forget call, if the server's answer has
// already arrived (no round trip); 0 if none. `clear` resets it.
int32_t ArrivedAsyncError(bool clear);

// True when CUDA_LAUNCH_BLOCKING is set: error queries then wait for every
// queued call, so a failed launch is reported by the next cudaGetLastError
// (CUDA's own debugging switch for asynchronous errors).
bool LaunchBlocking();

// Generated stubs for libcudart exports Tuna does not remote. Logs the name
// once (with TUNA_LOG=1) and returns cudaErrorNotSupported.
int Unsupported(const char* name);

// Bumped by cudaSetDevice and cudaDeviceReset: cached per-context state
// (function handles, symbol addresses) from older generations is refetched.
uint64_t ContextGeneration();
void BumpContextGeneration();

// ------------------------------------------------------------- connection
//
// The shim connects lazily, on the first CUDA call, using environment
// variables (a launcher like `tuna run` sets them):
//   TUNA_SERVER       host:port of the tuna_server listener (required)
//   TUNA_CA_CERT      CA that signed the server certificate     } mTLS; all three,
//   TUNA_CLIENT_CERT  this client's certificate                 } or none for a
//   TUNA_CLIENT_KEY   this client's private key                 } loopback server
//   TUNA_SERVER_NAME  identity to verify (default: the host)
//   TUNA_CONNECT_TIMEOUT_MS  retry window for the first connect (default 5000)
//   TUNA_LOG=1        diagnostics on stderr
//   TUNA_STATS=1      at exit, per-api counts of blocking and posted calls and
//                     the time blocking calls waited (what remoting costs)
// Without a reachable server every call fails with cudaErrorNoDevice, which
// applications already treat as "no CUDA GPU here".
bool Connect(std::string* error);

// Binds the shim to an established session (tests, or an embedding host).
void SetRpcClient(RpcClient* client);

// Test seam: route calls to an in-process function instead of the transport.
// `async` marks a Post(); its reply's result is latched like a real async
// failure. nullptr restores the transport path.
using CallHook = Reply (*)(uint32_t api_id, const std::string& args, const std::string& bulk_in, bool async);
void SetCallHook(CallHook hook);

// Drops every client-side cache of server state (modules uploaded, function
// handles, allocations). Tests call this between in-process "sessions".
void ResetClientStateForTesting();

// True when TUNA_LOG is set; Log() prints "tuna: ..." to stderr.
bool LogEnabled();
void Log(const std::string& message);

}  // namespace tuna::cudart

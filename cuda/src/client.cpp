// Client-side CUDA runtime shim. Exports the runtime symbols an app resolves
// (generated stubs + the hand-written cudaMemcpy), marshals each call, and
// sends it over the Phase 2 tunnel. Loaded ahead of the real libcudart via
// LD_LIBRARY_PATH (Linux) or the DLL search path (Windows).
//
// A test seam (SetCallHook) lets the marshalling tests route calls to an
// in-process dispatcher with a mock backend, so correctness is checked with no
// GPU and no transport.

#include "tuna/cuda/client.h"

#include <atomic>
#include <mutex>

#include <cuda_runtime_api.h>

#include "cuda_api_ids.h"

namespace tuna::cudart {

namespace {
std::atomic<CallHook> g_hook{nullptr};
std::mutex g_rpc_mutex;
RpcClient* g_rpc = nullptr;  // guarded by g_rpc_mutex; set once a session is up
std::chrono::milliseconds g_timeout{30000};
}  // namespace

void SetCallHook(CallHook hook) { g_hook = hook; }

void SetRpcClient(RpcClient* client) {
    std::lock_guard<std::mutex> lock(g_rpc_mutex);
    g_rpc = client;
}

Reply Call(uint32_t api_id, const ArgWriter& in, const std::string& bulk_in) {
    if (CallHook hook = g_hook.load()) return hook(api_id, in.str(), bulk_in);

    std::lock_guard<std::mutex> lock(g_rpc_mutex);
    Reply reply;
    if (!g_rpc) {
        reply.transport_error = true;
        return reply;
    }
    const CallResult r = g_rpc->Call(api_id, in.str(), /*synchronizing=*/false, bulk_in);
    if (r.result == kResultTransportError) {
        reply.transport_error = true;
        return reply;
    }
    reply.result = r.result;
    reply.out_args = std::move(r.out_args);
    reply.bulk_out = std::move(r.bulk_out);
    return reply;
}

}  // namespace tuna::cudart

extern "C" {

// Hand-written special case: cudaMemcpy. Direction follows `kind`; the payload
// uses the bulk channel. Device pointers are opaque server handles.
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    using namespace tuna::cudart;
    ArgWriter in;
    in.put_u64(reinterpret_cast<uint64_t>(dst));
    in.put_u64(reinterpret_cast<uint64_t>(src));
    in.put_u64(static_cast<uint64_t>(count));
    const uint32_t k = static_cast<uint32_t>(kind);
    in.put(&k, 4);

    std::string bulk_in;
    if (kind == cudaMemcpyHostToDevice) bulk_in.assign(static_cast<const char*>(src), count);
    Reply reply = Call(kApi_cudaMemcpy, in, bulk_in);
    if (reply.transport_error) return static_cast<cudaError_t>(999);
    if (kind == cudaMemcpyDeviceToHost && reply.result == 0) {
        if (reply.bulk_out.size() != count) return cudaErrorUnknown;
        std::memcpy(dst, reply.bulk_out.data(), count);
    }
    return static_cast<cudaError_t>(reply.result);
}

// Generated stubs for the annotated simple functions. They use the api-id
// enumerators and the marshalling helpers from tuna::cudart.
using namespace tuna::cudart;
#include "cuda_client_stubs.inc"

}  // extern "C"

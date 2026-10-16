// Server-side CUDA runtime dispatcher. The per-api switch is generated
// (cuda_server_dispatch.inc); cudaMemcpy and any other `special:` function are
// hand-written here. Device pointers arrive as uint64 handles the server
// issued from its own CUDA context; they are only ever passed back to the real
// runtime, never dereferenced by Tuna.

#include "tuna/cuda/backend.h"

#include <cuda_runtime_api.h>

#include "cuda_api_ids.h"

namespace tuna::cudart {

namespace {

constexpr int32_t kCudaErrorNotSupported = 801;
constexpr int32_t kCudaErrorInvalidValue = 1;

// Hand-written: cudaMemcpy. `kind` decides the direction; the payload rides the
// bulk channel so a large copy never blocks small control calls.
bool DispatchMemcpy(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out,
                    CudaBackend& backend, RpcResponse& resp) {
    if (request.api_id() != kApi_cudaMemcpy) return false;
    ArgReader in(request.args());
    const uint64_t dst = in.get_u64();
    const uint64_t src = in.get_u64();
    const uint64_t count = in.get_u64();
    uint32_t kind = 0;
    in.get(&kind, 4);

    auto fn = reinterpret_cast<cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind)>(
        backend.Resolve("cudaMemcpy"));
    if (!fn) {
        resp.set_result(kCudaErrorNotSupported);
        return true;
    }

    if (kind == cudaMemcpyHostToDevice) {
        if (bulk_in.size() != count) {
            resp.set_result(kCudaErrorInvalidValue);
            return true;
        }
        resp.set_result(fn(reinterpret_cast<void*>(dst), bulk_in.data(), count, cudaMemcpyHostToDevice));
    } else if (kind == cudaMemcpyDeviceToHost) {
        std::string host(count, '\0');
        resp.set_result(fn(host.data(), reinterpret_cast<const void*>(src), count, cudaMemcpyDeviceToHost));
        if (resp.result() == 0) *bulk_out = std::move(host);
    } else {  // D2D: both ends are server handles, no host payload
        resp.set_result(fn(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), count,
                           static_cast<cudaMemcpyKind>(kind)));
    }
    return true;
}

}  // namespace

RpcResponse CudaDispatch(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out,
                         CudaBackend& backend) {
    RpcResponse resp;
    try {
        if (DispatchMemcpy(request, bulk_in, bulk_out, backend, resp)) return resp;
#include "cuda_server_dispatch.inc"
    } catch (const std::exception&) {
        resp.set_result(kCudaErrorInvalidValue);  // short/garbled arg buffer
    }
    return resp;
}

bool IsCudaApi(uint32_t api_id) { return api_id >= kFirstCudaApi && api_id <= kLastCudaApi; }

}  // namespace tuna::cudart

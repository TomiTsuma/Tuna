// Server-side CUDA driver API dispatch (Phase 3e). The per-api cases are
// generated (driver_server_dispatch.inc); retaining a primary context and the
// device name (an OUT buffer) are hand-written here.
//
// Every generated driver call takes only scalars (device ordinals, attribute
// ids, flags): gen_cuda.py refuses contexts, modules and functions, which are
// host pointers in this worker. The client's context tokens stand for the
// primary contexts this session retained.

#include "tuna/cuda/backend.h"

#include <cuda.h>

#include <cstring>

#include "cuda_api_ids.h"

namespace tuna::cudart {

namespace {

constexpr int32_t kCudaErrorInvalidValue = 1;
constexpr int32_t kCudaErrorNotSupported = 801;

// Longer than any device name the driver reports (cudaDeviceProp::name is 256).
constexpr int kMaxDeviceName = 256;

}  // namespace

int32_t CudaSession::DriverInit() {
    if (driver_initialized_) return 0;
    auto init = reinterpret_cast<CUresult (*)(unsigned)>(backend_.Resolve("cuInit"));
    if (!init) return kCudaErrorNotSupported;
    if (CUresult r = init(0); r != CUDA_SUCCESS) return r;
    driver_initialized_ = true;
    return 0;
}

bool CudaSession::DispatchDriver(const RpcRequest& request, RpcResponse& resp) {
    switch (request.api_id()) {
    case kApi_cuDevicePrimaryCtxRetain: {
        ArgReader in(request.args());
        int device = 0;
        in.get(&device, 4);
        if (in.remaining() != 0) {
            resp.set_result(kCudaErrorInvalidValue);
            return true;
        }
        if (int32_t e = DriverInit()) {
            resp.set_result(e);
            return true;
        }
        if (retained_primary_.count(device)) return true;  // already retained this session
        auto get = reinterpret_cast<CUresult (*)(CUdevice*, int)>(backend_.Resolve("cuDeviceGet"));
        auto retain = reinterpret_cast<CUresult (*)(CUcontext*, CUdevice)>(backend_.Resolve("cuDevicePrimaryCtxRetain"));
        if (!get || !retain) {
            resp.set_result(kCudaErrorNotSupported);
            return true;
        }
        CUdevice dev = 0;
        CUcontext ctx = nullptr;
        CUresult r = get(&dev, device);
        if (r == CUDA_SUCCESS) r = retain(&ctx, dev);
        if (r == CUDA_SUCCESS) retained_primary_.insert(device);
        resp.set_result(r);
        return true;
    }
    case kApi_cuDeviceGetName: {
        ArgReader in(request.args());
        int device = 0;
        in.get(&device, 4);
        if (in.remaining() != 0) {
            resp.set_result(kCudaErrorInvalidValue);
            return true;
        }
        if (int32_t e = DriverInit()) {
            resp.set_result(e);
            return true;
        }
        auto get_name = reinterpret_cast<CUresult (*)(char*, int, CUdevice)>(backend_.Resolve("cuDeviceGetName"));
        if (!get_name) {
            resp.set_result(kCudaErrorNotSupported);
            return true;
        }
        char name[kMaxDeviceName] = {};
        const CUresult r = get_name(name, kMaxDeviceName, device);
        resp.set_result(r);
        if (r == CUDA_SUCCESS) resp.set_out_args(std::string(name, strnlen(name, kMaxDeviceName - 1)));
        return true;
    }
    default:
        break;
    }
#include "driver_server_dispatch.inc"
}

}  // namespace tuna::cudart

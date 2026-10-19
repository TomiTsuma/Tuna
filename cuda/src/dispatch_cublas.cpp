// Server-side cuBLAS dispatch (Phase 3d). The per-api cases are generated
// (cublas_server_dispatch.inc); the host<->device matrix copies, whose host
// side travels on the bulk channel, are hand-written here.
//
// cuBLAS runs in the session's worker, in the same CUDA context as the
// runtime calls: the app's device pointers and streams are valid as they are,
// and a cublasHandle_t is the real server handle, which the client only
// passes back.
//
// Handles: a cublasHandle_t is a pointer into cuBLAS's own host memory in this
// worker. The session therefore records the handles it created (and each
// one's pointer mode, which decides whether a scalar argument is read from
// host or device memory) and passes nothing else to cuBLAS; the generated
// cases check both. Create, destroy and set-pointer-mode maintain the table
// and are hand-written below.
//
// Status codes: a synchronous call answers with the cublasStatus_t. A posted
// (_async) call's failure is latched by the client and returned by the app's
// next synchronizing CUDA call, so it is reported as the closest cudaError_t
// (CublasAsyncError).

#include "tuna/cuda/backend.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>

#include "cuda_api_ids.h"
#include "dispatch_internal.h"
#include "tuna/tunnel/rpc.h"

namespace tuna::cudart {

int32_t CublasAsyncError(int status) {
    switch (static_cast<cublasStatus_t>(status)) {
    case CUBLAS_STATUS_SUCCESS: return cudaSuccess;
    case CUBLAS_STATUS_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUBLAS_STATUS_ALLOC_FAILED: return cudaErrorMemoryAllocation;
    case CUBLAS_STATUS_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUBLAS_STATUS_ARCH_MISMATCH: return cudaErrorNoKernelImageForDevice;
    case CUBLAS_STATUS_MAPPING_ERROR: return cudaErrorIllegalAddress;
    case CUBLAS_STATUS_EXECUTION_FAILED: return cudaErrorLaunchFailure;
    case CUBLAS_STATUS_NOT_SUPPORTED: return cudaErrorNotSupported;
    default: return cudaErrorUnknown;  // INTERNAL_ERROR, LICENSE_ERROR
    }
}

namespace {

constexpr int32_t kCudaErrorInvalidValue = 1;

bool IsAsyncCublasCopy(uint32_t api_id) {
    return api_id == kApi_cublasSetMatrixAsync || api_id == kApi_cublasGetMatrixAsync;
}

}  // namespace

bool CudaSession::DispatchCublas(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out,
                                 RpcResponse& resp) {
    try {
        switch (request.api_id()) {
        case kApi_cublasSetMatrix:
        case kApi_cublasSetMatrixAsync:
        case kApi_cublasGetMatrix:
        case kApi_cublasGetMatrixAsync:
            MatrixCopy(request, bulk_in, bulk_out, resp);
            return true;
        case kApi_cublasCreate_v2:
        case kApi_cublasDestroy_v2:
        case kApi_cublasSetPointerMode_v2:
            CublasHandleCall(request, resp);
            return true;
        default:
            break;
        }
#include "cublas_server_dispatch.inc"
    } catch (const std::exception&) {
        // A short or garbled arg buffer. Posted calls report in CUDA's codes.
        const bool posted = (request.flags() & tunnel::kFlagAsync) != 0;
        resp.set_result(posted ? kCudaErrorInvalidValue : CUBLAS_STATUS_INVALID_VALUE);
        resp.clear_out_args();
        bulk_out->clear();
        return true;
    }
}

// cublasCreate_v2 / cublasDestroy_v2 / cublasSetPointerMode_v2: the calls that
// change which handles exist and how their scalars are passed.
void CudaSession::CublasHandleCall(const RpcRequest& request, RpcResponse& resp) {
    ArgReader in(request.args());
    switch (request.api_id()) {
    case kApi_cublasCreate_v2: {
        if (in.remaining() != 0) {
            resp.set_result(CUBLAS_STATUS_INVALID_VALUE);
            return;
        }
        auto create = reinterpret_cast<cublasStatus_t (*)(cublasHandle_t*)>(backend_.Resolve("cublasCreate_v2"));
        if (!create) {
            resp.set_result(CUBLAS_STATUS_NOT_SUPPORTED);
            return;
        }
        cublasHandle_t handle = nullptr;
        const cublasStatus_t status = create(&handle);
        resp.set_result(status);
        if (status != CUBLAS_STATUS_SUCCESS) return;
        cublas_handles_[handle] = false;  // CUBLAS_POINTER_MODE_HOST
        ArgWriter out;
        out.put_u64(reinterpret_cast<uint64_t>(handle));
        resp.set_out_args(out.str());
        return;
    }
    case kApi_cublasDestroy_v2: {
        auto handle = reinterpret_cast<cublasHandle_t>(in.get_u64());
        if (in.remaining() != 0) {
            resp.set_result(CUBLAS_STATUS_INVALID_VALUE);
            return;
        }
        if (!cublas_handles_.count(handle)) {
            resp.set_result(CUBLAS_STATUS_NOT_INITIALIZED);
            return;
        }
        auto destroy = reinterpret_cast<cublasStatus_t (*)(cublasHandle_t)>(backend_.Resolve("cublasDestroy_v2"));
        if (!destroy) {
            resp.set_result(CUBLAS_STATUS_NOT_SUPPORTED);
            return;
        }
        resp.set_result(destroy(handle));
        cublas_handles_.erase(handle);  // cuBLAS has freed it whatever it returned
        return;
    }
    default: {  // cublasSetPointerMode_v2, posted: failures are cudaError_t
        auto handle = reinterpret_cast<cublasHandle_t>(in.get_u64());
        cublasPointerMode_t mode{};
        in.get(&mode, 4);
        auto known = cublas_handles_.find(handle);
        if (in.remaining() != 0 || (mode != CUBLAS_POINTER_MODE_HOST && mode != CUBLAS_POINTER_MODE_DEVICE)) {
            resp.set_result(kCudaErrorInvalidValue);
            return;
        }
        if (known == cublas_handles_.end()) {
            resp.set_result(CublasAsyncError(CUBLAS_STATUS_NOT_INITIALIZED));
            return;
        }
        auto set = reinterpret_cast<cublasStatus_t (*)(cublasHandle_t, cublasPointerMode_t)>(
            backend_.Resolve("cublasSetPointerMode_v2"));
        if (!set) {
            resp.set_result(kCudaErrorNotSupported);
            return;
        }
        const cublasStatus_t status = set(handle, mode);
        if (status == CUBLAS_STATUS_SUCCESS) known->second = mode == CUBLAS_POINTER_MODE_DEVICE;
        resp.set_result(CublasAsyncError(status));
        return;
    }
    }
}

// cublasSetMatrix(Async) / cublasGetMatrix(Async). The client sends the host
// side packed (leading dimension = rows): rows, cols, elemSize, the device
// pointer, its leading dimension and, for the async forms, the stream.
void CudaSession::MatrixCopy(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out,
                             RpcResponse& resp) {
    const uint32_t api = request.api_id();
    const bool async = IsAsyncCublasCopy(api);
    const bool to_device = api == kApi_cublasSetMatrix || api == kApi_cublasSetMatrixAsync;
    const int32_t invalid = async && to_device ? kCudaErrorInvalidValue : CUBLAS_STATUS_INVALID_VALUE;

    ArgReader in(request.args());
    int rows = 0, cols = 0, elem_size = 0, ld = 0;
    in.get(&rows, 4);
    in.get(&cols, 4);
    in.get(&elem_size, 4);
    void* device = reinterpret_cast<void*>(in.get_u64());
    in.get(&ld, 4);
    cudaStream_t stream = async ? reinterpret_cast<cudaStream_t>(in.get_u64()) : nullptr;
    if (in.remaining() != 0 || rows < 0 || cols < 0 || elem_size <= 0 || ld < std::max(1, rows)) {
        resp.set_result(invalid);
        return;
    }
    // rows, cols and elemSize are each < 2^31, so the product fits in 93 bits:
    // check it against the payload cap before multiplying everything out.
    const uint64_t column = static_cast<uint64_t>(rows) * static_cast<uint64_t>(elem_size);
    if (cols != 0 && column > tunnel::kMaxBulkTransferBytes / static_cast<uint64_t>(cols)) {
        resp.set_result(invalid);
        return;
    }
    const uint64_t bytes = column * static_cast<uint64_t>(cols);
    const int packed_ld = std::max(1, rows);

    if (to_device) {
        if (bulk_in.size() != bytes) {
            resp.set_result(invalid);
            return;
        }
        using SetFn = cublasStatus_t (*)(int, int, int, const void*, int, void*, int);
        using SetAsyncFn = cublasStatus_t (*)(int, int, int, const void*, int, void*, int, cudaStream_t);
        cublasStatus_t status;
        if (async) {
            auto set = reinterpret_cast<SetAsyncFn>(backend_.Resolve("cublasSetMatrixAsync"));
            if (!set) {
                resp.set_result(kCudaErrorNotSupported);
                return;
            }
            // From pageable memory the copy is staged before this returns, so
            // bulk_in may be released right after (as with cudaMemcpyAsync).
            status = set(rows, cols, elem_size, bulk_in.data(), packed_ld, device, ld, stream);
            resp.set_result(CublasAsyncError(status));
        } else {
            auto set = reinterpret_cast<SetFn>(backend_.Resolve("cublasSetMatrix"));
            if (!set) {
                resp.set_result(CUBLAS_STATUS_NOT_SUPPORTED);
                return;
            }
            resp.set_result(set(rows, cols, elem_size, bulk_in.data(), packed_ld, device, ld));
        }
        return;
    }

    using GetFn = cublasStatus_t (*)(int, int, int, const void*, int, void*, int);
    using GetAsyncFn = cublasStatus_t (*)(int, int, int, const void*, int, void*, int, cudaStream_t);
    std::string host(bytes, '\0');
    cublasStatus_t status;
    if (async) {
        auto get = reinterpret_cast<GetAsyncFn>(backend_.Resolve("cublasGetMatrixAsync"));
        auto stream_sync = reinterpret_cast<cudaError_t (*)(cudaStream_t)>(backend_.Resolve("cudaStreamSynchronize"));
        if (!get || !stream_sync) {
            resp.set_result(CUBLAS_STATUS_NOT_SUPPORTED);
            return;
        }
        status = get(rows, cols, elem_size, device, ld, host.data(), packed_ld, stream);
        // The client needs the bytes now: wait for this stream's work.
        if (status == CUBLAS_STATUS_SUCCESS && stream_sync(stream) != cudaSuccess) {
            status = CUBLAS_STATUS_EXECUTION_FAILED;
        }
    } else {
        auto get = reinterpret_cast<GetFn>(backend_.Resolve("cublasGetMatrix"));
        if (!get) {
            resp.set_result(CUBLAS_STATUS_NOT_SUPPORTED);
            return;
        }
        status = get(rows, cols, elem_size, device, ld, host.data(), packed_ld);
    }
    resp.set_result(status);
    if (status == CUBLAS_STATUS_SUCCESS) *bulk_out = std::move(host);
}

}  // namespace tuna::cudart

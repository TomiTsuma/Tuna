// Tuna's replacement libcublas (Phase 3d): the hand-written exports and the
// plumbing the generated stubs (cublas_client_stubs.inc) use.
//
// cuBLAS calls run on the server, in the same CUDA context as the app's
// runtime calls: everything goes through the runtime shim's session
// (tuna_cudart_call), so a cublasSgemm is ordered after the cudaMemcpy that
// filled its inputs and before the cudaMemcpy that reads its output. Device
// pointers, streams and cublasHandle_t values are server addresses the client
// only passes back.
//
// Client-side state is small: each handle's pointer mode, which decides
// whether alpha/beta are host values (copied into the request) or device
// addresses.

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <unordered_map>

#include "cuda_api_ids.h"
#include "tuna/cuda/bridge.h"
#include "tuna/cuda/cublas_client.h"

namespace tuna::cublas {

namespace {

constexpr int32_t kCudaErrorNoDevice = 100;
constexpr int32_t kCudaErrorNotSupported = 801;

// Largest host<->device copy sent as one bulk payload; bigger matrices go in
// column ranges (each piece is a full round trip anyway).
constexpr size_t kMaxCopyPiece = size_t{1} << 30;

std::mutex g_mutex;
std::unordered_map<const void*, bool> g_device_scalars;  // guarded by g_mutex; handle -> pointer mode is DEVICE

bool DeviceScalars(const void* handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_device_scalars.find(handle);
    return it != g_device_scalars.end() && it->second;
}

// Connection-level results (from the runtime shim, or a server that predates
// cuBLAS support) as cuBLAS statuses.
int StatusFromResult(int32_t result) {
    if (result == kCudaErrorNoDevice) return kStatusNotInitialized;
    if (result == kCudaErrorNotSupported) return kStatusNotSupported;
    return result;
}

bool LogEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("TUNA_LOG");
        return value && *value;
    }();
    return enabled;
}

}  // namespace

cudart::Reply Call(uint32_t api_id, const cudart::ArgWriter& in, const std::string& bulk_in, bool synchronizing) {
    TunaCallReply raw{};
    tuna_cudart_call(api_id, in.str().data(), in.str().size(), bulk_in.data(), bulk_in.size(),
                     synchronizing ? kTunaCallSynchronize : kTunaCall, &raw);
    cudart::Reply reply;
    reply.result = raw.result;
    reply.transport_error = raw.transport_error != 0;
    reply.out_args.assign(raw.out_args, raw.out_args_size);
    reply.bulk_out.assign(raw.bulk_out, raw.bulk_out_size);
    return reply;
}

int Post(uint32_t api_id, const cudart::ArgWriter& in, const std::string& bulk_in) {
    const int32_t queued =
        tuna_cudart_call(api_id, in.str().data(), in.str().size(), bulk_in.data(), bulk_in.size(), kTunaPost, nullptr);
    if (queued == 0) return kStatusSuccess;
    return queued == kCudaErrorNoDevice ? kStatusNotInitialized : kStatusInternalError;
}

int Finish(const cudart::Reply& reply) {
    if (reply.transport_error) return kStatusInternalError;
    return StatusFromResult(reply.result);
}

bool PutScalar(cudart::ArgWriter& in, const void* handle, const void* value, size_t size) {
    if (!value || size == 0 || size > cudart::kMaxScalarBytes) return false;
    if (DeviceScalars(handle)) {
        cudart::PutDeviceScalar(in, value);
    } else {
        cudart::PutHostScalar(in, value, size);
    }
    return true;
}

bool PutResult(cudart::ArgWriter& in, const void* handle, void* result) {
    if (DeviceScalars(handle)) {
        cudart::PutDeviceScalar(in, result);
        return true;
    }
    cudart::PutHostScalar(in, nullptr, 0);  // the server writes the value into the reply
    return false;
}

size_t ComputeScalarSize(int compute_type, int c_type) {
    size_t size = 0;
    switch (compute_type) {
    case CUBLAS_COMPUTE_16F:
    case CUBLAS_COMPUTE_16F_PEDANTIC:
        size = 2;
        break;
    case CUBLAS_COMPUTE_32F:
    case CUBLAS_COMPUTE_32F_PEDANTIC:
    case CUBLAS_COMPUTE_32F_FAST_16F:
    case CUBLAS_COMPUTE_32F_FAST_16BF:
    case CUBLAS_COMPUTE_32F_FAST_TF32:
    case CUBLAS_COMPUTE_32I:
    case CUBLAS_COMPUTE_32I_PEDANTIC:
        size = 4;
        break;
    case CUBLAS_COMPUTE_64F:
    case CUBLAS_COMPUTE_64F_PEDANTIC:
        size = 8;
        break;
    default:
        return 0;
    }
    switch (c_type) {  // complex C: complex scalars of the same precision
    case CUDA_C_16F:
    case CUDA_C_16BF:
    case CUDA_C_32F:
    case CUDA_C_64F:
    case CUDA_C_8I:
    case CUDA_C_32I:
        return size * 2;
    default:
        return size;
    }
}

int Unsupported(const char* name) {
    if (LogEnabled()) {
        static std::mutex mutex;
        static std::set<std::string> reported;
        std::lock_guard<std::mutex> lock(mutex);
        if (reported.insert(name).second) std::fprintf(stderr, "tuna: %s is not supported by Tuna yet\n", name);
    }
    return kStatusNotSupported;
}

void ResetStateForTesting() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_device_scalars.clear();
}

}  // namespace tuna::cublas

using namespace tuna::cudart;  // the generated stubs name kApi_* unqualified

// Generated stubs: the exported simple functions, plus static
// tuna_remote_<name> functions for the _wrap ones below.
#include "cublas_client_stubs.inc"

// ------------------------------------------------- handles (wrapped stubs)

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t status = tuna_remote_cublasCreate_v2(handle);
    if (status == CUBLAS_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(tuna::cublas::g_mutex);
        tuna::cublas::g_device_scalars[*handle] = false;  // CUBLAS_POINTER_MODE_HOST is the default
    }
    return status;
}

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle) {
    const cublasStatus_t status = tuna_remote_cublasDestroy_v2(handle);
    std::lock_guard<std::mutex> lock(tuna::cublas::g_mutex);
    tuna::cublas::g_device_scalars.erase(handle);
    return status;
}

// The pointer mode is tracked here (it decides how every alpha/beta is sent)
// and forwarded, so the server's handle interprets the scalars the same way.
TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle,
                                                                      cublasPointerMode_t mode) {
    if (mode != CUBLAS_POINTER_MODE_HOST && mode != CUBLAS_POINTER_MODE_DEVICE) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t status = tuna_remote_cublasSetPointerMode_v2(handle, mode);
    if (status == CUBLAS_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(tuna::cublas::g_mutex);
        tuna::cublas::g_device_scalars[handle] = mode == CUBLAS_POINTER_MODE_DEVICE;
    }
    return status;
}

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle,
                                                                      cublasPointerMode_t* mode) {
    if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = tuna::cublas::DeviceScalars(handle) ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST;
    return CUBLAS_STATUS_SUCCESS;
}

// ------------------------------------------- host <-> device matrix copies
//
// cublasSetMatrix copies a rows x cols column-major block from host memory
// (leading dimension lda) to the device (ldb). The client packs the host side
// densely and the server copies with lda = rows, so the payload carries no
// padding. Wire: rows, cols, elemSize (i32), the device pointer (u64), its
// leading dimension (i32) and, for the async forms, the stream (u64).

namespace {

using tuna::cublas::kMaxCopyPiece;

bool BadMatrix(int rows, int cols, int elem_size, int lda, int ldb) {
    return rows < 0 || cols < 0 || elem_size <= 0 || lda < std::max(1, rows) || ldb < std::max(1, rows);
}

// Columns per piece, so one piece stays under kMaxCopyPiece (at least one).
int ColumnsPerPiece(int rows, int elem_size) {
    const size_t column = static_cast<size_t>(rows) * static_cast<size_t>(elem_size);
    return static_cast<int>(std::max<size_t>(1, std::min<size_t>(kMaxCopyPiece / std::max<size_t>(column, 1),
                                                                 static_cast<size_t>(INT32_MAX))));
}

ArgWriter MatrixArgs(int rows, int cols, int elem_size, const void* device, int ld, bool async, cudaStream_t stream) {
    ArgWriter in;
    in.put(&rows, 4);
    in.put(&cols, 4);
    in.put(&elem_size, 4);
    in.put_u64(reinterpret_cast<uint64_t>(device));
    in.put(&ld, 4);
    if (async) in.put_u64(reinterpret_cast<uint64_t>(stream));
    return in;
}

cublasStatus_t SetMatrix(int rows, int cols, int elem_size, const void* A, int lda, void* B, int ldb, bool async,
                         cudaStream_t stream) {
    if (BadMatrix(rows, cols, elem_size, lda, ldb) || (!A && rows && cols)) return CUBLAS_STATUS_INVALID_VALUE;
    const size_t column = static_cast<size_t>(rows) * static_cast<size_t>(elem_size);
    const int step = ColumnsPerPiece(rows, elem_size);
    for (int first = 0; first < cols; first += step) {
        const int n = std::min(step, cols - first);
        std::string packed(column * static_cast<size_t>(n), '\0');
        for (int j = 0; j < n; ++j) {
            std::memcpy(packed.data() + column * j,
                        static_cast<const char*>(A) + static_cast<size_t>(first + j) * lda * elem_size, column);
        }
        void* piece = static_cast<char*>(B) + static_cast<size_t>(first) * ldb * elem_size;
        const ArgWriter in = MatrixArgs(rows, n, elem_size, piece, ldb, async, stream);
        const uint32_t api = async ? tuna::cudart::kApi_cublasSetMatrixAsync : tuna::cudart::kApi_cublasSetMatrix;
        const int status = async ? tuna::cublas::Post(api, in, packed)
                                 : tuna::cublas::Finish(tuna::cublas::Call(api, in, packed));
        if (status != CUBLAS_STATUS_SUCCESS) return static_cast<cublasStatus_t>(status);
    }
    return CUBLAS_STATUS_SUCCESS;
}

// The device -> host direction always waits for its bytes (the async form
// runs on `stream` on the server, then the server waits for that stream).
cublasStatus_t GetMatrix(int rows, int cols, int elem_size, const void* A, int lda, void* B, int ldb, bool async,
                         cudaStream_t stream) {
    if (BadMatrix(rows, cols, elem_size, lda, ldb) || (!B && rows && cols)) return CUBLAS_STATUS_INVALID_VALUE;
    const size_t column = static_cast<size_t>(rows) * static_cast<size_t>(elem_size);
    const int step = ColumnsPerPiece(rows, elem_size);
    for (int first = 0; first < cols; first += step) {
        const int n = std::min(step, cols - first);
        const void* piece = static_cast<const char*>(A) + static_cast<size_t>(first) * lda * elem_size;
        const ArgWriter in = MatrixArgs(rows, n, elem_size, piece, lda, async, stream);
        const uint32_t api = async ? tuna::cudart::kApi_cublasGetMatrixAsync : tuna::cudart::kApi_cublasGetMatrix;
        const tuna::cudart::Reply reply = tuna::cublas::Call(api, in);
        const int status = tuna::cublas::Finish(reply);
        if (status != CUBLAS_STATUS_SUCCESS) return static_cast<cublasStatus_t>(status);
        if (reply.bulk_out.size() != column * static_cast<size_t>(n)) return CUBLAS_STATUS_INTERNAL_ERROR;
        for (int j = 0; j < n; ++j) {
            std::memcpy(static_cast<char*>(B) + static_cast<size_t>(first + j) * ldb * elem_size,
                        reply.bulk_out.data() + column * j, column);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

bool BadVector(int n, int elem_size, int incx, int incy) { return n < 0 || elem_size <= 0 || incx <= 0 || incy <= 0; }

}  // namespace

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasSetMatrix(int rows, int cols, int elemSize, const void* A,
                                                              int lda, void* B, int ldb) {
    return SetMatrix(rows, cols, elemSize, A, lda, B, ldb, /*async=*/false, nullptr);
}

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasGetMatrix(int rows, int cols, int elemSize, const void* A,
                                                              int lda, void* B, int ldb) {
    return GetMatrix(rows, cols, elemSize, A, lda, B, ldb, /*async=*/false, nullptr);
}

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A,
                                                                   int lda, void* B, int ldb, cudaStream_t stream) {
    return SetMatrix(rows, cols, elemSize, A, lda, B, ldb, /*async=*/true, stream);
}

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A,
                                                                   int lda, void* B, int ldb, cudaStream_t stream) {
    return GetMatrix(rows, cols, elemSize, A, lda, B, ldb, /*async=*/true, stream);
}

// A vector with stride inc is a 1 x n matrix whose leading dimension is inc.
TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasSetVector(int n, int elemSize, const void* x, int incx,
                                                              void* devicePtr, int incy) {
    if (BadVector(n, elemSize, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
    return SetMatrix(1, n, elemSize, x, incx, devicePtr, incy, /*async=*/false, nullptr);
}

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasGetVector(int n, int elemSize, const void* x, int incx, void* y,
                                                              int incy) {
    if (BadVector(n, elemSize, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
    return GetMatrix(1, n, elemSize, x, incx, y, incy, /*async=*/false, nullptr);
}

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx,
                                                                   void* devicePtr, int incy, cudaStream_t stream) {
    if (BadVector(n, elemSize, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
    return SetMatrix(1, n, elemSize, hostPtr, incx, devicePtr, incy, /*async=*/true, stream);
}

TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasGetVectorAsync(int n, int elemSize, const void* devicePtr,
                                                                   int incx, void* hostPtr, int incy,
                                                                   cudaStream_t stream) {
    if (BadVector(n, elemSize, incx, incy)) return CUBLAS_STATUS_INVALID_VALUE;
    return GetMatrix(1, n, elemSize, devicePtr, incx, hostPtr, incy, /*async=*/true, stream);
}

// ------------------------------------------------------------ client-local

namespace {

struct StatusText {
    cublasStatus_t status;
    const char* name;
    const char* text;
};

// cuBLAS's own wording (cublasGetStatusString).
const StatusText kStatusTexts[] = {
    {CUBLAS_STATUS_SUCCESS, "CUBLAS_STATUS_SUCCESS", "the operation completed successfully"},
    {CUBLAS_STATUS_NOT_INITIALIZED, "CUBLAS_STATUS_NOT_INITIALIZED", "the library was not initialized"},
    {CUBLAS_STATUS_ALLOC_FAILED, "CUBLAS_STATUS_ALLOC_FAILED", "the resource allocation failed"},
    {CUBLAS_STATUS_INVALID_VALUE, "CUBLAS_STATUS_INVALID_VALUE", "an invalid numeric value was used as an argument"},
    {CUBLAS_STATUS_ARCH_MISMATCH, "CUBLAS_STATUS_ARCH_MISMATCH", "an absent device architectural feature is required"},
    {CUBLAS_STATUS_MAPPING_ERROR, "CUBLAS_STATUS_MAPPING_ERROR", "an access to GPU memory space failed"},
    {CUBLAS_STATUS_EXECUTION_FAILED, "CUBLAS_STATUS_EXECUTION_FAILED", "the GPU program failed to execute"},
    {CUBLAS_STATUS_INTERNAL_ERROR, "CUBLAS_STATUS_INTERNAL_ERROR", "an internal operation failed"},
    {CUBLAS_STATUS_NOT_SUPPORTED, "CUBLAS_STATUS_NOT_SUPPORTED", "the feature required is not supported"},
    {CUBLAS_STATUS_LICENSE_ERROR, "CUBLAS_STATUS_LICENSE_ERROR",
     "the functionality requested requires some license and an error was detected when trying to check the current "
     "licensing"},
};

const StatusText* FindStatus(cublasStatus_t status) {
    for (const auto& s : kStatusTexts) {
        if (s.status == status) return &s;
    }
    return nullptr;
}

}  // namespace

TUNA_CUBLAS_EXPORT const char* CUBLASWINAPI cublasGetStatusName(cublasStatus_t status) {
    const StatusText* s = FindStatus(status);
    return s ? s->name : "CUBLAS_STATUS_UNKNOWN";
}

TUNA_CUBLAS_EXPORT const char* CUBLASWINAPI cublasGetStatusString(cublasStatus_t status) {
    const StatusText* s = FindStatus(status);
    return s ? s->text : "unknown error";
}

// The runtime ABI this library pairs with: Tuna's libcudart implements CUDA 12.
TUNA_CUBLAS_EXPORT size_t CUBLASWINAPI cublasGetCudartVersion(void) { return CUDART_VERSION; }

// Logging happens in the server's cuBLAS, if anywhere; accept and ignore.
TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI cublasLoggerConfigure(int, int, int, const char*) {
    return CUBLAS_STATUS_SUCCESS;
}

// Marker the server checks before loading a "real" libcublas, so a
// misconfigured server never loads this shim and calls itself.
TUNA_CUBLAS_EXPORT int tuna_cublas_shim(void) { return 1; }

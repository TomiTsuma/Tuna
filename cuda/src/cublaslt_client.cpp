// Tuna's replacement libcublasLt (Phase 3f): every remoted export, by hand.
//
// cuBLASLt describes a GEMM with objects: a matmul descriptor (compute and
// scale type, transposes, epilogue, bias pointer, ...), one layout per matrix
// and a preference, each configured attribute by attribute. PyTorch builds
// all of them for every addmm, queries the heuristic once, runs the matmul
// and destroys them again.
//
// Remoting each object call synchronously would cost ~15 round trips per
// GEMM. Instead the client names the objects: a create call allocates an id
// here and is fire-and-forget like set-attribute, destroy and the matmul
// itself. The server maps the id to the real object (dispatch_cublaslt.cpp)
// and refuses ids it does not hold, so a failed create surfaces at the next
// call that uses the object. Only the heuristic query waits for the server:
// one round trip per GEMM.
//
// The client also tracks what decides how alpha/beta cross the wire: each
// matmul descriptor's scale type (their width) and pointer mode (host values
// or device addresses).
//
// The handle: PyTorch passes its cuBLAS handle as a cublasLtHandle_t, which
// NVIDIA allows; the server accepts the cuBLAS handles it created as well as
// handles from cublasLtCreate.

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

#include "cuda_api_ids.h"
#include "tuna/cuda/bridge.h"
#include "tuna/cuda/cublaslt_client.h"
#include "tuna/cuda/marshal.h"

namespace tuna::cublaslt {

namespace {

constexpr int32_t kCudaErrorNoDevice = 100;
constexpr int32_t kCudaErrorNotSupported = 801;

// Object ids: an address range no allocation uses, one 16-byte slot each.
// Only ever compared, never dereferenced.
constexpr uint64_t kObjectBase = 0x7d7d00000000ull;
std::atomic<uint64_t> g_next_object{0};

uint64_t NewObjectId() { return kObjectBase + (++g_next_object << 4); }

struct MatmulDesc {
    cudaDataType_t scale_type;
    int32_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
};

std::mutex g_mutex;
std::unordered_map<uint64_t, MatmulDesc> g_descs;  // guarded by g_mutex

bool LogEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("TUNA_LOG");
        return value && *value;
    }();
    return enabled;
}

cudart::Reply Call(uint32_t api_id, const cudart::ArgWriter& in) {
    TunaCallReply raw{};
    tuna_cudart_call(api_id, in.str().data(), in.str().size(), nullptr, 0, kTunaCall, &raw);
    cudart::Reply reply;
    reply.result = raw.result;
    reply.transport_error = raw.transport_error != 0;
    reply.out_args.assign(raw.out_args, raw.out_args_size);
    return reply;
}

cublasStatus_t Post(uint32_t api_id, const cudart::ArgWriter& in) {
    const int32_t queued = tuna_cudart_call(api_id, in.str().data(), in.str().size(), nullptr, 0, kTunaPost, nullptr);
    if (queued == 0) return CUBLAS_STATUS_SUCCESS;
    return queued == kCudaErrorNoDevice ? CUBLAS_STATUS_NOT_INITIALIZED : CUBLAS_STATUS_INTERNAL_ERROR;
}

// Connection-level results (no server, a server without cuBLAS) as statuses.
cublasStatus_t Finish(const cudart::Reply& reply) {
    if (reply.transport_error) return CUBLAS_STATUS_INTERNAL_ERROR;
    if (reply.result == kCudaErrorNoDevice) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (reply.result == kCudaErrorNotSupported) return CUBLAS_STATUS_NOT_SUPPORTED;
    return static_cast<cublasStatus_t>(reply.result);
}

// Bytes in one scalar of `type` (alpha/beta follow the scale type); 0 if
// it cannot be a scale type.
size_t ScalarSize(cudaDataType_t type) {
    switch (type) {
    case CUDA_R_16F:
    case CUDA_R_16BF: return 2;
    case CUDA_R_32F:
    case CUDA_R_32I:
    case CUDA_C_16F:
    case CUDA_C_16BF: return 4;
    case CUDA_R_64F:
    case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default: return 0;
    }
}

// Which of alpha/beta are device pointers under a descriptor's pointer mode.
bool AlphaOnDevice(int32_t mode) { return mode != CUBLASLT_POINTER_MODE_HOST; }
bool BetaOnDevice(int32_t mode) {
    return mode == CUBLASLT_POINTER_MODE_DEVICE || mode == CUBLASLT_POINTER_MODE_DEVICE_VECTOR;
}

// A host scalar may be null only when cuBLASLt ignores it (beta in the
// "beta zero" mode); the server then passes zeros.
void PutScalar(cudart::ArgWriter& in, const void* value, bool on_device, size_t size) {
    if (on_device) {
        cudart::PutDeviceScalar(in, value);
    } else {
        cudart::PutHostScalar(in, value, value ? size : 0);
    }
}

// The three object kinds share create/set/destroy wire formats.
cublasStatus_t SetAttribute(uint32_t api, const void* object, uint32_t attr, const void* buf, size_t size) {
    if (!object || !buf || size == 0 || size > kMaxAttributeBytes) return CUBLAS_STATUS_INVALID_VALUE;
    cudart::ArgWriter in;
    in.put_u64(reinterpret_cast<uint64_t>(object));
    in.put_u32(attr);
    in.put(buf, size);
    return Post(api, in);
}

cublasStatus_t Destroy(uint32_t api, const void* object) {
    if (!object) return CUBLAS_STATUS_SUCCESS;  // like cuBLASLt, destroying null is a no-op
    cudart::ArgWriter in;
    in.put_u64(reinterpret_cast<uint64_t>(object));
    return Post(api, in);
}

}  // namespace

int Unsupported(const char* name) {
    if (LogEnabled()) {
        static std::mutex mutex;
        static std::set<std::string> reported;
        std::lock_guard<std::mutex> lock(mutex);
        if (reported.insert(name).second) std::fprintf(stderr, "tuna: %s is not supported by Tuna yet\n", name);
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

void ResetStateForTesting() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_descs.clear();
}

}  // namespace tuna::cublaslt

using namespace tuna::cudart;  // kApi_* ids
using namespace tuna::cublaslt;

// ----------------------------------------------------------------- handles

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* lightHandle) {
    if (!lightHandle) return CUBLAS_STATUS_INVALID_VALUE;
    Reply reply = Call(kApi_cublasLtCreate, ArgWriter{});
    const cublasStatus_t status = Finish(reply);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    try {
        ArgReader out(reply.out_args);
        *lightHandle = reinterpret_cast<cublasLtHandle_t>(out.get_u64());
    } catch (const std::exception&) {
        return CUBLAS_STATUS_INTERNAL_ERROR;  // malformed reply
    }
    return CUBLAS_STATUS_SUCCESS;
}

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t lightHandle) {
    ArgWriter in;
    in.put_u64(reinterpret_cast<uint64_t>(lightHandle));
    return Finish(Call(kApi_cublasLtDestroy, in));
}

// ------------------------------------------------------------- descriptors

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
                                                                          cublasComputeType_t computeType,
                                                                          cudaDataType_t scaleType) {
    if (!matmulDesc) return CUBLAS_STATUS_INVALID_VALUE;
    const uint64_t id = NewObjectId();
    ArgWriter in;
    in.put_u64(id);
    in.put_u32(static_cast<uint32_t>(computeType));
    in.put_u32(static_cast<uint32_t>(scaleType));
    const cublasStatus_t status = Post(kApi_cublasLtMatmulDescCreate, in);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_descs[id] = MatmulDesc{scaleType};
    }
    *matmulDesc = reinterpret_cast<cublasLtMatmulDesc_t>(id);
    return CUBLAS_STATUS_SUCCESS;
}

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_descs.erase(reinterpret_cast<uint64_t>(matmulDesc));
    }
    return Destroy(kApi_cublasLtMatmulDescDestroy, matmulDesc);
}

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                                                                cublasLtMatmulDescAttributes_t attr,
                                                                                const void* buf, size_t sizeInBytes) {
    const cublasStatus_t status =
        SetAttribute(kApi_cublasLtMatmulDescSetAttribute, matmulDesc, static_cast<uint32_t>(attr), buf, sizeInBytes);
    // Mirror what decides alpha/beta's encoding (the server tracks the same).
    if (status == CUBLAS_STATUS_SUCCESS && sizeInBytes == 4 &&
        (attr == CUBLASLT_MATMUL_DESC_POINTER_MODE || attr == CUBLASLT_MATMUL_DESC_SCALE_TYPE)) {
        int32_t value = 0;
        std::memcpy(&value, buf, 4);
        std::lock_guard<std::mutex> lock(g_mutex);
        if (auto it = g_descs.find(reinterpret_cast<uint64_t>(matmulDesc)); it != g_descs.end()) {
            if (attr == CUBLASLT_MATMUL_DESC_POINTER_MODE) {
                it->second.pointer_mode = value;
            } else {
                it->second.scale_type = static_cast<cudaDataType_t>(value);
            }
        }
    }
    return status;
}

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout,
                                                                            cudaDataType type, uint64_t rows,
                                                                            uint64_t cols, int64_t ld) {
    if (!matLayout) return CUBLAS_STATUS_INVALID_VALUE;
    const uint64_t id = NewObjectId();
    ArgWriter in;
    in.put_u64(id);
    in.put_u32(static_cast<uint32_t>(type));
    in.put_u64(rows);
    in.put_u64(cols);
    in.put_u64(static_cast<uint64_t>(ld));
    const cublasStatus_t status = Post(kApi_cublasLtMatrixLayoutCreate, in);
    if (status == CUBLAS_STATUS_SUCCESS) *matLayout = reinterpret_cast<cublasLtMatrixLayout_t>(id);
    return status;
}

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    return Destroy(kApi_cublasLtMatrixLayoutDestroy, matLayout);
}

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                                                  cublasLtMatrixLayoutAttribute_t attr,
                                                                                  const void* buf, size_t sizeInBytes) {
    return SetAttribute(kApi_cublasLtMatrixLayoutSetAttribute, matLayout, static_cast<uint32_t>(attr), buf,
                        sizeInBytes);
}

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref) {
    if (!pref) return CUBLAS_STATUS_INVALID_VALUE;
    const uint64_t id = NewObjectId();
    ArgWriter in;
    in.put_u64(id);
    const cublasStatus_t status = Post(kApi_cublasLtMatmulPreferenceCreate, in);
    if (status == CUBLAS_STATUS_SUCCESS) *pref = reinterpret_cast<cublasLtMatmulPreference_t>(id);
    return status;
}

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    return Destroy(kApi_cublasLtMatmulPreferenceDestroy, pref);
}

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void* buf, size_t sizeInBytes) {
    return SetAttribute(kApi_cublasLtMatmulPreferenceSetAttribute, pref, static_cast<uint32_t>(attr), buf,
                        sizeInBytes);
}

// ------------------------------------------------------- heuristic, matmul

// Wire: handle, desc, A/B/C/D layouts, preference (u64 each), requested
// count (i32). Reply: returned count (i32), then that many results as they
// are laid out in memory (the client and server share cublasLt.h).
TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount) {
    if (requestedAlgoCount <= 0 || !heuristicResultsArray || !returnAlgoCount) return CUBLAS_STATUS_INVALID_VALUE;
    const int requested = std::min(requestedAlgoCount, kMaxHeuristicResults);
    ArgWriter in;
    for (const void* object : {static_cast<const void*>(lightHandle), static_cast<const void*>(operationDesc),
                               static_cast<const void*>(Adesc), static_cast<const void*>(Bdesc),
                               static_cast<const void*>(Cdesc), static_cast<const void*>(Ddesc),
                               static_cast<const void*>(preference)}) {
        in.put_u64(reinterpret_cast<uint64_t>(object));
    }
    in.put(&requested, 4);
    Reply reply = Call(kApi_cublasLtMatmulAlgoGetHeuristic, in);
    const cublasStatus_t status = Finish(reply);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    try {
        ArgReader out(reply.out_args);
        int count = 0;
        out.get(&count, 4);
        if (count < 0 || count > requested) return CUBLAS_STATUS_INTERNAL_ERROR;
        out.get(heuristicResultsArray, sizeof(cublasLtMatmulHeuristicResult_t) * static_cast<size_t>(count));
        *returnAlgoCount = count;
    } catch (const std::exception&) {
        return CUBLAS_STATUS_INTERNAL_ERROR;  // malformed reply
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Fire-and-forget, like cublasSgemm: a failure surfaces at the next
// synchronizing CUDA call. Wire: handle, desc, alpha, A, Adesc, B, Bdesc,
// beta, C, Cdesc, D, Ddesc (u64 / scalar refs), algo (u32 present flag + its
// 64 bytes), workspace, workspace size, stream (u64).
TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtMatmul(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void* alpha, const void* A,
    cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta, const void* C,
    cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
    MatmulDesc desc{};
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_descs.find(reinterpret_cast<uint64_t>(computeDesc));
        if (it == g_descs.end()) return CUBLAS_STATUS_INVALID_VALUE;
        desc = it->second;
    }
    const size_t scalar = ScalarSize(desc.scale_type);
    const bool alpha_device = AlphaOnDevice(desc.pointer_mode);
    const bool beta_device = BetaOnDevice(desc.pointer_mode);
    if (scalar == 0 || !alpha || (beta_device && !beta)) return CUBLAS_STATUS_INVALID_VALUE;

    ArgWriter in;
    in.put_u64(reinterpret_cast<uint64_t>(lightHandle));
    in.put_u64(reinterpret_cast<uint64_t>(computeDesc));
    PutScalar(in, alpha, alpha_device, scalar);
    in.put_u64(reinterpret_cast<uint64_t>(A));
    in.put_u64(reinterpret_cast<uint64_t>(Adesc));
    in.put_u64(reinterpret_cast<uint64_t>(B));
    in.put_u64(reinterpret_cast<uint64_t>(Bdesc));
    PutScalar(in, beta, beta_device, scalar);
    in.put_u64(reinterpret_cast<uint64_t>(C));
    in.put_u64(reinterpret_cast<uint64_t>(Cdesc));
    in.put_u64(reinterpret_cast<uint64_t>(D));
    in.put_u64(reinterpret_cast<uint64_t>(Ddesc));
    in.put_u32(algo ? 1 : 0);
    if (algo) in.put(algo, sizeof(*algo));
    in.put_u64(reinterpret_cast<uint64_t>(workspace));
    in.put_u64(workspaceSizeInBytes);
    in.put_u64(reinterpret_cast<uint64_t>(stream));
    return Post(kApi_cublasLtMatmul, in);
}

// ------------------------------------------------------------ client-local

// The cuBLASLt this shim was built against (the server runs the same major
// version; CI pins the headers).
TUNA_CUBLASLT_EXPORT size_t CUBLASWINAPI cublasLtGetVersion(void) { return CUBLAS_VERSION; }
TUNA_CUBLASLT_EXPORT size_t CUBLASWINAPI cublasLtGetCudartVersion(void) { return CUDART_VERSION; }

TUNA_CUBLASLT_EXPORT cublasStatus_t CUBLASWINAPI cublasLtGetProperty(libraryPropertyType type, int* value) {
    if (!value) return CUBLAS_STATUS_INVALID_VALUE;
    switch (type) {
    case MAJOR_VERSION: *value = CUBLAS_VER_MAJOR; break;
    case MINOR_VERSION: *value = CUBLAS_VER_MINOR; break;
    case PATCH_LEVEL: *value = CUBLAS_VER_PATCH; break;
    default: return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

TUNA_CUBLASLT_EXPORT const char* CUBLASWINAPI cublasLtGetStatusName(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

TUNA_CUBLASLT_EXPORT const char* CUBLASWINAPI cublasLtGetStatusString(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "the library was not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "the resource allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "an invalid numeric value was used as an argument";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "an absent device architectural feature is required";
    case CUBLAS_STATUS_MAPPING_ERROR: return "an access to GPU memory space failed";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "the GPU program failed to execute";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "an internal operation failed";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "the functionality requested is not supported";
    case CUBLAS_STATUS_LICENSE_ERROR: return "an error was detected when checking the current licensing";
    default: return "unknown error";
    }
}

// Marker the server checks before loading a "real" libcublasLt.
TUNA_CUBLASLT_EXPORT int tuna_cublaslt_shim(void) { return 1; }

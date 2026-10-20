// Tuna's replacement libcudnn (Phase 3g): every remoted export, by hand.
//
// What PyTorch uses (measured with tools/cudnn_trace on a ResNet-18 training
// loop):
//   * Convolutions go through the backend ("graph") API. Building an
//     execution plan creates dozens of descriptors (tensors, the convolution,
//     the operation, the graph, heuristics, engine configs, knobs), sets their
//     attributes one by one, finalizes them and reads results back with
//     GetAttribute. PyTorch caches the plans, so in steady state each
//     convolution is one variant pack (create, three attributes, finalize)
//     plus one execute.
//   * Batch norm goes through the legacy API with legacy tensor descriptors,
//     and asks for its workspace and reserve sizes on every call.
//
// As with cuBLASLt (cublaslt_client.cpp), the client names the objects: a
// create call allocates an id here and is fire-and-forget like set-attribute
// and destroy. The server maps ids to the real objects (dispatch_cudnn.cpp)
// and refuses ids it does not hold, so a failed create surfaces at the next
// call that uses the object. What still waits for the server:
//   * GetAttribute (the caller needs the values);
//   * Finalize of everything but a variant pack: the cuDNN frontend tries
//     engine configs in turn and needs each plan's real status. A variant
//     pack only bundles pointers, so its finalize is posted, and a training
//     step's convolutions cost no round trip at all;
//   * the batch-norm size queries, answered from a cache after the first
//     time: a size is a pure function of the handle, the mode and the tensor
//     descriptors' contents, which the client mirrors.
//
// Not supported, and refused here before anything is sent: by-value tensors
// (CUDNN_ATTR_TENSOR_IS_BY_VALUE), whose "data pointer" is host memory that
// cuDNN would read in the server process; activation descriptors.

#include <cudnn.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

#include "cuda_api_ids.h"
#include "tuna/cuda/bridge.h"
#include "tuna/cuda/cudnn_client.h"
#include "tuna/cuda/marshal.h"

// The shared element-size table (cudnn_client.h) is numbered by hand.
static_assert(CUDNN_TYPE_HANDLE == tuna::cudnn::kTypeHandle && CUDNN_TYPE_VOID_PTR == tuna::cudnn::kTypeVoidPtr &&
              CUDNN_TYPE_BACKEND_DESCRIPTOR == tuna::cudnn::kTypeDescriptor);
static_assert(CUDNN_TYPE_BOOLEAN == 2 && CUDNN_TYPE_INT64 == 3 && CUDNN_TYPE_DOUBLE == 5 && CUDNN_TYPE_CHAR == 24 &&
              CUDNN_TYPE_FRACTION == 26 && CUDNN_TYPE_RNG_DISTRIBUTION == 29);
static_assert(sizeof(cudnnFraction_t) == 16 && sizeof(bool) == 1 && sizeof(cudnnDataType_t) == 4 &&
              sizeof(cudnnBackendDescriptor_t) == 8);

namespace tuna::cudnn {

namespace {

constexpr int32_t kCudaErrorNoDevice = 100;
constexpr int32_t kCudaErrorNotSupported = 801;

// Object ids: an address range no allocation uses (and not cuBLASLt's), one
// 16-byte slot each. Only ever compared, never dereferenced.
constexpr uint64_t kObjectBase = 0x7e7e00000000ull;
std::atomic<uint64_t> g_next_object{0};

uint64_t NewObjectId() { return kObjectBase + (++g_next_object << 4); }

// A legacy tensor descriptor as the client last set it.
struct Tensor {
    bool set = false;
    int32_t type = 0;
    int nb_dims = 0;
    int dims[CUDNN_DIM_MAX] = {};
    int strides[CUDNN_DIM_MAX] = {};
};

std::mutex g_mutex;  // guards everything below
std::unordered_map<uint64_t, int32_t> g_backend;  // backend descriptor id -> cudnnBackendDescriptorType_t
std::unordered_map<uint64_t, Tensor> g_tensors;
std::unordered_map<uint64_t, uint64_t> g_streams;  // cudnnHandle_t -> its stream
std::unordered_map<std::string, uint64_t> g_sizes;  // batch-norm size queries already answered
std::unordered_map<uint32_t, size_t> g_versions;    // version queries already answered

bool LogEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("TUNA_LOG");
        return value && *value;
    }();
    return enabled;
}

void LogOnce(const std::string& message) {
    if (!LogEnabled()) return;
    static std::mutex mutex;
    static std::set<std::string> reported;
    std::lock_guard<std::mutex> lock(mutex);
    if (reported.insert(message).second) std::fprintf(stderr, "tuna: %s\n", message.c_str());
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

cudnnStatus_t Post(uint32_t api_id, const cudart::ArgWriter& in) {
    const int32_t queued = tuna_cudart_call(api_id, in.str().data(), in.str().size(), nullptr, 0, kTunaPost, nullptr);
    if (queued == 0) return CUDNN_STATUS_SUCCESS;
    return queued == kCudaErrorNoDevice ? CUDNN_STATUS_NOT_INITIALIZED : CUDNN_STATUS_INTERNAL_ERROR;
}

// Connection-level results (no server, a server without cuDNN) as statuses.
// cuDNN's own statuses are 0 or >= 1000, so they cannot collide with these.
cudnnStatus_t Finish(const cudart::Reply& reply) {
    if (reply.transport_error) return CUDNN_STATUS_INTERNAL_ERROR;
    if (reply.result == kCudaErrorNoDevice) return CUDNN_STATUS_NOT_INITIALIZED;
    if (reply.result == kCudaErrorNotSupported) return CUDNN_STATUS_NOT_SUPPORTED;
    return static_cast<cudnnStatus_t>(reply.result);
}

uint64_t Id(const void* object) { return reinterpret_cast<uint64_t>(object); }

// A version query's answer: the server's library, asked once (it cannot
// change during a session). Without a server, the build's own value.
size_t Version(uint32_t api_id, size_t fallback) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (auto it = g_versions.find(api_id); it != g_versions.end()) return it->second;
    }
    cudart::Reply reply = Call(api_id, cudart::ArgWriter{});
    if (Finish(reply) != CUDNN_STATUS_SUCCESS) return fallback;
    try {
        cudart::ArgReader out(reply.out_args);
        const size_t value = out.get_u64();
        std::lock_guard<std::mutex> lock(g_mutex);
        g_versions[api_id] = value;
        return value;
    } catch (const std::exception&) {
        return fallback;  // malformed reply
    }
}

// Appends a tensor descriptor's contents to a size-cache key; false if the
// descriptor is not one the client knows (the query then goes to the server,
// which decides, and is not cached). Called with g_mutex held.
bool AppendTensorKey(std::string& key, const void* desc) {
    if (!desc) {
        key += 'n';
        return true;
    }
    auto it = g_tensors.find(Id(desc));
    if (it == g_tensors.end() || !it->second.set) return false;
    const Tensor& t = it->second;
    key += 't';
    key.append(reinterpret_cast<const char*>(&t.type), sizeof(t.type));
    key.append(reinterpret_cast<const char*>(&t.nb_dims), sizeof(t.nb_dims));
    key.append(reinterpret_cast<const char*>(t.dims), sizeof(int) * static_cast<size_t>(t.nb_dims));
    key.append(reinterpret_cast<const char*>(t.strides), sizeof(int) * static_cast<size_t>(t.nb_dims));
    return true;
}

// The batch-norm size queries share one wire format: handle, mode, ops, the
// tensor descriptors (u64 ids, 0 for null), the activation descriptor.
// Reply: the size (u64).
cudnnStatus_t SizeQuery(uint32_t api_id, cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t ops,
                        std::initializer_list<const void*> tensors, const void* activation, size_t* size) {
    if (!size) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    std::string key;
    bool cacheable = activation == nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        key.append(reinterpret_cast<const char*>(&api_id), sizeof(api_id));
        const uint64_t h = Id(handle);
        key.append(reinterpret_cast<const char*>(&h), sizeof(h));
        key.append(reinterpret_cast<const char*>(&mode), sizeof(mode));
        key.append(reinterpret_cast<const char*>(&ops), sizeof(ops));
        for (const void* t : tensors) cacheable = cacheable && AppendTensorKey(key, t);
        if (cacheable) {
            if (auto it = g_sizes.find(key); it != g_sizes.end()) {
                *size = static_cast<size_t>(it->second);
                return CUDNN_STATUS_SUCCESS;
            }
        }
    }
    cudart::ArgWriter in;
    in.put_u64(Id(handle));
    in.put_u32(static_cast<uint32_t>(mode));
    in.put_u32(static_cast<uint32_t>(ops));
    for (const void* t : tensors) in.put_u64(Id(t));
    in.put_u64(Id(activation));
    cudart::Reply reply = Call(api_id, in);
    const cudnnStatus_t status = Finish(reply);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    uint64_t value = 0;
    try {
        cudart::ArgReader out(reply.out_args);
        value = out.get_u64();
    } catch (const std::exception&) {
        return CUDNN_STATUS_INTERNAL_ERROR;  // malformed reply
    }
    *size = static_cast<size_t>(value);
    if (cacheable) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_sizes.size() < kMaxCachedSizes) g_sizes.emplace(std::move(key), value);
    }
    return CUDNN_STATUS_SUCCESS;
}

// Width of the batch-norm scalars (alpha/beta): double for double tensors,
// float otherwise, as cuDNN specifies; 0 if the descriptor is unknown.
size_t ScalarWidth(const void* x_desc) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_tensors.find(Id(x_desc));
    if (it == g_tensors.end() || !it->second.set) return 0;
    return it->second.type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
}

void PutDouble(cudart::ArgWriter& in, double value) { in.put(&value, sizeof(value)); }

}  // namespace

int Unsupported(const char* name) {
    LogOnce(std::string(name) + " is not supported by Tuna yet");
    return CUDNN_STATUS_NOT_SUPPORTED;
}

void ResetStateForTesting() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_backend.clear();
    g_tensors.clear();
    g_streams.clear();
    g_sizes.clear();
    g_versions.clear();
}

}  // namespace tuna::cudnn

using namespace tuna::cudart;  // kApi_* ids, ArgWriter
using namespace tuna::cudnn;

// ------------------------------------------------------------------ handles

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t* handle) {
    if (!handle) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    Reply reply = Call(kApi_cudnnCreate, ArgWriter{});
    const cudnnStatus_t status = Finish(reply);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    uint64_t value = 0;
    try {
        ArgReader out(reply.out_args);
        value = out.get_u64();
    } catch (const std::exception&) {
        return CUDNN_STATUS_INTERNAL_ERROR;  // malformed reply
    }
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_streams[value] = 0;
    }
    *handle = reinterpret_cast<cudnnHandle_t>(value);
    return CUDNN_STATUS_SUCCESS;
}

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_streams.erase(Id(handle));
    }
    ArgWriter in;
    in.put_u64(Id(handle));
    return Finish(Call(kApi_cudnnDestroy, in));
}

// Called before every cuDNN operation by PyTorch; posted, and mirrored so
// cudnnGetStream needs no round trip.
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream) {
    ArgWriter in;
    in.put_u64(Id(handle));
    in.put_u64(reinterpret_cast<uint64_t>(stream));
    const cudnnStatus_t status = Post(kApi_cudnnSetStream, in);
    if (status == CUDNN_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (auto it = g_streams.find(Id(handle)); it != g_streams.end()) it->second = reinterpret_cast<uint64_t>(stream);
    }
    return status;
}

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t* stream) {
    if (!stream) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_streams.find(Id(handle));
    if (it == g_streams.end()) return CUDNN_STATUS_BAD_PARAM;
    *stream = reinterpret_cast<cudaStream_t>(it->second);
    return CUDNN_STATUS_SUCCESS;
}

// ----------------------------------------------------- versions and errors

TUNA_CUDNN_EXPORT size_t CUDNNWINAPI cudnnGetVersion(void) { return Version(kApi_cudnnGetVersion, CUDNN_VERSION); }
TUNA_CUDNN_EXPORT size_t CUDNNWINAPI cudnnGetMaxDeviceVersion(void) {
    return Version(kApi_cudnnGetMaxDeviceVersion, 0);
}
TUNA_CUDNN_EXPORT size_t CUDNNWINAPI cudnnGetCudartVersion(void) {
    return Version(kApi_cudnnGetCudartVersion, CUDART_VERSION);
}

// From the server's version, so that it agrees with cudnnGetVersion
// (cuDNN 9 encodes major * 10000 + minor * 100 + patch).
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType type, int* value) {
    if (!value) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    const auto version = static_cast<int>(cudnnGetVersion());
    switch (type) {
    case MAJOR_VERSION: *value = version / 10000; break;
    case MINOR_VERSION: *value = version % 10000 / 100; break;
    case PATCH_LEVEL: *value = version % 100; break;
    default: return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
}

// The shim and the server's library are one consistent unit to the caller.
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnGraphVersionCheck(void) { return CUDNN_STATUS_SUCCESS; }
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnOpsVersionCheck(void) { return CUDNN_STATUS_SUCCESS; }
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnCnnVersionCheck(void) { return CUDNN_STATUS_SUCCESS; }
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnAdvVersionCheck(void) { return CUDNN_STATUS_SUCCESS; }

namespace {
#include "cudnn_status_names.inc"
}  // namespace

TUNA_CUDNN_EXPORT const char* CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status) {
    for (const auto& entry : kCudnnStatusNames) {
        if (entry.value == static_cast<int>(status)) return entry.name;
    }
    return "CUDNN_STATUS_UNKNOWN";
}

// The server's thread-local "last error" message (PyTorch prints it when a
// call fails). Only called on error paths, so a round trip is fine.
TUNA_CUDNN_EXPORT void CUDNNWINAPI cudnnGetLastErrorString(char* message, size_t max_size) {
    if (!message || max_size == 0) return;
    message[0] = '\0';
    ArgWriter in;
    in.put_u64(std::min<uint64_t>(max_size, kMaxErrorStringBytes));
    Reply reply = Call(kApi_cudnnGetLastErrorString, in);
    if (Finish(reply) != CUDNN_STATUS_SUCCESS) return;
    try {
        ArgReader out(reply.out_args);
        const std::string text = out.get_string();
        const size_t n = std::min(text.size(), max_size - 1);
        std::memcpy(message, text.data(), n);
        message[n] = '\0';
    } catch (const std::exception&) {
        // malformed reply: leave the message empty
    }
}

// --------------------------------------------------- legacy tensor descriptors

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* desc) {
    if (!desc) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    const uint64_t id = NewObjectId();
    ArgWriter in;
    in.put_u64(id);
    const cudnnStatus_t status = Post(kApi_cudnnCreateTensorDescriptor, in);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_tensors[id] = Tensor{};
    }
    *desc = reinterpret_cast<cudnnTensorDescriptor_t>(id);
    return CUDNN_STATUS_SUCCESS;
}

// Wire: id, data type, nbDims (u32 each), dims, strides (nbDims i32 each).
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t desc,
                                                                       cudnnDataType_t dataType, int nbDims,
                                                                       const int dimA[], const int strideA[]) {
    if (!desc || !dimA || !strideA) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    if (nbDims <= 0 || nbDims > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
    const size_t bytes = sizeof(int) * static_cast<size_t>(nbDims);
    ArgWriter in;
    in.put_u64(Id(desc));
    in.put_u32(static_cast<uint32_t>(dataType));
    in.put_u32(static_cast<uint32_t>(nbDims));
    in.put(dimA, bytes);
    in.put(strideA, bytes);
    const cudnnStatus_t status = Post(kApi_cudnnSetTensorNdDescriptor, in);
    if (status == CUDNN_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (auto it = g_tensors.find(Id(desc)); it != g_tensors.end()) {
            Tensor& t = it->second;
            t.set = true;
            t.type = static_cast<int32_t>(dataType);
            t.nb_dims = nbDims;
            std::memcpy(t.dims, dimA, bytes);
            std::memcpy(t.strides, strideA, bytes);
        }
    }
    return status;
}

// Answered from the mirror: the client set every value itself.
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t desc,
                                                                       int nbDimsRequested,
                                                                       cudnnDataType_t* dataType, int* nbDims,
                                                                       int dimA[], int strideA[]) {
    if (nbDimsRequested < 0) return CUDNN_STATUS_BAD_PARAM;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_tensors.find(Id(desc));
    if (it == g_tensors.end() || !it->second.set) return CUDNN_STATUS_BAD_PARAM;
    const Tensor& t = it->second;
    if (dataType) *dataType = static_cast<cudnnDataType_t>(t.type);
    if (nbDims) *nbDims = t.nb_dims;
    const int n = std::min(nbDimsRequested, t.nb_dims);
    for (int i = 0; i < n; ++i) {
        if (dimA) dimA[i] = t.dims[i];
        if (strideA) strideA[i] = t.strides[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t desc) {
    if (!desc) return CUDNN_STATUS_SUCCESS;  // like cuDNN, destroying null is a no-op
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_tensors.erase(Id(desc));
    }
    ArgWriter in;
    in.put_u64(Id(desc));
    return Post(kApi_cudnnDestroyTensorDescriptor, in);
}

// ----------------------------------------------------------- batch norm

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t* sizeInBytes) {
    return SizeQuery(kApi_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize, handle, mode, bnOps,
                     {xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc}, activationDesc, sizeInBytes);
}

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t* sizeInBytes) {
    return SizeQuery(kApi_cudnnGetBatchNormalizationBackwardExWorkspaceSize, handle, mode, bnOps,
                     {xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc}, activationDesc, sizeInBytes);
}

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes) {
    return SizeQuery(kApi_cudnnGetBatchNormalizationTrainingExReserveSpaceSize, handle, mode, bnOps, {xDesc},
                     activationDesc, sizeInBytes);
}

// Fire-and-forget, like a kernel launch. Wire: handle, mode, ops; alpha, beta
// (host scalars, as wide as xDesc's data type says); then descriptors (u64
// ids), device pointers (u64), doubles and sizes in declaration order.
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xDesc, const void* xData, const cudnnTensorDescriptor_t zDesc, const void* zData,
    const cudnnTensorDescriptor_t yDesc, void* yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* bnScale, const void* bnBias, double exponentialAverageFactor, void* resultRunningMean,
    void* resultRunningVariance, double epsilon, void* resultSaveMean, void* resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    const size_t width = ScalarWidth(xDesc);
    if (!alpha || !beta) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    if (width == 0) return CUDNN_STATUS_BAD_PARAM;
    ArgWriter in;
    in.put_u64(Id(handle));
    in.put_u32(static_cast<uint32_t>(mode));
    in.put_u32(static_cast<uint32_t>(bnOps));
    PutHostScalar(in, alpha, width);
    PutHostScalar(in, beta, width);
    for (const void* p : {static_cast<const void*>(xDesc), xData, static_cast<const void*>(zDesc), zData,
                          static_cast<const void*>(yDesc), static_cast<const void*>(yData),
                          static_cast<const void*>(bnScaleBiasMeanVarDesc), bnScale, bnBias}) {
        in.put_u64(Id(p));
    }
    PutDouble(in, exponentialAverageFactor);
    in.put_u64(Id(resultRunningMean));
    in.put_u64(Id(resultRunningVariance));
    PutDouble(in, epsilon);
    in.put_u64(Id(resultSaveMean));
    in.put_u64(Id(resultSaveInvVariance));
    in.put_u64(Id(activationDesc));
    in.put_u64(Id(workspace));
    in.put_u64(workSpaceSizeInBytes);
    in.put_u64(Id(reserveSpace));
    in.put_u64(reserveSpaceSizeInBytes);
    return Post(kApi_cudnnBatchNormalizationForwardTrainingEx, in);
}

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void* alphaDataDiff,
    const void* betaDataDiff, const void* alphaParamDiff, const void* betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void* xData, const cudnnTensorDescriptor_t yDesc, const void* yData,
    const cudnnTensorDescriptor_t dyDesc, const void* dyData, const cudnnTensorDescriptor_t dzDesc, void* dzData,
    const cudnnTensorDescriptor_t dxDesc, void* dxData, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void* bnScaleData, const void* bnBiasData, void* dBnScaleData, void* dBnBiasData, double epsilon,
    const void* savedMean, const void* savedInvVariance, cudnnActivationDescriptor_t activationDesc, void* workSpace,
    size_t workSpaceSizeInBytes, void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    const size_t width = ScalarWidth(xDesc);
    if (!alphaDataDiff || !betaDataDiff || !alphaParamDiff || !betaParamDiff) {
        return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    }
    if (width == 0) return CUDNN_STATUS_BAD_PARAM;
    ArgWriter in;
    in.put_u64(Id(handle));
    in.put_u32(static_cast<uint32_t>(mode));
    in.put_u32(static_cast<uint32_t>(bnOps));
    for (const void* s : {alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff}) PutHostScalar(in, s, width);
    for (const void* p : {static_cast<const void*>(xDesc), xData, static_cast<const void*>(yDesc), yData,
                          static_cast<const void*>(dyDesc), dyData, static_cast<const void*>(dzDesc),
                          static_cast<const void*>(dzData), static_cast<const void*>(dxDesc),
                          static_cast<const void*>(dxData), static_cast<const void*>(dBnScaleBiasDesc), bnScaleData,
                          bnBiasData, static_cast<const void*>(dBnScaleData),
                          static_cast<const void*>(dBnBiasData)}) {
        in.put_u64(Id(p));
    }
    PutDouble(in, epsilon);
    in.put_u64(Id(savedMean));
    in.put_u64(Id(savedInvVariance));
    in.put_u64(Id(activationDesc));
    in.put_u64(Id(workSpace));
    in.put_u64(workSpaceSizeInBytes);
    in.put_u64(Id(reserveSpace));
    in.put_u64(reserveSpaceSizeInBytes);
    return Post(kApi_cudnnBatchNormalizationBackwardEx, in);
}

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t yDesc, void* y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias,
    const void* estimatedMean, const void* estimatedVariance, double epsilon) {
    const size_t width = ScalarWidth(xDesc);
    if (!alpha || !beta) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    if (width == 0) return CUDNN_STATUS_BAD_PARAM;
    ArgWriter in;
    in.put_u64(Id(handle));
    in.put_u32(static_cast<uint32_t>(mode));
    PutHostScalar(in, alpha, width);
    PutHostScalar(in, beta, width);
    for (const void* p : {static_cast<const void*>(xDesc), x, static_cast<const void*>(yDesc),
                          static_cast<const void*>(y), static_cast<const void*>(bnScaleBiasMeanVarDesc), bnScale,
                          bnBias, estimatedMean, estimatedVariance}) {
        in.put_u64(Id(p));
    }
    PutDouble(in, epsilon);
    return Post(kApi_cudnnBatchNormalizationForwardInference, in);
}

// -------------------------------------------------------------- backend API

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptorType,
                                                                         cudnnBackendDescriptor_t* descriptor) {
    if (!descriptor) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    const uint64_t id = NewObjectId();
    ArgWriter in;
    in.put_u64(id);
    in.put_u32(static_cast<uint32_t>(descriptorType));
    const cudnnStatus_t status = Post(kApi_cudnnBackendCreateDescriptor, in);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_backend[id] = static_cast<int32_t>(descriptorType);
    }
    *descriptor = reinterpret_cast<cudnnBackendDescriptor_t>(id);
    return CUDNN_STATUS_SUCCESS;
}

TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor) {
    if (!descriptor) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_backend.erase(Id(descriptor));
    }
    ArgWriter in;
    in.put_u64(Id(descriptor));
    return Post(kApi_cudnnBackendDestroyDescriptor, in);
}

// Wire: id, attribute name, attribute type (u32 each), element count (u64),
// then the elements exactly as they are in memory. Descriptors are client
// ids and handles / device addresses server values, all 8 bytes, which the
// server translates or checks (dispatch_cudnn.cpp).
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                                                                     cudnnBackendAttributeName_t attributeName,
                                                                     cudnnBackendAttributeType_t attributeType,
                                                                     int64_t elementCount,
                                                                     const void* arrayOfElements) {
    if (!descriptor) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    const int64_t size = BackendElementSize(static_cast<uint32_t>(attributeType));
    if (size == 0) return CUDNN_STATUS_NOT_SUPPORTED;
    if (elementCount < 0 || elementCount > kMaxAttributeElements ||
        static_cast<uint64_t>(elementCount * size) > kMaxAttributeBytes) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (elementCount > 0 && !arrayOfElements) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    if (attributeName == CUDNN_ATTR_TENSOR_IS_BY_VALUE && attributeType == CUDNN_TYPE_BOOLEAN && elementCount > 0 &&
        *static_cast<const bool*>(arrayOfElements)) {
        LogOnce("cuDNN by-value tensors (CUDNN_ATTR_TENSOR_IS_BY_VALUE) are not supported by Tuna");
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    ArgWriter in;
    in.put_u64(Id(descriptor));
    in.put_u32(static_cast<uint32_t>(attributeName));
    in.put_u32(static_cast<uint32_t>(attributeType));
    in.put_u64(static_cast<uint64_t>(elementCount));
    if (elementCount > 0) in.put(arrayOfElements, static_cast<size_t>(elementCount * size));
    return Post(kApi_cudnnBackendSetAttribute, in);
}

// Waits for the server. Wire: id, name, type (u32), requested count (u64),
// whether an array was passed (u32), and for descriptor arrays the ids the
// caller created for cuDNN to fill. Reply: the element count (u64), then for
// value types min(count, requested) elements.
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnBackendGetAttribute(cudnnBackendDescriptor_t const descriptor,
                                                                     cudnnBackendAttributeName_t attributeName,
                                                                     cudnnBackendAttributeType_t attributeType,
                                                                     int64_t requestedElementCount,
                                                                     int64_t* elementCount, void* arrayOfElements) {
    if (!descriptor) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    const int64_t size = BackendElementSize(static_cast<uint32_t>(attributeType));
    if (size == 0) return CUDNN_STATUS_NOT_SUPPORTED;
    if (requestedElementCount < 0 || requestedElementCount > kMaxAttributeElements ||
        static_cast<uint64_t>(requestedElementCount * size) > kMaxAttributeBytes) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const bool has_array = arrayOfElements != nullptr && requestedElementCount > 0;
    const bool descriptors = attributeType == CUDNN_TYPE_BACKEND_DESCRIPTOR;
    ArgWriter in;
    in.put_u64(Id(descriptor));
    in.put_u32(static_cast<uint32_t>(attributeName));
    in.put_u32(static_cast<uint32_t>(attributeType));
    in.put_u64(static_cast<uint64_t>(requestedElementCount));
    in.put_u32(has_array ? 1 : 0);
    if (has_array && descriptors) in.put(arrayOfElements, static_cast<size_t>(requestedElementCount * size));
    Reply reply = Call(kApi_cudnnBackendGetAttribute, in);
    const cudnnStatus_t status = Finish(reply);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    try {
        ArgReader out(reply.out_args);
        const auto count = static_cast<int64_t>(out.get_u64());
        if (has_array && !descriptors) {
            const int64_t n = std::clamp<int64_t>(count, 0, requestedElementCount);
            out.get(arrayOfElements, static_cast<size_t>(n * size));
        }
        if (out.remaining() != 0) return CUDNN_STATUS_INTERNAL_ERROR;
        if (elementCount) *elementCount = count;
    } catch (const std::exception&) {
        return CUDNN_STATUS_INTERNAL_ERROR;  // malformed reply
    }
    return CUDNN_STATUS_SUCCESS;
}

// A variant pack's finalize is posted (see the top of this file); every
// other descriptor's waits for its real status.
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor) {
    if (!descriptor) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    bool variant_pack = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_backend.find(Id(descriptor));
        variant_pack = it != g_backend.end() && it->second == CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
    }
    ArgWriter in;
    in.put_u64(Id(descriptor));
    if (variant_pack) return Post(kApi_cudnnBackendFinalize, in);
    return Finish(Call(kApi_cudnnBackendFinalize, in));
}

// Fire-and-forget, like a kernel launch. Wire: handle, plan id, variant pack id.
TUNA_CUDNN_EXPORT cudnnStatus_t CUDNNWINAPI cudnnBackendExecute(cudnnHandle_t handle,
                                                                cudnnBackendDescriptor_t executionPlan,
                                                                cudnnBackendDescriptor_t variantPack) {
    ArgWriter in;
    in.put_u64(Id(handle));
    in.put_u64(Id(executionPlan));
    in.put_u64(Id(variantPack));
    return Post(kApi_cudnnBackendExecute, in);
}

// Marker the server checks before loading a "real" libcudnn.
TUNA_CUDNN_EXPORT int tuna_cudnn_shim(void) { return 1; }

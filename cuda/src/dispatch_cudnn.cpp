// Server-side cuDNN dispatch (Phase 3g). All hand-written: see
// cuda/src/cudnn_client.cpp for why the client names the descriptor objects.
//
// What the server enforces, since every value here comes off the wire:
//   * objects: a client id maps to the real descriptor this session created,
//     of the kind the call expects (a legacy tensor descriptor is never used
//     as a backend descriptor); unknown ids are refused. Which backend type a
//     descriptor is, cuDNN checks itself;
//   * handles: only the cudnnHandle_t values this session created, whether
//     passed as an argument or as a CUDNN_TYPE_HANDLE attribute;
//   * attribute values: an element type Tuna does not know is refused, and a
//     value is copied into an aligned buffer of exactly the size sent, so
//     cuDNN never reads past what the client provided. Descriptor-typed
//     elements are translated from ids to the real objects;
//   * host memory: cuDNN dereferences two kinds of pointers in this process,
//     scalars and by-value tensors. Scalars are copied into local storage
//     (ScalarRef, never a client address); by-value tensors are refused, and
//     CUDNN_TYPE_VOID_PTR (device addresses, never dereferenced here) is
//     accepted only by a variant pack;
//   * activation descriptors are not supported (none can be created), so
//     every activation argument must be null.
//
// Status codes: synchronous calls answer with the cudnnStatus_t; posted ones
// (creates, setters, destroys, variant-pack finalizes, executions, batch
// norm) report a failure as the closest cudaError_t, at the app's next
// synchronizing CUDA call.

#include "tuna/cuda/backend.h"

#include <cudnn.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "cuda_api_ids.h"
#include "dispatch_internal.h"
#include "tuna/cuda/cudnn_client.h"
#include "tuna/tunnel/rpc.h"

namespace tuna::cudart {

int32_t CudnnAsyncError(int status) {
    if (status == CUDNN_STATUS_SUCCESS) return cudaSuccess;
    switch (status / 1000 * 1000) {  // cuDNN 9's status categories
    case CUDNN_STATUS_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDNN_STATUS_BAD_PARAM: return cudaErrorInvalidValue;
    case CUDNN_STATUS_NOT_SUPPORTED: return cudaErrorNotSupported;
    case CUDNN_STATUS_EXECUTION_FAILED: return cudaErrorLaunchFailure;
    default:
        if (status == CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED ||
            status == CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED) {
            return cudaErrorMemoryAllocation;
        }
        return cudaErrorUnknown;
    }
}

namespace {

template <typename Fn>
Fn Dnn(CudaBackend& backend, const char* symbol) {
    return reinterpret_cast<Fn>(backend.Resolve(symbol));
}

double GetDouble(ArgReader& in) {
    double value = 0;
    in.get(&value, sizeof(value));
    return value;
}

void* Ptr(uint64_t value) { return reinterpret_cast<void*>(value); }

}  // namespace

bool CudaSession::DispatchCudnn(const RpcRequest& request, RpcResponse& resp) {
    const uint32_t api = request.api_id();
    const bool posted = (request.flags() & tunnel::kFlagAsync) != 0;
    auto finish = [&](cudnnStatus_t status) {
        resp.set_result(posted ? CudnnAsyncError(status) : static_cast<int32_t>(status));
    };
    auto unsupported = [&] { resp.set_result(posted ? kCudaErrorNotSupported : CUDNN_STATUS_NOT_SUPPORTED); };
    // The object behind `id` if it is of `kind` (nullptr otherwise).
    auto object = [&](uint64_t id, int kind) -> CudnnObject* {
        auto it = cudnn_objects_.find(id);
        return it != cudnn_objects_.end() && it->second.kind == kind ? &it->second : nullptr;
    };
    // A legacy tensor descriptor argument: 0 is null (cuDNN decides whether
    // that is allowed); any other value must be this session's descriptor.
    auto tensor = [&](uint64_t id, cudnnTensorDescriptor_t* out) {
        if (id == 0) {
            *out = nullptr;
            return true;
        }
        CudnnObject* o = object(id, kCudnnTensor);
        *out = o ? static_cast<cudnnTensorDescriptor_t>(o->ptr) : nullptr;
        return o != nullptr;
    };
    auto known_handle = [&](uint64_t handle) { return cudnn_handles_.count(Ptr(handle)) != 0; };

    try {
        ArgReader in(request.args());
        switch (api) {
        // ------------------------------------------------------------ handles
        case kApi_cudnnCreate: {
            if (in.remaining() != 0) return finish(CUDNN_STATUS_BAD_PARAM), true;
            auto create = Dnn<cudnnStatus_t (*)(cudnnHandle_t*)>(backend_, "cudnnCreate");
            if (!create) return unsupported(), true;
            cudnnHandle_t handle = nullptr;
            const cudnnStatus_t status = create(&handle);
            finish(status);
            if (status != CUDNN_STATUS_SUCCESS) return true;
            cudnn_handles_.insert(handle);
            ArgWriter out;
            out.put_u64(reinterpret_cast<uint64_t>(handle));
            resp.set_out_args(out.str());
            return true;
        }
        case kApi_cudnnDestroy: {
            const uint64_t handle = in.get_u64();
            if (in.remaining() != 0) return finish(CUDNN_STATUS_BAD_PARAM), true;
            if (!known_handle(handle)) return finish(CUDNN_STATUS_BAD_PARAM), true;
            auto destroy = Dnn<cudnnStatus_t (*)(cudnnHandle_t)>(backend_, "cudnnDestroy");
            if (!destroy) return unsupported(), true;
            finish(destroy(static_cast<cudnnHandle_t>(Ptr(handle))));
            cudnn_handles_.erase(Ptr(handle));  // gone whatever it returned
            return true;
        }
        case kApi_cudnnSetStream: {
            const uint64_t handle = in.get_u64();
            const uint64_t stream = in.get_u64();
            if (in.remaining() != 0 || !known_handle(handle)) return finish(CUDNN_STATUS_BAD_PARAM), true;
            auto set = Dnn<cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t)>(backend_, "cudnnSetStream");
            if (!set) return unsupported(), true;
            finish(set(static_cast<cudnnHandle_t>(Ptr(handle)), static_cast<cudaStream_t>(Ptr(stream))));
            return true;
        }

        // ------------------------------------------------ versions and errors
        case kApi_cudnnGetVersion:
        case kApi_cudnnGetMaxDeviceVersion:
        case kApi_cudnnGetCudartVersion: {
            if (in.remaining() != 0) return finish(CUDNN_STATUS_BAD_PARAM), true;
            const char* symbol = api == kApi_cudnnGetVersion            ? "cudnnGetVersion"
                                 : api == kApi_cudnnGetMaxDeviceVersion ? "cudnnGetMaxDeviceVersion"
                                                                        : "cudnnGetCudartVersion";
            auto query = Dnn<size_t (*)()>(backend_, symbol);
            if (!query) return unsupported(), true;
            ArgWriter out;
            out.put_u64(query());
            finish(CUDNN_STATUS_SUCCESS);
            resp.set_out_args(out.str());
            return true;
        }
        case kApi_cudnnGetLastErrorString: {
            const uint64_t max_size = in.get_u64();
            if (in.remaining() != 0 || max_size == 0 || max_size > tuna::cudnn::kMaxErrorStringBytes) {
                return finish(CUDNN_STATUS_BAD_PARAM), true;
            }
            auto get = Dnn<void (*)(char*, size_t)>(backend_, "cudnnGetLastErrorString");
            if (!get) return unsupported(), true;
            std::string message(max_size, '\0');
            get(message.data(), message.size());
            message.resize(strnlen(message.data(), message.size()));
            ArgWriter out;
            out.put_string(message);
            finish(CUDNN_STATUS_SUCCESS);
            resp.set_out_args(out.str());
            return true;
        }

        // --------------------------------------- legacy tensor descriptors
        case kApi_cudnnCreateTensorDescriptor: {
            const uint64_t id = in.get_u64();
            if (in.remaining() != 0 || id == 0 || cudnn_objects_.count(id)) {
                return finish(CUDNN_STATUS_BAD_PARAM), true;
            }
            auto create = Dnn<cudnnStatus_t (*)(cudnnTensorDescriptor_t*)>(backend_, "cudnnCreateTensorDescriptor");
            if (!create) return unsupported(), true;
            cudnnTensorDescriptor_t desc = nullptr;
            const cudnnStatus_t status = create(&desc);
            if (status == CUDNN_STATUS_SUCCESS) cudnn_objects_[id] = {kCudnnTensor, 0, desc};
            finish(status);
            return true;
        }
        case kApi_cudnnSetTensorNdDescriptor: {
            CudnnObject* desc = object(in.get_u64(), kCudnnTensor);
            const auto type = static_cast<cudnnDataType_t>(in.get_u32());
            const uint32_t nb_dims = in.get_u32();
            if (!desc || nb_dims == 0 || nb_dims > CUDNN_DIM_MAX) return finish(CUDNN_STATUS_BAD_PARAM), true;
            int dims[CUDNN_DIM_MAX];
            int strides[CUDNN_DIM_MAX];
            in.get(dims, sizeof(int) * nb_dims);
            in.get(strides, sizeof(int) * nb_dims);
            if (in.remaining() != 0) return finish(CUDNN_STATUS_BAD_PARAM), true;
            auto set = Dnn<cudnnStatus_t (*)(cudnnTensorDescriptor_t, cudnnDataType_t, int, const int*, const int*)>(
                backend_, "cudnnSetTensorNdDescriptor");
            if (!set) return unsupported(), true;
            finish(set(static_cast<cudnnTensorDescriptor_t>(desc->ptr), type, static_cast<int>(nb_dims), dims,
                       strides));
            return true;
        }
        case kApi_cudnnDestroyTensorDescriptor: {
            const uint64_t id = in.get_u64();
            CudnnObject* desc = object(id, kCudnnTensor);
            if (in.remaining() != 0 || !desc) return finish(CUDNN_STATUS_BAD_PARAM), true;
            auto destroy =
                Dnn<cudnnStatus_t (*)(cudnnTensorDescriptor_t)>(backend_, "cudnnDestroyTensorDescriptor");
            if (!destroy) return unsupported(), true;
            finish(destroy(static_cast<cudnnTensorDescriptor_t>(desc->ptr)));
            cudnn_objects_.erase(id);
            return true;
        }

        // --------------------------------------------------- batch-norm sizes
        case kApi_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize:
        case kApi_cudnnGetBatchNormalizationBackwardExWorkspaceSize:
        case kApi_cudnnGetBatchNormalizationTrainingExReserveSpaceSize: {
            const uint64_t handle = in.get_u64();
            const auto mode = static_cast<cudnnBatchNormMode_t>(in.get_u32());
            const auto ops = static_cast<cudnnBatchNormOps_t>(in.get_u32());
            const size_t count = api == kApi_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize ? 4
                                 : api == kApi_cudnnGetBatchNormalizationBackwardExWorkspaceSize      ? 6
                                                                                                      : 1;
            cudnnTensorDescriptor_t t[6] = {};
            bool ok = true;
            for (size_t i = 0; i < count; ++i) ok = tensor(in.get_u64(), &t[i]) && ok;
            const uint64_t activation = in.get_u64();
            if (in.remaining() != 0 || !ok) return finish(CUDNN_STATUS_BAD_PARAM), true;
            if (activation != 0) return finish(CUDNN_STATUS_NOT_SUPPORTED), true;
            if (!known_handle(handle)) return finish(CUDNN_STATUS_BAD_PARAM), true;
            auto h = static_cast<cudnnHandle_t>(Ptr(handle));
            size_t size = 0;
            cudnnStatus_t status;
            if (api == kApi_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize) {
                auto fn = Dnn<cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                                                cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                                                cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                                                cudnnActivationDescriptor_t, size_t*)>(
                    backend_, "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
                if (!fn) return unsupported(), true;
                status = fn(h, mode, ops, t[0], t[1], t[2], t[3], nullptr, &size);
            } else if (api == kApi_cudnnGetBatchNormalizationBackwardExWorkspaceSize) {
                auto fn = Dnn<cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                                                cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                                                cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                                                cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                                                cudnnActivationDescriptor_t, size_t*)>(
                    backend_, "cudnnGetBatchNormalizationBackwardExWorkspaceSize");
                if (!fn) return unsupported(), true;
                status = fn(h, mode, ops, t[0], t[1], t[2], t[3], t[4], t[5], nullptr, &size);
            } else {
                auto fn = Dnn<cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                                                cudnnActivationDescriptor_t, cudnnTensorDescriptor_t, size_t*)>(
                    backend_, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
                if (!fn) return unsupported(), true;
                status = fn(h, mode, ops, nullptr, t[0], &size);
            }
            finish(status);
            if (status != CUDNN_STATUS_SUCCESS) return true;
            ArgWriter out;
            out.put_u64(size);
            resp.set_out_args(out.str());
            return true;
        }

        // ----------------------------------------------------------- batch norm
        case kApi_cudnnBatchNormalizationForwardTrainingEx: {
            const uint64_t handle = in.get_u64();
            const auto mode = static_cast<cudnnBatchNormMode_t>(in.get_u32());
            const auto ops = static_cast<cudnnBatchNormOps_t>(in.get_u32());
            ScalarRef alpha(in);
            ScalarRef beta(in);
            cudnnTensorDescriptor_t x_desc, z_desc, y_desc, bn_desc;
            bool ok = tensor(in.get_u64(), &x_desc);
            const void* x = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &z_desc) && ok;
            const void* z = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &y_desc) && ok;
            void* y = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &bn_desc) && ok;
            const void* scale = Ptr(in.get_u64());
            const void* bias = Ptr(in.get_u64());
            const double factor = GetDouble(in);
            void* running_mean = Ptr(in.get_u64());
            void* running_var = Ptr(in.get_u64());
            const double epsilon = GetDouble(in);
            void* save_mean = Ptr(in.get_u64());
            void* save_inv_var = Ptr(in.get_u64());
            const uint64_t activation = in.get_u64();
            void* workspace = Ptr(in.get_u64());
            const uint64_t workspace_size = in.get_u64();
            void* reserve = Ptr(in.get_u64());
            const uint64_t reserve_size = in.get_u64();
            if (in.remaining() != 0 || !ok || alpha.on_device() || beta.on_device()) {
                return finish(CUDNN_STATUS_BAD_PARAM), true;
            }
            if (activation != 0) return finish(CUDNN_STATUS_NOT_SUPPORTED), true;
            if (!known_handle(handle)) return finish(CUDNN_STATUS_BAD_PARAM), true;
            using Fn = cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const void*,
                                         const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t,
                                         const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t,
                                         const void*, const void*, double, void*, void*, double, void*, void*,
                                         cudnnActivationDescriptor_t, void*, size_t, void*, size_t);
            auto fn = Dnn<Fn>(backend_, "cudnnBatchNormalizationForwardTrainingEx");
            if (!fn) return unsupported(), true;
            finish(fn(static_cast<cudnnHandle_t>(Ptr(handle)), mode, ops, alpha.ptr(), beta.ptr(), x_desc, x, z_desc,
                      z, y_desc, y, bn_desc, scale, bias, factor, running_mean, running_var, epsilon, save_mean,
                      save_inv_var, nullptr, workspace, workspace_size, reserve, reserve_size));
            return true;
        }
        case kApi_cudnnBatchNormalizationBackwardEx: {
            const uint64_t handle = in.get_u64();
            const auto mode = static_cast<cudnnBatchNormMode_t>(in.get_u32());
            const auto ops = static_cast<cudnnBatchNormOps_t>(in.get_u32());
            ScalarRef alpha_data(in);
            ScalarRef beta_data(in);
            ScalarRef alpha_param(in);
            ScalarRef beta_param(in);
            cudnnTensorDescriptor_t x_desc, y_desc, dy_desc, dz_desc, dx_desc, dbn_desc;
            bool ok = tensor(in.get_u64(), &x_desc);
            const void* x = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &y_desc) && ok;
            const void* y = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &dy_desc) && ok;
            const void* dy = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &dz_desc) && ok;
            void* dz = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &dx_desc) && ok;
            void* dx = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &dbn_desc) && ok;
            const void* scale = Ptr(in.get_u64());
            const void* bias = Ptr(in.get_u64());
            void* dscale = Ptr(in.get_u64());
            void* dbias = Ptr(in.get_u64());
            const double epsilon = GetDouble(in);
            const void* saved_mean = Ptr(in.get_u64());
            const void* saved_inv_var = Ptr(in.get_u64());
            const uint64_t activation = in.get_u64();
            void* workspace = Ptr(in.get_u64());
            const uint64_t workspace_size = in.get_u64();
            void* reserve = Ptr(in.get_u64());
            const uint64_t reserve_size = in.get_u64();
            if (in.remaining() != 0 || !ok || alpha_data.on_device() || beta_data.on_device() ||
                alpha_param.on_device() || beta_param.on_device()) {
                return finish(CUDNN_STATUS_BAD_PARAM), true;
            }
            if (activation != 0) return finish(CUDNN_STATUS_NOT_SUPPORTED), true;
            if (!known_handle(handle)) return finish(CUDNN_STATUS_BAD_PARAM), true;
            using Fn = cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const void*,
                                         const void*, const void*, const void*, cudnnTensorDescriptor_t, const void*,
                                         cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*,
                                         cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, void*,
                                         cudnnTensorDescriptor_t, const void*, const void*, void*, void*, double,
                                         const void*, const void*, cudnnActivationDescriptor_t, void*, size_t, void*,
                                         size_t);
            auto fn = Dnn<Fn>(backend_, "cudnnBatchNormalizationBackwardEx");
            if (!fn) return unsupported(), true;
            finish(fn(static_cast<cudnnHandle_t>(Ptr(handle)), mode, ops, alpha_data.ptr(), beta_data.ptr(),
                      alpha_param.ptr(), beta_param.ptr(), x_desc, x, y_desc, y, dy_desc, dy, dz_desc, dz, dx_desc,
                      dx, dbn_desc, scale, bias, dscale, dbias, epsilon, saved_mean, saved_inv_var, nullptr,
                      workspace, workspace_size, reserve, reserve_size));
            return true;
        }
        case kApi_cudnnBatchNormalizationForwardInference: {
            const uint64_t handle = in.get_u64();
            const auto mode = static_cast<cudnnBatchNormMode_t>(in.get_u32());
            ScalarRef alpha(in);
            ScalarRef beta(in);
            cudnnTensorDescriptor_t x_desc, y_desc, bn_desc;
            bool ok = tensor(in.get_u64(), &x_desc);
            const void* x = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &y_desc) && ok;
            void* y = Ptr(in.get_u64());
            ok = tensor(in.get_u64(), &bn_desc) && ok;
            const void* scale = Ptr(in.get_u64());
            const void* bias = Ptr(in.get_u64());
            const void* mean = Ptr(in.get_u64());
            const void* var = Ptr(in.get_u64());
            const double epsilon = GetDouble(in);
            if (in.remaining() != 0 || !ok || alpha.on_device() || beta.on_device()) {
                return finish(CUDNN_STATUS_BAD_PARAM), true;
            }
            if (!known_handle(handle)) return finish(CUDNN_STATUS_BAD_PARAM), true;
            using Fn = cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, const void*, const void*,
                                         cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*,
                                         cudnnTensorDescriptor_t, const void*, const void*, const void*, const void*,
                                         double);
            auto fn = Dnn<Fn>(backend_, "cudnnBatchNormalizationForwardInference");
            if (!fn) return unsupported(), true;
            finish(fn(static_cast<cudnnHandle_t>(Ptr(handle)), mode, alpha.ptr(), beta.ptr(), x_desc, x, y_desc, y,
                      bn_desc, scale, bias, mean, var, epsilon));
            return true;
        }

        // ---------------------------------------------------------- backend API
        case kApi_cudnnBackendCreateDescriptor: {
            const uint64_t id = in.get_u64();
            const auto type = static_cast<int32_t>(in.get_u32());
            if (in.remaining() != 0 || id == 0 || cudnn_objects_.count(id)) {
                return finish(CUDNN_STATUS_BAD_PARAM), true;
            }
            auto create = Dnn<cudnnStatus_t (*)(cudnnBackendDescriptorType_t, cudnnBackendDescriptor_t*)>(
                backend_, "cudnnBackendCreateDescriptor");
            if (!create) return unsupported(), true;
            cudnnBackendDescriptor_t desc = nullptr;
            const cudnnStatus_t status = create(static_cast<cudnnBackendDescriptorType_t>(type), &desc);
            if (status == CUDNN_STATUS_SUCCESS) cudnn_objects_[id] = {kCudnnBackend, type, desc};
            finish(status);
            return true;
        }
        case kApi_cudnnBackendDestroyDescriptor: {
            const uint64_t id = in.get_u64();
            CudnnObject* desc = object(id, kCudnnBackend);
            if (in.remaining() != 0 || !desc) return finish(CUDNN_STATUS_BAD_PARAM), true;
            auto destroy =
                Dnn<cudnnStatus_t (*)(cudnnBackendDescriptor_t)>(backend_, "cudnnBackendDestroyDescriptor");
            if (!destroy) return unsupported(), true;
            finish(destroy(desc->ptr));
            cudnn_objects_.erase(id);
            return true;
        }
        case kApi_cudnnBackendSetAttribute: {
            CudnnObject* desc = object(in.get_u64(), kCudnnBackend);
            const uint32_t name = in.get_u32();
            const uint32_t type = in.get_u32();
            const uint64_t count = in.get_u64();
            const int64_t size = tuna::cudnn::BackendElementSize(type);
            if (size == 0) return finish(CUDNN_STATUS_NOT_SUPPORTED), true;
            if (!desc || count > static_cast<uint64_t>(tuna::cudnn::kMaxAttributeElements) ||
                in.remaining() != count * static_cast<uint64_t>(size)) {
                return finish(CUDNN_STATUS_BAD_PARAM), true;
            }
            // 8-byte aligned, exactly the size sent (rounded up only for alignment).
            std::vector<uint64_t> buffer((in.remaining() + 7) / 8);
            in.get(buffer.data(), in.remaining());
            if (type == tuna::cudnn::kTypeDescriptor) {
                for (uint64_t& element : buffer) {
                    CudnnObject* target = object(element, kCudnnBackend);
                    if (!target) return finish(CUDNN_STATUS_BAD_PARAM), true;
                    element = reinterpret_cast<uint64_t>(target->ptr);
                }
            } else if (type == tuna::cudnn::kTypeHandle) {
                for (uint64_t element : buffer) {
                    if (!known_handle(element)) return finish(CUDNN_STATUS_BAD_PARAM), true;
                }
            } else if (type == tuna::cudnn::kTypeVoidPtr && desc->type != CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR) {
                return finish(CUDNN_STATUS_NOT_SUPPORTED), true;
            } else if (name == CUDNN_ATTR_TENSOR_IS_BY_VALUE && type == CUDNN_TYPE_BOOLEAN) {
                const auto* bytes = reinterpret_cast<const unsigned char*>(buffer.data());
                if (std::any_of(bytes, bytes + count, [](unsigned char b) { return b != 0; })) {
                    return finish(CUDNN_STATUS_NOT_SUPPORTED), true;
                }
            }
            auto set = Dnn<cudnnStatus_t (*)(cudnnBackendDescriptor_t, cudnnBackendAttributeName_t,
                                             cudnnBackendAttributeType_t, int64_t, const void*)>(
                backend_, "cudnnBackendSetAttribute");
            if (!set) return unsupported(), true;
            finish(set(desc->ptr, static_cast<cudnnBackendAttributeName_t>(name),
                       static_cast<cudnnBackendAttributeType_t>(type), static_cast<int64_t>(count),
                       count ? buffer.data() : nullptr));
            return true;
        }
        case kApi_cudnnBackendGetAttribute: {
            CudnnObject* desc = object(in.get_u64(), kCudnnBackend);
            const uint32_t name = in.get_u32();
            const uint32_t type = in.get_u32();
            const uint64_t requested = in.get_u64();
            const uint32_t has_array = in.get_u32();
            const int64_t size = tuna::cudnn::BackendElementSize(type);
            if (size == 0) return finish(CUDNN_STATUS_NOT_SUPPORTED), true;
            if (!desc || has_array > 1 || requested > static_cast<uint64_t>(tuna::cudnn::kMaxAttributeElements) ||
                (has_array && requested == 0)) {
                return finish(CUDNN_STATUS_BAD_PARAM), true;
            }
            const bool descriptors = type == tuna::cudnn::kTypeDescriptor;
            std::vector<uint64_t> buffer;
            if (has_array) buffer.resize((requested * static_cast<uint64_t>(size) + 7) / 8);
            if (has_array && descriptors) {
                // The descriptors the caller created for cuDNN to fill.
                for (uint64_t& element : buffer) {
                    CudnnObject* target = object(in.get_u64(), kCudnnBackend);
                    if (!target) return finish(CUDNN_STATUS_BAD_PARAM), true;
                    element = reinterpret_cast<uint64_t>(target->ptr);
                }
            }
            if (in.remaining() != 0) return finish(CUDNN_STATUS_BAD_PARAM), true;
            auto get = Dnn<cudnnStatus_t (*)(cudnnBackendDescriptor_t, cudnnBackendAttributeName_t,
                                             cudnnBackendAttributeType_t, int64_t, int64_t*, void*)>(
                backend_, "cudnnBackendGetAttribute");
            if (!get) return unsupported(), true;
            int64_t count = 0;
            const cudnnStatus_t status =
                get(desc->ptr, static_cast<cudnnBackendAttributeName_t>(name),
                    static_cast<cudnnBackendAttributeType_t>(type), static_cast<int64_t>(requested), &count,
                    has_array ? buffer.data() : nullptr);
            finish(status);
            if (status != CUDNN_STATUS_SUCCESS) return true;
            ArgWriter out;
            out.put_u64(static_cast<uint64_t>(count));
            if (has_array && !descriptors) {
                const int64_t n = std::clamp<int64_t>(count, 0, static_cast<int64_t>(requested));
                out.put(buffer.data(), static_cast<size_t>(n * size));
            }
            resp.set_out_args(out.str());
            return true;
        }
        case kApi_cudnnBackendFinalize: {
            CudnnObject* desc = object(in.get_u64(), kCudnnBackend);
            if (in.remaining() != 0 || !desc) return finish(CUDNN_STATUS_BAD_PARAM), true;
            auto fin = Dnn<cudnnStatus_t (*)(cudnnBackendDescriptor_t)>(backend_, "cudnnBackendFinalize");
            if (!fin) return unsupported(), true;
            finish(fin(desc->ptr));
            return true;
        }
        case kApi_cudnnBackendExecute: {
            const uint64_t handle = in.get_u64();
            CudnnObject* plan = object(in.get_u64(), kCudnnBackend);
            CudnnObject* pack = object(in.get_u64(), kCudnnBackend);
            if (in.remaining() != 0 || !plan || !pack || !known_handle(handle)) {
                return finish(CUDNN_STATUS_BAD_PARAM), true;
            }
            auto execute = Dnn<cudnnStatus_t (*)(cudnnHandle_t, cudnnBackendDescriptor_t, cudnnBackendDescriptor_t)>(
                backend_, "cudnnBackendExecute");
            if (!execute) return unsupported(), true;
            finish(execute(static_cast<cudnnHandle_t>(Ptr(handle)), plan->ptr, pack->ptr));
            return true;
        }

        default:
            return false;
        }
    } catch (const std::exception&) {
        // A short or garbled arg buffer.
        resp.clear_out_args();
        finish(CUDNN_STATUS_BAD_PARAM);
        return true;
    }
}

}  // namespace tuna::cudart

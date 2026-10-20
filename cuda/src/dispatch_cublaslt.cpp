// Server-side cuBLASLt dispatch (Phase 3f). All hand-written: see
// cuda/src/cublaslt_client.cpp for why the client names the descriptor
// objects.
//
// What the server enforces, since every value here comes off the wire:
//   * objects: a client id maps to the real descriptor this session created,
//     of the kind the call expects (a layout id is never used as a matmul
//     descriptor); unknown ids are refused;
//   * handles: only cuBLAS handles this session created (PyTorch passes those
//     as Lt handles) or handles from cublasLtCreate;
//   * algos: cublasLtMatmulAlgo_t is an opaque blob cuBLASLt interprets, so a
//     matmul only accepts an algo the heuristic handed out in this session
//     (or none);
//   * scalars: alpha/beta must be encoded the way the descriptor's pointer
//     mode says (host value vs device address), or cuBLASLt would read a
//     client-chosen address as host memory;
//   * attribute values are copied into a buffer of exactly the size sent, so
//     cuBLASLt's own size check sees the truth.
//
// Status codes: synchronous calls answer with the cublasStatus_t; posted
// ones (creates, setters, destroys, matmul) report a failure as the closest
// cudaError_t, at the app's next synchronizing CUDA call.

#include "tuna/cuda/backend.h"

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "cuda_api_ids.h"
#include "dispatch_internal.h"
#include "tuna/cuda/cublaslt_client.h"
#include "tuna/tunnel/rpc.h"

namespace tuna::cudart {

namespace {

constexpr size_t kMaxLtAlgos = 65536;  // distinct algos remembered per session

bool AlphaOnDevice(int32_t mode) { return mode != CUBLASLT_POINTER_MODE_HOST; }
bool BetaOnDevice(int32_t mode) {
    return mode == CUBLASLT_POINTER_MODE_DEVICE || mode == CUBLASLT_POINTER_MODE_DEVICE_VECTOR;
}

std::string AlgoKey(const cublasLtMatmulAlgo_t& algo) {
    return std::string(reinterpret_cast<const char*>(&algo), sizeof(algo));
}

template <typename Fn>
Fn Lt(CudaBackend& backend, const char* symbol) {
    return reinterpret_cast<Fn>(backend.Resolve(symbol));
}

}  // namespace

void* CudaSession::LtObjectPtr(uint64_t id, int kind) {
    auto it = lt_objects_.find(id);
    return it != lt_objects_.end() && it->second.kind == kind ? it->second.ptr : nullptr;
}

bool CudaSession::DispatchCublasLt(const RpcRequest& request, RpcResponse& resp) {
    const uint32_t api = request.api_id();
    const bool posted = (request.flags() & tunnel::kFlagAsync) != 0;
    auto finish = [&](cublasStatus_t status) {
        resp.set_result(posted ? CublasAsyncError(status) : static_cast<int32_t>(status));
    };
    auto unsupported = [&] { resp.set_result(posted ? kCudaErrorNotSupported : CUBLAS_STATUS_NOT_SUPPORTED); };
    auto known_handle = [&](void* handle) { return cublas_handles_.count(handle) || lt_handles_.count(handle); };

    // The three object kinds share their wire formats.
    struct Kind {
        int kind;
        const char* set;
        const char* destroy;
    };
    auto kind_of = [](uint32_t id) -> Kind {
        if (id == kApi_cublasLtMatmulDescSetAttribute || id == kApi_cublasLtMatmulDescDestroy) {
            return {kLtMatmulDesc, "cublasLtMatmulDescSetAttribute", "cublasLtMatmulDescDestroy"};
        }
        if (id == kApi_cublasLtMatrixLayoutSetAttribute || id == kApi_cublasLtMatrixLayoutDestroy) {
            return {kLtLayout, "cublasLtMatrixLayoutSetAttribute", "cublasLtMatrixLayoutDestroy"};
        }
        return {kLtPreference, "cublasLtMatmulPreferenceSetAttribute", "cublasLtMatmulPreferenceDestroy"};
    };

    try {
        ArgReader in(request.args());
        switch (api) {
        case kApi_cublasLtCreate: {
            if (in.remaining() != 0) return finish(CUBLAS_STATUS_INVALID_VALUE), true;
            auto create = Lt<cublasStatus_t (*)(cublasLtHandle_t*)>(backend_, "cublasLtCreate");
            if (!create) return unsupported(), true;
            cublasLtHandle_t handle = nullptr;
            const cublasStatus_t status = create(&handle);
            finish(status);
            if (status != CUBLAS_STATUS_SUCCESS) return true;
            lt_handles_.insert(handle);
            ArgWriter out;
            out.put_u64(reinterpret_cast<uint64_t>(handle));
            resp.set_out_args(out.str());
            return true;
        }
        case kApi_cublasLtDestroy: {
            void* handle = reinterpret_cast<void*>(in.get_u64());
            if (in.remaining() != 0) return finish(CUBLAS_STATUS_INVALID_VALUE), true;
            if (!lt_handles_.count(handle)) return finish(CUBLAS_STATUS_NOT_INITIALIZED), true;
            auto destroy = Lt<cublasStatus_t (*)(cublasLtHandle_t)>(backend_, "cublasLtDestroy");
            if (!destroy) return unsupported(), true;
            finish(destroy(static_cast<cublasLtHandle_t>(handle)));
            lt_handles_.erase(handle);  // gone whatever it returned
            return true;
        }

        case kApi_cublasLtMatmulDescCreate:
        case kApi_cublasLtMatrixLayoutCreate:
        case kApi_cublasLtMatmulPreferenceCreate: {
            const uint64_t id = in.get_u64();
            LtObject object{};
            cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
            if (api == kApi_cublasLtMatmulDescCreate) {
                const auto compute = static_cast<cublasComputeType_t>(in.get_u32());
                const auto scale = static_cast<cudaDataType_t>(in.get_u32());
                if (in.remaining() != 0 || lt_objects_.count(id)) return finish(CUBLAS_STATUS_INVALID_VALUE), true;
                auto create = Lt<cublasStatus_t (*)(cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t)>(
                    backend_, "cublasLtMatmulDescCreate");
                if (!create) return unsupported(), true;
                cublasLtMatmulDesc_t desc = nullptr;
                status = create(&desc, compute, scale);
                object = {kLtMatmulDesc, desc, static_cast<int32_t>(scale), CUBLASLT_POINTER_MODE_HOST};
            } else if (api == kApi_cublasLtMatrixLayoutCreate) {
                const auto type = static_cast<cudaDataType>(in.get_u32());
                const uint64_t rows = in.get_u64();
                const uint64_t cols = in.get_u64();
                const auto ld = static_cast<int64_t>(in.get_u64());
                if (in.remaining() != 0 || lt_objects_.count(id)) return finish(CUBLAS_STATUS_INVALID_VALUE), true;
                auto create = Lt<cublasStatus_t (*)(cublasLtMatrixLayout_t*, cudaDataType, uint64_t, uint64_t,
                                                    int64_t)>(backend_, "cublasLtMatrixLayoutCreate");
                if (!create) return unsupported(), true;
                cublasLtMatrixLayout_t layout = nullptr;
                status = create(&layout, type, rows, cols, ld);
                object = {kLtLayout, layout, 0, 0};
            } else {
                if (in.remaining() != 0 || lt_objects_.count(id)) return finish(CUBLAS_STATUS_INVALID_VALUE), true;
                auto create = Lt<cublasStatus_t (*)(cublasLtMatmulPreference_t*)>(backend_,
                                                                                  "cublasLtMatmulPreferenceCreate");
                if (!create) return unsupported(), true;
                cublasLtMatmulPreference_t pref = nullptr;
                status = create(&pref);
                object = {kLtPreference, pref, 0, 0};
            }
            if (status == CUBLAS_STATUS_SUCCESS) lt_objects_[id] = object;
            finish(status);
            return true;
        }

        case kApi_cublasLtMatmulDescSetAttribute:
        case kApi_cublasLtMatrixLayoutSetAttribute:
        case kApi_cublasLtMatmulPreferenceSetAttribute: {
            const Kind kind = kind_of(api);
            const uint64_t id = in.get_u64();
            const uint32_t attr = in.get_u32();
            const std::string value = in.rest();
            void* object = LtObjectPtr(id, kind.kind);
            if (!object || value.empty() || value.size() > tuna::cublaslt::kMaxAttributeBytes) {
                return finish(CUBLAS_STATUS_INVALID_VALUE), true;
            }
            // All three setters share this signature (the attribute enums are ints).
            auto set = Lt<cublasStatus_t (*)(void*, uint32_t, const void*, size_t)>(backend_, kind.set);
            if (!set) return unsupported(), true;
            const cublasStatus_t status = set(object, attr, value.data(), value.size());
            if (status == CUBLAS_STATUS_SUCCESS && kind.kind == kLtMatmulDesc && value.size() == 4) {
                int32_t v = 0;
                std::memcpy(&v, value.data(), 4);
                if (attr == CUBLASLT_MATMUL_DESC_POINTER_MODE) lt_objects_[id].pointer_mode = v;
                if (attr == CUBLASLT_MATMUL_DESC_SCALE_TYPE) lt_objects_[id].scale_type = v;
            }
            finish(status);
            return true;
        }

        case kApi_cublasLtMatmulDescDestroy:
        case kApi_cublasLtMatrixLayoutDestroy:
        case kApi_cublasLtMatmulPreferenceDestroy: {
            const Kind kind = kind_of(api);
            const uint64_t id = in.get_u64();
            void* object = LtObjectPtr(id, kind.kind);
            if (in.remaining() != 0 || !object) return finish(CUBLAS_STATUS_INVALID_VALUE), true;
            auto destroy = Lt<cublasStatus_t (*)(void*)>(backend_, kind.destroy);
            if (!destroy) return unsupported(), true;
            finish(destroy(object));
            lt_objects_.erase(id);
            return true;
        }

        case kApi_cublasLtMatmulAlgoGetHeuristic: {
            void* handle = reinterpret_cast<void*>(in.get_u64());
            void* desc = LtObjectPtr(in.get_u64(), kLtMatmulDesc);
            void* layouts[4];
            for (void*& layout : layouts) layout = LtObjectPtr(in.get_u64(), kLtLayout);
            void* pref = LtObjectPtr(in.get_u64(), kLtPreference);
            int requested = 0;
            in.get(&requested, 4);
            if (in.remaining() != 0 || !desc || !pref || std::count(layouts, layouts + 4, nullptr) != 0 ||
                requested <= 0 || requested > tuna::cublaslt::kMaxHeuristicResults) {
                return finish(CUBLAS_STATUS_INVALID_VALUE), true;
            }
            if (!known_handle(handle)) return finish(CUBLAS_STATUS_NOT_INITIALIZED), true;
            using HeuristicFn = cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
                                                   cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                                   cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int,
                                                   cublasLtMatmulHeuristicResult_t*, int*);
            auto heuristic = Lt<HeuristicFn>(backend_, "cublasLtMatmulAlgoGetHeuristic");
            if (!heuristic) return unsupported(), true;
            std::vector<cublasLtMatmulHeuristicResult_t> results(static_cast<size_t>(requested));
            int count = 0;
            const cublasStatus_t status = heuristic(
                static_cast<cublasLtHandle_t>(handle), static_cast<cublasLtMatmulDesc_t>(desc),
                static_cast<cublasLtMatrixLayout_t>(layouts[0]), static_cast<cublasLtMatrixLayout_t>(layouts[1]),
                static_cast<cublasLtMatrixLayout_t>(layouts[2]), static_cast<cublasLtMatrixLayout_t>(layouts[3]),
                static_cast<cublasLtMatmulPreference_t>(pref), requested, results.data(), &count);
            finish(status);
            if (status != CUBLAS_STATUS_SUCCESS) return true;
            count = std::clamp(count, 0, requested);
            for (int i = 0; i < count; ++i) {
                if (results[i].state == CUBLAS_STATUS_SUCCESS && lt_algos_.size() < kMaxLtAlgos) {
                    lt_algos_.insert(AlgoKey(results[i].algo));
                }
            }
            ArgWriter out;
            out.put(&count, 4);
            out.put(results.data(), sizeof(cublasLtMatmulHeuristicResult_t) * static_cast<size_t>(count));
            resp.set_out_args(out.str());
            return true;
        }

        case kApi_cublasLtMatmul: {
            void* handle = reinterpret_cast<void*>(in.get_u64());
            const uint64_t desc_id = in.get_u64();
            ScalarRef alpha(in);
            const void* a = reinterpret_cast<const void*>(in.get_u64());
            void* a_layout = LtObjectPtr(in.get_u64(), kLtLayout);
            const void* b = reinterpret_cast<const void*>(in.get_u64());
            void* b_layout = LtObjectPtr(in.get_u64(), kLtLayout);
            ScalarRef beta(in);
            const void* c = reinterpret_cast<const void*>(in.get_u64());
            void* c_layout = LtObjectPtr(in.get_u64(), kLtLayout);
            void* d = reinterpret_cast<void*>(in.get_u64());
            void* d_layout = LtObjectPtr(in.get_u64(), kLtLayout);
            const uint32_t has_algo = in.get_u32();
            cublasLtMatmulAlgo_t algo{};
            if (has_algo) in.get(&algo, sizeof(algo));
            void* workspace = reinterpret_cast<void*>(in.get_u64());
            const uint64_t workspace_size = in.get_u64();
            auto stream = reinterpret_cast<cudaStream_t>(in.get_u64());

            auto desc = lt_objects_.find(desc_id);
            if (in.remaining() != 0 || desc == lt_objects_.end() || desc->second.kind != kLtMatmulDesc || !a_layout ||
                !b_layout || !c_layout || !d_layout || has_algo > 1 ||
                (has_algo && !lt_algos_.count(AlgoKey(algo))) ||
                alpha.on_device() != AlphaOnDevice(desc->second.pointer_mode) ||
                beta.on_device() != BetaOnDevice(desc->second.pointer_mode)) {
                return finish(CUBLAS_STATUS_INVALID_VALUE), true;
            }
            if (!known_handle(handle)) return finish(CUBLAS_STATUS_NOT_INITIALIZED), true;
            using MatmulFn = cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void*, const void*,
                                                cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t,
                                                const void*, const void*, cublasLtMatrixLayout_t, void*,
                                                cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*, void*, size_t,
                                                cudaStream_t);
            auto matmul = Lt<MatmulFn>(backend_, "cublasLtMatmul");
            if (!matmul) return unsupported(), true;
            finish(matmul(static_cast<cublasLtHandle_t>(handle), static_cast<cublasLtMatmulDesc_t>(desc->second.ptr),
                          alpha.ptr(), a, static_cast<cublasLtMatrixLayout_t>(a_layout), b,
                          static_cast<cublasLtMatrixLayout_t>(b_layout), beta.ptr(), c,
                          static_cast<cublasLtMatrixLayout_t>(c_layout), d,
                          static_cast<cublasLtMatrixLayout_t>(d_layout), has_algo ? &algo : nullptr, workspace,
                          workspace_size, stream));
            return true;
        }

        default:
            return false;
        }
    } catch (const std::exception&) {
        // A short or garbled arg buffer.
        resp.clear_out_args();
        finish(CUBLAS_STATUS_INVALID_VALUE);
        return true;
    }
}

}  // namespace tuna::cudart

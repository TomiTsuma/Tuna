#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tuna/cuda/marshal.h"
#include "tuna/tunnel/rpc.h"
#include "tuna_frames.pb.h"

namespace tuna::cudart {

// The real CUDA stack, as the server dispatcher sees it: runtime symbols
// (cudaMalloc, ...), the few driver symbols kernel launch needs
// (cuModuleLoadData, cuLaunchKernel, ...) and cuBLAS (cublasSgemm_v2, ...).
// The production backend dlopen's libcudart + libcuda (+ libcublas on first
// use); tests supply a mock whose Resolve() returns stand-ins.
class CudaBackend {
public:
    virtual ~CudaBackend() = default;
    // The real function for `symbol`, or nullptr if this backend lacks it.
    virtual void* Resolve(const char* symbol) = 0;
};

// One client session's CUDA state on the server: uploaded fatbin images, the
// kernels resolved from them, and which modules are loaded into which CUDA
// context. Lives in the per-session worker process, so a session's GPU state
// dies with its worker.
//
// Not thread-safe: RpcServer dispatches one call at a time, in order.
class CudaSession {
public:
    explicit CudaSession(CudaBackend& backend) : backend_(backend) {}

    // Executes one remoted runtime call. Reads request.args() (and `bulk_in`
    // for host->device copies and module uploads), calls the real function
    // through the backend, and returns the response (with `bulk_out` set for
    // device->host copies). Never throws: a malformed request becomes
    // cudaErrorInvalidValue.
    RpcResponse Dispatch(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out);

    size_t module_count() const { return images_.size(); }

private:
    struct Image {
        std::vector<uint64_t> words;  // 8-byte aligned copy of the fatbin
    };
    struct Function {
        uint64_t module;   // 1-based index into images_
        std::string name;
        std::vector<std::pair<uint32_t, uint32_t>> params;  // (offset, size)
        size_t param_bytes = 0;
    };
    using Context = void*;  // CUcontext

    bool DispatchSpecial(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out,
                         RpcResponse& resp);
    // cuBLAS calls (dispatch_cublas.cpp; returns false when the build has no
    // cuBLAS headers, and the call is then not supported).
    bool DispatchCublas(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out,
                        RpcResponse& resp);
    void MatrixCopy(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out, RpcResponse& resp);
    void CublasHandleCall(const RpcRequest& request, RpcResponse& resp);
    void Memcpy(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out, RpcResponse& resp);
    void Reset(RpcResponse& resp);
    void RegisterModule(const std::string& bulk_in, RpcResponse& resp);
    void GetFunction(const RpcRequest& request, RpcResponse& resp);
    void GetVar(const RpcRequest& request, RpcResponse& resp);
    void LaunchKernel(const RpcRequest& request, RpcResponse& resp);
    void FuncCall(const RpcRequest& request, RpcResponse& resp);

    // The current CUDA context (initializing the runtime's primary context
    // first), or nullptr with *error set.
    Context CurrentContext(int32_t* error);
    // `module` loaded into the current context, loading it on first use.
    int32_t LoadedModule(uint64_t module, Context ctx, void** cu_module);
    // Function `id` (1-based) resolved in the current context.
    int32_t ResolvedFunction(uint64_t id, void** cu_function);

    CudaBackend& backend_;
    std::vector<Image> images_;
    std::vector<Function> functions_;
    std::map<std::pair<uint64_t, std::string>, uint64_t> function_ids_;  // (module, name) -> id
    std::map<std::pair<Context, uint64_t>, void*> loaded_;    // (ctx, module) -> CUmodule
    std::map<std::pair<Context, uint64_t>, void*> resolved_;  // (ctx, function) -> CUfunction
    // cuBLAS handles this session created -> pointer mode is DEVICE. A handle
    // is a host pointer in this process, so no other value reaches cuBLAS.
    std::map<void*, bool> cublas_handles_;
};

// True if `api_id` is a CUDA call this build knows (not the echo id).
bool IsCudaApi(uint32_t api_id);

// The api's name for logs ("api#<id>" if unknown).
std::string CudaApiName(uint32_t api_id);

}  // namespace tuna::cudart

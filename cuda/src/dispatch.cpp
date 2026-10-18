// Server-side CUDA runtime dispatcher. The per-api switch for simple functions
// is generated (cuda_server_dispatch.inc); the memcpy family, device reset and
// the Tuna-internal kernel calls are hand-written here.
//
// Handles (device pointers, streams, events, function ids) arrive as uint64
// values the server issued. Device pointers and streams are only ever passed
// back to the real runtime, never dereferenced by Tuna.
//
// Kernels go through the driver API: the client uploads the fatbin nvcc
// embedded in its program, the server loads it into the current context with
// cuModuleLoadData, and cuFuncGetParamInfo (CUDA 12.4+ drivers) gives each
// kernel's parameter layout so the client can pack arguments by value. The
// runtime and driver share the primary context, so device pointers from
// cudaMalloc and streams from cudaStreamCreate are valid in cuLaunchKernel.

#include "tuna/cuda/backend.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>

#include "cuda_api_ids.h"

namespace tuna::cudart {

namespace {

#include "cuda_coverage.inc"

constexpr int32_t kCudaErrorNotSupported = 801;
constexpr int32_t kCudaErrorInvalidValue = 1;
constexpr int32_t kCudaErrorInvalidKernelImage = 200;
constexpr int32_t kCudaErrorInvalidDeviceFunction = 98;
constexpr int32_t kCudaErrorInvalidResourceHandle = 400;

constexpr uint32_t kFatbinMagic = 0xBA55ED50;
constexpr size_t kMaxKernelParams = 4096;

template <typename Fn>
Fn Get(CudaBackend& backend, const char* symbol) {
    return reinterpret_cast<Fn>(backend.Resolve(symbol));
}

}  // namespace

bool IsCudaApi(uint32_t api_id) { return api_id >= kFirstCudaApi && api_id <= kLastCudaApi; }

std::string CudaApiName(uint32_t api_id) {
    for (const auto& entry : kCudaApiNames) {
        if (entry.id == api_id) return entry.name;
    }
    return "api#" + std::to_string(api_id);
}

RpcResponse CudaSession::Dispatch(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out) {
    RpcResponse resp;
    try {
        if (DispatchSpecial(request, bulk_in, bulk_out, resp)) return resp;
        if (DispatchCublas(request, bulk_in, bulk_out, resp)) return resp;
#include "cuda_server_dispatch.inc"
    } catch (const std::exception&) {
        resp.set_result(kCudaErrorInvalidValue);  // short/garbled arg buffer
        resp.clear_out_args();
        bulk_out->clear();
    }
    return resp;
}

#ifndef TUNA_CUDA_HAVE_CUBLAS
// Built without the cuBLAS headers: cuBLAS calls fall through to "not supported".
bool CudaSession::DispatchCublas(const RpcRequest&, const std::string&, std::string*, RpcResponse&) { return false; }
void CudaSession::MatrixCopy(const RpcRequest&, const std::string&, std::string*, RpcResponse&) {}
void CudaSession::CublasHandleCall(const RpcRequest&, RpcResponse&) {}
#endif

bool CudaSession::DispatchSpecial(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out,
                                  RpcResponse& resp) {
    switch (request.api_id()) {
    case kApi_cudaMemcpy:
    case kApi_cudaMemcpyAsync:
        Memcpy(request, bulk_in, bulk_out, resp);
        return true;
    case kApi_cudaDeviceReset:
        Reset(resp);
        return true;
    case kApi_cudaGetLastError:
    case kApi_cudaPeekAtLastError: {
        auto fn = Get<cudaError_t (*)()>(backend_, request.api_id() == kApi_cudaGetLastError ? "cudaGetLastError"
                                                                                          : "cudaPeekAtLastError");
        resp.set_result(fn ? fn() : kCudaErrorNotSupported);
        return true;
    }
    case kApi___tunaRegisterModule:
        RegisterModule(bulk_in, resp);
        return true;
    case kApi___tunaGetFunction:
        GetFunction(request, resp);
        return true;
    case kApi___tunaGetVar:
        GetVar(request, resp);
        return true;
    case kApi___tunaLaunchKernel:
        LaunchKernel(request, resp);
        return true;
    case kApi___tunaFuncGetAttributes:
    case kApi___tunaFuncSetAttribute:
    case kApi___tunaOccupancy:
        FuncCall(request, resp);
        return true;
    default:
        return false;
    }
}

// cudaMemcpy / cudaMemcpyAsync. `kind` decides the direction; payloads ride the
// bulk channel so a large copy never blocks small control calls.
void CudaSession::Memcpy(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out,
                         RpcResponse& resp) {
    const bool async = request.api_id() == kApi_cudaMemcpyAsync;
    ArgReader in(request.args());
    void* dst = reinterpret_cast<void*>(in.get_u64());
    const void* src = reinterpret_cast<const void*>(in.get_u64());
    const uint64_t count = in.get_u64();
    const auto kind = static_cast<cudaMemcpyKind>(in.get_u32());
    cudaStream_t stream = async ? reinterpret_cast<cudaStream_t>(in.get_u64()) : nullptr;
    if (in.remaining() != 0) {
        resp.set_result(kCudaErrorInvalidValue);
        return;
    }

    using MemcpyFn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
    using MemcpyAsyncFn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
    auto copy = Get<MemcpyFn>(backend_, "cudaMemcpy");
    auto copy_async = Get<MemcpyAsyncFn>(backend_, "cudaMemcpyAsync");
    auto stream_sync = Get<cudaError_t (*)(cudaStream_t)>(backend_, "cudaStreamSynchronize");
    if (!copy || (async && (!copy_async || !stream_sync))) {
        resp.set_result(kCudaErrorNotSupported);
        return;
    }

    if (kind == cudaMemcpyHostToDevice) {
        if (bulk_in.size() != count) {
            resp.set_result(kCudaErrorInvalidValue);
            return;
        }
        // From pageable memory, cudaMemcpyAsync returns once the source is
        // staged, so bulk_in may be freed as soon as this returns.
        resp.set_result(async ? copy_async(dst, bulk_in.data(), count, kind, stream)
                              : copy(dst, bulk_in.data(), count, kind));
    } else if (kind == cudaMemcpyDeviceToHost) {
        std::string host(count, '\0');
        cudaError_t e = async ? copy_async(host.data(), src, count, kind, stream) : copy(host.data(), src, count, kind);
        // The client needs the bytes now: wait for this stream's work.
        if (async && e == cudaSuccess) e = stream_sync(stream);
        resp.set_result(e);
        if (e == cudaSuccess) *bulk_out = std::move(host);
    } else if (kind == cudaMemcpyDeviceToDevice) {
        resp.set_result(async ? copy_async(dst, src, count, kind, stream) : copy(dst, src, count, kind));
    } else {
        resp.set_result(kCudaErrorInvalidValue);  // the client resolves Default and HostToHost itself
    }
}

void CudaSession::Reset(RpcResponse& resp) {
    auto reset = Get<cudaError_t (*)()>(backend_, "cudaDeviceReset");
    if (!reset) {
        resp.set_result(kCudaErrorNotSupported);
        return;
    }
    int32_t ignored = 0;
    Context ctx = CurrentContext(&ignored);
    resp.set_result(reset());
    // The context's modules are gone; a later use reloads them.
    std::erase_if(loaded_, [ctx](const auto& entry) { return entry.first.first == ctx; });
    std::erase_if(resolved_, [ctx](const auto& entry) { return entry.first.first == ctx; });
}

CudaSession::Context CudaSession::CurrentContext(int32_t* error) {
    auto get_current = Get<CUresult (*)(CUcontext*)>(backend_, "cuCtxGetCurrent");
    auto runtime_free = Get<cudaError_t (*)(void*)>(backend_, "cudaFree");
    if (!get_current || !runtime_free) {
        *error = kCudaErrorNotSupported;
        return nullptr;
    }
    CUcontext ctx = nullptr;
    get_current(&ctx);
    if (!ctx) {
        // The runtime creates (and makes current) the primary context on its
        // first real call; cudaFree(nullptr) is the documented way to force it.
        if (cudaError_t e = runtime_free(nullptr); e != cudaSuccess) {
            *error = e;
            return nullptr;
        }
        get_current(&ctx);
    }
    if (!ctx) *error = kCudaErrorInvalidResourceHandle;
    return ctx;
}

void CudaSession::RegisterModule(const std::string& bulk_in, RpcResponse& resp) {
    uint32_t magic = 0;
    if (bulk_in.size() < 16) {
        resp.set_result(kCudaErrorInvalidKernelImage);
        return;
    }
    std::memcpy(&magic, bulk_in.data(), 4);
    if (magic != kFatbinMagic) {
        resp.set_result(kCudaErrorInvalidKernelImage);
        return;
    }
    Image image;
    image.words.resize((bulk_in.size() + 7) / 8);
    std::memcpy(image.words.data(), bulk_in.data(), bulk_in.size());
    images_.push_back(std::move(image));
    ArgWriter out;
    out.put_u64(images_.size());
    resp.set_out_args(out.str());
}

int32_t CudaSession::LoadedModule(uint64_t module, Context ctx, void** cu_module) {
    if (module == 0 || module > images_.size()) return kCudaErrorInvalidResourceHandle;
    auto it = loaded_.find({ctx, module});
    if (it != loaded_.end()) {
        *cu_module = it->second;
        return 0;
    }
    auto load = Get<CUresult (*)(CUmodule*, const void*)>(backend_, "cuModuleLoadData");
    if (!load) return kCudaErrorNotSupported;
    CUmodule loaded = nullptr;
    if (CUresult r = load(&loaded, images_[module - 1].words.data()); r != CUDA_SUCCESS) return r;
    loaded_[{ctx, module}] = loaded;
    *cu_module = loaded;
    return 0;
}

// (module id, kernel name) -> function id + parameter layout. The id is
// context-independent; ResolvedFunction() maps it per context at launch.
void CudaSession::GetFunction(const RpcRequest& request, RpcResponse& resp) {
    ArgReader in(request.args());
    const uint64_t module = in.get_u64();
    const std::string name = in.get_string();
    if (in.remaining() != 0) {
        resp.set_result(kCudaErrorInvalidValue);
        return;
    }

    uint64_t id = 0;
    if (auto it = function_ids_.find({module, name}); it != function_ids_.end()) {
        id = it->second;
    } else {
        int32_t error = 0;
        Context ctx = CurrentContext(&error);
        if (!ctx) {
            resp.set_result(error);
            return;
        }
        void* cu_module = nullptr;
        if (int32_t e = LoadedModule(module, ctx, &cu_module)) {
            resp.set_result(e);
            return;
        }
        auto get_function = Get<CUresult (*)(CUfunction*, CUmodule, const char*)>(backend_, "cuModuleGetFunction");
        auto param_info = Get<CUresult (*)(CUfunction, size_t, size_t*, size_t*)>(backend_, "cuFuncGetParamInfo");
        if (!get_function || !param_info) {
            resp.set_result(kCudaErrorNotSupported);  // cuFuncGetParamInfo needs a CUDA 12.4+ driver
            return;
        }
        CUfunction function = nullptr;
        if (CUresult r = get_function(&function, static_cast<CUmodule>(cu_module), name.c_str()); r != CUDA_SUCCESS) {
            resp.set_result(r == CUDA_ERROR_NOT_FOUND ? kCudaErrorInvalidDeviceFunction : r);
            return;
        }
        Function info{module, name, {}, 0};
        for (size_t i = 0; i < kMaxKernelParams; ++i) {
            size_t offset = 0, size = 0;
            if (param_info(function, i, &offset, &size) != CUDA_SUCCESS) break;  // past the last parameter
            info.params.emplace_back(static_cast<uint32_t>(offset), static_cast<uint32_t>(size));
            info.param_bytes = std::max(info.param_bytes, offset + size);
        }
        functions_.push_back(std::move(info));
        id = functions_.size();
        function_ids_[{module, name}] = id;
        resolved_[{ctx, id}] = function;
    }

    const Function& info = functions_[id - 1];
    ArgWriter out;
    out.put_u64(id);
    out.put_u32(static_cast<uint32_t>(info.params.size()));
    for (const auto& [offset, size] : info.params) {
        out.put_u32(offset);
        out.put_u32(size);
    }
    resp.set_out_args(out.str());
}

int32_t CudaSession::ResolvedFunction(uint64_t id, void** cu_function) {
    if (id == 0 || id > functions_.size()) return kCudaErrorInvalidDeviceFunction;
    int32_t error = 0;
    Context ctx = CurrentContext(&error);
    if (!ctx) return error;
    if (auto it = resolved_.find({ctx, id}); it != resolved_.end()) {
        *cu_function = it->second;
        return 0;
    }
    // First use of this kernel in this context (another device, or after a reset).
    const Function& info = functions_[id - 1];
    void* cu_module = nullptr;
    if (int32_t e = LoadedModule(info.module, ctx, &cu_module)) return e;
    auto get_function = Get<CUresult (*)(CUfunction*, CUmodule, const char*)>(backend_, "cuModuleGetFunction");
    if (!get_function) return kCudaErrorNotSupported;
    CUfunction function = nullptr;
    if (CUresult r = get_function(&function, static_cast<CUmodule>(cu_module), info.name.c_str()); r != CUDA_SUCCESS) {
        return r;
    }
    resolved_[{ctx, id}] = function;
    *cu_function = function;
    return 0;
}

void CudaSession::GetVar(const RpcRequest& request, RpcResponse& resp) {
    ArgReader in(request.args());
    const uint64_t module = in.get_u64();
    const std::string name = in.get_string();
    if (in.remaining() != 0) {
        resp.set_result(kCudaErrorInvalidValue);
        return;
    }
    int32_t error = 0;
    Context ctx = CurrentContext(&error);
    if (!ctx) {
        resp.set_result(error);
        return;
    }
    void* cu_module = nullptr;
    if (int32_t e = LoadedModule(module, ctx, &cu_module)) {
        resp.set_result(e);
        return;
    }
    auto get_global = Get<CUresult (*)(CUdeviceptr*, size_t*, CUmodule, const char*)>(backend_, "cuModuleGetGlobal_v2");
    if (!get_global) {
        resp.set_result(kCudaErrorNotSupported);
        return;
    }
    CUdeviceptr address = 0;
    size_t size = 0;
    if (CUresult r = get_global(&address, &size, static_cast<CUmodule>(cu_module), name.c_str()); r != CUDA_SUCCESS) {
        resp.set_result(r == CUDA_ERROR_NOT_FOUND ? 13 /* cudaErrorInvalidSymbol */ : r);
        return;
    }
    ArgWriter out;
    out.put_u64(address);
    out.put_u64(size);
    resp.set_out_args(out.str());
}

void CudaSession::LaunchKernel(const RpcRequest& request, RpcResponse& resp) {
    ArgReader in(request.args());
    const uint64_t id = in.get_u64();
    uint32_t dims[6];
    for (uint32_t& d : dims) d = in.get_u32();
    const uint64_t shared_mem = in.get_u64();
    auto stream = reinterpret_cast<CUstream>(in.get_u64());
    const std::string params = in.rest();

    void* cu_function = nullptr;
    if (int32_t e = ResolvedFunction(id, &cu_function)) {
        resp.set_result(e);
        return;
    }
    const Function& info = functions_[id - 1];
    if (params.size() != info.param_bytes) {
        resp.set_result(kCudaErrorInvalidValue);
        return;
    }
    // Each kernelParams entry points at its argument's bytes; keep them in
    // 8-byte aligned storage at the offsets the driver reported.
    std::vector<uint64_t> storage((params.size() + 7) / 8);
    if (!params.empty()) std::memcpy(storage.data(), params.data(), params.size());
    std::vector<void*> pointers;
    pointers.reserve(info.params.size());
    for (const auto& [offset, size] : info.params) {
        pointers.push_back(reinterpret_cast<char*>(storage.data()) + offset);
    }

    using LaunchFn = CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                  CUstream, void**, void**);
    auto launch = Get<LaunchFn>(backend_, "cuLaunchKernel");
    if (!launch) {
        resp.set_result(kCudaErrorNotSupported);
        return;
    }
    // cudaError_t and CUresult share values for every launch failure
    // (invalid value, out of resources, invalid handle, ...).
    resp.set_result(launch(static_cast<CUfunction>(cu_function), dims[0], dims[1], dims[2], dims[3], dims[4], dims[5],
                           static_cast<unsigned>(shared_mem), stream, pointers.empty() ? nullptr : pointers.data(),
                           nullptr));
}

// cudaFuncGetAttributes / cudaFuncSetAttribute / occupancy, by function id.
void CudaSession::FuncCall(const RpcRequest& request, RpcResponse& resp) {
    ArgReader in(request.args());
    const uint64_t id = in.get_u64();
    void* cu_function = nullptr;
    if (int32_t e = ResolvedFunction(id, &cu_function)) {
        resp.set_result(e);
        return;
    }
    auto function = static_cast<CUfunction>(cu_function);

    if (request.api_id() == kApi___tunaFuncGetAttributes) {
        auto get = Get<CUresult (*)(int*, CUfunction_attribute, CUfunction)>(backend_, "cuFuncGetAttribute");
        if (!get) {
            resp.set_result(kCudaErrorNotSupported);
            return;
        }
        auto attribute = [&](CUfunction_attribute which) {
            int value = 0;
            get(&value, which, function);
            return value;
        };
        cudaFuncAttributes attr{};
        attr.maxThreadsPerBlock = attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        attr.sharedSizeBytes = static_cast<size_t>(attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES));
        attr.constSizeBytes = static_cast<size_t>(attribute(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES));
        attr.localSizeBytes = static_cast<size_t>(attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES));
        attr.numRegs = attribute(CU_FUNC_ATTRIBUTE_NUM_REGS);
        attr.ptxVersion = attribute(CU_FUNC_ATTRIBUTE_PTX_VERSION);
        attr.binaryVersion = attribute(CU_FUNC_ATTRIBUTE_BINARY_VERSION);
        attr.cacheModeCA = attribute(CU_FUNC_ATTRIBUTE_CACHE_MODE_CA);
        attr.maxDynamicSharedSizeBytes = attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
        attr.preferredShmemCarveout = attribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
        resp.set_out_args(std::string(reinterpret_cast<const char*>(&attr), sizeof(attr)));
    } else if (request.api_id() == kApi___tunaFuncSetAttribute) {
        const uint32_t which = in.get_u32();
        const auto value = static_cast<int>(in.get_u32());
        auto set = Get<CUresult (*)(CUfunction, CUfunction_attribute, int)>(backend_, "cuFuncSetAttribute");
        // cudaFuncAttribute and CUfunction_attribute share values for the
        // settable attributes (max dynamic shared memory, carveout, ...).
        resp.set_result(set ? set(function, static_cast<CUfunction_attribute>(which), value) : kCudaErrorNotSupported);
    } else {
        const auto block_size = static_cast<int>(in.get_u32());
        const uint64_t shared_mem = in.get_u64();
        const uint32_t flags = in.get_u32();
        using OccupancyFn = CUresult (*)(int*, CUfunction, int, size_t, unsigned);
        auto occupancy = Get<OccupancyFn>(backend_, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
        if (!occupancy) {
            resp.set_result(kCudaErrorNotSupported);
            return;
        }
        int blocks = 0;
        resp.set_result(occupancy(&blocks, function, block_size, shared_mem, flags));
        ArgWriter out;
        out.put(&blocks, 4);
        resp.set_out_args(out.str());
    }
}

}  // namespace tuna::cudart

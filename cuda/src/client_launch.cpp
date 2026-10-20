// Kernel registration and launch for the replacement libcudart.
//
// nvcc-compiled programs register their device code from static constructors:
// __cudaRegisterFatBinary hands over the embedded fatbin, __cudaRegisterFunction
// maps each kernel's host stub address to its device name, __cudaRegisterVar
// does the same for __device__/__constant__ variables. A `kernel<<<...>>>`
// becomes __cudaPushCallConfiguration + a stub that pops it and calls
// cudaLaunchKernel(host stub address, grid, block, args, smem, stream).
//
// The shim records all of that locally and talks to the server lazily: a
// fatbin is uploaded the first time one of its kernels or variables is used
// (programs link far more device code than they run; PyTorch ships hundreds of
// MB), a kernel's parameter layout is fetched once, and launches are
// fire-and-forget. The server owns modules and functions per CUDA context.

#include <cuda_runtime_api.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "client_internal.h"
#include "cuda_api_ids.h"
#include "tuna/cuda/client.h"

using namespace tuna::cudart;

namespace tuna::cudart {

namespace {

// Layout of what nvcc embeds (see crt/host_runtime.h, fatbinary_section.h).
struct FatbinWrapper {
    int magic;
    int version;
    const void* data;
    void* filename_or_fatbins;
};
constexpr int kFatbinWrapperMagic = 0x466243b1;

struct FatbinHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t fat_size;
};
constexpr uint32_t kFatbinMagic = 0xBA55ED50;

struct Module {
    const char* image = nullptr;  // in the program's own data section
    size_t size = 0;
    uint64_t remote_id = 0;       // 0 until uploaded
};

struct Kernel {
    Module* module;
    std::string name;
    // Fetched on first use.
    uint64_t remote_id = 0;
    std::vector<std::pair<uint32_t, uint32_t>> params;  // (offset, size)
    size_t param_bytes = 0;
};

struct Variable {
    Module* module;
    std::string name;
    size_t size;
    uint64_t address = 0;        // device address in the current context
    uint64_t generation = 0;     // ContextGeneration() when `address` was fetched
};

std::mutex g_mutex;
std::vector<std::unique_ptr<Module>> g_modules;                  // guarded by g_mutex
std::unordered_map<const void*, Kernel> g_kernels;               // guarded by g_mutex
std::unordered_map<const void*, Variable> g_variables;           // guarded by g_mutex

struct CallConfiguration {
    dim3 grid, block;
    size_t shared_mem;
    cudaStream_t stream;
};
thread_local std::vector<CallConfiguration> t_configurations;

// Uploads the module's fatbin if this session has not seen it.
int32_t EnsureUploaded(Module& module) {
    if (module.remote_id) return 0;
    if (!module.image) return kErrInvalidDeviceFunction;
    Reply reply = Call(kApi___tunaRegisterModule, ArgWriter{}, std::string(module.image, module.size));
    if (reply.transport_error || reply.result != 0) return Finish(reply);
    try {
        ArgReader out(reply.out_args);
        module.remote_id = out.get_u64();
    } catch (const std::exception&) {
        return kErrUnknown;
    }
    return 0;
}

// The kernel registered for host stub `func`, with its server id and
// parameter layout. Requires g_mutex.
int32_t ResolveKernel(const void* func, Kernel** out_kernel) {
    auto it = g_kernels.find(func);
    if (it == g_kernels.end()) return kErrInvalidDeviceFunction;
    Kernel& kernel = it->second;
    if (!kernel.remote_id) {
        if (int32_t e = EnsureUploaded(*kernel.module)) return e;
        ArgWriter in;
        in.put_u64(kernel.module->remote_id);
        in.put_string(kernel.name);
        Reply reply = Call(kApi___tunaGetFunction, in);
        if (reply.transport_error || reply.result != 0) return Finish(reply);
        try {
            ArgReader out(reply.out_args);
            const uint64_t id = out.get_u64();
            const uint32_t count = out.get_u32();
            std::vector<std::pair<uint32_t, uint32_t>> params;
            size_t bytes = 0;
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t offset = out.get_u32();
                const uint32_t size = out.get_u32();
                params.emplace_back(offset, size);
                bytes = std::max<size_t>(bytes, size_t{offset} + size);
            }
            kernel.params = std::move(params);
            kernel.param_bytes = bytes;
            kernel.remote_id = id;
        } catch (const std::exception&) {
            return kErrUnknown;
        }
    }
    *out_kernel = &kernel;
    return 0;
}

// The device address of a registered variable in the current context.
int32_t ResolveVariable(const void* symbol, uint64_t* address, size_t* size) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_variables.find(symbol);
    if (it == g_variables.end()) return kErrInvalidSymbol;
    Variable& var = it->second;
    const uint64_t generation = ContextGeneration();
    if (var.generation != generation) {
        if (int32_t e = EnsureUploaded(*var.module)) return e;
        ArgWriter in;
        in.put_u64(var.module->remote_id);
        in.put_string(var.name);
        Reply reply = Call(kApi___tunaGetVar, in);
        if (reply.transport_error || reply.result != 0) return Finish(reply);
        try {
            ArgReader out(reply.out_args);
            var.address = out.get_u64();
            var.size = out.get_u64();
        } catch (const std::exception&) {
            return kErrUnknown;
        }
        var.generation = generation;
        TrackDeviceRange(reinterpret_cast<void*>(var.address), var.size);
    }
    *address = var.address;
    if (size) *size = var.size;
    return 0;
}

int32_t FunctionCall(uint32_t api, const void* func, ArgWriter extra, Reply* reply) {
    uint64_t id = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        Kernel* kernel = nullptr;
        if (int32_t e = ResolveKernel(func, &kernel)) return Fail(e);
        id = kernel->remote_id;
    }
    ArgWriter in;
    in.put_u64(id);
    in.put(extra.str().data(), extra.str().size());
    *reply = Call(api, in);
    return Finish(*reply);
}

}  // namespace

void ResetLaunchState() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& module : g_modules) module->remote_id = 0;
    for (auto& [func, kernel] : g_kernels) kernel.remote_id = 0;
    for (auto& [symbol, var] : g_variables) var.generation = 0;
}

void ResetClientStateForTesting() {
    ResetLaunchState();
    ResetRuntimeState();
}

}  // namespace tuna::cudart

// ------------------------------------------------------------ registration

TUNA_CUDART_EXPORT void** __cudaRegisterFatBinary(void* fatCubin) {
    auto module = std::make_unique<Module>();
    const auto* wrapper = static_cast<const FatbinWrapper*>(fatCubin);
    if (wrapper && wrapper->magic == kFatbinWrapperMagic && wrapper->data) {
        const auto* header = static_cast<const FatbinHeader*>(wrapper->data);
        if (header->magic == kFatbinMagic) {
            module->image = static_cast<const char*>(wrapper->data);
            module->size = size_t{header->header_size} + header->fat_size;
        }
        // Version 2 wrappers (-rdc=true) need device-side linking of several
        // fatbins at load time; not supported yet. Their kernels fail to
        // launch with cudaErrorInvalidDeviceFunction.
        if (wrapper->version != 1 && LogEnabled()) {
            Log("fatbin wrapper version " + std::to_string(wrapper->version) + " (relocatable device code) "
                "is not supported yet");
            module->image = nullptr;
        }
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_modules.push_back(std::move(module));
    // The handle nvcc passes back to every other registration call.
    return reinterpret_cast<void**>(g_modules.back().get());
}

TUNA_CUDART_EXPORT void __cudaRegisterFatBinaryEnd(void**) {}

// Called from static destructors at exit; the session is gone or going, and the
// server frees the module with the worker, so there is nothing to do.
TUNA_CUDART_EXPORT void __cudaUnregisterFatBinary(void**) {}

TUNA_CUDART_EXPORT char __cudaInitModule(void**) { return 1; }

TUNA_CUDART_EXPORT void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char*,
                                               const char* deviceName, int, uint3*, uint3*, dim3*, dim3*, int*) {
    if (!fatCubinHandle || !hostFun || !deviceName) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    Kernel kernel{reinterpret_cast<Module*>(fatCubinHandle), deviceName, 0, {}, 0};
    g_kernels.insert_or_assign(hostFun, std::move(kernel));
}

TUNA_CUDART_EXPORT void __cudaRegisterVar(void** fatCubinHandle, char* hostVar, char*, const char* deviceName,
                                          int, size_t size, int, int) {
    if (!fatCubinHandle || !hostVar || !deviceName) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_variables.insert_or_assign(hostVar, Variable{reinterpret_cast<Module*>(fatCubinHandle), deviceName, size, 0, 0});
}

TUNA_CUDART_EXPORT unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                                        cudaStream_t stream) {
    t_configurations.push_back({gridDim, blockDim, sharedMem, stream});
    return 0;
}

TUNA_CUDART_EXPORT cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                                          void* stream) {
    if (t_configurations.empty()) return static_cast<cudaError_t>(Fail(cudaErrorMissingConfiguration));
    const CallConfiguration c = t_configurations.back();
    t_configurations.pop_back();
    *gridDim = c.grid;
    *blockDim = c.block;
    *sharedMem = c.shared_mem;
    *static_cast<cudaStream_t*>(stream) = c.stream;
    return cudaSuccess;
}

// ------------------------------------------------------------------ launch

// Wire: function id, grid xyz, block xyz (u32), shared mem, stream (u64),
// then the parameters packed at the offsets the server reported. Device
// pointers among them are server addresses already, so they pass unchanged.
TUNA_CUDART_EXPORT cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                                                size_t sharedMem, cudaStream_t stream) {
    // A bad configuration is reported by the launch itself, as CUDA does.
    if (int32_t e = ValidateLaunch(gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z)) {
        return static_cast<cudaError_t>(Fail(e));
    }
    ArgWriter in;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        Kernel* kernel = nullptr;
        if (int32_t e = ResolveKernel(func, &kernel)) return static_cast<cudaError_t>(Fail(e));
        in.put_u64(kernel->remote_id);
        for (unsigned v : {gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z}) in.put_u32(v);
        in.put_u64(sharedMem);
        in.put_u64(reinterpret_cast<uint64_t>(stream));
        std::string params(kernel->param_bytes, '\0');
        for (size_t i = 0; i < kernel->params.size(); ++i) {
            if (!args || !args[i]) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
            std::memcpy(params.data() + kernel->params[i].first, args[i], kernel->params[i].second);
        }
        in.put(params.data(), params.size());
    }
    return static_cast<cudaError_t>(Post(kApi___tunaLaunchKernel, in));
}

TUNA_CUDART_EXPORT cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) {
    if (!attr) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    Reply reply;
    if (int32_t e = FunctionCall(kApi___tunaFuncGetAttributes, func, ArgWriter{}, &reply)) {
        return static_cast<cudaError_t>(e);
    }
    if (reply.out_args.size() != sizeof(*attr)) return static_cast<cudaError_t>(Fail(cudaErrorUnknown));
    std::memcpy(attr, reply.out_args.data(), sizeof(*attr));
    return cudaSuccess;
}

TUNA_CUDART_EXPORT cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) {
    ArgWriter extra;
    extra.put_u32(static_cast<uint32_t>(attr));
    extra.put_u32(static_cast<uint32_t>(value));
    Reply reply;
    return static_cast<cudaError_t>(FunctionCall(kApi___tunaFuncSetAttribute, func, extra, &reply));
}

// A cache preference is a hint the runtime may ignore; so does Tuna.
TUNA_CUDART_EXPORT cudaError_t cudaFuncSetCacheConfig(const void*, cudaFuncCache) { return cudaSuccess; }

TUNA_CUDART_EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    if (!numBlocks) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    ArgWriter extra;
    extra.put_u32(static_cast<uint32_t>(blockSize));
    extra.put_u64(dynamicSMemSize);
    extra.put_u32(flags);
    Reply reply;
    if (int32_t e = FunctionCall(kApi___tunaOccupancy, func, extra, &reply)) return static_cast<cudaError_t>(e);
    if (reply.out_args.size() != 4) return static_cast<cudaError_t>(Fail(cudaErrorUnknown));
    std::memcpy(numBlocks, reply.out_args.data(), 4);
    return cudaSuccess;
}

TUNA_CUDART_EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func,
                                                                             int blockSize, size_t dynamicSMemSize) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, 0);
}

// ------------------------------------------------------- symbol variables

namespace {

cudaError_t SymbolAddress(const void* symbol, size_t offset, size_t count, char** device) {
    uint64_t address = 0;
    size_t size = 0;
    if (int32_t e = ResolveVariable(symbol, &address, &size)) return static_cast<cudaError_t>(Fail(e));
    if (offset > size || count > size - offset) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    *device = reinterpret_cast<char*>(address) + offset;
    return cudaSuccess;
}

}  // namespace

TUNA_CUDART_EXPORT cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
    if (!devPtr) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    char* device = nullptr;
    cudaError_t e = SymbolAddress(symbol, 0, 0, &device);
    if (e == cudaSuccess) *devPtr = device;
    return e;
}

TUNA_CUDART_EXPORT cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol) {
    if (!size) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_variables.find(symbol);
    if (it == g_variables.end()) return static_cast<cudaError_t>(Fail(cudaErrorInvalidSymbol));
    *size = it->second.size;
    return cudaSuccess;
}

TUNA_CUDART_EXPORT cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
                                                  cudaMemcpyKind kind) {
    char* device = nullptr;
    if (cudaError_t e = SymbolAddress(symbol, offset, count, &device)) return e;
    return cudaMemcpy(device, src, count, kind);
}

TUNA_CUDART_EXPORT cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
                                                    cudaMemcpyKind kind) {
    char* device = nullptr;
    if (cudaError_t e = SymbolAddress(symbol, offset, count, &device)) return e;
    return cudaMemcpy(dst, device, count, kind);
}

TUNA_CUDART_EXPORT cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count,
                                                       size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    char* device = nullptr;
    if (cudaError_t e = SymbolAddress(symbol, offset, count, &device)) return e;
    return cudaMemcpyAsync(device, src, count, kind, stream);
}

TUNA_CUDART_EXPORT cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset,
                                                         cudaMemcpyKind kind, cudaStream_t stream) {
    char* device = nullptr;
    if (cudaError_t e = SymbolAddress(symbol, offset, count, &device)) return e;
    return cudaMemcpyAsync(dst, device, count, kind, stream);
}

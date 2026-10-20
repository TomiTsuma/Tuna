// Tuna's replacement CUDA driver library (Phase 3e): the hand-written exports
// and the plumbing the generated stubs (driver_client_stubs.inc) use.
//
// Tuna remotes CUDA at the runtime API, so the driver shim is small: the
// device queries, the primary-context calls frameworks make (PyTorch checks
// cuDevicePrimaryCtxGetState before every op and makes the primary context
// current before creating a cuBLAS handle), error strings, and
// cuGetProcAddress. Everything else in the driver ABI is exported as a
// not-supported stub, so any program or library that links libcuda loads.
//
// Contexts. A CUcontext is a host pointer inside the server's worker, so it
// never crosses the wire. Tuna supports each device's primary context only,
// which is the one the server's runtime creates and uses; the client hands
// out a token per device for it. The current context is tracked per client
// thread: a thread that set one (cuCtxSetCurrent/cuCtxPushCurrent) sees what
// it set; any other thread sees the primary context of the runtime's current
// device once that context is active, which is what CUDA does after the
// runtime's first call on a thread. Making a token current also makes its
// device the runtime's current device, since every client thread shares one
// server thread (see cuda/README.md, "Limitations").

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "cuda_api_ids.h"
#include "tuna/cuda/bridge.h"
#include "tuna/cuda/driver_client.h"

namespace tuna::cudrv {

namespace {

#include "driver_error_strings.inc"

// Context tokens: an address range no allocation uses, one 16-byte slot per
// device. Only ever compared, never dereferenced.
constexpr uintptr_t kContextTokenBase = 0x7c7c00000000ull;
constexpr int kMaxDevices = 1 << 16;

CUcontext Token(int device) { return reinterpret_cast<CUcontext>(kContextTokenBase + (uintptr_t(device) << 4)); }

bool TokenDevice(CUcontext ctx, int* device) {
    const auto value = reinterpret_cast<uintptr_t>(ctx);
    if (value < kContextTokenBase || (value - kContextTokenBase) % 16 != 0) return false;
    const uintptr_t index = (value - kContextTokenBase) >> 4;
    if (index >= kMaxDevices) return false;
    *device = static_cast<int>(index);
    return true;
}

std::mutex g_mutex;
std::map<std::pair<int, int>, int> g_attributes;      // guarded by g_mutex; (attrib, dev) -> value
std::map<int, std::pair<uint64_t, unsigned>> g_active;  // guarded by g_mutex; dev -> (generation, flags)

// The thread's explicitly set context stack; `set` stays false until the
// thread calls cuCtxSetCurrent / cuCtxPushCurrent.
struct ThreadContexts {
    bool set = false;
    std::vector<CUcontext> stack;
};
thread_local ThreadContexts t_contexts;

bool LogEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("TUNA_LOG");
        return value && *value;
    }();
    return enabled;
}

const CuErrorText* FindError(int value) {
    for (const auto& e : kCuErrorTexts) {
        if (e.value == value) return &e;
    }
    return nullptr;
}

// Makes `device` the runtime's current device (the server thread every client
// thread shares), so a context token made current here is the one the
// following runtime and driver calls use.
CUresult Activate(int device) {
    int current = -1;
    if (cudaError_t e = cudaGetDevice(&current); e != cudaSuccess) return static_cast<CUresult>(e);
    if (current == device) return CUDA_SUCCESS;
    // cudaError_t and CUresult share these values (invalid device, no device).
    return static_cast<CUresult>(cudaSetDevice(device));
}

// This library's own module, for cuGetProcAddress.
#if defined(_WIN32)
void* SelfSymbol(const char* name) {
    static HMODULE self = [] {
        HMODULE module = nullptr;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&Token), &module);
        return module;
    }();
    return self ? reinterpret_cast<void*>(GetProcAddress(self, name)) : nullptr;
}
#else
void* SelfSymbol(const char* name) {
    static void* self = [] {
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void*>(&Token), &info) || !info.dli_fname) return static_cast<void*>(nullptr);
        return dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    }();
    return self ? dlsym(self, name) : nullptr;
}
#endif

}  // namespace

cudart::Reply Call(uint32_t api_id, const cudart::ArgWriter& in, bool synchronizing) {
    TunaCallReply raw{};
    tuna_cudart_call(api_id, in.str().data(), in.str().size(), nullptr, 0,
                     synchronizing ? kTunaCallSynchronize : kTunaCall, &raw);
    cudart::Reply reply;
    reply.result = raw.result;
    reply.transport_error = raw.transport_error != 0;
    reply.out_args.assign(raw.out_args, raw.out_args_size);
    reply.bulk_out.assign(raw.bulk_out, raw.bulk_out_size);
    return reply;
}

int Post(uint32_t api_id, const cudart::ArgWriter& in) {
    const int32_t queued = tuna_cudart_call(api_id, in.str().data(), in.str().size(), nullptr, 0, kTunaPost, nullptr);
    return queued == 0 || queued == kNoDevice ? queued : kUnknown;
}

int Finish(const cudart::Reply& reply) { return reply.transport_error ? kUnknown : reply.result; }

int Unsupported(const char* name) {
    if (LogEnabled()) {
        static std::mutex mutex;
        static std::set<std::string> reported;
        std::lock_guard<std::mutex> lock(mutex);
        if (reported.insert(name).second) std::fprintf(stderr, "tuna: %s is not supported by Tuna yet\n", name);
    }
    return kNotSupported;
}

void ResetStateForTesting() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_attributes.clear();
    g_active.clear();
    t_contexts = ThreadContexts{};
}

}  // namespace tuna::cudrv

using namespace tuna::cudart;  // the generated stubs name kApi_* unqualified
using namespace tuna::cudrv;

// Generated stubs: the exported simple functions, plus static
// tuna_remote_<name> functions for the _wrap ones below.
#include "driver_client_stubs.inc"

// ------------------------------------------------ cached device queries

// A device attribute never changes, and frameworks query them constantly.
TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    const auto key = std::make_pair(static_cast<int>(attrib), static_cast<int>(dev));
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (auto it = g_attributes.find(key); it != g_attributes.end()) {
            *pi = it->second;
            return CUDA_SUCCESS;
        }
    }
    const CUresult r = tuna_remote_cuDeviceGetAttribute(pi, attrib, dev);
    if (r == CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_attributes[key] = *pi;
    }
    return r;
}

// PyTorch asks before every op whether the primary context exists. Once it is
// active it stays active until the runtime resets the device, which bumps the
// runtime shim's context generation, so only "active" answers are cached.
TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    if (!flags || !active) return CUDA_ERROR_INVALID_VALUE;
    const uint64_t generation = tuna_cudart_context_generation();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (auto it = g_active.find(dev); it != g_active.end() && it->second.first == generation) {
            *flags = it->second.second;
            *active = 1;
            return CUDA_SUCCESS;
        }
    }
    const CUresult r = tuna_remote_cuDevicePrimaryCtxGetState(dev, flags, active);
    if (r == CUDA_SUCCESS && *active) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_active[dev] = {generation, *flags};
    }
    return r;
}

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice dev) {
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    ArgWriter in;
    in.put(&dev, 4);
    Reply reply = Call(kApi_cuDeviceGetName, in);
    const int r = Finish(reply);
    if (r != CUDA_SUCCESS) return static_cast<CUresult>(r);
    const size_t n = std::min(reply.out_args.size(), static_cast<size_t>(len) - 1);
    std::memcpy(name, reply.out_args.data(), n);
    name[n] = '\0';
    return CUDA_SUCCESS;
}

// ------------------------------------------------------ primary contexts

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || dev >= kMaxDevices) return CUDA_ERROR_INVALID_DEVICE;
    ArgWriter in;
    in.put(&dev, 4);
    const int r = Finish(Call(kApi_cuDevicePrimaryCtxRetain, in));
    if (r != CUDA_SUCCESS) return static_cast<CUresult>(r);
    *pctx = Token(dev);
    return CUDA_SUCCESS;
}

// The server keeps each primary context it retained for the whole session (the
// runtime holds it anyway), so releasing changes nothing.
TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    return dev < 0 || dev >= kMaxDevices ? CUDA_ERROR_INVALID_DEVICE : CUDA_SUCCESS;
}

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuCtxGetCurrent(CUcontext* pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (t_contexts.set) {
        *pctx = t_contexts.stack.empty() ? nullptr : t_contexts.stack.back();
        return CUDA_SUCCESS;
    }
    *pctx = nullptr;
    int device = 0;
    if (cudaError_t e = cudaGetDevice(&device); e != cudaSuccess) {
        return e == cudaErrorNoDevice ? CUDA_ERROR_NO_DEVICE : CUDA_SUCCESS;
    }
    unsigned int flags = 0;
    int active = 0;
    if (CUresult r = cuDevicePrimaryCtxGetState(device, &flags, &active); r != CUDA_SUCCESS) return r;
    if (active) *pctx = Token(device);
    return CUDA_SUCCESS;
}

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
    int device = 0;
    if (ctx) {
        if (!TokenDevice(ctx, &device)) return CUDA_ERROR_INVALID_CONTEXT;
        if (CUresult r = Activate(device); r != CUDA_SUCCESS) return r;
    }
    // Replaces the top of the thread's stack (or pushes onto an empty one).
    t_contexts.set = true;
    if (t_contexts.stack.empty()) {
        t_contexts.stack.push_back(ctx);
    } else {
        t_contexts.stack.back() = ctx;
    }
    return CUDA_SUCCESS;
}

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext ctx) {
    int device = 0;
    if (!ctx || !TokenDevice(ctx, &device)) return CUDA_ERROR_INVALID_CONTEXT;
    if (CUresult r = Activate(device); r != CUDA_SUCCESS) return r;
    if (!t_contexts.set) {
        // Keep what the thread implicitly had, so popping returns to it.
        CUcontext implicit = nullptr;
        cuCtxGetCurrent(&implicit);
        t_contexts.set = true;
        if (implicit) t_contexts.stack.push_back(implicit);
    }
    t_contexts.stack.push_back(ctx);
    return CUDA_SUCCESS;
}

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext* pctx) {
    if (!t_contexts.set || t_contexts.stack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    if (pctx) *pctx = t_contexts.stack.back();
    t_contexts.stack.pop_back();
    int device = 0;
    if (!t_contexts.stack.empty() && TokenDevice(t_contexts.stack.back(), &device)) return Activate(device);
    return CUDA_SUCCESS;
}

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    CUcontext ctx = nullptr;
    if (CUresult r = cuCtxGetCurrent(&ctx); r != CUDA_SUCCESS) return r;
    int index = 0;
    if (!ctx || !TokenDevice(ctx, &index)) return CUDA_ERROR_INVALID_CONTEXT;
    *device = index;
    return CUDA_SUCCESS;
}

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuCtxSynchronize(void) {
    CUcontext ctx = nullptr;
    if (CUresult r = cuCtxGetCurrent(&ctx); r != CUDA_SUCCESS) return r;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    // A failed fire-and-forget call is reported here, as the runtime's code;
    // the two share values for launch and memory errors.
    return static_cast<CUresult>(cudaDeviceSynchronize());
}

// ---------------------------------------------------------- error strings

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuGetErrorName(CUresult error, const char** pStr) {
    if (!pStr) return CUDA_ERROR_INVALID_VALUE;
    const CuErrorText* e = FindError(static_cast<int>(error));
    *pStr = e ? e->name : nullptr;
    return e ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuGetErrorString(CUresult error, const char** pStr) {
    if (!pStr) return CUDA_ERROR_INVALID_VALUE;
    const CuErrorText* e = FindError(static_cast<int>(error));
    *pStr = e ? e->text : nullptr;
    return e ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

// ------------------------------------------------------- cuGetProcAddress

// The runtime and NVIDIA's libraries look driver entry points up by base name
// ("cuMemAlloc") and get the ABI variant for their CUDA version
// ("cuMemAlloc_v2"). Every name resolves to this shim's own export, newest
// variant first, so nothing reaches past Tuna to a local driver. The
// per-thread-default-stream variants (_ptds/_ptsz) are not offered: all client
// threads share one server thread, so the legacy default stream is the only one.
TUNA_CUDA_DRIVER_EXPORT CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** pfn, int, cuuint64_t,
                                                              CUdriverProcAddressQueryResult* symbolStatus) {
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
    *pfn = nullptr;
    if (std::strncmp(symbol, "cu", 2) == 0 && std::strncmp(symbol, "cuda", 4) != 0) {
        for (const char* suffix : {"_v4", "_v3", "_v2", ""}) {
            if (void* fn = SelfSymbol((std::string(symbol) + suffix).c_str())) {
                *pfn = fn;
                break;
            }
        }
    }
    if (symbolStatus) *symbolStatus = *pfn ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return *pfn ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

// Marker the server checks before loading a "real" libcuda, so a misconfigured
// server never loads this shim and calls itself.
TUNA_CUDA_DRIVER_EXPORT int tuna_cuda_driver_shim(void) { return 1; }

// Hand-written runtime exports of the replacement libcudart: the memcpy
// family (direction-dependent, payloads on the bulk channel), host "pinned"
// memory (plain local memory: a socket cannot back a DMA mapping), error
// strings, versions, and the wrappers around generated stubs that add caching
// or allocation tracking. The generated stubs are included at the bottom.

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <utility>

#include "client_internal.h"
#include "cuda_api_ids.h"
#include "tuna/cuda/client.h"

using namespace tuna::cudart;

// Generated stubs: the exported simple functions, plus static
// tuna_remote_<name> functions for the _wrap ones used below.
#include "cuda_client_stubs.inc"

namespace tuna::cudart {

namespace {

#include "cuda_error_strings.inc"

std::mutex g_mutex;
std::map<uintptr_t, size_t> g_device_ranges;           // guarded by g_mutex
std::set<void*> g_host_allocations;                    // guarded by g_mutex
std::map<int, cudaDeviceProp> g_props;                 // guarded by g_mutex
std::map<std::pair<int, int>, int> g_attributes;       // guarded by g_mutex; (attr, device)
int g_current_device = -1;                             // guarded by g_mutex; -1 = not yet set

// One bulk payload is capped (kMaxBulkTransferBytes); larger copies go in
// pieces, which costs nothing extra since each piece is a full round trip.
constexpr size_t kMaxCopyPiece = size_t{1} << 30;

const CudaErrorText* FindError(int value) {
    for (const auto& e : kCudaErrorTexts) {
        if (e.value == value) return &e;
    }
    return nullptr;
}

// cudaMemcpyDefault: the client cannot ask UVA, but it knows which addresses
// are server allocations.
cudaMemcpyKind Resolve(cudaMemcpyKind kind, void* dst, const void* src) {
    if (kind != cudaMemcpyDefault) return kind;
    const bool dst_dev = IsDevicePointer(dst);
    const bool src_dev = IsDevicePointer(src);
    if (dst_dev && src_dev) return cudaMemcpyDeviceToDevice;
    if (dst_dev) return cudaMemcpyHostToDevice;
    if (src_dev) return cudaMemcpyDeviceToHost;
    return cudaMemcpyHostToHost;
}

// One remoted copy (or piece). Wire: dst, src, count (u64), kind (u32), and
// for the async form the stream (u64). H2D payloads ride the bulk channel in;
// D2H payloads come back on it.
cudaError_t RemoteCopy(void* dst, const void* src, size_t count, cudaMemcpyKind kind, bool async,
                       cudaStream_t stream) {
    ArgWriter in;
    in.put_u64(reinterpret_cast<uint64_t>(dst));
    in.put_u64(reinterpret_cast<uint64_t>(src));
    in.put_u64(count);
    in.put_u32(static_cast<uint32_t>(kind));
    if (async) in.put_u64(reinterpret_cast<uint64_t>(stream));
    const uint32_t api = async ? kApi_cudaMemcpyAsync : kApi_cudaMemcpy;

    std::string bulk_in;
    if (kind == cudaMemcpyHostToDevice) bulk_in.assign(static_cast<const char*>(src), count);
    // An async copy that returns nothing to the host can be fire-and-forget;
    // a D2H copy must wait for its bytes either way.
    if (async && kind != cudaMemcpyDeviceToHost) return static_cast<cudaError_t>(Post(api, in, bulk_in));

    Reply reply = Call(api, in, bulk_in, /*synchronizing=*/!async);
    if (!reply.transport_error && reply.result == 0 && kind == cudaMemcpyDeviceToHost) {
        if (reply.bulk_out.size() != count) return static_cast<cudaError_t>(Fail(cudaErrorUnknown));
        std::memcpy(dst, reply.bulk_out.data(), count);
    }
    return static_cast<cudaError_t>(Finish(reply));
}

cudaError_t Copy(void* dst, const void* src, size_t count, cudaMemcpyKind kind, bool async, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    kind = Resolve(kind, dst, src);
    if (kind == cudaMemcpyHostToHost) {
        std::memmove(dst, src, count);
        return cudaSuccess;
    }
    if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice) {
        return static_cast<cudaError_t>(Fail(cudaErrorInvalidMemcpyDirection));
    }
    for (size_t done = 0; done < count;) {
        const size_t piece = std::min(count - done, kMaxCopyPiece);
        cudaError_t e = RemoteCopy(static_cast<char*>(dst) + done, static_cast<const char*>(src) + done, piece,
                                   kind, async, stream);
        if (e != cudaSuccess) return e;
        done += piece;
    }
    return cudaSuccess;
}

cudaError_t HostAlloc(void** ptr, size_t size) {
    if (!ptr) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    // Page-aligned like real pinned memory, so code that checks alignment works.
    const size_t rounded = ((size ? size : 1) + 4095) & ~size_t{4095};
#ifdef _WIN32
    void* p = _aligned_malloc(rounded, 4096);
#else
    void* p = std::aligned_alloc(4096, rounded);
#endif
    if (!p) return static_cast<cudaError_t>(Fail(cudaErrorMemoryAllocation));
    std::lock_guard<std::mutex> lock(g_mutex);
    g_host_allocations.insert(p);
    *ptr = p;
    return cudaSuccess;
}

}  // namespace

void TrackDeviceRange(const void* base, size_t size) {
    if (!base) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_device_ranges[reinterpret_cast<uintptr_t>(base)] = size ? size : 1;
}

void UntrackDeviceRange(const void* base) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_device_ranges.erase(reinterpret_cast<uintptr_t>(base));
}

bool IsDevicePointer(const void* ptr) {
    const auto p = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_device_ranges.upper_bound(p);
    if (it == g_device_ranges.begin()) return false;
    --it;
    return p < it->first + it->second;
}

void ResetRuntimeState() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_device_ranges.clear();
    g_props.clear();
    g_attributes.clear();
    g_current_device = -1;
}

}  // namespace tuna::cudart

// ------------------------------------------------------------------ memcpy

TUNA_CUDART_EXPORT cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    return Copy(dst, src, count, kind, /*async=*/false, nullptr);
}

TUNA_CUDART_EXPORT cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                                               cudaStream_t stream) {
    return Copy(dst, src, count, kind, /*async=*/true, stream);
}

// ------------------------------------------------- allocations (wrapped stubs)

TUNA_CUDART_EXPORT cudaError_t cudaMalloc(void** devPtr, size_t size) {
    cudaError_t e = tuna_remote_cudaMalloc(devPtr, size);
    if (e == cudaSuccess && devPtr) TrackDeviceRange(*devPtr, size);
    return e;
}

TUNA_CUDART_EXPORT cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
    cudaError_t e = tuna_remote_cudaMallocPitch(devPtr, pitch, width, height);
    if (e == cudaSuccess && devPtr && pitch) TrackDeviceRange(*devPtr, *pitch * height);
    return e;
}

TUNA_CUDART_EXPORT cudaError_t cudaFree(void* devPtr) {
    cudaError_t e = tuna_remote_cudaFree(devPtr);
    if (e == cudaSuccess && devPtr) UntrackDeviceRange(devPtr);
    return e;
}

// ------------------------------------------- host memory (client-local)

TUNA_CUDART_EXPORT cudaError_t cudaMallocHost(void** ptr, size_t size) { return HostAlloc(ptr, size); }

TUNA_CUDART_EXPORT cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int) { return HostAlloc(pHost, size); }

TUNA_CUDART_EXPORT cudaError_t cudaFreeHost(void* ptr) {
    if (!ptr) return cudaSuccess;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_host_allocations.erase(ptr) == 0) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    }
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
    return cudaSuccess;
}

// Registering memory is a DMA optimization; over a network every copy is
// staged anyway, so registration succeeds and changes nothing.
TUNA_CUDART_EXPORT cudaError_t cudaHostRegister(void*, size_t, unsigned int) { return cudaSuccess; }
TUNA_CUDART_EXPORT cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }

TUNA_CUDART_EXPORT cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!pFlags || !g_host_allocations.count(pHost)) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    *pFlags = 0;
    return cudaSuccess;
}

// --------------------------------------------- device state (wrapped stubs)

TUNA_CUDART_EXPORT cudaError_t cudaSetDevice(int device) {
    cudaError_t e = tuna_remote_cudaSetDevice(device);
    if (e == cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_current_device != device) {
            if (g_current_device != -1) BumpContextGeneration();  // symbol addresses are per context
            g_current_device = device;
        }
    }
    return e;
}

// Device properties and attributes never change for a device, and frameworks
// query them constantly (PyTorch asks per op), so each is fetched once.
TUNA_CUDART_EXPORT cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device) {
    if (!prop) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_props.find(device);
        if (it != g_props.end()) {
            *prop = it->second;
            return cudaSuccess;
        }
    }
    cudaError_t e = tuna_remote_cudaGetDeviceProperties_v2(prop, device);
    if (e == cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_props[device] = *prop;
    }
    return e;
}

TUNA_CUDART_EXPORT cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    if (!value) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    const auto key = std::make_pair(static_cast<int>(attr), device);
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_attributes.find(key);
        if (it != g_attributes.end()) {
            *value = it->second;
            return cudaSuccess;
        }
    }
    cudaError_t e = tuna_remote_cudaDeviceGetAttribute(value, attr, device);
    if (e == cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_attributes[key] = *value;
    }
    return e;
}

TUNA_CUDART_EXPORT cudaError_t cudaDeviceReset(void) {
    Reply reply = Call(kApi_cudaDeviceReset, ArgWriter{}, {}, /*synchronizing=*/true);
    // The context is gone either way: device pointers and module state with it.
    BumpContextGeneration();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_device_ranges.clear();
    }
    return static_cast<cudaError_t>(Finish(reply));
}

// ------------------------------------------------------------------ errors

// The client's own last error (a failed call it made, or one it rejected
// locally) wins; otherwise the server runtime's, which also flushes and
// reports any failed fire-and-forget call.
TUNA_CUDART_EXPORT cudaError_t cudaGetLastError(void) {
    const int32_t local = LastError(/*clear=*/true);
    Reply reply = Call(kApi_cudaGetLastError, ArgWriter{}, {}, /*synchronizing=*/true);
    const int32_t remote = reply.transport_error ? kErrUnknown : reply.result;
    return static_cast<cudaError_t>(local ? local : remote);
}

TUNA_CUDART_EXPORT cudaError_t cudaPeekAtLastError(void) {
    const int32_t local = LastError(/*clear=*/false);
    if (local) return static_cast<cudaError_t>(local);
    Reply reply = Call(kApi_cudaPeekAtLastError, ArgWriter{}, {}, /*synchronizing=*/true);
    const int32_t remote = reply.transport_error ? kErrUnknown : reply.result;
    if (remote) Fail(remote);  // a latched async error stays visible until cleared
    return static_cast<cudaError_t>(remote);
}

TUNA_CUDART_EXPORT const char* cudaGetErrorName(cudaError_t error) {
    const CudaErrorText* e = FindError(static_cast<int>(error));
    return e ? e->name : "cudaErrorUnknown";
}

TUNA_CUDART_EXPORT const char* cudaGetErrorString(cudaError_t error) {
    const CudaErrorText* e = FindError(static_cast<int>(error));
    return e ? e->text : "unrecognized error code";
}

// ---------------------------------------------------------------- versions

TUNA_CUDART_EXPORT cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
    if (!runtimeVersion) return static_cast<cudaError_t>(Fail(cudaErrorInvalidValue));
    *runtimeVersion = CUDART_VERSION;  // the runtime ABI this shim implements
    return cudaSuccess;
}

TUNA_CUDART_EXPORT cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
TUNA_CUDART_EXPORT cudaError_t cudaProfilerStop(void) { return cudaSuccess; }

// Marker the server checks before loading a "real" libcudart, so a
// misconfigured server never loads this shim and calls itself.
TUNA_CUDART_EXPORT int tuna_cudart_shim(void) { return 1; }

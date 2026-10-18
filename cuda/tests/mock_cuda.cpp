// A mock CUDA stack (runtime + the driver calls kernel launch uses) that
// models a tiny device in host memory, so the whole remoting path - client
// shim, transport, server dispatch - is testable with no GPU.
//
// "Device memory" is host buffers addressed by fake device pointers far from
// any real host address. Any fatbin with a valid header loads as a module that
// contains three kernels and one __constant__ variable:
//   vecAdd(const float* a, const float* b, float* c, int n)   c[i] = a[i] + b[i]
//   scale(float* data, int n)                                  data[i] *= gScale
//   fill(int* data, int value, int n)                          data[i] = value
//   __constant__ float gScale (initially 1.0)
// Kernels execute synchronously on the host with real grid/block bounds, so
// launch-configuration bugs show up as wrong results.
//
// It is linked into tuna_cuda_tests (in-process) and also built as a shared
// library that tuna_server loads in place of libcudart: it exports only
// tuna_cuda_resolve, which backend_real.cpp uses as its symbol source.

#include "mock_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kFatbinMagic = 0xBA55ED50;

struct Module {
    uint64_t constant_base;  // device address of gScale
};

struct Function {
    std::string name;
};

struct Event {
    uint64_t sequence = 0;
};

struct MockDevice {
    std::map<uint64_t, std::vector<char>> buffers;  // device address -> bytes
    uint64_t next_address = 0x7f0000100000ull;
    int device_count = 2;
    int current_device = 0;
    bool initialized = false;
    size_t total_mem = 8ull << 30;
    std::map<uint64_t, Module> modules;
    std::map<uint64_t, Function> functions;
    std::map<uint64_t, Event> events;
    uint64_t next_handle = 0x5000;
    uint64_t sequence = 0;
    MockCudaStats stats;
};
MockDevice g;

size_t UsedMemory() {
    size_t used = 0;
    for (const auto& [address, bytes] : g.buffers) used += bytes.size();
    return used;
}

uint64_t Allocate(size_t size) {
    const uint64_t address = g.next_address;
    g.next_address += (size + 0xfff) & ~uint64_t{0xfff};
    g.next_address += 0x1000;  // guard gap: an overrun lands in no buffer
    g.buffers[address].resize(size);
    return address;
}

// Host bytes behind [address, address + count), or nullptr if that range is
// not inside one allocation. Interior pointers are allowed, like the real thing.
char* Find(uint64_t address, size_t count) {
    auto it = g.buffers.upper_bound(address);
    if (it == g.buffers.begin()) return nullptr;
    --it;
    const uint64_t offset = address - it->first;
    if (offset > it->second.size() || count > it->second.size() - offset) return nullptr;
    return it->second.data() + offset;
}

template <typename T>
T* Device(uint64_t address, size_t count) {
    return reinterpret_cast<T*>(Find(address, count * sizeof(T)));
}

// ------------------------------------------------------------------ runtime

cudaError_t GetDeviceCount(int* count) {
    *count = g.device_count;
    return cudaSuccess;
}

cudaError_t GetDevice(int* device) {
    *device = g.current_device;
    return cudaSuccess;
}

cudaError_t SetDevice(int device) {
    if (device < 0 || device >= g.device_count) return cudaErrorInvalidDevice;
    g.current_device = device;
    return cudaSuccess;
}

cudaError_t DeviceSynchronize() { return cudaSuccess; }

cudaError_t DeviceReset() {
    g.buffers.clear();
    g.modules.clear();
    g.functions.clear();
    ++g.stats.resets;
    return cudaSuccess;
}

cudaError_t GetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (device < 0 || device >= g.device_count) return cudaErrorInvalidDevice;
    ++g.stats.property_queries;
    std::memset(prop, 0, sizeof(*prop));
    std::snprintf(prop->name, sizeof(prop->name), "Tuna Mock GPU %d", device);
    prop->major = 8;
    prop->minor = 9;
    prop->totalGlobalMem = g.total_mem;
    prop->multiProcessorCount = 20;
    prop->maxThreadsPerBlock = 1024;
    prop->warpSize = 32;
    return cudaSuccess;
}

cudaError_t DeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    if (device < 0 || device >= g.device_count) return cudaErrorInvalidDevice;
    ++g.stats.attribute_queries;
    switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock: *value = 1024; break;
    case cudaDevAttrMultiProcessorCount: *value = 20; break;
    case cudaDevAttrComputeCapabilityMajor: *value = 8; break;
    case cudaDevAttrComputeCapabilityMinor: *value = 9; break;
    case cudaDevAttrWarpSize: *value = 32; break;
    default: *value = 0; break;
    }
    return cudaSuccess;
}

cudaError_t DriverGetVersion(int* version) {
    *version = 12080;
    return cudaSuccess;
}

cudaError_t Malloc(void** ptr, size_t size) {
    if (size > g.total_mem - UsedMemory()) return cudaErrorMemoryAllocation;
    *ptr = reinterpret_cast<void*>(Allocate(size));
    return cudaSuccess;
}

cudaError_t Free(void* ptr) {
    g.initialized = true;  // cudaFree(nullptr) is how callers force initialization
    if (!ptr) return cudaSuccess;
    if (g.buffers.erase(reinterpret_cast<uint64_t>(ptr)) == 0) return cudaErrorInvalidValue;
    return cudaSuccess;
}

cudaError_t MemGetInfo(size_t* free, size_t* total) {
    *total = g.total_mem;
    *free = g.total_mem - UsedMemory();
    return cudaSuccess;
}

cudaError_t Memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    const auto d = reinterpret_cast<uint64_t>(dst);
    const auto s = reinterpret_cast<uint64_t>(src);
    char* target = nullptr;
    const char* source = nullptr;
    switch (kind) {
    case cudaMemcpyHostToDevice: target = Find(d, count); source = static_cast<const char*>(src); break;
    case cudaMemcpyDeviceToHost: target = static_cast<char*>(dst); source = Find(s, count); break;
    case cudaMemcpyDeviceToDevice: target = Find(d, count); source = Find(s, count); break;
    default: return cudaErrorInvalidMemcpyDirection;
    }
    if (!target || !source) return cudaErrorInvalidValue;
    std::memmove(target, source, count);
    return cudaSuccess;
}

cudaError_t MemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t) {
    return Memcpy(dst, src, count, kind);
}

cudaError_t Memset(void* ptr, int value, size_t count) {
    char* target = Find(reinterpret_cast<uint64_t>(ptr), count);
    if (!target) return cudaErrorInvalidValue;
    std::memset(target, value, count);
    return cudaSuccess;
}

cudaError_t MemsetAsync(void* ptr, int value, size_t count, cudaStream_t) { return Memset(ptr, value, count); }

cudaError_t StreamCreate(cudaStream_t* stream) {
    *stream = reinterpret_cast<cudaStream_t>(g.next_handle++);
    return cudaSuccess;
}
cudaError_t StreamCreateWithFlags(cudaStream_t* stream, unsigned int) { return StreamCreate(stream); }
cudaError_t StreamDestroy(cudaStream_t) { return cudaSuccess; }
cudaError_t StreamSynchronize(cudaStream_t) { return cudaSuccess; }

cudaError_t EventCreate(cudaEvent_t* event) {
    const uint64_t handle = g.next_handle++;
    g.events[handle] = Event{};
    *event = reinterpret_cast<cudaEvent_t>(handle);
    return cudaSuccess;
}
cudaError_t EventCreateWithFlags(cudaEvent_t* event, unsigned int) { return EventCreate(event); }

cudaError_t EventRecord(cudaEvent_t event, cudaStream_t) {
    auto it = g.events.find(reinterpret_cast<uint64_t>(event));
    if (it == g.events.end()) return cudaErrorInvalidResourceHandle;
    it->second.sequence = ++g.sequence;
    return cudaSuccess;
}

cudaError_t EventSynchronize(cudaEvent_t event) {
    return g.events.count(reinterpret_cast<uint64_t>(event)) ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

// Elapsed "time" is the number of operations between the two records, in ms.
cudaError_t EventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    auto a = g.events.find(reinterpret_cast<uint64_t>(start));
    auto b = g.events.find(reinterpret_cast<uint64_t>(end));
    if (a == g.events.end() || b == g.events.end()) return cudaErrorInvalidResourceHandle;
    *ms = static_cast<float>(b->second.sequence - a->second.sequence);
    return cudaSuccess;
}

cudaError_t EventDestroy(cudaEvent_t event) {
    return g.events.erase(reinterpret_cast<uint64_t>(event)) ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

cudaError_t GetLastError() { return cudaSuccess; }

// ------------------------------------------------------------------- driver

CUresult CtxGetCurrent(CUcontext* ctx) {
    *ctx = g.initialized ? reinterpret_cast<CUcontext>(0xC000 + g.current_device) : nullptr;
    return CUDA_SUCCESS;
}

CUresult ModuleLoadData(CUmodule* module, const void* image) {
    uint32_t magic = 0;
    std::memcpy(&magic, image, 4);
    if (magic != kFatbinMagic) return CUDA_ERROR_INVALID_IMAGE;
    const uint64_t handle = g.next_handle++;
    const uint64_t constant = Allocate(sizeof(float));
    const float one = 1.0f;
    std::memcpy(Find(constant, sizeof(float)), &one, sizeof(float));
    g.modules[handle] = Module{constant};
    ++g.stats.module_loads;
    *module = reinterpret_cast<CUmodule>(handle);
    return CUDA_SUCCESS;
}

CUresult ModuleGetFunction(CUfunction* function, CUmodule module, const char* name) {
    if (!g.modules.count(reinterpret_cast<uint64_t>(module))) return CUDA_ERROR_INVALID_HANDLE;
    const std::string n = name;
    if (n != "vecAdd" && n != "scale" && n != "fill") return CUDA_ERROR_NOT_FOUND;
    const uint64_t handle = g.next_handle++;
    g.functions[handle] = Function{n};
    // Remember which module a "scale" reads gScale from: the latest one loaded.
    *function = reinterpret_cast<CUfunction>(handle);
    return CUDA_SUCCESS;
}

CUresult ModuleGetGlobal(CUdeviceptr* address, size_t* size, CUmodule module, const char* name) {
    auto it = g.modules.find(reinterpret_cast<uint64_t>(module));
    if (it == g.modules.end()) return CUDA_ERROR_INVALID_HANDLE;
    if (std::string(name) != "gScale") return CUDA_ERROR_NOT_FOUND;
    *address = it->second.constant_base;
    *size = sizeof(float);
    return CUDA_SUCCESS;
}

CUresult FuncGetParamInfo(CUfunction function, size_t index, size_t* offset, size_t* size) {
    auto it = g.functions.find(reinterpret_cast<uint64_t>(function));
    if (it == g.functions.end()) return CUDA_ERROR_INVALID_HANDLE;
    static const std::map<std::string, std::vector<std::pair<size_t, size_t>>> layouts = {
        {"vecAdd", {{0, 8}, {8, 8}, {16, 8}, {24, 4}}},
        {"scale", {{0, 8}, {8, 4}}},
        {"fill", {{0, 8}, {8, 4}, {12, 4}}},
    };
    const auto& layout = layouts.at(it->second.name);
    if (index >= layout.size()) return CUDA_ERROR_INVALID_VALUE;
    *offset = layout[index].first;
    *size = layout[index].second;
    return CUDA_SUCCESS;
}

template <typename T>
T Param(void** params, int index) {
    T value;
    std::memcpy(&value, params[index], sizeof(T));
    return value;
}

CUresult LaunchKernel(CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                      unsigned bz, unsigned, CUstream, void** params, void**) {
    auto it = g.functions.find(reinterpret_cast<uint64_t>(function));
    if (it == g.functions.end()) return CUDA_ERROR_INVALID_HANDLE;
    if (bx * by * bz > 1024 || bx * by * bz == 0 || gx * gy * gz == 0) return CUDA_ERROR_INVALID_VALUE;
    ++g.stats.launches;
    const uint64_t threads = uint64_t{gx} * gy * gz * bx * by * bz;
    const std::string& name = it->second.name;
    if (name == "vecAdd") {
        const int n = Param<int>(params, 3);
        const float* a = Device<float>(Param<uint64_t>(params, 0), n);
        const float* b = Device<float>(Param<uint64_t>(params, 1), n);
        float* c = Device<float>(Param<uint64_t>(params, 2), n);
        if (!a || !b || !c) return CUDA_ERROR_ILLEGAL_ADDRESS;
        for (uint64_t i = 0; i < threads && i < static_cast<uint64_t>(n); ++i) c[i] = a[i] + b[i];
    } else if (name == "scale") {
        const int n = Param<int>(params, 1);
        float* data = Device<float>(Param<uint64_t>(params, 0), n);
        if (!data || g.modules.empty()) return CUDA_ERROR_ILLEGAL_ADDRESS;
        float factor = 0;
        std::memcpy(&factor, Find(g.modules.rbegin()->second.constant_base, sizeof(float)), sizeof(float));
        for (uint64_t i = 0; i < threads && i < static_cast<uint64_t>(n); ++i) data[i] *= factor;
    } else {  // fill
        const int value = Param<int>(params, 1);
        const int n = Param<int>(params, 2);
        int* data = Device<int>(Param<uint64_t>(params, 0), n);
        if (!data) return CUDA_ERROR_ILLEGAL_ADDRESS;
        for (uint64_t i = 0; i < threads && i < static_cast<uint64_t>(n); ++i) data[i] = value;
    }
    return CUDA_SUCCESS;
}

CUresult FuncGetAttribute(int* value, CUfunction_attribute attr, CUfunction function) {
    if (!g.functions.count(reinterpret_cast<uint64_t>(function))) return CUDA_ERROR_INVALID_HANDLE;
    switch (attr) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *value = 1024; break;
    case CU_FUNC_ATTRIBUTE_NUM_REGS: *value = 16; break;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION: *value = 89; break;
    case CU_FUNC_ATTRIBUTE_BINARY_VERSION: *value = 89; break;
    default: *value = 0; break;
    }
    return CUDA_SUCCESS;
}

CUresult FuncSetAttribute(CUfunction function, CUfunction_attribute, int) {
    return g.functions.count(reinterpret_cast<uint64_t>(function)) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult Occupancy(int* blocks, CUfunction function, int block_size, size_t, unsigned) {
    if (!g.functions.count(reinterpret_cast<uint64_t>(function)) || block_size <= 0) return CUDA_ERROR_INVALID_VALUE;
    *blocks = 2048 / block_size;
    return CUDA_SUCCESS;
}

struct Entry {
    const char* name;
    void* fn;
};

#define TUNA_MOCK(symbol, fn) {symbol, reinterpret_cast<void*>(fn)}
const Entry kEntries[] = {
    TUNA_MOCK("cudaGetDeviceCount", GetDeviceCount),
    TUNA_MOCK("cudaGetDevice", GetDevice),
    TUNA_MOCK("cudaSetDevice", SetDevice),
    TUNA_MOCK("cudaDeviceSynchronize", DeviceSynchronize),
    TUNA_MOCK("cudaDeviceReset", DeviceReset),
    TUNA_MOCK("cudaGetDeviceProperties_v2", GetDeviceProperties),
    TUNA_MOCK("cudaDeviceGetAttribute", DeviceGetAttribute),
    TUNA_MOCK("cudaDriverGetVersion", DriverGetVersion),
    TUNA_MOCK("cudaMalloc", Malloc),
    TUNA_MOCK("cudaFree", Free),
    TUNA_MOCK("cudaMemGetInfo", MemGetInfo),
    TUNA_MOCK("cudaMemcpy", Memcpy),
    TUNA_MOCK("cudaMemcpyAsync", MemcpyAsync),
    TUNA_MOCK("cudaMemset", Memset),
    TUNA_MOCK("cudaMemsetAsync", MemsetAsync),
    TUNA_MOCK("cudaStreamCreate", StreamCreate),
    TUNA_MOCK("cudaStreamCreateWithFlags", StreamCreateWithFlags),
    TUNA_MOCK("cudaStreamDestroy", StreamDestroy),
    TUNA_MOCK("cudaStreamSynchronize", StreamSynchronize),
    TUNA_MOCK("cudaEventCreate", EventCreate),
    TUNA_MOCK("cudaEventCreateWithFlags", EventCreateWithFlags),
    TUNA_MOCK("cudaEventRecord", EventRecord),
    TUNA_MOCK("cudaEventSynchronize", EventSynchronize),
    TUNA_MOCK("cudaEventElapsedTime", EventElapsedTime),
    TUNA_MOCK("cudaEventDestroy", EventDestroy),
    TUNA_MOCK("cudaGetLastError", GetLastError),
    TUNA_MOCK("cudaPeekAtLastError", GetLastError),
    TUNA_MOCK("cuCtxGetCurrent", CtxGetCurrent),
    TUNA_MOCK("cuModuleLoadData", ModuleLoadData),
    TUNA_MOCK("cuModuleGetFunction", ModuleGetFunction),
    TUNA_MOCK("cuModuleGetGlobal_v2", ModuleGetGlobal),
    TUNA_MOCK("cuFuncGetParamInfo", FuncGetParamInfo),
    TUNA_MOCK("cuLaunchKernel", LaunchKernel),
    TUNA_MOCK("cuFuncGetAttribute", FuncGetAttribute),
    TUNA_MOCK("cuFuncSetAttribute", FuncSetAttribute),
    TUNA_MOCK("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", Occupancy),
};
#undef TUNA_MOCK

}  // namespace

extern "C" TUNA_MOCK_EXPORT void* tuna_cuda_resolve(const char* symbol) {
    for (const Entry& e : kEntries) {
        if (std::strcmp(e.name, symbol) == 0) return e.fn;
    }
    return nullptr;
}

void MockCudaReset() { g = MockDevice{}; }

MockCudaStats MockCudaGetStats() { return g.stats; }

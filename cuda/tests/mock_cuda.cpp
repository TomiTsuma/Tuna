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
// Driver API: cuInit, device queries (failing before cuInit, as the real
// driver does), and primary-context state/retain/flags.
//
// With cuBLAS headers (TUNA_CUDA_HAVE_CUBLAS) it also models a small cuBLAS
// (and a cuBLASLt matmul with bias/ReLU epilogues, at the end of the file):
// handles with a stream and pointer mode, column-major S/D gemm (plain,
// strided-batched, GemmEx for fp32), axpy, scal, dot, nrm2 and the
// host<->device matrix copies, computed on the host from "device" memory.
// Scalars are read from host or device memory according to the pointer mode,
// so the remoting of both modes is checked by the results.
//
// With cuDNN headers (TUNA_CUDA_HAVE_CUDNN) it also models a small cuDNN:
// spatial batch norm and a backend API that runs a single-convolution graph
// on the host (see the cuDNN section at the end of the file).
//
// It is linked into tuna_cuda_tests (in-process) and also built as a shared
// library that tuna_server loads in place of libcudart: it exports only
// tuna_cuda_resolve, which backend_real.cpp uses as its symbol source.

#include "mock_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#ifdef TUNA_CUDA_HAVE_CUBLAS
#include <cublasLt.h>
#include <cublas_v2.h>
#endif
#ifdef TUNA_CUDA_HAVE_CUDNN
#include <cudnn.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
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

struct BlasHandle {
    uint64_t stream = 0;
    bool device_scalars = false;  // CUBLAS_POINTER_MODE_DEVICE
    int math_mode = 0;
};

#ifdef TUNA_CUDA_HAVE_CUBLAS
struct LtDesc {
    cudaDataType_t scale_type = CUDA_R_32F;
    int32_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    int32_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
    int32_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    uint64_t bias = 0;
};

struct LtLayout {
    cudaDataType type;
    uint64_t rows, cols;
    int64_t ld;
    int32_t batch = 1;
};
#endif

struct MockDevice {
    std::map<uint64_t, std::vector<char>> buffers;  // device address -> bytes
    uint64_t next_address = 0x7f0000100000ull;
    int device_count = 2;
    int current_device = 0;
    bool initialized = false;
    bool driver_initialized = false;          // cuInit called
    std::map<int, unsigned> retained_primary;  // device -> primary context flags (retained)
    size_t total_mem = 8ull << 30;
    std::map<uint64_t, Module> modules;
    std::map<uint64_t, Function> functions;
    std::map<uint64_t, Event> events;
    std::map<uint64_t, BlasHandle> blas_handles;
#ifdef TUNA_CUDA_HAVE_CUBLAS
    std::map<uint64_t, LtDesc> lt_descs;
    std::map<uint64_t, LtLayout> lt_layouts;
    std::map<uint64_t, uint64_t> lt_prefs;  // -> max workspace bytes
    std::map<uint64_t, bool> lt_handles;
#endif
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
    prop->maxThreadsDim[0] = prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 0x7fffffff;
    prop->maxGridSize[1] = prop->maxGridSize[2] = 65535;
    prop->sharedMemPerBlock = 48 << 10;
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
    g.initialized = true;  // like the real runtime, the first allocation creates the context
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

CUresult Init(unsigned flags) {
    if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
    g.driver_initialized = true;
    return CUDA_SUCCESS;
}

// Like the real driver, every device query fails before cuInit.
CUresult CheckDevice(CUdevice dev) {
    if (!g.driver_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    ++g.stats.driver_queries;
    return dev < 0 || dev >= g.device_count ? CUDA_ERROR_INVALID_DEVICE : CUDA_SUCCESS;
}

CUresult DriverGetVersionDrv(int* version) {
    *version = 12080;
    return CUDA_SUCCESS;
}

CUresult DeviceGet(CUdevice* device, int ordinal) {
    if (CUresult r = CheckDevice(ordinal)) return r;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult DeviceGetCount(int* count) {
    if (!g.driver_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    *count = g.device_count;
    return CUDA_SUCCESS;
}

CUresult DeviceGetAttributeDrv(int* value, CUdevice_attribute attr, CUdevice dev) {
    if (CUresult r = CheckDevice(dev)) return r;
    // The runtime's attribute ids are the driver's.
    return static_cast<CUresult>(DeviceGetAttribute(value, static_cast<cudaDeviceAttr>(attr), dev));
}

CUresult DeviceTotalMem(size_t* bytes, CUdevice dev) {
    if (CUresult r = CheckDevice(dev)) return r;
    *bytes = g.total_mem;
    return CUDA_SUCCESS;
}

CUresult DeviceGetUuid(CUuuid* uuid, CUdevice dev) {
    if (CUresult r = CheckDevice(dev)) return r;
    for (int i = 0; i < 16; ++i) uuid->bytes[i] = static_cast<char>(0xA0 + i + dev);
    return CUDA_SUCCESS;
}

CUresult DeviceGetName(char* name, int len, CUdevice dev) {
    if (CUresult r = CheckDevice(dev)) return r;
    std::snprintf(name, static_cast<size_t>(len), "Tuna Mock GPU %d", dev);
    return CUDA_SUCCESS;
}

// A device's primary context is active once the runtime initialized it (on the
// current device) or someone retained it.
CUresult PrimaryCtxGetState(CUdevice dev, unsigned* flags, int* active) {
    if (CUresult r = CheckDevice(dev)) return r;
    auto it = g.retained_primary.find(dev);
    *flags = it == g.retained_primary.end() ? 0 : it->second;
    *active = it != g.retained_primary.end() || (g.initialized && dev == g.current_device);
    return CUDA_SUCCESS;
}

CUresult PrimaryCtxSetFlags(CUdevice dev, unsigned flags) {
    if (CUresult r = CheckDevice(dev)) return r;
    g.retained_primary[dev] = flags;
    return CUDA_SUCCESS;
}

CUresult PrimaryCtxRetain(CUcontext* ctx, CUdevice dev) {
    if (CUresult r = CheckDevice(dev)) return r;
    g.retained_primary.emplace(dev, 0);
    *ctx = reinterpret_cast<CUcontext>(0xC000 + dev);
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
                      unsigned bz, unsigned shared_mem, CUstream, void** params, void**) {
    auto it = g.functions.find(reinterpret_cast<uint64_t>(function));
    if (it == g.functions.end()) return CUDA_ERROR_INVALID_HANDLE;
    if (bx * by * bz > 1024 || bx * by * bz == 0 || gx * gy * gz == 0) return CUDA_ERROR_INVALID_VALUE;
    // More dynamic shared memory than the kernel may use: a failure only the
    // device side can see (it depends on the function's attributes).
    if (shared_mem > (48u << 10)) return CUDA_ERROR_INVALID_VALUE;
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

#ifdef TUNA_CUDA_HAVE_CUBLAS
// ------------------------------------------------------------------- cuBLAS

BlasHandle* Handle(cublasHandle_t handle) {
    auto it = g.blas_handles.find(reinterpret_cast<uint64_t>(handle));
    return it == g.blas_handles.end() ? nullptr : &it->second;
}

// A scalar argument, from host or device memory per the handle's pointer mode.
template <typename T>
bool ReadScalar(const BlasHandle& h, const T* p, T* value) {
    if (!p) return false;
    if (!h.device_scalars) {
        *value = *p;
        return true;
    }
    const T* d = Device<T>(reinterpret_cast<uint64_t>(p), 1);
    if (!d) return false;
    *value = *d;
    return true;
}

template <typename T>
bool WriteScalar(const BlasHandle& h, T* p, T value) {
    if (!p) return false;
    if (!h.device_scalars) {
        *p = value;
        return true;
    }
    T* d = Device<T>(reinterpret_cast<uint64_t>(p), 1);
    if (!d) return false;
    *d = value;
    return true;
}

// A column-major rows x cols matrix with leading dimension ld in device memory.
template <typename T>
T* Matrix(const void* p, int rows, int cols, int ld) {
    if (rows == 0 || cols == 0) return reinterpret_cast<T*>(const_cast<void*>(p));
    return Device<T>(reinterpret_cast<uint64_t>(p), static_cast<size_t>(ld) * (cols - 1) + rows);
}

// A strided vector of n elements in device memory.
template <typename T>
T* Vector(const void* p, int n, int inc) {
    if (n == 0) return reinterpret_cast<T*>(const_cast<void*>(p));
    return Device<T>(reinterpret_cast<uint64_t>(p), static_cast<size_t>(inc) * (n - 1) + 1);
}

cublasStatus_t BlasCreate(cublasHandle_t* handle) {
    const uint64_t h = g.next_handle++;
    g.blas_handles[h] = BlasHandle{};
    *handle = reinterpret_cast<cublasHandle_t>(h);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasDestroy(cublasHandle_t handle) {
    return g.blas_handles.erase(reinterpret_cast<uint64_t>(handle)) ? CUBLAS_STATUS_SUCCESS
                                                                    : CUBLAS_STATUS_NOT_INITIALIZED;
}

cublasStatus_t BlasGetVersion(cublasHandle_t handle, int* version) {
    if (!Handle(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
    *version = 120804;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasSetStream(cublasHandle_t handle, cudaStream_t stream) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->stream = reinterpret_cast<uint64_t>(stream);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasGetStream(cublasHandle_t handle, cudaStream_t* stream) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    *stream = reinterpret_cast<cudaStream_t>(h->stream);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->device_scalars = mode == CUBLAS_POINTER_MODE_DEVICE;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->math_mode = static_cast<int>(mode);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    *mode = static_cast<cublasMath_t>(h->math_mode);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasSetWorkspace(cublasHandle_t handle, void* workspace, size_t size) {
    if (!Handle(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (size && !Find(reinterpret_cast<uint64_t>(workspace), size)) return CUBLAS_STATUS_INVALID_VALUE;
    return CUBLAS_STATUS_SUCCESS;
}

// C = alpha * op(A) * op(B) + beta * C, column-major (the reference BLAS).
template <typename T>
cublasStatus_t Gemm(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                    const T* alpha_p, const void* a_p, int lda, const void* b_p, int ldb, const T* beta_p, void* c_p,
                    int ldc) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    const bool at = ta != CUBLAS_OP_N, bt = tb != CUBLAS_OP_N;
    const int a_rows = at ? k : m, a_cols = at ? m : k;
    const int b_rows = bt ? n : k, b_cols = bt ? k : n;
    if (m < 0 || n < 0 || k < 0 || lda < std::max(1, a_rows) || ldb < std::max(1, b_rows) || ldc < std::max(1, m)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    T alpha{}, beta{};
    if (!ReadScalar(*h, alpha_p, &alpha) || !ReadScalar(*h, beta_p, &beta)) return CUBLAS_STATUS_INVALID_VALUE;
    const T* A = Matrix<T>(a_p, a_rows, a_cols, lda);
    const T* B = Matrix<T>(b_p, b_rows, b_cols, ldb);
    T* C = Matrix<T>(c_p, m, n, ldc);
    if (!A || !B || !C) return CUBLAS_STATUS_EXECUTION_FAILED;
    ++g.stats.blas_calls;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T sum{};
            for (int p = 0; p < k; ++p) {
                const T a = at ? A[p + static_cast<size_t>(i) * lda] : A[i + static_cast<size_t>(p) * lda];
                const T b = bt ? B[j + static_cast<size_t>(p) * ldb] : B[p + static_cast<size_t>(j) * ldb];
                sum += a * b;
            }
            T& c = C[i + static_cast<size_t>(j) * ldc];
            c = alpha * sum + (beta == T{} ? T{} : beta * c);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <typename T>
cublasStatus_t GemmStridedBatched(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                                  int k, const T* alpha, const T* a, int lda, long long stride_a, const T* b, int ldb,
                                  long long stride_b, const T* beta, T* c, int ldc, long long stride_c, int batch) {
    if (batch < 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batch; ++i) {
        const cublasStatus_t status = Gemm<T>(handle, ta, tb, m, n, k, alpha, a + i * stride_a, lda, b + i * stride_b,
                                              ldb, beta, c + i * stride_c, ldc);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasGemmEx(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                          const void* alpha, const void* a, cudaDataType a_type, int lda, const void* b,
                          cudaDataType b_type, int ldb, const void* beta, void* c, cudaDataType c_type, int ldc,
                          cublasComputeType_t compute, cublasGemmAlgo_t) {
    // The mock only does fp32 (enough to check the compute-type scalar sizing).
    if (a_type != CUDA_R_32F || b_type != CUDA_R_32F || c_type != CUDA_R_32F || compute != CUBLAS_COMPUTE_32F) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return Gemm<float>(handle, ta, tb, m, n, k, static_cast<const float*>(alpha), a, lda, b, ldb,
                       static_cast<const float*>(beta), c, ldc);
}

cublasStatus_t BlasSaxpy(cublasHandle_t handle, int n, const float* alpha_p, const float* x_p, int incx, float* y_p,
                         int incy) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    float alpha = 0;
    if (n < 0 || incx <= 0 || incy <= 0 || !ReadScalar(*h, alpha_p, &alpha)) return CUBLAS_STATUS_INVALID_VALUE;
    const float* x = Vector<float>(x_p, n, incx);
    float* y = Vector<float>(y_p, n, incy);
    if (!x || !y) return CUBLAS_STATUS_EXECUTION_FAILED;
    ++g.stats.blas_calls;
    for (int i = 0; i < n; ++i) y[static_cast<size_t>(i) * incy] += alpha * x[static_cast<size_t>(i) * incx];
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasSscal(cublasHandle_t handle, int n, const float* alpha_p, float* x_p, int incx) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    float alpha = 0;
    if (n < 0 || incx <= 0 || !ReadScalar(*h, alpha_p, &alpha)) return CUBLAS_STATUS_INVALID_VALUE;
    float* x = Vector<float>(x_p, n, incx);
    if (!x) return CUBLAS_STATUS_EXECUTION_FAILED;
    ++g.stats.blas_calls;
    for (int i = 0; i < n; ++i) x[static_cast<size_t>(i) * incx] *= alpha;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasSdot(cublasHandle_t handle, int n, const float* x_p, int incx, const float* y_p, int incy,
                        float* result) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || incx <= 0 || incy <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    const float* x = Vector<float>(x_p, n, incx);
    const float* y = Vector<float>(y_p, n, incy);
    if (!x || !y) return CUBLAS_STATUS_EXECUTION_FAILED;
    ++g.stats.blas_calls;
    float sum = 0;
    for (int i = 0; i < n; ++i) sum += x[static_cast<size_t>(i) * incx] * y[static_cast<size_t>(i) * incy];
    return WriteScalar(*h, result, sum) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t BlasSnrm2(cublasHandle_t handle, int n, const float* x_p, int incx, float* result) {
    BlasHandle* h = Handle(handle);
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    const float* x = Vector<float>(x_p, n, incx);
    if (!x) return CUBLAS_STATUS_EXECUTION_FAILED;
    ++g.stats.blas_calls;
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += double{x[static_cast<size_t>(i) * incx]} * x[static_cast<size_t>(i) * incx];
    return WriteScalar(*h, result, static_cast<float>(std::sqrt(sum))) ? CUBLAS_STATUS_SUCCESS
                                                                       : CUBLAS_STATUS_INVALID_VALUE;
}

// Host <-> device copies of a rows x cols block (any leading dimensions).
cublasStatus_t BlasSetMatrix(int rows, int cols, int elem, const void* a, int lda, void* b, int ldb) {
    if (rows < 0 || cols < 0 || elem <= 0 || lda < std::max(1, rows) || ldb < std::max(1, rows)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < cols; ++j) {
        char* dst = Find(reinterpret_cast<uint64_t>(b) + static_cast<uint64_t>(j) * ldb * elem,
                         static_cast<size_t>(rows) * elem);
        if (!dst) return CUBLAS_STATUS_MAPPING_ERROR;
        std::memcpy(dst, static_cast<const char*>(a) + static_cast<size_t>(j) * lda * elem,
                    static_cast<size_t>(rows) * elem);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasGetMatrix(int rows, int cols, int elem, const void* a, int lda, void* b, int ldb) {
    if (rows < 0 || cols < 0 || elem <= 0 || lda < std::max(1, rows) || ldb < std::max(1, rows)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < cols; ++j) {
        const char* src = Find(reinterpret_cast<uint64_t>(a) + static_cast<uint64_t>(j) * lda * elem,
                               static_cast<size_t>(rows) * elem);
        if (!src) return CUBLAS_STATUS_MAPPING_ERROR;
        std::memcpy(static_cast<char*>(b) + static_cast<size_t>(j) * ldb * elem, src, static_cast<size_t>(rows) * elem);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t BlasSetMatrixAsync(int rows, int cols, int elem, const void* a, int lda, void* b, int ldb,
                                  cudaStream_t) {
    return BlasSetMatrix(rows, cols, elem, a, lda, b, ldb);
}

cublasStatus_t BlasGetMatrixAsync(int rows, int cols, int elem, const void* a, int lda, void* b, int ldb,
                                  cudaStream_t) {
    return BlasGetMatrix(rows, cols, elem, a, lda, b, ldb);
}
// ----------------------------------------------------------------- cuBLASLt
//
// Enough of cuBLASLt for PyTorch's addmm path: fp32 matmul descriptors with
// transposes, a bias / ReLU epilogue and either pointer mode, column-major
// layouts, one heuristic result (a tagged algo), and the matmul itself.

constexpr uint64_t kMockAlgoTag = 0x7A1607A1ull;

template <typename T>
T* LtObject(std::map<uint64_t, T>& objects, const void* id) {
    auto it = objects.find(reinterpret_cast<uint64_t>(id));
    return it == objects.end() ? nullptr : &it->second;
}

cublasStatus_t LtCreate(cublasLtHandle_t* handle) {
    const uint64_t h = g.next_handle++;
    g.lt_handles[h] = true;
    *handle = reinterpret_cast<cublasLtHandle_t>(h);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t LtDestroy(cublasLtHandle_t handle) {
    return g.lt_handles.erase(reinterpret_cast<uint64_t>(handle)) ? CUBLAS_STATUS_SUCCESS
                                                                   : CUBLAS_STATUS_NOT_INITIALIZED;
}

// A cuBLAS handle works as an Lt handle, as with the real libraries.
bool LtHandleKnown(cublasLtHandle_t handle) {
    const auto h = reinterpret_cast<uint64_t>(handle);
    return g.lt_handles.count(h) || g.blas_handles.count(h);
}

cublasStatus_t LtDescCreate(cublasLtMatmulDesc_t* desc, cublasComputeType_t, cudaDataType_t scale) {
    const uint64_t id = g.next_handle++;
    LtDesc d;
    d.scale_type = scale;
    g.lt_descs[id] = d;
    *desc = reinterpret_cast<cublasLtMatmulDesc_t>(id);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t LtDescDestroy(cublasLtMatmulDesc_t desc) {
    return g.lt_descs.erase(reinterpret_cast<uint64_t>(desc)) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

// Attribute values must have their exact size, as in cuBLASLt.
template <typename T>
bool SetValue(T* field, const void* buf, size_t size) {
    if (size != sizeof(T)) return false;
    std::memcpy(field, buf, sizeof(T));
    return true;
}

cublasStatus_t LtDescSetAttribute(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr, const void* buf,
                                  size_t size) {
    LtDesc* d = LtObject(g.lt_descs, desc);
    if (!d || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    bool ok = true;
    switch (attr) {
    case CUBLASLT_MATMUL_DESC_TRANSA: ok = SetValue(&d->transa, buf, size); break;
    case CUBLASLT_MATMUL_DESC_TRANSB: ok = SetValue(&d->transb, buf, size); break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE: ok = SetValue(&d->epilogue, buf, size); break;
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER: ok = SetValue(&d->bias, buf, size); break;
    case CUBLASLT_MATMUL_DESC_POINTER_MODE: ok = SetValue(&d->pointer_mode, buf, size); break;
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE: ok = SetValue(&d->scale_type, buf, size); break;
    default: break;  // accepted and ignored
    }
    return ok ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t LtLayoutCreate(cublasLtMatrixLayout_t* layout, cudaDataType type, uint64_t rows, uint64_t cols,
                              int64_t ld) {
    if (ld < static_cast<int64_t>(std::max<uint64_t>(1, rows))) return CUBLAS_STATUS_INVALID_VALUE;
    const uint64_t id = g.next_handle++;
    g.lt_layouts[id] = LtLayout{type, rows, cols, ld};
    *layout = reinterpret_cast<cublasLtMatrixLayout_t>(id);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t LtLayoutDestroy(cublasLtMatrixLayout_t layout) {
    return g.lt_layouts.erase(reinterpret_cast<uint64_t>(layout)) ? CUBLAS_STATUS_SUCCESS
                                                                   : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t LtLayoutSetAttribute(cublasLtMatrixLayout_t layout, cublasLtMatrixLayoutAttribute_t attr,
                                    const void* buf, size_t size) {
    LtLayout* l = LtObject(g.lt_layouts, layout);
    if (!l || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    if (attr == CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT && !SetValue(&l->batch, buf, size)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t LtPrefCreate(cublasLtMatmulPreference_t* pref) {
    const uint64_t id = g.next_handle++;
    g.lt_prefs[id] = 0;
    *pref = reinterpret_cast<cublasLtMatmulPreference_t>(id);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t LtPrefDestroy(cublasLtMatmulPreference_t pref) {
    return g.lt_prefs.erase(reinterpret_cast<uint64_t>(pref)) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

cublasStatus_t LtPrefSetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr,
                                  const void* buf, size_t size) {
    uint64_t* workspace = LtObject(g.lt_prefs, pref);
    if (!workspace || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    if (attr == CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES && !SetValue(workspace, buf, size)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t LtHeuristic(cublasLtHandle_t handle, cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t a,
                           cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
                           cublasLtMatmulPreference_t pref, int requested, cublasLtMatmulHeuristicResult_t* results,
                           int* count) {
    if (!LtHandleKnown(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!LtObject(g.lt_descs, desc) || !LtObject(g.lt_layouts, a) || !LtObject(g.lt_layouts, b) ||
        !LtObject(g.lt_layouts, c) || !LtObject(g.lt_layouts, d) || !LtObject(g.lt_prefs, pref) || requested < 1) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::memset(results, 0, sizeof(*results));
    results[0].algo.data[0] = kMockAlgoTag;
    results[0].state = CUBLAS_STATUS_SUCCESS;
    results[0].wavesCount = 1.0f;
    *count = 1;
    return CUBLAS_STATUS_SUCCESS;
}

// One scalar from host memory or the mock's device memory.
bool LtScalar(const void* p, bool on_device, float* value) {
    if (!p) return false;
    const float* source = on_device ? Device<float>(reinterpret_cast<uint64_t>(p), 1) : static_cast<const float*>(p);
    if (!source) return false;
    *value = *source;
    return true;
}

// D = epilogue(alpha * op(A) op(B) + beta * C [+ bias per row]), fp32, column-major.
cublasStatus_t LtMatmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t desc_id, const void* alpha_p, const void* a_p,
                        cublasLtMatrixLayout_t a_id, const void* b_p, cublasLtMatrixLayout_t b_id, const void* beta_p,
                        const void* c_p, cublasLtMatrixLayout_t c_id, void* d_p, cublasLtMatrixLayout_t d_id,
                        const cublasLtMatmulAlgo_t* algo, void*, size_t, cudaStream_t) {
    if (!LtHandleKnown(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
    const LtDesc* desc = LtObject(g.lt_descs, desc_id);
    const LtLayout* la = LtObject(g.lt_layouts, a_id);
    const LtLayout* lb = LtObject(g.lt_layouts, b_id);
    const LtLayout* lc = LtObject(g.lt_layouts, c_id);
    const LtLayout* ld = LtObject(g.lt_layouts, d_id);
    if (!desc || !la || !lb || !lc || !ld || (algo && algo->data[0] != kMockAlgoTag)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (const LtLayout* l : {la, lb, lc, ld}) {
        if (l->type != CUDA_R_32F || l->batch != 1) return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (desc->scale_type != CUDA_R_32F) return CUBLAS_STATUS_NOT_SUPPORTED;
    const bool at = desc->transa != CUBLAS_OP_N;
    const bool bt = desc->transb != CUBLAS_OP_N;
    const uint64_t m = ld->rows, n = ld->cols;
    const uint64_t k = at ? la->rows : la->cols;
    if ((at ? la->cols : la->rows) != m || (bt ? lb->cols : lb->rows) != k || (bt ? lb->rows : lb->cols) != n ||
        lc->rows != m || lc->cols != n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    // alpha follows the pointer mode; beta is on the device only in the
    // DEVICE mode (the mock models scalars, not the vector modes).
    float alpha = 0, beta = 0;
    if (!LtScalar(alpha_p, desc->pointer_mode != CUBLASLT_POINTER_MODE_HOST, &alpha) ||
        !LtScalar(beta_p, desc->pointer_mode == CUBLASLT_POINTER_MODE_DEVICE, &beta)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const auto dims = [](const LtLayout* l) {
        return std::make_pair(static_cast<int>(l->rows), static_cast<int>(l->cols));
    };
    const float* A = Matrix<float>(a_p, dims(la).first, dims(la).second, static_cast<int>(la->ld));
    const float* B = Matrix<float>(b_p, dims(lb).first, dims(lb).second, static_cast<int>(lb->ld));
    const float* C = Matrix<float>(c_p, dims(lc).first, dims(lc).second, static_cast<int>(lc->ld));
    float* D = Matrix<float>(d_p, dims(ld).first, dims(ld).second, static_cast<int>(ld->ld));
    const bool has_bias = (desc->epilogue & CUBLASLT_EPILOGUE_BIAS) == CUBLASLT_EPILOGUE_BIAS;
    const bool relu = (desc->epilogue & CUBLASLT_EPILOGUE_RELU) == CUBLASLT_EPILOGUE_RELU;
    const float* bias = has_bias ? Device<float>(desc->bias, m) : nullptr;
    if (!A || !B || !C || !D || (has_bias && !bias)) return CUBLAS_STATUS_INVALID_VALUE;
    ++g.stats.blas_calls;
    std::vector<float> result(m * n);  // C may alias D
    for (uint64_t j = 0; j < n; ++j) {
        for (uint64_t i = 0; i < m; ++i) {
            double sum = 0;
            for (uint64_t p = 0; p < k; ++p) {
                const float av = at ? A[p + i * la->ld] : A[i + p * la->ld];
                const float bv = bt ? B[j + p * lb->ld] : B[p + j * lb->ld];
                sum += static_cast<double>(av) * bv;
            }
            float value = alpha * static_cast<float>(sum);
            if (beta != 0) value += beta * C[i + j * lc->ld];
            if (has_bias) value += bias[i];
            if (relu) value = std::max(value, 0.0f);
            result[i + j * m] = value;
        }
    }
    for (uint64_t j = 0; j < n; ++j) {
        for (uint64_t i = 0; i < m; ++i) D[i + j * ld->ld] = result[i + j * m];
    }
    return CUBLAS_STATUS_SUCCESS;
}

#endif  // TUNA_CUDA_HAVE_CUBLAS

#ifdef TUNA_CUDA_HAVE_CUDNN
// ------------------------------------------------------------------- cuDNN
//
// A small cuDNN: handles with a stream, legacy tensor descriptors, spatial
// batch norm (forward training, backward, inference) on float NCHW tensors,
// and a backend API that stores attributes generically and runs one graph
// shape, a single 2-D convolution forward, on the host. That is enough to
// check the descriptor translation end to end: tensors, the convolution, the
// operation, the graph, heuristic results copied into caller-created engine
// configs, the plan and a variant pack all have to reach the "library" as
// the right objects for the convolution to come out right.

struct DnnTensor {
    bool set = false;
    cudnnDataType_t type = CUDNN_DATA_FLOAT;
    int nb = 0;
    int dims[CUDNN_DIM_MAX] = {};
    int strides[CUDNN_DIM_MAX] = {};
};

struct DnnAttr {
    cudnnBackendAttributeType_t type;
    int64_t count = 0;
    std::vector<char> bytes;
};

struct DnnDesc {
    cudnnBackendDescriptorType_t type;
    bool finalized = false;
    std::map<int, DnnAttr> attrs;
};

struct MockDnn {
    std::map<uint64_t, uint64_t> handles;  // -> stream
    std::map<uint64_t, DnnTensor> tensors;
    std::map<uint64_t, DnnDesc> descs;
    std::string last_error;
};
MockDnn gd;

cudnnStatus_t DnnFail(cudnnStatus_t status, const char* message) {
    gd.last_error = message;
    return status;
}

size_t DnnElementSize(cudnnBackendAttributeType_t type) {
    switch (type) {
    case CUDNN_TYPE_HANDLE:
    case CUDNN_TYPE_INT64:
    case CUDNN_TYPE_DOUBLE:
    case CUDNN_TYPE_VOID_PTR:
    case CUDNN_TYPE_BACKEND_DESCRIPTOR: return 8;
    case CUDNN_TYPE_BOOLEAN: return 1;
    case CUDNN_TYPE_DATA_TYPE:
    case CUDNN_TYPE_FLOAT:
    case CUDNN_TYPE_CONVOLUTION_MODE:
    case CUDNN_TYPE_HEUR_MODE:
    case CUDNN_TYPE_INT32: return 4;
    default: return 0;
    }
}

DnnDesc* DnnFind(const void* p) {
    auto it = gd.descs.find(reinterpret_cast<uint64_t>(p));
    return it == gd.descs.end() ? nullptr : &it->second;
}

const DnnAttr* DnnGet(const DnnDesc& d, int name, cudnnBackendAttributeType_t type) {
    auto it = d.attrs.find(name);
    return it != d.attrs.end() && it->second.type == type ? &it->second : nullptr;
}

template <typename T>
std::vector<T> DnnValues(const DnnDesc& d, int name, cudnnBackendAttributeType_t type) {
    const DnnAttr* a = DnnGet(d, name, type);
    std::vector<T> values;
    if (a) {
        values.resize(static_cast<size_t>(a->count));
        std::memcpy(values.data(), a->bytes.data(), a->bytes.size());
    }
    return values;
}

// The descriptor a single-element descriptor attribute refers to.
DnnDesc* DnnRef(const DnnDesc& d, int name) {
    auto refs = DnnValues<void*>(d, name, CUDNN_TYPE_BACKEND_DESCRIPTOR);
    return refs.size() == 1 ? DnnFind(refs[0]) : nullptr;
}

void DnnPut(DnnDesc& d, int name, cudnnBackendAttributeType_t type, const void* values, int64_t count) {
    DnnAttr& a = d.attrs[name];
    a.type = type;
    a.count = count;
    a.bytes.assign(static_cast<const char*>(values),
                   static_cast<const char*>(values) + DnnElementSize(type) * static_cast<size_t>(count));
}

uint64_t DnnNewDesc(cudnnBackendDescriptorType_t type) {
    const uint64_t id = g.next_handle++;
    gd.descs[id].type = type;
    return id;
}

cudnnStatus_t DnnCreate(cudnnHandle_t* handle) {
    const uint64_t h = g.next_handle++;
    gd.handles[h] = 0;
    *handle = reinterpret_cast<cudnnHandle_t>(h);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t DnnDestroy(cudnnHandle_t handle) {
    return gd.handles.erase(reinterpret_cast<uint64_t>(handle)) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

bool DnnHandleKnown(const void* handle) { return gd.handles.count(reinterpret_cast<uint64_t>(handle)) != 0; }

cudnnStatus_t DnnSetStream(cudnnHandle_t handle, cudaStream_t stream) {
    auto it = gd.handles.find(reinterpret_cast<uint64_t>(handle));
    if (it == gd.handles.end()) return CUDNN_STATUS_BAD_PARAM;
    it->second = reinterpret_cast<uint64_t>(stream);
    return CUDNN_STATUS_SUCCESS;
}

size_t DnnGetVersion() { return 91002; }
size_t DnnGetMaxDeviceVersion() { return 1200; }
size_t DnnGetCudartVersion() { return 12080; }

void DnnGetLastErrorString(char* message, size_t max_size) {
    std::snprintf(message, max_size, "mock: %s", gd.last_error.c_str());
}

cudnnStatus_t DnnCreateTensor(cudnnTensorDescriptor_t* desc) {
    const uint64_t id = g.next_handle++;
    gd.tensors[id] = DnnTensor{};
    *desc = reinterpret_cast<cudnnTensorDescriptor_t>(id);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t DnnSetTensorNd(cudnnTensorDescriptor_t desc, cudnnDataType_t type, int nb, const int* dims,
                             const int* strides) {
    auto it = gd.tensors.find(reinterpret_cast<uint64_t>(desc));
    if (it == gd.tensors.end() || nb < 1 || nb > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
    DnnTensor& t = it->second;
    t.set = true;
    t.type = type;
    t.nb = nb;
    std::copy(dims, dims + nb, t.dims);
    std::copy(strides, strides + nb, t.strides);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t DnnDestroyTensor(cudnnTensorDescriptor_t desc) {
    return gd.tensors.erase(reinterpret_cast<uint64_t>(desc)) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

const DnnTensor* DnnTensorOf(const void* desc) {
    auto it = gd.tensors.find(reinterpret_cast<uint64_t>(desc));
    return it != gd.tensors.end() && it->second.set ? &it->second : nullptr;
}

// A float NCHW tensor; nullptr unless the descriptor is one.
const DnnTensor* DnnNchw(const void* desc) {
    const DnnTensor* t = DnnTensorOf(desc);
    return t && t->nb == 4 && t->type == CUDNN_DATA_FLOAT ? t : nullptr;
}

size_t DnnSpan(const DnnTensor& t) {  // elements from the first to the last one, plus one
    size_t span = 1;
    for (int i = 0; i < t.nb; ++i) span += static_cast<size_t>(t.dims[i] - 1) * static_cast<size_t>(t.strides[i]);
    return span;
}

size_t DnnAt(const DnnTensor& t, int n, int c, int h, int w) {
    return static_cast<size_t>(n) * t.strides[0] + static_cast<size_t>(c) * t.strides[1] +
           static_cast<size_t>(h) * t.strides[2] + static_cast<size_t>(w) * t.strides[3];
}

// Batch-norm sizes: made up, but distinct per query and per channel count.
cudnnStatus_t DnnBnSize(const void* x_desc, size_t per_channel, size_t* size) {
    ++g.stats.dnn_size_queries;
    const DnnTensor* x = DnnNchw(x_desc);
    if (!x) return DnnFail(CUDNN_STATUS_BAD_PARAM, "batch norm needs a float NCHW x");
    *size = per_channel * static_cast<size_t>(x->dims[1]);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t DnnBnFwdWorkspace(cudnnHandle_t handle, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                                cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                                cudnnTensorDescriptor_t, cudnnActivationDescriptor_t act, size_t* size) {
    if (!DnnHandleKnown(handle) || act) return CUDNN_STATUS_BAD_PARAM;
    return DnnBnSize(x, 32, size);
}

cudnnStatus_t DnnBnBwdWorkspace(cudnnHandle_t handle, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                                cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                                cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                                cudnnActivationDescriptor_t act, size_t* size) {
    if (!DnnHandleKnown(handle) || act) return CUDNN_STATUS_BAD_PARAM;
    return DnnBnSize(x, 48, size);
}

cudnnStatus_t DnnBnReserve(cudnnHandle_t handle, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                           cudnnActivationDescriptor_t act, cudnnTensorDescriptor_t x, size_t* size) {
    if (!DnnHandleKnown(handle) || act) return CUDNN_STATUS_BAD_PARAM;
    return DnnBnSize(x, 16, size);
}

// Workspace / reserve buffers must be device memory of at least the size given.
bool DnnBuffer(const void* p, size_t size) { return size == 0 || Find(reinterpret_cast<uint64_t>(p), size); }

struct DnnBnShape {
    const DnnTensor* x;
    int n, c, h, w;
    double m;  // elements per channel
};

bool DnnBnCheck(const void* x_desc, const void* y_desc, const void* bn_desc, DnnBnShape* shape) {
    const DnnTensor* x = DnnNchw(x_desc);
    const DnnTensor* y = DnnNchw(y_desc);
    const DnnTensor* bn = DnnNchw(bn_desc);
    if (!x || !y || !bn || bn->dims[1] != x->dims[1]) return false;
    for (int i = 0; i < 4; ++i) {
        if (x->dims[i] != y->dims[i] || x->strides[i] != y->strides[i]) return false;
    }
    *shape = {x, x->dims[0], x->dims[1], x->dims[2], x->dims[3],
              static_cast<double>(x->dims[0]) * x->dims[2] * x->dims[3]};
    return true;
}

cudnnStatus_t DnnBnForwardTraining(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t ops,
                                   const void* alpha_p, const void* beta_p, cudnnTensorDescriptor_t x_desc,
                                   const void* x_p, cudnnTensorDescriptor_t z_desc, const void*,
                                   cudnnTensorDescriptor_t y_desc, void* y_p, cudnnTensorDescriptor_t bn_desc,
                                   const void* scale_p, const void* bias_p, double factor, void* rmean_p,
                                   void* rvar_p, double epsilon, void* smean_p, void* sinv_p,
                                   cudnnActivationDescriptor_t act, void* ws, size_t ws_size, void* reserve,
                                   size_t reserve_size) {
    DnnBnShape s;
    if (!DnnHandleKnown(handle) || mode != CUDNN_BATCHNORM_SPATIAL || ops != CUDNN_BATCHNORM_OPS_BN || z_desc ||
        act || !DnnBnCheck(x_desc, y_desc, bn_desc, &s) || !DnnBuffer(ws, ws_size) ||
        !DnnBuffer(reserve, reserve_size)) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "bad batch-norm forward arguments");
    }
    const size_t span = DnnSpan(*s.x);
    const float* x = Device<float>(reinterpret_cast<uint64_t>(x_p), span);
    float* y = Device<float>(reinterpret_cast<uint64_t>(y_p), span);
    const float* scale = Device<float>(reinterpret_cast<uint64_t>(scale_p), s.c);
    const float* bias = Device<float>(reinterpret_cast<uint64_t>(bias_p), s.c);
    float* rmean = rmean_p ? Device<float>(reinterpret_cast<uint64_t>(rmean_p), s.c) : nullptr;
    float* rvar = rvar_p ? Device<float>(reinterpret_cast<uint64_t>(rvar_p), s.c) : nullptr;
    float* smean = smean_p ? Device<float>(reinterpret_cast<uint64_t>(smean_p), s.c) : nullptr;
    float* sinv = sinv_p ? Device<float>(reinterpret_cast<uint64_t>(sinv_p), s.c) : nullptr;
    if (!x || !y || !scale || !bias || (rmean_p && !rmean) || (rvar_p && !rvar) || (smean_p && !smean) ||
        (sinv_p && !sinv)) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "batch-norm pointer outside device memory");
    }
    const float alpha = *static_cast<const float*>(alpha_p), beta = *static_cast<const float*>(beta_p);
    for (int c = 0; c < s.c; ++c) {
        double sum = 0, sq = 0;
        for (int n = 0; n < s.n; ++n)
            for (int h = 0; h < s.h; ++h)
                for (int w = 0; w < s.w; ++w) {
                    const double v = x[DnnAt(*s.x, n, c, h, w)];
                    sum += v;
                    sq += v * v;
                }
        const double mean = sum / s.m, var = sq / s.m - mean * mean;
        const double inv = 1.0 / std::sqrt(var + epsilon);
        for (int n = 0; n < s.n; ++n)
            for (int h = 0; h < s.h; ++h)
                for (int w = 0; w < s.w; ++w) {
                    const size_t i = DnnAt(*s.x, n, c, h, w);
                    const double out = scale[c] * (x[i] - mean) * inv + bias[c];
                    y[i] = static_cast<float>(alpha * out + beta * y[i]);
                }
        if (rmean) rmean[c] = static_cast<float>((1 - factor) * rmean[c] + factor * mean);
        if (rvar) rvar[c] = static_cast<float>((1 - factor) * rvar[c] + factor * var * s.m / (s.m - 1));
        if (smean) smean[c] = static_cast<float>(mean);
        if (sinv) sinv[c] = static_cast<float>(inv);
    }
    ++g.stats.dnn_executes;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t DnnBnBackward(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t ops,
                            const void* alpha_data_p, const void* beta_data_p, const void* alpha_param_p,
                            const void* beta_param_p, cudnnTensorDescriptor_t x_desc, const void* x_p,
                            cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t dy_desc, const void* dy_p,
                            cudnnTensorDescriptor_t dz_desc, void*, cudnnTensorDescriptor_t dx_desc, void* dx_p,
                            cudnnTensorDescriptor_t dbn_desc, const void* scale_p, const void*, void* dscale_p,
                            void* dbias_p, double, const void* smean_p, const void* sinv_p,
                            cudnnActivationDescriptor_t act, void* ws, size_t ws_size, void* reserve,
                            size_t reserve_size) {
    DnnBnShape s, sdx;
    if (!DnnHandleKnown(handle) || mode != CUDNN_BATCHNORM_SPATIAL || ops != CUDNN_BATCHNORM_OPS_BN || dz_desc ||
        act || !DnnBnCheck(x_desc, dy_desc, dbn_desc, &s) || !DnnBnCheck(x_desc, dx_desc, dbn_desc, &sdx) ||
        !smean_p || !sinv_p || !DnnBuffer(ws, ws_size) || !DnnBuffer(reserve, reserve_size)) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "bad batch-norm backward arguments");
    }
    const size_t span = DnnSpan(*s.x);
    const float* x = Device<float>(reinterpret_cast<uint64_t>(x_p), span);
    const float* dy = Device<float>(reinterpret_cast<uint64_t>(dy_p), span);
    float* dx = Device<float>(reinterpret_cast<uint64_t>(dx_p), span);
    const float* scale = Device<float>(reinterpret_cast<uint64_t>(scale_p), s.c);
    float* dscale = Device<float>(reinterpret_cast<uint64_t>(dscale_p), s.c);
    float* dbias = Device<float>(reinterpret_cast<uint64_t>(dbias_p), s.c);
    const float* smean = Device<float>(reinterpret_cast<uint64_t>(smean_p), s.c);
    const float* sinv = Device<float>(reinterpret_cast<uint64_t>(sinv_p), s.c);
    if (!x || !dy || !dx || !scale || !dscale || !dbias || !smean || !sinv) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "batch-norm pointer outside device memory");
    }
    const float ad = *static_cast<const float*>(alpha_data_p), bd = *static_cast<const float*>(beta_data_p);
    const float ap = *static_cast<const float*>(alpha_param_p), bp = *static_cast<const float*>(beta_param_p);
    for (int c = 0; c < s.c; ++c) {
        double gsum = 0, gxhat = 0;
        for (int n = 0; n < s.n; ++n)
            for (int h = 0; h < s.h; ++h)
                for (int w = 0; w < s.w; ++w) {
                    const size_t i = DnnAt(*s.x, n, c, h, w);
                    gsum += dy[i];
                    gxhat += dy[i] * (x[i] - smean[c]) * sinv[c];
                }
        for (int n = 0; n < s.n; ++n)
            for (int h = 0; h < s.h; ++h)
                for (int w = 0; w < s.w; ++w) {
                    const size_t i = DnnAt(*s.x, n, c, h, w);
                    const double xhat = (x[i] - smean[c]) * sinv[c];
                    const double g = scale[c] * sinv[c] / s.m * (s.m * dy[i] - gsum - xhat * gxhat);
                    dx[i] = static_cast<float>(ad * g + bd * dx[i]);
                }
        dscale[c] = static_cast<float>(ap * gxhat + bp * dscale[c]);
        dbias[c] = static_cast<float>(ap * gsum + bp * dbias[c]);
    }
    ++g.stats.dnn_executes;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t DnnBnInference(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha_p,
                             const void* beta_p, cudnnTensorDescriptor_t x_desc, const void* x_p,
                             cudnnTensorDescriptor_t y_desc, void* y_p, cudnnTensorDescriptor_t bn_desc,
                             const void* scale_p, const void* bias_p, const void* mean_p, const void* var_p,
                             double epsilon) {
    DnnBnShape s;
    if (!DnnHandleKnown(handle) || mode != CUDNN_BATCHNORM_SPATIAL || !DnnBnCheck(x_desc, y_desc, bn_desc, &s)) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "bad batch-norm inference arguments");
    }
    const size_t span = DnnSpan(*s.x);
    const float* x = Device<float>(reinterpret_cast<uint64_t>(x_p), span);
    float* y = Device<float>(reinterpret_cast<uint64_t>(y_p), span);
    const float* scale = Device<float>(reinterpret_cast<uint64_t>(scale_p), s.c);
    const float* bias = Device<float>(reinterpret_cast<uint64_t>(bias_p), s.c);
    const float* mean = Device<float>(reinterpret_cast<uint64_t>(mean_p), s.c);
    const float* var = Device<float>(reinterpret_cast<uint64_t>(var_p), s.c);
    if (!x || !y || !scale || !bias || !mean || !var) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "batch-norm pointer outside device memory");
    }
    const float alpha = *static_cast<const float*>(alpha_p), beta = *static_cast<const float*>(beta_p);
    for (int c = 0; c < s.c; ++c) {
        const double inv = 1.0 / std::sqrt(var[c] + epsilon);
        for (int n = 0; n < s.n; ++n)
            for (int h = 0; h < s.h; ++h)
                for (int w = 0; w < s.w; ++w) {
                    const size_t i = DnnAt(*s.x, n, c, h, w);
                    y[i] = static_cast<float>(alpha * (scale[c] * (x[i] - mean[c]) * inv + bias[c]) + beta * y[i]);
                }
    }
    ++g.stats.dnn_executes;
    return CUDNN_STATUS_SUCCESS;
}

// --- backend API

cudnnStatus_t DnnBackendCreate(cudnnBackendDescriptorType_t type, cudnnBackendDescriptor_t* desc) {
    *desc = reinterpret_cast<cudnnBackendDescriptor_t>(DnnNewDesc(type));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t DnnBackendDestroy(cudnnBackendDescriptor_t desc) {
    return gd.descs.erase(reinterpret_cast<uint64_t>(desc)) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t DnnBackendSet(cudnnBackendDescriptor_t desc, cudnnBackendAttributeName_t name,
                            cudnnBackendAttributeType_t type, int64_t count, const void* elements) {
    DnnDesc* d = DnnFind(desc);
    if (!d) return DnnFail(CUDNN_STATUS_BAD_PARAM, "set on an unknown descriptor");
    if (DnnElementSize(type) == 0) return DnnFail(CUDNN_STATUS_NOT_SUPPORTED, "attribute type not modelled");
    if (count < 0 || (count > 0 && !elements)) return DnnFail(CUDNN_STATUS_BAD_PARAM, "bad element count");
    // Pointer-typed elements must be what they claim, which checks Tuna's translation.
    const bool pointers = type == CUDNN_TYPE_BACKEND_DESCRIPTOR || type == CUDNN_TYPE_HANDLE;
    for (int64_t i = 0; pointers && i < count; ++i) {
        const void* p = static_cast<void* const*>(elements)[i];
        if (type == CUDNN_TYPE_BACKEND_DESCRIPTOR && !DnnFind(p)) {
            return DnnFail(CUDNN_STATUS_BAD_PARAM, "descriptor element is not a descriptor");
        }
        if (type == CUDNN_TYPE_HANDLE && !DnnHandleKnown(p)) return DnnFail(CUDNN_STATUS_BAD_PARAM, "bad handle");
    }
    DnnPut(*d, name, type, elements, count);
    return CUDNN_STATUS_SUCCESS;
}

// The convolution forward operation a plan runs, or nullptr.
DnnDesc* DnnPlanOperation(const DnnDesc& plan) {
    DnnDesc* cfg = DnnRef(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
    DnnDesc* engine = cfg ? DnnRef(*cfg, CUDNN_ATTR_ENGINECFG_ENGINE) : nullptr;
    DnnDesc* graph = engine ? DnnRef(*engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH) : nullptr;
    if (!graph) return nullptr;
    auto ops = DnnValues<void*>(*graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR);
    DnnDesc* op = ops.size() == 1 ? DnnFind(ops[0]) : nullptr;
    return op && op->type == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR ? op : nullptr;
}

constexpr int64_t kDnnPlanWorkspace = 256;

cudnnStatus_t DnnBackendFinalize(cudnnBackendDescriptor_t desc) {
    DnnDesc* d = DnnFind(desc);
    if (!d) return DnnFail(CUDNN_STATUS_BAD_PARAM, "finalize of an unknown descriptor");
    auto has = [&](std::initializer_list<int> names) {
        return std::all_of(names.begin(), names.end(), [&](int n) { return d->attrs.count(n) != 0; });
    };
    bool ok = true;
    switch (d->type) {
    case CUDNN_BACKEND_TENSOR_DESCRIPTOR:
        ok = has({CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_ATTR_TENSOR_STRIDES,
                  CUDNN_ATTR_TENSOR_UNIQUE_ID});
        break;
    case CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
        ok = has({CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                  CUDNN_ATTR_CONVOLUTION_DILATIONS});
        break;
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
        ok = DnnRef(*d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X) &&
             DnnRef(*d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W) &&
             DnnRef(*d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y) &&
             DnnRef(*d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC);
        break;
    case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
        ok = has({CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_ATTR_OPERATIONGRAPH_OPS});
        break;
    case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: {
        // Two "engines" for the graph, each wrapped in an engine config: the
        // results a caller reads back into descriptors it created itself.
        DnnDesc* graph = DnnRef(*d, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH);
        if (!graph || !d->attrs.count(CUDNN_ATTR_ENGINEHEUR_MODE)) {
            ok = false;
            break;
        }
        void* graph_ptr = DnnValues<void*>(*d, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR)[0];
        std::vector<void*> results;
        for (int64_t i = 0; i < 2; ++i) {
            const uint64_t engine = DnnNewDesc(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            DnnPut(gd.descs[engine], CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, &graph_ptr, 1);
            DnnPut(gd.descs[engine], CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, &i, 1);
            gd.descs[engine].finalized = true;
            const uint64_t cfg = DnnNewDesc(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
            void* engine_ptr = reinterpret_cast<void*>(engine);
            DnnPut(gd.descs[cfg], CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, &engine_ptr, 1);
            gd.descs[cfg].finalized = true;
            results.push_back(reinterpret_cast<void*>(cfg));
        }
        d = DnnFind(desc);  // gd.descs grew; std::map keeps references stable, but be explicit
        DnnPut(*d, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, results.data(),
               static_cast<int64_t>(results.size()));
        break;
    }
    case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
        ok = has({CUDNN_ATTR_EXECUTION_PLAN_HANDLE}) && DnnPlanOperation(*d);
        if (ok) DnnPut(*d, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, &kDnnPlanWorkspace, 1);
        break;
    case CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: {
        auto ids = DnnValues<int64_t>(*d, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64);
        auto ptrs = DnnValues<void*>(*d, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR);
        ok = !ids.empty() && ids.size() == ptrs.size();
        break;
    }
    default: break;
    }
    if (!ok) return DnnFail(CUDNN_STATUS_BAD_PARAM, "finalize: required attributes missing");
    d->finalized = true;
    ++g.stats.dnn_finalizes;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t DnnBackendGet(cudnnBackendDescriptor_t desc, cudnnBackendAttributeName_t name,
                            cudnnBackendAttributeType_t type, int64_t requested, int64_t* count, void* elements) {
    DnnDesc* d = DnnFind(desc);
    if (!d || !count) return DnnFail(CUDNN_STATUS_BAD_PARAM, "get on an unknown descriptor");
    auto it = d->attrs.find(name);
    if (it == d->attrs.end()) return DnnFail(CUDNN_STATUS_BAD_PARAM, "attribute not set");
    const DnnAttr a = it->second;  // a copy: filling descriptors below may touch gd.descs
    if (a.type != type) return DnnFail(CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE, "attribute type mismatch");
    *count = a.count;
    if (!elements) return CUDNN_STATUS_SUCCESS;
    const int64_t n = std::min(requested, a.count);
    if (type == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
        // cuDNN copies each result into the descriptor the caller passed in.
        for (int64_t i = 0; i < n; ++i) {
            void* source = nullptr;
            std::memcpy(&source, a.bytes.data() + 8 * i, 8);
            DnnDesc* from = DnnFind(source);
            DnnDesc* to = DnnFind(static_cast<void**>(elements)[i]);
            if (!from || !to || from->type != to->type) {
                return DnnFail(CUDNN_STATUS_BAD_PARAM, "output descriptor of the wrong type");
            }
            *to = *from;
        }
    } else {
        std::memcpy(elements, a.bytes.data(), DnnElementSize(type) * static_cast<size_t>(n));
    }
    return CUDNN_STATUS_SUCCESS;
}

// Runs the plan's convolution (cross-correlation, NCHW float, 2-D) on the host.
cudnnStatus_t DnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t plan_p, cudnnBackendDescriptor_t pack_p) {
    DnnDesc* plan = DnnFind(plan_p);
    DnnDesc* pack = DnnFind(pack_p);
    if (!DnnHandleKnown(handle) || !plan || !pack || !plan->finalized || !pack->finalized ||
        plan->type != CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR || pack->type != CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "execute needs a finalized plan and variant pack");
    }
    DnnDesc* op = DnnPlanOperation(*plan);
    DnnDesc* conv = DnnRef(*op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC);
    auto ids = DnnValues<int64_t>(*pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64);
    auto ptrs = DnnValues<void*>(*pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR);
    auto workspace = DnnValues<void*>(*pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR);
    if (workspace.size() != 1 || !Find(reinterpret_cast<uint64_t>(workspace[0]), kDnnPlanWorkspace)) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "workspace missing or too small");
    }
    struct View {
        std::vector<int64_t> dims, strides;
        float* data = nullptr;
    };
    auto view = [&](int name, View* v) {
        DnnDesc* t = DnnRef(*op, name);
        if (!t) return false;
        v->dims = DnnValues<int64_t>(*t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64);
        v->strides = DnnValues<int64_t>(*t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64);
        auto uid = DnnValues<int64_t>(*t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64);
        if (v->dims.size() != 4 || v->strides.size() != 4 || uid.size() != 1) return false;
        auto at = std::find(ids.begin(), ids.end(), uid[0]);
        if (at == ids.end()) return false;
        size_t span = 1;
        for (int i = 0; i < 4; ++i) span += static_cast<size_t>((v->dims[i] - 1) * v->strides[i]);
        v->data = Device<float>(reinterpret_cast<uint64_t>(ptrs[static_cast<size_t>(at - ids.begin())]), span);
        return v->data != nullptr;
    };
    View x, w, y;
    if (!view(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, &x) || !view(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, &w) ||
        !view(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, &y) || !conv) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "variant pack does not cover the graph's tensors");
    }
    auto pad = DnnValues<int64_t>(*conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64);
    auto stride = DnnValues<int64_t>(*conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64);
    auto dilation = DnnValues<int64_t>(*conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64);
    auto alpha = DnnValues<float>(*op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT);
    auto beta = DnnValues<float>(*op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT);
    if (pad.size() != 2 || stride.size() != 2 || dilation.size() != 2) {
        return DnnFail(CUDNN_STATUS_BAD_PARAM, "2-D convolutions only");
    }
    const float a = alpha.empty() ? 1.0f : alpha[0], b = beta.empty() ? 0.0f : beta[0];
    auto at = [](const View& v, int64_t n, int64_t c, int64_t h, int64_t wi) {
        return n * v.strides[0] + c * v.strides[1] + h * v.strides[2] + wi * v.strides[3];
    };
    for (int64_t n = 0; n < y.dims[0]; ++n)
        for (int64_t k = 0; k < y.dims[1]; ++k)
            for (int64_t p = 0; p < y.dims[2]; ++p)
                for (int64_t q = 0; q < y.dims[3]; ++q) {
                    double sum = 0;
                    for (int64_t c = 0; c < x.dims[1]; ++c)
                        for (int64_t r = 0; r < w.dims[2]; ++r)
                            for (int64_t s = 0; s < w.dims[3]; ++s) {
                                const int64_t ih = p * stride[0] - pad[0] + r * dilation[0];
                                const int64_t iw = q * stride[1] - pad[1] + s * dilation[1];
                                if (ih < 0 || iw < 0 || ih >= x.dims[2] || iw >= x.dims[3]) continue;
                                sum += x.data[at(x, n, c, ih, iw)] * w.data[at(w, k, c, r, s)];
                            }
                    float& out = y.data[at(y, n, k, p, q)];
                    out = static_cast<float>(a * sum + b * out);
                }
    ++g.stats.dnn_executes;
    return CUDNN_STATUS_SUCCESS;
}
#endif  // TUNA_CUDA_HAVE_CUDNN

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
    TUNA_MOCK("cuInit", Init),
    TUNA_MOCK("cuDriverGetVersion", DriverGetVersionDrv),
    TUNA_MOCK("cuDeviceGet", DeviceGet),
    TUNA_MOCK("cuDeviceGetCount", DeviceGetCount),
    TUNA_MOCK("cuDeviceGetAttribute", DeviceGetAttributeDrv),
    TUNA_MOCK("cuDeviceTotalMem_v2", DeviceTotalMem),
    TUNA_MOCK("cuDeviceGetUuid_v2", DeviceGetUuid),
    TUNA_MOCK("cuDeviceGetName", DeviceGetName),
    TUNA_MOCK("cuDevicePrimaryCtxGetState", PrimaryCtxGetState),
    TUNA_MOCK("cuDevicePrimaryCtxSetFlags_v2", PrimaryCtxSetFlags),
    TUNA_MOCK("cuDevicePrimaryCtxRetain", PrimaryCtxRetain),
    TUNA_MOCK("cuModuleLoadData", ModuleLoadData),
    TUNA_MOCK("cuModuleGetFunction", ModuleGetFunction),
    TUNA_MOCK("cuModuleGetGlobal_v2", ModuleGetGlobal),
    TUNA_MOCK("cuFuncGetParamInfo", FuncGetParamInfo),
    TUNA_MOCK("cuLaunchKernel", LaunchKernel),
    TUNA_MOCK("cuFuncGetAttribute", FuncGetAttribute),
    TUNA_MOCK("cuFuncSetAttribute", FuncSetAttribute),
    TUNA_MOCK("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", Occupancy),
#ifdef TUNA_CUDA_HAVE_CUBLAS
    TUNA_MOCK("cublasCreate_v2", BlasCreate),
    TUNA_MOCK("cublasDestroy_v2", BlasDestroy),
    TUNA_MOCK("cublasGetVersion_v2", BlasGetVersion),
    TUNA_MOCK("cublasSetStream_v2", BlasSetStream),
    TUNA_MOCK("cublasGetStream_v2", BlasGetStream),
    TUNA_MOCK("cublasSetPointerMode_v2", BlasSetPointerMode),
    TUNA_MOCK("cublasSetMathMode", BlasSetMathMode),
    TUNA_MOCK("cublasGetMathMode", BlasGetMathMode),
    TUNA_MOCK("cublasSetWorkspace_v2", BlasSetWorkspace),
    TUNA_MOCK("cublasSgemm_v2", Gemm<float>),
    TUNA_MOCK("cublasDgemm_v2", Gemm<double>),
    TUNA_MOCK("cublasSgemmStridedBatched", GemmStridedBatched<float>),
    TUNA_MOCK("cublasDgemmStridedBatched", GemmStridedBatched<double>),
    TUNA_MOCK("cublasGemmEx", BlasGemmEx),
    TUNA_MOCK("cublasSaxpy_v2", BlasSaxpy),
    TUNA_MOCK("cublasSscal_v2", BlasSscal),
    TUNA_MOCK("cublasSdot_v2", BlasSdot),
    TUNA_MOCK("cublasSnrm2_v2", BlasSnrm2),
    TUNA_MOCK("cublasSetMatrix", BlasSetMatrix),
    TUNA_MOCK("cublasGetMatrix", BlasGetMatrix),
    TUNA_MOCK("cublasSetMatrixAsync", BlasSetMatrixAsync),
    TUNA_MOCK("cublasGetMatrixAsync", BlasGetMatrixAsync),
    TUNA_MOCK("cublasLtCreate", LtCreate),
    TUNA_MOCK("cublasLtDestroy", LtDestroy),
    TUNA_MOCK("cublasLtMatmulDescCreate", LtDescCreate),
    TUNA_MOCK("cublasLtMatmulDescDestroy", LtDescDestroy),
    TUNA_MOCK("cublasLtMatmulDescSetAttribute", LtDescSetAttribute),
    TUNA_MOCK("cublasLtMatrixLayoutCreate", LtLayoutCreate),
    TUNA_MOCK("cublasLtMatrixLayoutDestroy", LtLayoutDestroy),
    TUNA_MOCK("cublasLtMatrixLayoutSetAttribute", LtLayoutSetAttribute),
    TUNA_MOCK("cublasLtMatmulPreferenceCreate", LtPrefCreate),
    TUNA_MOCK("cublasLtMatmulPreferenceDestroy", LtPrefDestroy),
    TUNA_MOCK("cublasLtMatmulPreferenceSetAttribute", LtPrefSetAttribute),
    TUNA_MOCK("cublasLtMatmulAlgoGetHeuristic", LtHeuristic),
    TUNA_MOCK("cublasLtMatmul", LtMatmul),
#endif
#ifdef TUNA_CUDA_HAVE_CUDNN
    TUNA_MOCK("cudnnCreate", DnnCreate),
    TUNA_MOCK("cudnnDestroy", DnnDestroy),
    TUNA_MOCK("cudnnSetStream", DnnSetStream),
    TUNA_MOCK("cudnnGetVersion", DnnGetVersion),
    TUNA_MOCK("cudnnGetMaxDeviceVersion", DnnGetMaxDeviceVersion),
    TUNA_MOCK("cudnnGetCudartVersion", DnnGetCudartVersion),
    TUNA_MOCK("cudnnGetLastErrorString", DnnGetLastErrorString),
    TUNA_MOCK("cudnnCreateTensorDescriptor", DnnCreateTensor),
    TUNA_MOCK("cudnnSetTensorNdDescriptor", DnnSetTensorNd),
    TUNA_MOCK("cudnnDestroyTensorDescriptor", DnnDestroyTensor),
    TUNA_MOCK("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize", DnnBnFwdWorkspace),
    TUNA_MOCK("cudnnGetBatchNormalizationBackwardExWorkspaceSize", DnnBnBwdWorkspace),
    TUNA_MOCK("cudnnGetBatchNormalizationTrainingExReserveSpaceSize", DnnBnReserve),
    TUNA_MOCK("cudnnBatchNormalizationForwardTrainingEx", DnnBnForwardTraining),
    TUNA_MOCK("cudnnBatchNormalizationBackwardEx", DnnBnBackward),
    TUNA_MOCK("cudnnBatchNormalizationForwardInference", DnnBnInference),
    TUNA_MOCK("cudnnBackendCreateDescriptor", DnnBackendCreate),
    TUNA_MOCK("cudnnBackendDestroyDescriptor", DnnBackendDestroy),
    TUNA_MOCK("cudnnBackendSetAttribute", DnnBackendSet),
    TUNA_MOCK("cudnnBackendGetAttribute", DnnBackendGet),
    TUNA_MOCK("cudnnBackendFinalize", DnnBackendFinalize),
    TUNA_MOCK("cudnnBackendExecute", DnnBackendExecute),
#endif
};
#undef TUNA_MOCK

}  // namespace

extern "C" TUNA_MOCK_EXPORT void* tuna_cuda_resolve(const char* symbol) {
    for (const Entry& e : kEntries) {
        if (std::strcmp(e.name, symbol) == 0) return e.fn;
    }
    return nullptr;
}

void MockCudaReset() {
    g = MockDevice{};
#ifdef TUNA_CUDA_HAVE_CUDNN
    gd = MockDnn{};
#endif
}

MockCudaStats MockCudaGetStats() { return g.stats; }

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
// With cuBLAS headers (TUNA_CUDA_HAVE_CUBLAS) it also models a small cuBLAS:
// handles with a stream and pointer mode, column-major S/D gemm (plain,
// strided-batched, GemmEx for fp32), axpy, scal, dot, nrm2 and the
// host<->device matrix copies, computed on the host from "device" memory.
// Scalars are read from host or device memory according to the pointer mode,
// so the remoting of both modes is checked by the results.
//
// It is linked into tuna_cuda_tests (in-process) and also built as a shared
// library that tuna_server loads in place of libcudart: it exports only
// tuna_cuda_resolve, which backend_real.cpp uses as its symbol source.

#include "mock_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#ifdef TUNA_CUDA_HAVE_CUBLAS
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
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

struct BlasHandle {
    uint64_t stream = 0;
    bool device_scalars = false;  // CUBLAS_POINTER_MODE_DEVICE
    int math_mode = 0;
};

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
#endif  // TUNA_CUDA_HAVE_CUBLAS

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

void MockCudaReset() { g = MockDevice{}; }

MockCudaStats MockCudaGetStats() { return g.stats; }

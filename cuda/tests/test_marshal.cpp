// Marshalling round-trip for the generated CUDA runtime remoting, with no GPU
// and no transport: the client stubs encode -> CudaDispatch decodes -> a mock
// "driver" runs -> encodes outs -> the client stub decodes. The mock models a
// tiny device: cudaMalloc hands out opaque handles backed by host buffers, and
// cudaMemcpy moves bytes in and out of them, so handles and the bulk path are
// exercised exactly as the real server would.

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "tuna/cuda/backend.h"
#include "tuna/cuda/client.h"

#include "cuda_api_ids.h"

using namespace tuna;
using namespace tuna::cudart;

namespace {

// A fake device: handles are 1-based indices into `buffers`.
struct FakeDevice {
    std::map<uint64_t, std::vector<char>> buffers;
    uint64_t next_handle = 0x100000;  // not near 0, to catch null/handle confusion
    int device_count = 2;
    size_t free_mem = 8ull << 30;
};
FakeDevice g_dev;

cudaError_t MockGetDeviceCount(int* count) { *count = g_dev.device_count; return cudaSuccess; }
cudaError_t MockGetDevice(int* device) { *device = 1; return cudaSuccess; }
cudaError_t MockSetDevice(int device) { return device < g_dev.device_count ? cudaSuccess : cudaErrorInvalidDevice; }
cudaError_t MockDeviceSynchronize() { return cudaSuccess; }

cudaError_t MockGetDeviceProperties(cudaDeviceProp* prop, int device) {
    std::memset(prop, 0, sizeof(*prop));
    std::snprintf(prop->name, sizeof(prop->name), "Mock GPU %d", device);
    prop->major = 8;
    prop->minor = 9;
    prop->totalGlobalMem = g_dev.free_mem;
    return cudaSuccess;
}

cudaError_t MockMalloc(void** devPtr, size_t size) {
    const uint64_t handle = g_dev.next_handle;
    g_dev.next_handle += 0x1000;
    g_dev.buffers[handle].resize(size);
    *devPtr = reinterpret_cast<void*>(handle);
    g_dev.free_mem -= size;
    return cudaSuccess;
}

cudaError_t MockFree(void* devPtr) {
    auto it = g_dev.buffers.find(reinterpret_cast<uint64_t>(devPtr));
    if (it == g_dev.buffers.end()) return cudaErrorInvalidValue;
    g_dev.free_mem += it->second.size();
    g_dev.buffers.erase(it);
    return cudaSuccess;
}

cudaError_t MockMemGetInfo(size_t* free, size_t* total) {
    *free = g_dev.free_mem;
    *total = 8ull << 30;
    return cudaSuccess;
}

cudaError_t MockMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyHostToDevice) {
        auto it = g_dev.buffers.find(reinterpret_cast<uint64_t>(dst));
        if (it == g_dev.buffers.end() || it->second.size() < count) return cudaErrorInvalidValue;
        std::memcpy(it->second.data(), src, count);
    } else if (kind == cudaMemcpyDeviceToHost) {
        auto it = g_dev.buffers.find(reinterpret_cast<uint64_t>(const_cast<void*>(src)));
        if (it == g_dev.buffers.end() || it->second.size() < count) return cudaErrorInvalidValue;
        std::memcpy(dst, it->second.data(), count);
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

class MockBackend : public CudaBackend {
public:
    void* Resolve(const char* symbol) override {
        const std::string name = symbol;
        if (name == "cudaGetDeviceCount") return reinterpret_cast<void*>(MockGetDeviceCount);
        if (name == "cudaGetDevice") return reinterpret_cast<void*>(MockGetDevice);
        if (name == "cudaSetDevice") return reinterpret_cast<void*>(MockSetDevice);
        if (name == "cudaDeviceSynchronize") return reinterpret_cast<void*>(MockDeviceSynchronize);
        if (name == "cudaGetDeviceProperties_v2") return reinterpret_cast<void*>(MockGetDeviceProperties);
        if (name == "cudaMalloc") return reinterpret_cast<void*>(MockMalloc);
        if (name == "cudaFree") return reinterpret_cast<void*>(MockFree);
        if (name == "cudaMemGetInfo") return reinterpret_cast<void*>(MockMemGetInfo);
        if (name == "cudaMemcpy") return reinterpret_cast<void*>(MockMemcpy);
        return nullptr;
    }
};

MockBackend g_backend;

// In-process stand-in for the transport: run the dispatcher directly.
Reply InProcessCall(uint32_t api_id, const std::string& args, const std::string& bulk_in) {
    RpcRequest request;
    request.set_api_id(api_id);
    request.set_args(args);
    std::string bulk_out;
    RpcResponse resp = CudaDispatch(request, bulk_in, &bulk_out, g_backend);
    Reply reply;
    reply.result = resp.result();
    reply.out_args = resp.out_args();
    reply.bulk_out = std::move(bulk_out);
    return reply;
}

struct HookGuard {
    HookGuard() { g_dev = FakeDevice{}; SetCallHook(InProcessCall); }
    ~HookGuard() { SetCallHook(nullptr); }
};

}  // namespace

TEST_CASE("Scalar out params round-trip", "[cuda]") {
    HookGuard guard;
    int count = -1;
    REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
    CHECK(count == 2);

    int device = -1;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    CHECK(device == 1);

    CHECK(cudaSetDevice(0) == cudaSuccess);
    CHECK(cudaSetDevice(99) == cudaErrorInvalidDevice);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
}

TEST_CASE("Struct out param round-trips the full cudaDeviceProp", "[cuda]") {
    HookGuard guard;
    cudaDeviceProp prop;
    REQUIRE(cudaGetDeviceProperties(&prop, 1) == cudaSuccess);
    CHECK(std::string(prop.name) == "Mock GPU 1");
    CHECK(prop.major == 8);
    CHECK(prop.minor == 9);
    CHECK(prop.totalGlobalMem == (8ull << 30));
}

TEST_CASE("Device handles and the bulk path move real bytes", "[cuda]") {
    HookGuard guard;
    void* d = nullptr;
    const size_t n = 4096;
    REQUIRE(cudaMalloc(&d, n) == cudaSuccess);
    REQUIRE(d != nullptr);

    std::vector<char> host_in(n), host_out(n, 0);
    for (size_t i = 0; i < n; ++i) host_in[i] = static_cast<char>(i * 31 + 7);
    REQUIRE(cudaMemcpy(d, host_in.data(), n, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(host_out.data(), d, n, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(host_out == host_in);

    size_t free_mem = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_mem, &total) == cudaSuccess);
    CHECK(total == (8ull << 30));
    CHECK(free_mem == (8ull << 30) - n);  // the allocation is reflected

    REQUIRE(cudaFree(d) == cudaSuccess);
    REQUIRE(cudaMemGetInfo(&free_mem, &total) == cudaSuccess);
    CHECK(free_mem == (8ull << 30));
}

TEST_CASE("An unknown api is reported as not supported", "[cuda]") {
    HookGuard guard;
    // api_id past the generated range: the dispatcher must not crash.
    RpcRequest request;
    request.set_api_id(999999);
    std::string bulk_out;
    RpcResponse resp = CudaDispatch(request, "", &bulk_out, g_backend);
    CHECK(resp.result() == 801);  // cudaErrorNotSupported
}

TEST_CASE("A truncated arg buffer is rejected, not read out of bounds", "[cuda]") {
    HookGuard guard;
    RpcRequest request;
    request.set_api_id(kApi_cudaSetDevice);
    request.set_args("");  // cudaSetDevice needs 4 bytes
    std::string bulk_out;
    RpcResponse resp = CudaDispatch(request, "", &bulk_out, g_backend);
    CHECK(resp.result() == 1);  // cudaErrorInvalidValue
}

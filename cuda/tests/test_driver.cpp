// Round trip of the driver-API remoting with no GPU and no transport: the
// driver shim encodes -> tuna_cudart_call (the runtime shim's bridge) ->
// CudaSession decodes -> the mock driver (mock_cuda.cpp) -> outs are encoded
// -> the shim decodes. Also covers the shim's client-side state: context
// tokens, the per-thread current context, and the caches PyTorch's per-op
// primary-context checks rely on.

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_api_ids.h"
#include "mock_cuda.h"
#include "tuna/cuda/backend.h"
#include "tuna/cuda/client.h"
#include "tuna/cuda/driver_client.h"

using namespace tuna;
using namespace tuna::cudart;

namespace {

class MockBackend : public CudaBackend {
public:
    void* Resolve(const char* symbol) override { return tuna_cuda_resolve(symbol); }
};

MockBackend g_backend;
std::unique_ptr<CudaSession> g_session;
uint64_t g_calls = 0;  // requests that reached the server

Reply InProcessCall(uint32_t api_id, const std::string& args, const std::string& bulk_in, bool) {
    ++g_calls;
    RpcRequest request;
    request.set_api_id(api_id);
    request.set_args(args);
    std::string bulk_out;
    RpcResponse resp = g_session->Dispatch(request, bulk_in, &bulk_out);
    Reply reply;
    reply.result = resp.result();
    reply.out_args = resp.out_args();
    reply.bulk_out = std::move(bulk_out);
    return reply;
}

struct Fixture {
    Fixture() {
        MockCudaReset();
        g_session = std::make_unique<CudaSession>(g_backend);
        ResetClientStateForTesting();
        tuna::cudrv::ResetStateForTesting();
        SetCallHook(InProcessCall);
        LastError(/*clear=*/true);
    }
    ~Fixture() { SetCallHook(nullptr); }
};

RpcResponse Raw(uint32_t api_id, const std::string& args) {
    RpcRequest request;
    request.set_api_id(api_id);
    request.set_args(args);
    std::string bulk_out;
    return g_session->Dispatch(request, {}, &bulk_out);
}

}  // namespace

TEST_CASE_METHOD(Fixture, "driver: device queries reach the server's driver", "[cuda][driver]") {
    // No cuInit from the client: the worker initializes its driver itself.
    int count = 0;
    REQUIRE(cuDeviceGetCount(&count) == CUDA_SUCCESS);
    CHECK(count == 2);
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CHECK(cuInit(1) == CUDA_ERROR_INVALID_VALUE);

    int version = 0;
    REQUIRE(cuDriverGetVersion(&version) == CUDA_SUCCESS);
    CHECK(version == 12080);

    CUdevice dev = -1;
    REQUIRE(cuDeviceGet(&dev, 1) == CUDA_SUCCESS);
    CHECK(dev == 1);
    CHECK(cuDeviceGet(&dev, 7) == CUDA_ERROR_INVALID_DEVICE);

    char name[64] = {};
    REQUIRE(cuDeviceGetName(name, sizeof(name), 1) == CUDA_SUCCESS);
    CHECK(std::string(name) == "Tuna Mock GPU 1");
    char shorter[5];
    std::memset(shorter, 'x', sizeof(shorter));
    REQUIRE(cuDeviceGetName(shorter, sizeof(shorter), 0) == CUDA_SUCCESS);
    CHECK(std::string(shorter) == "Tuna");  // truncated and terminated, like the driver
    CHECK(cuDeviceGetName(name, 0, 0) == CUDA_ERROR_INVALID_VALUE);

    size_t bytes = 0;
    REQUIRE(cuDeviceTotalMem(&bytes, 0) == CUDA_SUCCESS);
    CHECK(bytes == (8ull << 30));

    CUuuid uuid{};
    REQUIRE(cuDeviceGetUuid_v2(&uuid, 1) == CUDA_SUCCESS);
    CHECK(static_cast<unsigned char>(uuid.bytes[0]) == 0xA1);
}

TEST_CASE_METHOD(Fixture, "driver: device attributes are fetched once", "[cuda][driver]") {
    int value = 0;
    REQUIRE(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) == CUDA_SUCCESS);
    CHECK(value == 8);
    const uint64_t queries = MockCudaGetStats().driver_queries;
    for (int i = 0; i < 10; ++i) {
        REQUIRE(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) == CUDA_SUCCESS);
    }
    CHECK(MockCudaGetStats().driver_queries == queries);
    CHECK(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 5) == CUDA_ERROR_INVALID_DEVICE);
    CHECK(cuDeviceGetAttribute(nullptr, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0) == CUDA_ERROR_INVALID_VALUE);
}

TEST_CASE_METHOD(Fixture, "driver: primary context state follows the runtime and is cached once active",
                 "[cuda][driver]") {
    unsigned flags = 1;
    int active = 1;
    REQUIRE(cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_SUCCESS);
    CHECK(active == 0);  // nothing has used the device yet
    CUcontext ctx = reinterpret_cast<CUcontext>(1);
    REQUIRE(cuCtxGetCurrent(&ctx) == CUDA_SUCCESS);
    CHECK(ctx == nullptr);

    // The runtime's first real call creates the primary context, as on a GPU.
    REQUIRE(cudaFree(nullptr) == cudaSuccess);
    REQUIRE(cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_SUCCESS);
    CHECK(active == 1);
    REQUIRE(cuCtxGetCurrent(&ctx) == CUDA_SUCCESS);
    CHECK(ctx != nullptr);
    CUdevice dev = -1;
    REQUIRE(cuCtxGetDevice(&dev) == CUDA_SUCCESS);
    CHECK(dev == 0);

    // PyTorch asks before every op: once active, the answer is local.
    const uint64_t calls = g_calls;
    for (int i = 0; i < 100; ++i) REQUIRE(cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_SUCCESS);
    CHECK(g_calls == calls);

    // A device reset invalidates it.
    REQUIRE(cudaDeviceReset() == cudaSuccess);
    REQUIRE(cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_SUCCESS);
    CHECK(g_calls > calls);

    CHECK(cuDevicePrimaryCtxGetState(0, nullptr, &active) == CUDA_ERROR_INVALID_VALUE);
}

TEST_CASE_METHOD(Fixture, "driver: retained primary contexts are tokens that select the device",
                 "[cuda][driver]") {
    CUcontext ctx1 = nullptr;
    REQUIRE(cuDevicePrimaryCtxRetain(&ctx1, 1) == CUDA_SUCCESS);
    REQUIRE(ctx1 != nullptr);
    unsigned flags = 0;
    int active = 0;
    REQUIRE(cuDevicePrimaryCtxGetState(1, &flags, &active) == CUDA_SUCCESS);
    CHECK(active == 1);
    // Retaining again is answered without retaining twice on the server.
    CUcontext again = nullptr;
    REQUIRE(cuDevicePrimaryCtxRetain(&again, 1) == CUDA_SUCCESS);
    CHECK(again == ctx1);
    CHECK(cuDevicePrimaryCtxRetain(&again, 9) == CUDA_ERROR_INVALID_DEVICE);

    // Making it current makes device 1 the runtime's device too.
    REQUIRE(cuCtxSetCurrent(ctx1) == CUDA_SUCCESS);
    int device = -1;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    CHECK(device == 1);
    CUcontext current = nullptr;
    REQUIRE(cuCtxGetCurrent(&current) == CUDA_SUCCESS);
    CHECK(current == ctx1);

    // Push/pop across devices.
    CUcontext ctx0 = nullptr;
    REQUIRE(cuDevicePrimaryCtxRetain(&ctx0, 0) == CUDA_SUCCESS);
    REQUIRE(cuCtxPushCurrent(ctx0) == CUDA_SUCCESS);
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    CHECK(device == 0);
    CUcontext popped = nullptr;
    REQUIRE(cuCtxPopCurrent(&popped) == CUDA_SUCCESS);
    CHECK(popped == ctx0);
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    CHECK(device == 1);
    REQUIRE(cuCtxSynchronize() == CUDA_SUCCESS);

    // Unsetting it leaves the thread without a context.
    REQUIRE(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS);
    REQUIRE(cuCtxGetCurrent(&current) == CUDA_SUCCESS);
    CHECK(current == nullptr);
    CHECK(cuCtxSynchronize() == CUDA_ERROR_INVALID_CONTEXT);
    CUdevice dev = 0;
    CHECK(cuCtxGetDevice(&dev) == CUDA_ERROR_INVALID_CONTEXT);

    REQUIRE(cuDevicePrimaryCtxRelease(1) == CUDA_SUCCESS);
}

TEST_CASE_METHOD(Fixture, "driver: only Tuna's own context tokens are accepted", "[cuda][driver]") {
    CHECK(cuCtxSetCurrent(reinterpret_cast<CUcontext>(0x1234)) == CUDA_ERROR_INVALID_CONTEXT);
    CHECK(cuCtxPushCurrent(nullptr) == CUDA_ERROR_INVALID_CONTEXT);
    CUcontext ctx = nullptr;
    REQUIRE(cuDevicePrimaryCtxRetain(&ctx, 0) == CUDA_SUCCESS);
    CHECK(cuCtxSetCurrent(reinterpret_cast<CUcontext>(reinterpret_cast<uintptr_t>(ctx) + 1)) ==
          CUDA_ERROR_INVALID_CONTEXT);
    CHECK(cuCtxPopCurrent(&ctx) == CUDA_ERROR_INVALID_CONTEXT);  // this thread pushed nothing
}

TEST_CASE_METHOD(Fixture, "driver: the current context is per thread", "[cuda][driver]") {
    CUcontext ctx1 = nullptr;
    REQUIRE(cuDevicePrimaryCtxRetain(&ctx1, 1) == CUDA_SUCCESS);
    REQUIRE(cuCtxSetCurrent(ctx1) == CUDA_SUCCESS);
    CUcontext seen = reinterpret_cast<CUcontext>(1);
    std::thread other([&] { cuCtxGetCurrent(&seen); });
    other.join();
    // The other thread set nothing: it sees the runtime device's primary
    // context (device 1, retained), not this thread's explicit stack.
    CHECK(seen == ctx1);
    REQUIRE(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS);
    std::thread third([&] { cuCtxGetCurrent(&seen); });
    third.join();
    CHECK(seen == ctx1);
    CUcontext mine = reinterpret_cast<CUcontext>(1);
    REQUIRE(cuCtxGetCurrent(&mine) == CUDA_SUCCESS);
    CHECK(mine == nullptr);
}

TEST_CASE_METHOD(Fixture, "driver: error names and strings are local", "[cuda][driver]") {
    const char* text = nullptr;
    REQUIRE(cuGetErrorName(CUDA_ERROR_INVALID_VALUE, &text) == CUDA_SUCCESS);
    CHECK(std::string(text) == "CUDA_ERROR_INVALID_VALUE");
    REQUIRE(cuGetErrorString(CUDA_ERROR_OUT_OF_MEMORY, &text) == CUDA_SUCCESS);
    CHECK(std::string(text).find("memory") != std::string::npos);
    CHECK(cuGetErrorName(static_cast<CUresult>(12345), &text) == CUDA_ERROR_INVALID_VALUE);
    CHECK(text == nullptr);
    CHECK(g_calls == 0);
}

TEST_CASE_METHOD(Fixture, "driver: server rejects malformed driver requests", "[cuda][driver]") {
    CHECK(Raw(kApi_cuDeviceGetCount, std::string(3, '\0')).result() == CUDA_ERROR_INVALID_VALUE);
    CHECK(Raw(kApi_cuDevicePrimaryCtxRetain, std::string(2, '\0')).result() == CUDA_ERROR_INVALID_VALUE);
    CHECK(Raw(kApi_cuDeviceGetName, std::string(9, '\0')).result() == CUDA_ERROR_INVALID_VALUE);
}

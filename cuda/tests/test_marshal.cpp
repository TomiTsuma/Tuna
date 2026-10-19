// Round trip of the CUDA runtime remoting with no GPU and no transport: the
// client shim encodes -> CudaSession decodes -> the mock CUDA stack
// (mock_cuda.cpp) runs -> outs are encoded -> the shim decodes. The mock
// models a device in host memory and executes a few known kernels, so handles,
// the bulk path, kernel registration/launch and symbol access are exercised
// exactly as the real server would run them.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "cuda_api_ids.h"
#include "mock_cuda.h"
#include "tuna/cuda/backend.h"
#include "tuna/cuda/client.h"

using namespace tuna;
using namespace tuna::cudart;

// What nvcc's generated host code calls (crt/host_runtime.h).
extern "C" {
void** __cudaRegisterFatBinary(void* fatCubin);
void __cudaRegisterFatBinaryEnd(void** handle);
void __cudaRegisterFunction(void** handle, const char* hostFun, char* deviceFun, const char* deviceName,
                            int thread_limit, uint3* tid, uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
void __cudaRegisterVar(void** handle, char* hostVar, char* deviceAddress, const char* deviceName, int ext,
                       size_t size, int constant, int global);
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, void* stream);
}

namespace {

class MockBackend : public CudaBackend {
public:
    void* Resolve(const char* symbol) override { return tuna_cuda_resolve(symbol); }
};

MockBackend g_backend;
std::unique_ptr<CudaSession> g_session;
uint64_t g_round_trips = 0;  // blocking calls that reached the "server"

// In-process stand-in for the transport: run the session's dispatcher directly.
Reply InProcessCall(uint32_t api_id, const std::string& args, const std::string& bulk_in, bool async) {
    if (!async) ++g_round_trips;
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
        SetCallHook(InProcessCall);
        LastError(/*clear=*/true);
    }
    ~Fixture() { SetCallHook(nullptr); }
};

// A minimal fatbin as nvcc embeds it: the wrapper points at a fatbin header
// (magic, version, header size, payload size) followed by the payload.
struct alignas(8) FakeFatbin {
    uint32_t magic = 0xBA55ED50;
    uint16_t version = 1;
    uint16_t header_size = 16;
    uint64_t fat_size = 32;
    char payload[32] = "mock device code";
};
struct FakeWrapper {
    int magic;
    int version;
    const void* data;
    void* filename_or_fatbins;
};
FakeFatbin g_fatbin;
FakeWrapper g_wrapper{0x466243b1, 1, &g_fatbin, nullptr};

// Host-side "kernel stubs" and a "__constant__ variable": only their
// addresses matter, exactly like nvcc's.
char g_vec_add_stub, g_scale_stub, g_fill_stub, g_missing_stub, g_unregistered_stub;
float g_scale_symbol;

// Registers the module once per process, the way static constructors do.
void RegisterProgram() {
    static bool done = false;
    if (done) return;
    done = true;
    void** handle = __cudaRegisterFatBinary(&g_wrapper);
    __cudaRegisterFunction(handle, &g_vec_add_stub, nullptr, "vecAdd", -1, nullptr, nullptr, nullptr, nullptr, nullptr);
    __cudaRegisterFunction(handle, &g_scale_stub, nullptr, "scale", -1, nullptr, nullptr, nullptr, nullptr, nullptr);
    __cudaRegisterFunction(handle, &g_fill_stub, nullptr, "fill", -1, nullptr, nullptr, nullptr, nullptr, nullptr);
    __cudaRegisterFunction(handle, &g_missing_stub, nullptr, "notInModule", -1, nullptr, nullptr, nullptr, nullptr,
                           nullptr);
    __cudaRegisterVar(handle, reinterpret_cast<char*>(&g_scale_symbol), nullptr, "gScale", 0, sizeof(float), 1, 0);
    __cudaRegisterFatBinaryEnd(handle);
}

// Allocates `n` floats on the device and fills them from `values`.
float* DeviceFloats(const std::vector<float>& values) {
    void* d = nullptr;
    REQUIRE(cudaMalloc(&d, values.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    return static_cast<float*>(d);
}

std::vector<float> HostFloats(const float* d, size_t n) {
    std::vector<float> values(n);
    REQUIRE(cudaMemcpy(values.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    return values;
}

}  // namespace

TEST_CASE_METHOD(Fixture, "Scalar out params round-trip", "[cuda]") {
    int count = -1;
    REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
    CHECK(count == 2);

    CHECK(cudaSetDevice(1) == cudaSuccess);
    int device = -1;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    CHECK(device == 1);
    CHECK(cudaSetDevice(99) == cudaErrorInvalidDevice);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    int version = 0;
    REQUIRE(cudaDriverGetVersion(&version) == cudaSuccess);
    CHECK(version == 12080);
    REQUIRE(cudaRuntimeGetVersion(&version) == cudaSuccess);  // client-local
    CHECK(version == CUDART_VERSION);
}

TEST_CASE_METHOD(Fixture, "Device properties round-trip in full and are fetched once", "[cuda]") {
    cudaDeviceProp prop;
    REQUIRE(cudaGetDeviceProperties(&prop, 1) == cudaSuccess);
    CHECK(std::string(prop.name) == "Tuna Mock GPU 1");
    CHECK(prop.major == 8);
    CHECK(prop.minor == 9);
    CHECK(prop.totalGlobalMem == (8ull << 30));

    cudaDeviceProp again;
    REQUIRE(cudaGetDeviceProperties(&again, 1) == cudaSuccess);
    CHECK(std::memcmp(&prop, &again, sizeof(prop)) == 0);
    CHECK(MockCudaGetStats().property_queries == 1);  // the second came from the client cache

    int value = 0;
    REQUIRE(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess);
    REQUIRE(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess);
    CHECK(value == 20);
    CHECK(MockCudaGetStats().attribute_queries == 1);
}

TEST_CASE_METHOD(Fixture, "Device handles and the bulk path move real bytes", "[cuda]") {
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
    CHECK(free_mem == total - n);  // the allocation is reflected

    REQUIRE(cudaFree(d) == cudaSuccess);
    REQUIRE(cudaMemGetInfo(&free_mem, &total) == cudaSuccess);
    CHECK(free_mem == total);
}

TEST_CASE_METHOD(Fixture, "cudaMemcpyDefault infers direction from known allocations", "[cuda]") {
    std::vector<float> values{1, 2, 3, 4};
    void* d = nullptr;
    REQUIRE(cudaMalloc(&d, 64) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, values.data(), 16, cudaMemcpyDefault) == cudaSuccess);  // H2D
    // An interior device pointer is still recognised.
    REQUIRE(cudaMemcpy(static_cast<char*>(d) + 16, d, 16, cudaMemcpyDefault) == cudaSuccess);  // D2D
    std::vector<float> out(8, 0);
    REQUIRE(cudaMemcpy(out.data(), d, 32, cudaMemcpyDefault) == cudaSuccess);  // D2H
    CHECK(out == std::vector<float>{1, 2, 3, 4, 1, 2, 3, 4});

    std::vector<float> copy(4, 0);
    REQUIRE(cudaMemcpy(copy.data(), values.data(), 16, cudaMemcpyDefault) == cudaSuccess);  // H2H, local
    CHECK(copy == values);
}

TEST_CASE_METHOD(Fixture, "memset, streams, events and async copies", "[cuda]") {
    void* d = nullptr;
    REQUIRE(cudaMalloc(&d, 256) == cudaSuccess);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(stream != nullptr);
    cudaEvent_t start = nullptr, stop = nullptr;
    REQUIRE(cudaEventCreate(&start) == cudaSuccess);
    REQUIRE(cudaEventCreate(&stop) == cudaSuccess);

    REQUIRE(cudaEventRecord(start, stream) == cudaSuccess);
    REQUIRE(cudaMemsetAsync(d, 0x5a, 256, stream) == cudaSuccess);
    std::vector<unsigned char> pattern(64, 0x11);
    REQUIRE(cudaMemcpyAsync(d, pattern.data(), 64, cudaMemcpyHostToDevice, stream) == cudaSuccess);
    REQUIRE(cudaEventRecord(stop, stream) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(stop) == cudaSuccess);

    std::vector<unsigned char> out(256, 0);
    REQUIRE(cudaMemcpyAsync(out.data(), d, 256, cudaMemcpyDeviceToHost, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(out[0] == 0x11);
    CHECK(out[63] == 0x11);
    CHECK(out[64] == 0x5a);
    CHECK(out[255] == 0x5a);

    float ms = 0;
    REQUIRE(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess);
    CHECK(ms == 1.0f);  // the mock's "time" counts records between the events

    REQUIRE(cudaMemset(d, 0, 256) == cudaSuccess);
    CHECK(cudaEventDestroy(start) == cudaSuccess);
    CHECK(cudaEventDestroy(stop) == cudaSuccess);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE_METHOD(Fixture, "A registered kernel launches with by-value parameters", "[cuda][launch]") {
    RegisterProgram();
    const int n = 1000;
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }
    float* da = DeviceFloats(a);
    float* db = DeviceFloats(b);
    float* dc = DeviceFloats(std::vector<float>(n, -1));

    int count = n;
    void* args[] = {&da, &db, &dc, &count};
    REQUIRE(cudaLaunchKernel(&g_vec_add_stub, dim3((n + 255) / 256), dim3(256), args, 0, nullptr) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> c = HostFloats(dc, n);
    for (int i = 0; i < n; ++i) REQUIRE(c[i] == static_cast<float>(3 * i));

    // A second launch reuses the uploaded module and the cached layout.
    REQUIRE(cudaLaunchKernel(&g_vec_add_stub, dim3(4), dim3(256), args, 0, nullptr) == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    CHECK(MockCudaGetStats().module_loads == 1);
    CHECK(MockCudaGetStats().launches == 2);
}

TEST_CASE_METHOD(Fixture, "The <<<>>> call-configuration path reaches cudaLaunchKernel", "[cuda][launch]") {
    RegisterProgram();
    void* d = nullptr;
    REQUIRE(cudaMalloc(&d, 64 * sizeof(int)) == cudaSuccess);

    // What nvcc generates for fill<<<2, 32>>>(d, 7, 64):
    REQUIRE(__cudaPushCallConfiguration(dim3(2), dim3(32), 0, nullptr) == 0);
    dim3 grid, block;
    size_t shared = 1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(1);
    REQUIRE(__cudaPopCallConfiguration(&grid, &block, &shared, &stream) == cudaSuccess);
    int value = 7, n = 64;
    void* args[] = {&d, &value, &n};
    REQUIRE(cudaLaunchKernel(&g_fill_stub, grid, block, args, shared, stream) == cudaSuccess);

    std::vector<int> out(64);
    REQUIRE(cudaMemcpy(out.data(), d, out.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(out == std::vector<int>(64, 7));
}

TEST_CASE_METHOD(Fixture, "A failed async launch surfaces on the next synchronizing call", "[cuda][launch]") {
    RegisterProgram();
    void* d = nullptr;
    REQUIRE(cudaMalloc(&d, 16) == cudaSuccess);
    int value = 1, n = 4;
    void* args[] = {&d, &value, &n};
    // Too much dynamic shared memory fails on the device side; the launch
    // itself is fire-and-forget.
    REQUIRE(cudaLaunchKernel(&g_fill_stub, dim3(1), dim3(4), args, 1 << 20, nullptr) == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaErrorInvalidValue);
    CHECK(cudaGetLastError() == cudaErrorInvalidValue);  // recorded as the last error...
    CHECK(cudaGetLastError() == cudaSuccess);             // ...and cleared by reading it
}

// PyTorch asks for the current device hundreds of times per training step and
// checks the last error after every launch; neither may cost a round trip.
TEST_CASE_METHOD(Fixture, "Device and error queries per op are answered without a round trip", "[cuda][launch]") {
    RegisterProgram();
    void* d = nullptr;
    REQUIRE(cudaMalloc(&d, 16) == cudaSuccess);
    int device = -1;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);  // the first answer comes from the server
    CHECK(device == 0);

    int value = 1, n = 4;
    void* args[] = {&d, &value, &n};
    // The first launch uploads the module and fetches the kernel's layout.
    REQUIRE(cudaLaunchKernel(&g_fill_stub, dim3(1), dim3(4), args, 0, nullptr) == cudaSuccess);
    const uint64_t before = g_round_trips;
    for (int i = 0; i < 100; ++i) {
        REQUIRE(cudaGetDevice(&device) == cudaSuccess);
        REQUIRE(cudaLaunchKernel(&g_fill_stub, dim3(1), dim3(4), args, 0, nullptr) == cudaSuccess);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusActive;
        REQUIRE(cudaStreamIsCapturing(nullptr, &capture) == cudaSuccess);
        REQUIRE(capture == cudaStreamCaptureStatusNone);
    }
    CHECK(g_round_trips == before);

    // cudaSetDevice keeps the cached device current.
    REQUIRE(cudaSetDevice(1) == cudaSuccess);
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    CHECK(device == 1);

    // A bad configuration is caught by the launch itself, as in CUDA...
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    const uint64_t at_config = g_round_trips;
    CHECK(cudaLaunchKernel(&g_fill_stub, dim3(1), dim3(2048), args, 0, nullptr) == cudaErrorInvalidConfiguration);
    CHECK(cudaLaunchKernel(&g_fill_stub, dim3(0), dim3(4), args, 0, nullptr) == cudaErrorInvalidConfiguration);
    CHECK(cudaGetLastError() == cudaErrorInvalidConfiguration);
    CHECK(g_round_trips == at_config);
    CHECK(MockCudaGetStats().launches == 101);  // neither reached the server

    // ...and a device-side failure whose answer has arrived is reported, still locally.
    REQUIRE(cudaLaunchKernel(&g_fill_stub, dim3(1), dim3(4), args, 1 << 20, nullptr) == cudaSuccess);
    const uint64_t at_launch = g_round_trips;
    CHECK(cudaPeekAtLastError() == cudaErrorInvalidValue);
    CHECK(cudaGetLastError() == cudaErrorInvalidValue);
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(g_round_trips == at_launch);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);  // reported once, not again
}

TEST_CASE_METHOD(Fixture, "Unknown kernels are rejected", "[cuda][launch]") {
    RegisterProgram();
    void* args[] = {nullptr};
    CHECK(cudaLaunchKernel(&g_unregistered_stub, dim3(1), dim3(1), args, 0, nullptr) ==
          cudaErrorInvalidDeviceFunction);
    CHECK(cudaGetLastError() == cudaErrorInvalidDeviceFunction);
    CHECK(cudaLaunchKernel(&g_missing_stub, dim3(1), dim3(1), args, 0, nullptr) == cudaErrorInvalidDeviceFunction);
}

TEST_CASE_METHOD(Fixture, "__constant__ symbols are written, read and addressed", "[cuda][launch]") {
    RegisterProgram();
    size_t size = 0;
    REQUIRE(cudaGetSymbolSize(&size, &g_scale_symbol) == cudaSuccess);
    CHECK(size == sizeof(float));

    const float factor = 3.0f;
    REQUIRE(cudaMemcpyToSymbol(&g_scale_symbol, &factor, sizeof(factor), 0, cudaMemcpyHostToDevice) == cudaSuccess);
    float read_back = 0;
    REQUIRE(cudaMemcpyFromSymbol(&read_back, &g_scale_symbol, sizeof(float), 0, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    CHECK(read_back == 3.0f);
    CHECK(cudaMemcpyToSymbol(&g_scale_symbol, &factor, sizeof(factor), 4, cudaMemcpyHostToDevice) ==
          cudaErrorInvalidValue);  // past the end of the symbol

    float* d = DeviceFloats({1, 2, 3});
    int n = 3;
    void* args[] = {&d, &n};
    REQUIRE(cudaLaunchKernel(&g_scale_stub, dim3(1), dim3(32), args, 0, nullptr) == cudaSuccess);
    CHECK(HostFloats(d, 3) == std::vector<float>{3, 6, 9});

    void* address = nullptr;
    REQUIRE(cudaGetSymbolAddress(&address, &g_scale_symbol) == cudaSuccess);
    float via_address = 0;
    REQUIRE(cudaMemcpy(&via_address, address, sizeof(float), cudaMemcpyDefault) == cudaSuccess);
    CHECK(via_address == 3.0f);
}

TEST_CASE_METHOD(Fixture, "Function attributes and occupancy go to the right kernel", "[cuda][launch]") {
    RegisterProgram();
    cudaFuncAttributes attr{};
    REQUIRE(cudaFuncGetAttributes(&attr, &g_vec_add_stub) == cudaSuccess);
    CHECK(attr.maxThreadsPerBlock == 1024);
    CHECK(attr.numRegs == 16);
    int blocks = 0;
    REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, &g_vec_add_stub, 256, 0) == cudaSuccess);
    CHECK(blocks == 8);
    CHECK(cudaFuncSetAttribute(&g_vec_add_stub, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096) == cudaSuccess);
}

TEST_CASE_METHOD(Fixture, "A device reset drops loaded modules and they reload on use", "[cuda][launch]") {
    RegisterProgram();
    void* d = nullptr;
    REQUIRE(cudaMalloc(&d, 16) == cudaSuccess);
    int value = 2, n = 4;
    void* args[] = {&d, &value, &n};
    REQUIRE(cudaLaunchKernel(&g_fill_stub, dim3(1), dim3(4), args, 0, nullptr) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(MockCudaGetStats().module_loads == 1);

    REQUIRE(cudaDeviceReset() == cudaSuccess);
    REQUIRE(cudaMalloc(&d, 16) == cudaSuccess);
    REQUIRE(cudaLaunchKernel(&g_fill_stub, dim3(1), dim3(4), args, 0, nullptr) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(MockCudaGetStats().module_loads == 2);
    std::vector<int> out(4);
    REQUIRE(cudaMemcpy(out.data(), d, sizeof(int) * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(out == std::vector<int>(4, 2));
}

TEST_CASE_METHOD(Fixture, "Host allocations are local and page aligned", "[cuda]") {
    void* p = nullptr;
    REQUIRE(cudaMallocHost(&p, 100) == cudaSuccess);
    CHECK(reinterpret_cast<uintptr_t>(p) % 4096 == 0);
    unsigned flags = 99;
    CHECK(cudaHostGetFlags(&flags, p) == cudaSuccess);
    CHECK(cudaFreeHost(p) == cudaSuccess);
    CHECK(cudaFreeHost(p) == cudaErrorInvalidValue);  // already freed
}

TEST_CASE_METHOD(Fixture, "Error names and strings come from the CUDA headers", "[cuda]") {
    CHECK(std::string(cudaGetErrorName(cudaErrorInvalidValue)) == "cudaErrorInvalidValue");
    CHECK_THAT(std::string(cudaGetErrorString(cudaErrorMemoryAllocation)),
               Catch::Matchers::ContainsSubstring("allocate enough memory"));
    CHECK(std::string(cudaGetErrorString(static_cast<cudaError_t>(123456))) == "unrecognized error code");
}

TEST_CASE_METHOD(Fixture, "Unimplemented runtime exports report not supported", "[cuda]") {
    cudaGraph_t graph = nullptr;
    CHECK(cudaGraphCreate(&graph, 0) == cudaErrorNotSupported);
    CHECK(cudaGetLastError() == cudaErrorNotSupported);
}

TEST_CASE_METHOD(Fixture, "An unknown api is reported as not supported", "[cuda]") {
    RpcRequest request;
    request.set_api_id(999999);
    std::string bulk_out;
    RpcResponse resp = g_session->Dispatch(request, "", &bulk_out);
    CHECK(resp.result() == 801);  // cudaErrorNotSupported
    CHECK(CudaApiName(kApi_cudaMalloc) == "cudaMalloc");
}

TEST_CASE_METHOD(Fixture, "Malformed arg buffers are rejected, not read out of bounds", "[cuda]") {
    std::string bulk_out;
    RpcRequest request;
    request.set_api_id(kApi_cudaSetDevice);
    request.set_args("");  // cudaSetDevice needs 4 bytes
    CHECK(g_session->Dispatch(request, "", &bulk_out).result() == 1);  // cudaErrorInvalidValue

    request.set_args(std::string(12, '\0'));  // trailing garbage
    CHECK(g_session->Dispatch(request, "", &bulk_out).result() == 1);

    request.set_api_id(kApi___tunaGetFunction);
    ArgWriter lying;  // claims a 1000-byte name, provides none
    lying.put_u64(1);
    lying.put_u32(1000);
    request.set_args(lying.str());
    CHECK(g_session->Dispatch(request, "", &bulk_out).result() == 1);

    request.set_api_id(kApi___tunaRegisterModule);
    request.set_args("");
    CHECK(g_session->Dispatch(request, "not a fatbin at all", &bulk_out).result() == 200);  // invalid kernel image

    request.set_api_id(kApi___tunaLaunchKernel);
    ArgWriter launch;  // function id 77 was never resolved
    launch.put_u64(77);
    for (int i = 0; i < 6; ++i) launch.put_u32(1);
    launch.put_u64(0);
    launch.put_u64(0);
    request.set_args(launch.str());
    CHECK(g_session->Dispatch(request, "", &bulk_out).result() == 98);  // invalid device function
}

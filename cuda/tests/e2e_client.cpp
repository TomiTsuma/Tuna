// An unmodified-style CUDA program for the GPU-free end-to-end test. It links
// against libcudart (Tuna's shim, built as tuna_cudart) and makes the same
// calls nvcc-generated code makes: it registers a fatbin and kernels from a
// static constructor, then allocates, copies and launches. The shim forwards
// everything to tuna_server, which runs it on the mock CUDA stack
// (mock_cuda.cpp). Exit code 0 only if every check passes.
//
// Usage: tuna_cuda_e2e_client [--expect-no-server]
//   --expect-no-server  check the no-server behavior instead (no TUNA_SERVER)

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
void** __cudaRegisterFatBinary(void* fatCubin);
void __cudaRegisterFatBinaryEnd(void** handle);
void __cudaRegisterFunction(void** handle, const char* hostFun, char* deviceFun, const char* deviceName,
                            int thread_limit, uint3* tid, uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
void __cudaRegisterVar(void** handle, char* hostVar, char* deviceAddress, const char* deviceName, int ext,
                       size_t size, int constant, int global);
}

namespace {

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
const FakeFatbin kFatbin;
const FakeWrapper kWrapper{0x466243b1, 1, &kFatbin, nullptr};

char vec_add_stub, scale_stub;
float g_scale;

// What nvcc emits as a static constructor in every translation unit with
// device code: register before main() runs.
struct Registration {
    Registration() {
        void** handle = __cudaRegisterFatBinary(const_cast<FakeWrapper*>(&kWrapper));
        __cudaRegisterFunction(handle, &vec_add_stub, nullptr, "vecAdd", -1, nullptr, nullptr, nullptr, nullptr,
                               nullptr);
        __cudaRegisterFunction(handle, &scale_stub, nullptr, "scale", -1, nullptr, nullptr, nullptr, nullptr, nullptr);
        __cudaRegisterVar(handle, reinterpret_cast<char*>(&g_scale), nullptr, "gScale", 0, sizeof(float), 1, 0);
        __cudaRegisterFatBinaryEnd(handle);
    }
} registration;

int failures = 0;

void Check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

bool Ok(cudaError_t e, const char* call) {
    if (e == cudaSuccess) return true;
    std::printf("FAIL: %s -> %s (%d)\n", call, cudaGetErrorName(e), static_cast<int>(e));
    ++failures;
    return false;
}

int NoServer() {
    int count = -1;
    const cudaError_t e = cudaGetDeviceCount(&count);
    Check(e == cudaErrorNoDevice, std::string("without a server, cudaGetDeviceCount reports ") + cudaGetErrorName(e));
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--expect-no-server") == 0) return NoServer();

    int count = 0;
    if (!Ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) return 1;
    cudaDeviceProp prop{};
    if (!Ok(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties")) return 1;
    Check(count == 2 && std::string(prop.name) == "Tuna Mock GPU 0",
          "remote device: " + std::to_string(count) + " x " + prop.name);

    // vecAdd over the network.
    const int n = 1 << 16;
    std::vector<float> a(n), b(n), c(n, 0);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 0.5f * static_cast<float>(i);
    }
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    Ok(cudaMalloc(reinterpret_cast<void**>(&da), n * sizeof(float)), "cudaMalloc a");
    Ok(cudaMalloc(reinterpret_cast<void**>(&db), n * sizeof(float)), "cudaMalloc b");
    Ok(cudaMalloc(reinterpret_cast<void**>(&dc), n * sizeof(float)), "cudaMalloc c");
    Ok(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D a");
    Ok(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D b");
    int count_arg = n;
    void* args[] = {&da, &db, &dc, &count_arg};
    Ok(cudaLaunchKernel(&vec_add_stub, dim3((n + 255) / 256), dim3(256), args, 0, nullptr), "cudaLaunchKernel");
    Ok(cudaGetLastError(), "cudaGetLastError after launch");
    Ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    Ok(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H c");
    bool correct = true;
    for (int i = 0; i < n && correct; ++i) correct = c[i] == 1.5f * static_cast<float>(i);
    Check(correct, "vecAdd of " + std::to_string(n) + " floats is correct");

    // __constant__ memory + a second kernel, on a stream, timed with events.
    const float factor = 2.0f;
    Ok(cudaMemcpyToSymbol(&g_scale, &factor, sizeof(factor), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    Ok(cudaStreamCreate(&stream), "cudaStreamCreate");
    Ok(cudaEventCreate(&start), "cudaEventCreate");
    Ok(cudaEventCreate(&stop), "cudaEventCreate");
    Ok(cudaEventRecord(start, stream), "cudaEventRecord");
    void* scale_args[] = {&dc, &count_arg};
    Ok(cudaLaunchKernel(&scale_stub, dim3((n + 255) / 256), dim3(256), scale_args, 0, stream), "launch scale");
    Ok(cudaEventRecord(stop, stream), "cudaEventRecord");
    Ok(cudaMemcpyAsync(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
    Ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    float ms = -1;
    Ok(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    correct = true;
    for (int i = 0; i < n && correct; ++i) correct = c[i] == 3.0f * static_cast<float>(i);
    Check(correct && ms >= 0, "scale by __constant__ gScale on a stream is correct");

    // A payload larger than one 16 MiB frame rides the bulk channel in chunks.
    const size_t big = 40u << 20;
    std::vector<unsigned char> out(big), back(big, 0);
    for (size_t i = 0; i < big; ++i) out[i] = static_cast<unsigned char>((i * 2654435761u) >> 24);
    void* dbig = nullptr;
    Ok(cudaMalloc(&dbig, big), "cudaMalloc big");
    Ok(cudaMemcpy(dbig, out.data(), big, cudaMemcpyHostToDevice), "cudaMemcpy H2D 40 MiB");
    Ok(cudaMemcpy(back.data(), dbig, big, cudaMemcpyDeviceToHost), "cudaMemcpy D2H 40 MiB");
    Check(out == back, "40 MiB round trip over the bulk channel");

    // A failed asynchronous launch is reported at the next synchronization.
    Ok(cudaLaunchKernel(&vec_add_stub, dim3(1), dim3(4096), args, 0, nullptr), "invalid launch (queued)");
    const cudaError_t sync = cudaDeviceSynchronize();
    Check(sync == cudaErrorInvalidValue, std::string("invalid launch reported at sync: ") + cudaGetErrorName(sync));
    cudaGetLastError();

    cudaGraph_t graph = nullptr;
    Check(cudaGraphCreate(&graph, 0) == cudaErrorNotSupported, "an unimplemented API reports not supported");

    size_t free_mem = 0, total = 0;
    Ok(cudaFree(da), "cudaFree");
    Ok(cudaFree(db), "cudaFree");
    Ok(cudaFree(dc), "cudaFree");
    Ok(cudaFree(dbig), "cudaFree");
    Ok(cudaMemGetInfo(&free_mem, &total), "cudaMemGetInfo");
    Check(free_mem + sizeof(float) == total, "all device memory released (except the module's gScale)");

    std::printf("%s\n", failures == 0 ? "PASS" : "FAILED");
    return failures == 0 ? 0 : 1;
}

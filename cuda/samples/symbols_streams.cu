// Runtime features beyond plain launches: __constant__ and __device__
// variables, struct-by-value and double kernel parameters, several streams
// with cross-stream events, memset, occupancy and function attributes, dynamic
// shared memory, and an error that must surface at synchronization.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

__constant__ float kCoefficients[4];
__device__ int gCounter;

struct Params {
    float bias;
    int n;
    double scale;
};

__global__ void polynomial(float* out, const float* in, Params p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    float x = in[i];
    float y = kCoefficients[0] + x * (kCoefficients[1] + x * (kCoefficients[2] + x * kCoefficients[3]));
    out[i] = static_cast<float>(y * p.scale) + p.bias;
    atomicAdd(&gCounter, 1);
}

__global__ void reverseShared(int* data, int n) {
    extern __shared__ int tile[];
    int t = threadIdx.x;
    if (t < n) tile[t] = data[t];
    __syncthreads();
    if (t < n) data[t] = tile[n - 1 - t];
}

__global__ void addOne(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}

#define CHECK(call)                                                                      \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err_));     \
            return EXIT_FAILURE;                                                         \
        }                                                                                \
    } while (0)

int main() {
    // __constant__ / __device__ variables.
    const float coeffs[4] = {1.0f, 2.0f, 0.5f, 0.25f};
    CHECK(cudaMemcpyToSymbol(kCoefficients, coeffs, sizeof(coeffs)));
    const int zero = 0;
    CHECK(cudaMemcpyToSymbol(gCounter, &zero, sizeof(zero)));

    const int n = 10000;
    std::vector<float> in(n), out(n);
    for (int i = 0; i < n; ++i) in[i] = static_cast<float>(i % 13) * 0.1f;
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    Params p{0.5f, n, 2.0};
    polynomial<<<(n + 127) / 128, 128>>>(d_out, d_in, p);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        float x = in[i];
        float ref = (coeffs[0] + x * (coeffs[1] + x * (coeffs[2] + x * coeffs[3]))) * 2.0f + 0.5f;
        if (std::fabs(out[i] - ref) > 1e-4f) {
            std::fprintf(stderr, "polynomial mismatch at %d: %f vs %f\n", i, out[i], ref);
            return EXIT_FAILURE;
        }
    }
    int counter = 0;
    CHECK(cudaMemcpyFromSymbol(&counter, gCounter, sizeof(counter)));
    if (counter != n) {
        std::fprintf(stderr, "gCounter = %d, expected %d\n", counter, n);
        return EXIT_FAILURE;
    }
    std::printf("constant/device symbols and struct-by-value params: ok (gCounter=%d)\n", counter);

    // Dynamic shared memory.
    std::vector<int> seq(256);
    for (int i = 0; i < 256; ++i) seq[i] = i;
    int* d_seq;
    CHECK(cudaMalloc(&d_seq, 256 * sizeof(int)));
    CHECK(cudaMemcpy(d_seq, seq.data(), 256 * sizeof(int), cudaMemcpyHostToDevice));
    reverseShared<<<1, 256, 256 * sizeof(int)>>>(d_seq, 256);
    CHECK(cudaMemcpy(seq.data(), d_seq, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    if (seq[0] != 255 || seq[255] != 0) {
        std::fprintf(stderr, "dynamic shared memory reverse failed\n");
        return EXIT_FAILURE;
    }
    std::printf("dynamic shared memory: ok\n");

    // Two streams ordered by an event: memset on s1, then increments on s2.
    const int m = 1 << 20;
    int* d_m;
    CHECK(cudaMalloc(&d_m, m * sizeof(int)));
    cudaStream_t s1, s2;
    CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    cudaEvent_t ready;
    CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CHECK(cudaMemsetAsync(d_m, 0, m * sizeof(int), s1));
    CHECK(cudaEventRecord(ready, s1));
    CHECK(cudaStreamWaitEvent(s2, ready, 0));
    for (int k = 0; k < 10; ++k) addOne<<<(m + 255) / 256, 256, 0, s2>>>(d_m, m);
    std::vector<int> back(m);
    CHECK(cudaMemcpyAsync(back.data(), d_m, m * sizeof(int), cudaMemcpyDeviceToHost, s2));
    CHECK(cudaStreamSynchronize(s2));
    for (int i = 0; i < m; i += 4099) {
        if (back[i] != 10) {
            std::fprintf(stderr, "stream ordering failed at %d: %d\n", i, back[i]);
            return EXIT_FAILURE;
        }
    }
    std::printf("two streams + event ordering: ok\n");

    // Occupancy and attributes.
    int blocks = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, addOne, 256, 0));
    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr, polynomial));
    std::printf("occupancy: %d blocks/SM of 256; polynomial uses %d registers, max %d threads/block\n", blocks,
                attr.numRegs, attr.maxThreadsPerBlock);
    if (blocks <= 0 || attr.maxThreadsPerBlock <= 0) return EXIT_FAILURE;

    // An invalid configuration is reported (here, at the next sync).
    addOne<<<1, 4096>>>(d_m, m);
    cudaError_t e = cudaGetLastError();
    if (e != cudaErrorInvalidValue && e != cudaErrorInvalidConfiguration) {
        std::fprintf(stderr, "invalid launch not reported: %s\n", cudaGetErrorString(e));
        return EXIT_FAILURE;
    }
    std::printf("invalid launch reported: %s\n", cudaGetErrorName(e));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_seq));
    CHECK(cudaFree(d_m));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    CHECK(cudaEventDestroy(ready));
    std::printf("PASS\n");
    return 0;
}

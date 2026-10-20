// vectorAdd, as in NVIDIA's cuda-samples: milestone 2 of the execution plan.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] + B[i] + 0.0f;
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
    const int numElements = 50000;
    const size_t size = numElements * sizeof(float);
    float* h_A = static_cast<float*>(std::malloc(size));
    float* h_B = static_cast<float*>(std::malloc(size));
    float* h_C = static_cast<float*>(std::malloc(size));
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = std::rand() / static_cast<float>(RAND_MAX);
        h_B[i] = std::rand() / static_cast<float>(RAND_MAX);
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK(cudaMalloc(&d_A, size));
    CHECK(cudaMalloc(&d_B, size));
    CHECK(cudaMalloc(&d_C, size));
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::fprintf(stderr, "Result verification failed at element %d!\n", i);
            return EXIT_FAILURE;
        }
    }
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    std::printf("Test PASSED\nPASS\n");
    return 0;
}

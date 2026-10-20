// matrixMul in the style of NVIDIA's cuda-samples (milestone 3, kernel half):
// a templated shared-memory tiled kernel, pinned host buffers, a non-blocking
// stream, async copies, events for timing, many launches, CPU verification.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float* C, const float* A, const float* B, int wA, int wB) {
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = wA * BLOCK_SIZE * by, aEnd = aBegin + wA - 1, aStep = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx, bStep = BLOCK_SIZE * wB;
    float Csub = 0;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) Csub += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
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
    const int block = 32;
    const int hA = 10 * block, wA = 10 * block, hB = wA, wB = 20 * block;
    const size_t sizeA = sizeof(float) * hA * wA, sizeB = sizeof(float) * hB * wB, sizeC = sizeof(float) * hA * wB;

    float *h_A, *h_B, *h_C;
    CHECK(cudaMallocHost(&h_A, sizeA));
    CHECK(cudaMallocHost(&h_B, sizeB));
    CHECK(cudaMallocHost(&h_C, sizeC));
    for (int i = 0; i < hA * wA; ++i) h_A[i] = static_cast<float>(i % 7) * 0.25f;
    for (int i = 0; i < hB * wB; ++i) h_B[i] = static_cast<float>(i % 5) * 0.5f;

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, sizeA));
    CHECK(cudaMalloc(&d_B, sizeB));
    CHECK(cudaMalloc(&d_C, sizeC));
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaMemcpyAsync(d_A, h_A, sizeA, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_B, h_B, sizeB, cudaMemcpyHostToDevice, stream));

    dim3 threads(block, block), grid(wB / threads.x, hA / threads.y);
    MatrixMulCUDA<block><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, wA, wB);  // warm-up
    CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    const int iterations = 300;
    CHECK(cudaEventRecord(start, stream));
    for (int j = 0; j < iterations; ++j) MatrixMulCUDA<block><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, wA, wB);
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));
    float msecTotal = 0;
    CHECK(cudaEventElapsedTime(&msecTotal, start, stop));
    const double flops = 2.0 * hA * wA * wB;
    const double gflops = flops * 1e-9 / (msecTotal / iterations / 1000.0);
    std::printf("Performance= %.2f GFlop/s, Time= %.3f msec per kernel, %d launches\n", gflops,
                msecTotal / iterations, iterations);

    CHECK(cudaMemcpyAsync(h_C, d_C, sizeC, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    // Verify against the CPU on a sample of entries.
    for (int r = 0; r < hA; r += 7) {
        for (int c = 0; c < wB; c += 11) {
            double ref = 0;
            for (int k = 0; k < wA; ++k) ref += static_cast<double>(h_A[r * wA + k]) * h_B[k * wB + c];
            if (std::fabs(h_C[r * wB + c] - ref) > 1e-3 * std::fabs(ref) + 1e-3) {
                std::fprintf(stderr, "mismatch at (%d,%d): %f vs %f\n", r, c, h_C[r * wB + c], ref);
                return EXIT_FAILURE;
            }
        }
    }
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(h_C));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(stream));
    std::printf("Result = PASS\nPASS\n");
    return 0;
}

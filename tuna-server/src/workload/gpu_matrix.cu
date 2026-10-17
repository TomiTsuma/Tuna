#include "gpu_matrix.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kTileSize = 16;
constexpr int kRequiredComputeCapabilityMajor = 8;
constexpr int kRequiredComputeCapabilityMinor = 6;

__global__ void matrix_multiply_kernel(const float* left,
                                       const float* right,
                                       float* result,
                                       unsigned int size) {
    __shared__ float left_tile[kTileSize][kTileSize];
    __shared__ float right_tile[kTileSize][kTileSize];

    const unsigned int row = blockIdx.y * kTileSize + threadIdx.y;
    const unsigned int column = blockIdx.x * kTileSize + threadIdx.x;
    float sum = 0.0F;

    for (unsigned int tile = 0; tile < size; tile += kTileSize) {
        left_tile[threadIdx.y][threadIdx.x] =
            row < size && tile + threadIdx.x < size
                ? left[row * size + tile + threadIdx.x]
                : 0.0F;
        right_tile[threadIdx.y][threadIdx.x] =
            tile + threadIdx.y < size && column < size
                ? right[(tile + threadIdx.y) * size + column]
                : 0.0F;
        __syncthreads();
        for (unsigned int index = 0; index < kTileSize; ++index) {
            sum += left_tile[threadIdx.y][index] * right_tile[index][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < size && column < size) {
        result[row * size + column] = sum;
    }
}

bool set_cuda_error(cudaError_t status, const char* operation, std::string* error) {
    if (status == cudaSuccess) {
        return true;
    }
    if (error != nullptr) {
        *error = std::string(operation) + ": " + cudaGetErrorString(status);
    }
    return false;
}

}

bool verify_gpu_available(std::string* error) {
    int device_count = 0;
    if (!set_cuda_error(cudaGetDeviceCount(&device_count), "query CUDA devices", error)) {
        return false;
    }
    if (device_count < 1) {
        if (error != nullptr) {
            *error = "no CUDA-capable GPU was detected";
        }
        return false;
    }

    cudaDeviceProp properties{};
    if (!set_cuda_error(cudaGetDeviceProperties(&properties, 0), "query CUDA device", error)) {
        return false;
    }
    if (properties.major < kRequiredComputeCapabilityMajor ||
        (properties.major == kRequiredComputeCapabilityMajor &&
         properties.minor < kRequiredComputeCapabilityMinor)) {
        if (error != nullptr) {
            *error = "GPU compute capability 8.6 or newer is required";
        }
        return false;
    }
    if (!set_cuda_error(cudaSetDevice(0), "select CUDA device 0", error)) {
        return false;
    }
    return true;
}

bool multiply_matrices_on_gpu(std::uint32_t matrix_size,
                              const std::vector<float>& left,
                              const std::vector<float>& right,
                              std::vector<float>* result,
                              std::string* error) {
    if (result == nullptr) {
        if (error != nullptr) {
            *error = "result pointer is null";
        }
        return false;
    }
    if (!set_cuda_error(cudaSetDevice(0), "select CUDA device 0", error)) {
        return false;
    }

    const std::size_t elements =
        static_cast<std::size_t>(matrix_size) * static_cast<std::size_t>(matrix_size);
    if (matrix_size == 0 || left.size() != elements || right.size() != elements ||
        elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        if (error != nullptr) {
            *error = "matrix buffers do not match the requested dimensions";
        }
        return false;
    }

    const std::size_t bytes = elements * sizeof(float);
    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_result = nullptr;
    const auto release_buffers = [&]() {
        if (device_result != nullptr) cudaFree(device_result);
        if (device_right != nullptr) cudaFree(device_right);
        if (device_left != nullptr) cudaFree(device_left);
    };

    if (!set_cuda_error(cudaMalloc(&device_left, bytes), "allocate left matrix", error) ||
        !set_cuda_error(cudaMalloc(&device_right, bytes), "allocate right matrix", error) ||
        !set_cuda_error(cudaMalloc(&device_result, bytes), "allocate result matrix", error) ||
        !set_cuda_error(cudaMemcpy(device_left, left.data(), bytes, cudaMemcpyHostToDevice),
                        "copy left matrix", error) ||
        !set_cuda_error(cudaMemcpy(device_right, right.data(), bytes, cudaMemcpyHostToDevice),
                        "copy right matrix", error)) {
        release_buffers();
        return false;
    }

    const dim3 threads(kTileSize, kTileSize);
    const dim3 blocks((matrix_size + kTileSize - 1) / kTileSize,
                      (matrix_size + kTileSize - 1) / kTileSize);
    matrix_multiply_kernel<<<blocks, threads>>>(device_left, device_right, device_result,
                                                 matrix_size);
    if (!set_cuda_error(cudaGetLastError(), "launch matrix multiplication", error) ||
        !set_cuda_error(cudaDeviceSynchronize(), "execute matrix multiplication", error)) {
        release_buffers();
        return false;
    }

    result->resize(elements);
    const bool copied = set_cuda_error(
        cudaMemcpy(result->data(), device_result, bytes, cudaMemcpyDeviceToHost),
        "copy result matrix", error);
    release_buffers();
    if (!copied) {
        result->clear();
        return false;
    }
    for (const float value : *result) {
        if (!std::isfinite(value)) {
            result->clear();
            if (error != nullptr) {
                *error = "matrix result contains a non-finite value";
            }
            return false;
        }
    }
    return true;
}

#pragma once

// Shared between the server dispatch translation units; not part of the
// public CudaSession interface.

#include <cstdint>

namespace tuna::cudart {

constexpr int32_t kCudaErrorNotSupported = 801;

// A cuBLAS / cuBLASLt status from a posted call, as the cudaError_t the app's
// next synchronizing CUDA call reports (dispatch_cublas.cpp).
int32_t CublasAsyncError(int status);

// The same for a cudnnStatus_t (dispatch_cudnn.cpp).
int32_t CudnnAsyncError(int status);

}  // namespace tuna::cudart

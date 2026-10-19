#pragma once

// A mock CUDA runtime + driver modelling a small device in host memory. See
// mock_cuda.cpp for what it implements.

#include <cstdint>

#if defined(_WIN32)
#define TUNA_MOCK_EXPORT __declspec(dllexport)
#else
#define TUNA_MOCK_EXPORT __attribute__((visibility("default")))
#endif

// Resolves a CUDA symbol ("cudaMalloc", "cuLaunchKernel", ...) to the mock's
// implementation, or nullptr. backend_real.cpp uses this export when the
// "runtime library" it opened is the mock.
extern "C" TUNA_MOCK_EXPORT void* tuna_cuda_resolve(const char* symbol);

struct MockCudaStats {
    uint64_t module_loads = 0;
    uint64_t launches = 0;
    uint64_t resets = 0;
    uint64_t property_queries = 0;
    uint64_t attribute_queries = 0;
    uint64_t blas_calls = 0;  // cuBLAS compute calls executed (gemm, axpy, dot, ...)
    uint64_t driver_queries = 0;  // driver device / primary-context queries answered
};

// In-process use (tuna_cuda_tests): a fresh device, and counters.
void MockCudaReset();
MockCudaStats MockCudaGetStats();

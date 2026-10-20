#pragma once

#include <memory>
#include <string>

#include "tuna/cuda/backend.h"

namespace tuna::cudart {

// Where the real CUDA libraries are. Empty paths use the platform defaults:
// libcudart.so.12, libcuda.so.1, libcublas.so.12, libcudnn.so.9 (or
// cudart64_12.dll, nvcuda.dll, cublas64_12.dll, cudnn64_9.dll).
struct CudaLibraries {
    std::string runtime;
    std::string driver;
    std::string cublas;
    std::string cudnn;
};

// Opens the real CUDA stack (dlopen/LoadLibrary) and returns a backend that
// resolves runtime symbols from libcudart, driver symbols (cu*) from libcuda,
// cuBLAS symbols (cublas*) from libcublas and cuDNN symbols (cudnn*) from
// libcudnn. libcublas and libcudnn are opened on their first call, so
// sessions that never use them never load them (and a server without them
// still serves the runtime). Returns nullptr on failure, with the reason in
// *error.
//
// Refuses a "runtime" (or cuBLAS, cuDNN) that is actually Tuna's own client
// shim (it exports tuna_cudart_shim / tuna_cublas_shim / tuna_cudnn_shim): a
// server with the shim on its library path would otherwise forward every
// call to itself.
//
// Test hook: a library that exports `void* tuna_cuda_resolve(const char*)`
// resolves every symbol through that function instead (the mock CUDA library
// used by the GPU-free end-to-end test).
std::unique_ptr<CudaBackend> OpenRealCuda(const CudaLibraries& libraries = {}, std::string* error = nullptr);

}  // namespace tuna::cudart

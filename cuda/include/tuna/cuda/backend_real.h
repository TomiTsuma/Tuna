#pragma once

#include <memory>
#include <string>

#include "tuna/cuda/backend.h"

namespace tuna::cudart {

// Opens the real CUDA stack (dlopen/LoadLibrary) and returns a backend that
// resolves runtime symbols from libcudart and driver symbols (cu*) from
// libcuda. Empty paths use the platform defaults (libcudart.so.12 +
// libcuda.so.1, or cudart64_12.dll + nvcuda.dll). Returns nullptr on failure,
// with the reason in *error.
//
// Refuses a "runtime" that is actually Tuna's own client shim (it exports
// tuna_cudart_shim): a server with the shim on its library path would
// otherwise forward every call to itself.
//
// Test hook: a library that exports `void* tuna_cuda_resolve(const char*)`
// resolves every symbol through that function instead (the mock CUDA library
// used by the GPU-free end-to-end test).
std::unique_ptr<CudaBackend> OpenRealCuda(const std::string& runtime_library = {},
                                          const std::string& driver_library = {}, std::string* error = nullptr);

}  // namespace tuna::cudart

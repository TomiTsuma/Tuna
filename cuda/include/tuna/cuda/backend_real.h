#pragma once

#include <memory>
#include <string>

#include "tuna/cuda/backend.h"

namespace tuna::cudart {

// Opens the real CUDA runtime (dlopen/LoadLibrary) and returns a backend that
// resolves symbols from it. `library` is a path or soname; nullptr uses the
// platform default (libcudart.so.12 / cudart64_12.dll). Returns nullptr on
// failure, with the reason in *error.
std::unique_ptr<CudaBackend> OpenRealCuda(const char* library = nullptr, std::string* error = nullptr);

}  // namespace tuna::cudart

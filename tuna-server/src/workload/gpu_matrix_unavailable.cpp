#include "gpu_matrix.h"

bool verify_gpu_available(std::string* error) {
    if (error != nullptr) {
        *error = "server was built without CUDA support";
    }
    return false;
}

bool multiply_matrices_on_gpu(std::uint32_t,
                              const std::vector<float>&,
                              const std::vector<float>&,
                              std::vector<float>*,
                              std::string* error) {
    if (error != nullptr) {
        *error = "matrix multiplication requires a CUDA-enabled server build";
    }
    return false;
}

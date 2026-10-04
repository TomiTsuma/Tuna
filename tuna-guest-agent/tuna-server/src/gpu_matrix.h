#pragma once

#include <cstdint>
#include <string>
#include <vector>

bool verify_gpu_available(std::string* error);
bool multiply_matrices_on_gpu(std::uint32_t matrix_size,
                              const std::vector<float>& left,
                              const std::vector<float>& right,
                              std::vector<float>* result,
                              std::string* error);

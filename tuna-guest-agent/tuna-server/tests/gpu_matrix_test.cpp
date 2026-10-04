#include "gpu_matrix.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::string error;
    if (!verify_gpu_available(&error)) {
        std::cerr << "GPU validation could not start: " << error << '\n';
        return 1;
    }

    constexpr std::uint32_t size = 130;
    const std::size_t elements = static_cast<std::size_t>(size) * size;
    const std::vector<float> left(elements, 1.0F);
    const std::vector<float> right(elements, 2.0F);
    std::vector<float> result;
    if (!multiply_matrices_on_gpu(size, left, right, &result, &error)) {
        std::cerr << "GPU multiplication failed: " << error << '\n';
        return 1;
    }
    if (result.size() != elements) {
        std::cerr << "GPU multiplication returned an unexpected element count\n";
        return 1;
    }
    for (const float value : result) {
        if (value != static_cast<float>(size) * 2.0F) {
            std::cerr << "GPU multiplication returned an incorrect value: " << value << '\n';
            return 1;
        }
    }

    std::vector<float> invalid_result;
    if (multiply_matrices_on_gpu(size, left, {}, &invalid_result, &error)) {
        std::cerr << "GPU backend accepted a matrix with an invalid element count\n";
        return 1;
    }
    return 0;
}

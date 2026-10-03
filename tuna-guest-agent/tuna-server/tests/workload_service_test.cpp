#include "workload_service.h"

#include <iostream>
#include <limits>

namespace {

bool expect_status(grpc::StatusCode expected, const grpc::Status& actual, const char* name) {
    if (actual.error_code() == expected) return true;
    std::cerr << name << ": expected status " << expected << ", got " << actual.error_code()
              << " (" << actual.error_message() << ")\n";
    return false;
}

}  // namespace

int main() {
    std::uint64_t result = 0;
    if (!calculate_sum_of_squares({3, 4}, &result).ok() || result != 25) {
        std::cerr << "valid sum-of-squares request failed\n";
        return 1;
    }

    if (!expect_status(grpc::StatusCode::FAILED_PRECONDITION,
                       validate_request_metadata(2, "request-1"), "protocol version")) return 1;
    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       validate_request_metadata(1, ""), "empty request id")) return 1;
    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       validate_request_metadata(1, std::string(129, 'x')), "long request id")) return 1;

    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       calculate_sum_of_squares({}, &result), "empty batch")) return 1;
    if (!expect_status(grpc::StatusCode::OUT_OF_RANGE,
                       calculate_sum_of_squares({std::numeric_limits<std::uint64_t>::max()}, &result),
                       "overflow")) return 1;
    if (!expect_status(grpc::StatusCode::OUT_OF_RANGE,
                       calculate_sum_of_squares({4294967295ULL, 4294967295ULL},
                                                &result),
                       "sum overflow")) return 1;
    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       calculate_sum_of_squares({1}, nullptr), "null result")) return 1;

    std::vector<std::uint64_t> too_many(4097, 1);
    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       calculate_sum_of_squares(too_many, &result), "batch limit")) return 1;

    return 0;
}

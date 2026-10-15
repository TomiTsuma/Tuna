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

    tuna::v1::RemoteMatrixMultiplyRequest matrix_request;
    matrix_request.set_protocol_version(1);
    matrix_request.set_request_id("matrix-1");
    matrix_request.set_matrix_size(128);
    for (int index = 0; index < 128 * 128; ++index) {
        matrix_request.add_left(1.0F);
        matrix_request.add_right(2.0F);
    }
    if (!validate_matrix_request(matrix_request).ok()) {
        std::cerr << "valid matrix request was rejected\n";
        return 1;
    }

    matrix_request.set_matrix_size(127);
    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       validate_matrix_request(matrix_request), "matrix lower bound")) return 1;
    matrix_request.set_matrix_size(513);
    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       validate_matrix_request(matrix_request), "matrix upper bound")) return 1;
    matrix_request.set_matrix_size(128);
    matrix_request.mutable_left()->RemoveLast();
    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       validate_matrix_request(matrix_request), "matrix element count")) return 1;
    matrix_request.add_left(std::numeric_limits<float>::quiet_NaN());
    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       validate_matrix_request(matrix_request), "non-finite matrix value")) return 1;

    std::vector<std::uint64_t> too_many(4097, 1);
    if (!expect_status(grpc::StatusCode::INVALID_ARGUMENT,
                       calculate_sum_of_squares(too_many, &result), "batch limit")) return 1;

    return 0;
}

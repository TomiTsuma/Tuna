#include "workload_service.h"

#include "gpu_matrix.h"

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMaximumValues = 4096;
constexpr std::uint32_t kMinimumMatrixSize = 128;
constexpr std::uint32_t kMaximumMatrixSize = 512;
constexpr std::uint32_t kProtocolVersion = 1;
constexpr char kSanProperty[] = "x509_subject_alternative_name";

bool peer_has_allowed_san(const std::shared_ptr<const grpc::AuthContext>& auth_context,
                          const std::string& expected_san) {
    if (!auth_context || !auth_context->IsPeerAuthenticated()) {
        return false;
    }
    const auto sans = auth_context->FindPropertyValues(kSanProperty);
    for (const auto& san : sans) {
        if (std::string(san.data(), san.size()) == expected_san) {
            return true;
        }
    }
    return false;
}

}  // namespace

WorkloadService::WorkloadService(std::string allowed_client_san)
    : allowed_client_san_(std::move(allowed_client_san)) {}

grpc::Status calculate_sum_of_squares(const std::vector<std::uint64_t>& values,
                                      std::uint64_t* result) {
    if (result == nullptr) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "result pointer is null"};
    }
    if (values.empty() || values.size() > kMaximumValues) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "values must contain 1 to 4096 items"};
    }

    std::uint64_t sum = 0;
    for (const std::uint64_t value : values) {
        if (value != 0 && value > std::numeric_limits<std::uint64_t>::max() / value) {
            return {grpc::StatusCode::OUT_OF_RANGE, "sum of squares overflows uint64"};
        }
        const std::uint64_t square = value * value;
        if (square > std::numeric_limits<std::uint64_t>::max() - sum) {
            return {grpc::StatusCode::OUT_OF_RANGE, "sum of squares overflows uint64"};
        }
        sum += square;
    }
    *result = sum;
    return grpc::Status::OK;
}

grpc::Status validate_request_metadata(std::uint32_t protocol_version,
                                       const std::string& request_id) {
    if (protocol_version != kProtocolVersion) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "unsupported protocol version"};
    }
    if (request_id.empty() || request_id.size() > 128) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "request_id must contain 1 to 128 bytes"};
    }
    return grpc::Status::OK;
}

grpc::Status validate_matrix_request(const tuna::v1::RemoteMatrixMultiplyRequest& request) {
    const std::uint32_t size = request.matrix_size();
    if (size < kMinimumMatrixSize || size > kMaximumMatrixSize) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "matrix_size must be between 128 and 512"};
    }
    const std::size_t expected =
        static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    if (static_cast<std::size_t>(request.left_size()) != expected ||
        static_cast<std::size_t>(request.right_size()) != expected) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "left and right matrices must each contain matrix_size squared values"};
    }
    for (const float value : request.left()) {
        if (!std::isfinite(value)) {
            return {grpc::StatusCode::INVALID_ARGUMENT,
                    "matrix values must be finite"};
        }
    }
    for (const float value : request.right()) {
        if (!std::isfinite(value)) {
            return {grpc::StatusCode::INVALID_ARGUMENT,
                    "matrix values must be finite"};
        }
    }
    return grpc::Status::OK;
}

grpc::Status WorkloadService::RemoteSumOfSquares(
    grpc::ServerContext* context,
    const tuna::v1::RemoteSumOfSquaresRequest* request,
    tuna::v1::RemoteSumOfSquaresResponse* response) {
    if (!peer_has_allowed_san(context->auth_context(), allowed_client_san_)) {
        return {grpc::StatusCode::PERMISSION_DENIED, "client certificate identity is not authorized"};
    }
    const grpc::Status metadata_status =
        validate_request_metadata(request->protocol_version(), request->request_id());
    if (!metadata_status.ok()) {
        return metadata_status;
    }
    if (request->values_size() <= 0 ||
        static_cast<std::size_t>(request->values_size()) > kMaximumValues) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "values must contain 1 to 4096 items"};
    }

    std::vector<std::uint64_t> values(request->values().begin(), request->values().end());
    std::uint64_t result_value = 0;
    const grpc::Status computation = calculate_sum_of_squares(values, &result_value);
    if (!computation.ok()) {
        return computation;
    }

    response->set_protocol_version(kProtocolVersion);
    response->set_request_id(request->request_id());
    response->set_result(result_value);
    return grpc::Status::OK;
}

grpc::Status WorkloadService::RemoteMatrixMultiply(
    grpc::ServerContext* context,
    const tuna::v1::RemoteMatrixMultiplyRequest* request,
    tuna::v1::RemoteMatrixMultiplyResponse* response) {
    if (!peer_has_allowed_san(context->auth_context(), allowed_client_san_)) {
        return {grpc::StatusCode::PERMISSION_DENIED, "client certificate identity is not authorized"};
    }
    const grpc::Status metadata_status =
        validate_request_metadata(request->protocol_version(), request->request_id());
    if (!metadata_status.ok()) {
        return metadata_status;
    }
    const grpc::Status validation_status = validate_matrix_request(*request);
    if (!validation_status.ok()) {
        return validation_status;
    }
#ifndef TUNA_ENABLE_CUDA
    (void)response;
    return {grpc::StatusCode::UNAVAILABLE,
            "matrix multiplication requires a CUDA-enabled server build"};
#else
    if (gpu_request_active_.test_and_set(std::memory_order_acquire)) {
        return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                "GPU is busy; retry the request later"};
    }
    struct GpuRequestGuard {
        std::atomic_flag& active;
        ~GpuRequestGuard() { active.clear(std::memory_order_release); }
    } request_guard{gpu_request_active_};

    std::vector<float> left(request->left().begin(), request->left().end());
    std::vector<float> right(request->right().begin(), request->right().end());
    std::vector<float> result;
    std::string error;
    if (!multiply_matrices_on_gpu(request->matrix_size(), left, right, &result, &error)) {
        return {grpc::StatusCode::INTERNAL, "GPU matrix multiplication failed: " + error};
    }

    response->set_protocol_version(kProtocolVersion);
    response->set_request_id(request->request_id());
    response->set_matrix_size(request->matrix_size());
    response->mutable_result()->Reserve(static_cast<int>(result.size()));
    for (const float value : result) {
        response->add_result(value);
    }
    return grpc::Status::OK;
#endif
}

#include "tuna/client/workload_client.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace tuna::client {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot read TLS credential file: " + path);
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

}  // namespace

WorkloadClient::WorkloadClient(const TlsConfig& tls_config) {
    if (tls_config.server_address.empty()) {
        throw std::invalid_argument("server address must not be empty");
    }

    grpc::SslCredentialsOptions options;
    options.pem_root_certs = read_file(tls_config.server_ca_file);
    options.pem_cert_chain = read_file(tls_config.client_certificate_file);
    options.pem_private_key = read_file(tls_config.client_private_key_file);
    if (options.pem_root_certs.empty() || options.pem_cert_chain.empty() ||
        options.pem_private_key.empty()) {
        throw std::runtime_error("TLS CA, client certificate, and private key must be non-empty");
    }

    grpc::ChannelArguments channel_arguments;
    channel_arguments.SetMaxSendMessageSize(4 * 1024 * 1024);
    channel_arguments.SetMaxReceiveMessageSize(4 * 1024 * 1024);
    channel_ = grpc::CreateCustomChannel(
        tls_config.server_address, grpc::SslCredentials(options), channel_arguments);
    stub_ = tuna::v1::WorkloadService::NewStub(channel_);
}

grpc::Status WorkloadClient::remote_sum_of_squares(const std::string& request_id,
                                                   const std::vector<std::uint64_t>& values,
                                                   std::uint64_t* result) const {
    if (result == nullptr) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "result pointer is null");
    }
    if (request_id.empty() || request_id.size() > 128) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "request_id must contain 1 to 128 bytes");
    }
    if (values.empty() || values.size() > 4096) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "values must contain 1 to 4096 items");
    }

    tuna::v1::RemoteSumOfSquaresRequest request;
    request.set_protocol_version(1);
    request.set_request_id(request_id);
    for (const auto value : values) {
        request.add_values(value);
    }

    tuna::v1::RemoteSumOfSquaresResponse response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    const grpc::Status status = stub_->RemoteSumOfSquares(&context, request, &response);
    if (!status.ok()) {
        return status;
    }
    if (response.protocol_version() != 1 || response.request_id() != request_id) {
        return grpc::Status(grpc::StatusCode::DATA_LOSS, "server response did not match the request");
    }

    *result = response.result();
    return grpc::Status::OK;
}

grpc::Status WorkloadClient::remote_matrix_multiply(
    const std::string& request_id,
    std::uint32_t matrix_size,
    const std::vector<float>& left,
    const std::vector<float>& right,
    std::vector<float>* result) const {
    if (result == nullptr) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "result pointer is null"};
    }
    if (request_id.empty() || request_id.size() > 128) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "request_id must contain 1 to 128 bytes"};
    }
    if (matrix_size < 128 || matrix_size > 512) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "matrix_size must be between 128 and 512"};
    }
    const std::size_t expected =
        static_cast<std::size_t>(matrix_size) * static_cast<std::size_t>(matrix_size);
    if (left.size() != expected || right.size() != expected) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "matrix buffers must each contain matrix_size squared values"};
    }
    if (!std::all_of(left.begin(), left.end(), [](float value) { return std::isfinite(value); }) ||
        !std::all_of(right.begin(), right.end(), [](float value) { return std::isfinite(value); })) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "matrix values must be finite"};
    }

    tuna::v1::RemoteMatrixMultiplyRequest request;
    request.set_protocol_version(1);
    request.set_request_id(request_id);
    request.set_matrix_size(matrix_size);
    request.mutable_left()->Reserve(static_cast<int>(left.size()));
    request.mutable_right()->Reserve(static_cast<int>(right.size()));
    for (const float value : left) request.add_left(value);
    for (const float value : right) request.add_right(value);

    tuna::v1::RemoteMatrixMultiplyResponse response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    const grpc::Status status = stub_->RemoteMatrixMultiply(&context, request, &response);
    if (!status.ok()) {
        return status;
    }
    if (response.protocol_version() != 1 || response.request_id() != request_id ||
        response.matrix_size() != matrix_size ||
        static_cast<std::size_t>(response.result_size()) != expected) {
        return {grpc::StatusCode::DATA_LOSS, "server matrix response did not match the request"};
    }
    result->assign(response.result().begin(), response.result().end());
    if (!std::all_of(result->begin(), result->end(),
                     [](float value) { return std::isfinite(value); })) {
        result->clear();
        return {grpc::StatusCode::DATA_LOSS, "server returned a non-finite matrix value"};
    }
    return grpc::Status::OK;
}

}  // namespace tuna::client

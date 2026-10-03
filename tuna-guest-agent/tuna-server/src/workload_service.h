#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "tuna/v1/workload.grpc.pb.h"

class WorkloadService final : public tuna::v1::WorkloadService::Service {
public:
    explicit WorkloadService(std::string allowed_client_san);

    grpc::Status RemoteSumOfSquares(
        grpc::ServerContext* context,
        const tuna::v1::RemoteSumOfSquaresRequest* request,
        tuna::v1::RemoteSumOfSquaresResponse* response) override;

private:
    std::string allowed_client_san_;
};

grpc::Status calculate_sum_of_squares(const std::vector<std::uint64_t>& values,
                                      std::uint64_t* result);
grpc::Status validate_request_metadata(std::uint32_t protocol_version,
                                       const std::string& request_id);

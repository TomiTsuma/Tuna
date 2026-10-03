#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "tuna/v1/workload.grpc.pb.h"

namespace tuna::client {

struct TlsConfig {
    std::string server_address;
    std::string server_ca_file;
    std::string client_certificate_file;
    std::string client_private_key_file;
};

class WorkloadClient {
public:
    explicit WorkloadClient(const TlsConfig& tls_config);

    grpc::Status remote_sum_of_squares(const std::string& request_id,
                                       const std::vector<std::uint64_t>& values,
                                       std::uint64_t* result) const;

private:
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<tuna::v1::WorkloadService::Stub> stub_;
};

}  // namespace tuna::client

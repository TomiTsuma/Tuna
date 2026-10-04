#include "workload_service.h"
#include "gpu_matrix.h"

#include <iostream>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

#include <grpcpp/grpcpp.h>

namespace {

struct Options {
    std::string listen_address = "127.0.0.1:50051";
    std::string certificate_file;
    std::string private_key_file;
    std::string client_ca_file;
    std::string allowed_client_san;
};

void usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " --cert SERVER_CERT.pem --key SERVER_KEY.pem --client-ca CLIENT_CA.pem"
                 " --allowed-client-san DNS:client-name [--listen HOST:PORT]\n";
}

bool parse_options(int argc, char* argv[], Options* options) {
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + name);
        }
        const std::string value = argv[++i];
        if (name == "--listen") options->listen_address = value;
        else if (name == "--cert") options->certificate_file = value;
        else if (name == "--key") options->private_key_file = value;
        else if (name == "--client-ca") options->client_ca_file = value;
        else if (name == "--allowed-client-san") options->allowed_client_san = value;
        else throw std::invalid_argument("unknown option: " + name);
    }
    if (options->certificate_file.empty() || options->private_key_file.empty() ||
        options->client_ca_file.empty() || options->allowed_client_san.empty()) {
        usage(argv[0]);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            usage(argv[0]);
            return 0;
        }
    }

    Options options;
    try {
        if (!parse_options(argc, argv, &options)) {
            return 2;
        }

#ifdef TUNA_ENABLE_CUDA
        std::string gpu_error;
        if (!verify_gpu_available(&gpu_error)) {
            throw std::runtime_error("required CUDA GPU is unavailable: " + gpu_error);
        }
#endif

        grpc::SslServerCredentialsOptions credentials_options;
        credentials_options.pem_root_certs = [&options] {
            std::ifstream file(options.client_ca_file, std::ios::binary);
            if (!file) throw std::runtime_error("cannot read client CA file");
            return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        }();
        if (credentials_options.pem_root_certs.empty()) {
            throw std::runtime_error("client CA file is empty");
        }
        credentials_options.client_certificate_request =
            GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
        credentials_options.pem_key_cert_pairs.push_back({
            [&options] {
                std::ifstream file(options.private_key_file, std::ios::binary);
                if (!file) throw std::runtime_error("cannot read server private key file");
                return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            }(),
            [&options] {
                std::ifstream file(options.certificate_file, std::ios::binary);
                if (!file) throw std::runtime_error("cannot read server certificate file");
                return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            }()
        });

        WorkloadService service(options.allowed_client_san);
        grpc::ServerBuilder builder;
        builder.SetMaxReceiveMessageSize(4 * 1024 * 1024);
        builder.SetMaxSendMessageSize(4 * 1024 * 1024);
        builder.AddListeningPort(options.listen_address, grpc::SslServerCredentials(credentials_options));
        builder.RegisterService(&service);
        std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
        if (!server) {
            throw std::runtime_error("failed to bind gRPC listener at " + options.listen_address);
        }
        std::cout << "Tuna sample server listening on " << options.listen_address << '\n';
        server->Wait();
    } catch (const std::exception& error) {
        std::cerr << "Server startup failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

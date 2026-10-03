#include "tuna/client/workload_client.h"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " --server HOST:PORT --ca SERVER_CA.pem --cert CLIENT_CERT.pem"
                 " --key CLIENT_KEY.pem VALUE [VALUE ...]\n";
}

std::uint64_t parse_value(const std::string& text) {
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument("workload values must be unsigned decimal integers");
    }
    return value;
}

}  // namespace

int main(int argc, char* argv[]) {
    tuna::client::TlsConfig tls;
    std::vector<std::uint64_t> values;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--server" || argument == "--ca" || argument == "--cert" ||
            argument == "--key") {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            const std::string value = argv[++i];
            if (argument == "--server") tls.server_address = value;
            if (argument == "--ca") tls.server_ca_file = value;
            if (argument == "--cert") tls.client_certificate_file = value;
            if (argument == "--key") tls.client_private_key_file = value;
        } else if (argument == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            try {
                values.push_back(parse_value(argument));
            } catch (const std::invalid_argument& error) {
                std::cerr << error.what() << ": " << argument << '\n';
                print_usage(argv[0]);
                return 2;
            }
        }
    }

    if (tls.server_address.empty() || tls.server_ca_file.empty() ||
        tls.client_certificate_file.empty() || tls.client_private_key_file.empty() ||
        values.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    if (values.size() > 4096) {
        std::cerr << "A request may contain at most 4096 values\n";
        return 2;
    }

    try {
        tuna::client::WorkloadClient client(tls);
        const auto request_id = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        std::uint64_t result = 0;
        const grpc::Status status = client.remote_sum_of_squares(request_id, values, &result);
        if (!status.ok()) {
            std::cerr << "Remote workload failed (" << status.error_code()
                      << "): " << status.error_message() << '\n';
            return 1;
        }
        std::cout << "request_id=" << request_id << " sum_of_squares=" << result << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Client setup failed: " << error.what() << '\n';
        return 1;
    }
}

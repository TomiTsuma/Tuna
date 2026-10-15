#include "tuna/client/workload_client.h"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " --server HOST:PORT --ca SERVER_CA.pem --cert CLIENT_CERT.pem"
                 " --key CLIENT_KEY.pem VALUE [VALUE ...]\n"
              << "       " << executable
              << " --server HOST:PORT --ca SERVER_CA.pem --cert CLIENT_CERT.pem"
                 " --key CLIENT_KEY.pem --matrix-size 128..512\n";
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
    std::uint32_t matrix_size = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--server" || argument == "--ca" || argument == "--cert" ||
            argument == "--key" || argument == "--matrix-size") {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            const std::string value = argv[++i];
            if (argument == "--server") tls.server_address = value;
            if (argument == "--ca") tls.server_ca_file = value;
            if (argument == "--cert") tls.client_certificate_file = value;
            if (argument == "--key") tls.client_private_key_file = value;
            if (argument == "--matrix-size") {
                try {
                    const std::uint64_t parsed = parse_value(value);
                    if (parsed < 128 || parsed > 512) {
                        throw std::invalid_argument("matrix size must be between 128 and 512");
                    }
                    matrix_size = static_cast<std::uint32_t>(parsed);
                } catch (const std::invalid_argument& error) {
                    std::cerr << error.what() << '\n';
                    print_usage(argv[0]);
                    return 2;
                }
            }
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

    const bool has_matrix_workload = matrix_size != 0;
    const bool has_sum_workload = !values.empty();
    if (tls.server_address.empty() || tls.server_ca_file.empty() ||
        tls.client_certificate_file.empty() || tls.client_private_key_file.empty() ||
        has_matrix_workload == has_sum_workload) {
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
        if (matrix_size != 0) {
            const std::size_t elements =
                static_cast<std::size_t>(matrix_size) * static_cast<std::size_t>(matrix_size);
            std::vector<float> left(elements);
            std::vector<float> right(elements);
            for (std::uint32_t row = 0; row < matrix_size; ++row) {
                for (std::uint32_t column = 0; column < matrix_size; ++column) {
                    const std::size_t index =
                        static_cast<std::size_t>(row) * matrix_size + column;
                    left[index] = static_cast<float>((row + column) % 5U + 1U);
                    right[index] = static_cast<float>((row * 2U + column) % 7U + 1U);
                }
            }
            std::vector<float> result;
            const grpc::Status status =
                client.remote_matrix_multiply(request_id, matrix_size, left, right, &result);
            if (!status.ok()) {
                std::cerr << "Remote GPU workload failed (" << status.error_code()
                          << "): " << status.error_message() << '\n';
                return 1;
            }
            double checksum = 0.0;
            for (const float value : result) checksum += static_cast<double>(value);
            std::cout << std::setprecision(12) << "request_id=" << request_id
                      << " matrix_size=" << matrix_size
                      << " result_checksum=" << checksum
                      << " result_0_0=" << result.front() << '\n';
            return 0;
        }
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

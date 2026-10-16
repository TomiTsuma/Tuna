#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace tuna::tunnel {

// Operator certificate tooling for Track B mTLS (EC P-256, SHA-256).
// Every function refuses to overwrite an existing file, and on POSIX creates
// private keys with mode 0600.

// Writes <dir>/ca.key and <dir>/ca.crt: a self-signed CA allowed to sign leaf
// certificates only (pathlen 0).
bool GenerateCa(const std::filesystem::path& dir, const std::string& common_name, int days,
                std::string* error = nullptr);

struct IssueOptions {
    std::string name;                // CN, and the default SAN (DNS:<name>)
    bool server = false;             // EKU serverAuth (true) or clientAuth (false)
    std::vector<std::string> sans;   // OpenSSL syntax, e.g. "DNS:gpu-box", "IP:10.0.0.5"
    int days = 397;
};

// Signs a new key pair with the CA in `ca_dir` and writes
// <out_dir>/<name>.key and <out_dir>/<name>.crt.
bool IssueCertificate(const std::filesystem::path& ca_dir, const std::filesystem::path& out_dir,
                      const IssueOptions& options, std::string* error = nullptr);

}  // namespace tuna::tunnel

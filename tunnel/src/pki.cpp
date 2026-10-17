#include "tuna/tunnel/pki.h"

#include <cerrno>
#include <cstdio>
#include <memory>

#ifdef _WIN32
// Before OpenSSL, and lean: wincrypt.h's X509_NAME macro clashes with OpenSSL.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>

namespace tuna::tunnel {

namespace {

struct Free {
    void operator()(EVP_PKEY* p) const { EVP_PKEY_free(p); }
    void operator()(X509* p) const { X509_free(p); }
    void operator()(BIO* p) const { BIO_free(p); }
    void operator()(BIGNUM* p) const { BN_free(p); }
};
template <typename T>
using Owned = std::unique_ptr<T, Free>;

bool Fail(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
        while (const unsigned long code = ERR_get_error()) {
            char buffer[256];
            ERR_error_string_n(code, buffer, sizeof(buffer));
            *error += "; ";
            *error += buffer;
        }
    }
    return false;
}

// Creates `path` exclusively (fails if it exists) and writes `bytes`.
bool WriteNewFile(const std::filesystem::path& path, const std::string& bytes, bool secret,
                  std::string* error) {
#ifdef _WIN32
    // The file inherits the directory's ACL; operators keep CA and key
    // directories private (documented in security_model.md).
    (void)secret;
    HANDLE file = CreateFileW(path.wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return Fail(error, GetLastError() == ERROR_FILE_EXISTS
                               ? "refusing to overwrite " + path.string()
                               : "cannot create " + path.string());
    }
    DWORD written = 0;
    const bool ok = WriteFile(file, bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr) &&
                    written == bytes.size();
    CloseHandle(file);
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, secret ? 0600 : 0644);
    if (fd < 0) {
        return Fail(error, errno == EEXIST ? "refusing to overwrite " + path.string()
                                           : "cannot create " + path.string());
    }
    const bool ok = ::write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size());
    ::close(fd);
#endif
    return ok ? true : Fail(error, "cannot write " + path.string());
}

std::string BioContents(BIO* bio) {
    char* data = nullptr;
    const long size = BIO_get_mem_data(bio, &data);
    return size > 0 ? std::string(data, static_cast<size_t>(size)) : std::string();
}

bool WriteKeyAndCert(const std::filesystem::path& dir, const std::string& stem, EVP_PKEY* key,
                     X509* cert, std::string* error) {
    Owned<BIO> key_pem(BIO_new(BIO_s_mem()));
    Owned<BIO> cert_pem(BIO_new(BIO_s_mem()));
    if (!key_pem || !cert_pem ||
        PEM_write_bio_PrivateKey(key_pem.get(), key, nullptr, nullptr, 0, nullptr, nullptr) != 1 ||
        PEM_write_bio_X509(cert_pem.get(), cert) != 1) {
        return Fail(error, "cannot encode PEM");
    }
    const auto key_path = dir / (stem + ".key");
    const auto cert_path = dir / (stem + ".crt");
    // Check both up front so a refused overwrite never leaves half a pair.
    for (const auto& path : {key_path, cert_path}) {
        if (std::filesystem::exists(path)) return Fail(error, "refusing to overwrite " + path.string());
    }
    return WriteNewFile(key_path, BioContents(key_pem.get()), true, error) &&
           WriteNewFile(cert_path, BioContents(cert_pem.get()), false, error);
}

bool AddExtension(X509* cert, X509* issuer, int nid, const char* value) {
    X509V3_CTX ctx;
    X509V3_set_ctx(&ctx, issuer, cert, nullptr, nullptr, 0);
    X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &ctx, nid, value);
    if (!ext) return false;
    const bool ok = X509_add_ext(cert, ext, -1) == 1;
    X509_EXTENSION_free(ext);
    return ok;
}

// A certificate with a random 159-bit serial, the given subject CN and
// validity window, and `key`'s public half.
Owned<X509> NewCertificate(EVP_PKEY* key, const std::string& common_name, int days) {
    Owned<X509> cert(X509_new());
    Owned<BIGNUM> serial(BN_new());
    if (!cert || !serial || X509_set_version(cert.get(), 2) != 1 ||
        BN_rand(serial.get(), 159, BN_RAND_TOP_ANY, BN_RAND_BOTTOM_ANY) != 1 ||
        !BN_to_ASN1_INTEGER(serial.get(), X509_get_serialNumber(cert.get())) ||
        !X509_gmtime_adj(X509_getm_notBefore(cert.get()), -300) ||  // tolerate small clock skew
        !X509_gmtime_adj(X509_getm_notAfter(cert.get()), static_cast<long>(days) * 86400L) ||
        X509_set_pubkey(cert.get(), key) != 1) {
        return nullptr;
    }
    // A separate name object: X509_get_subject_name() returns const in
    // OpenSSL 4 but not in 3.0 (Ubuntu 22.04/24.04), and this works on both.
    X509_NAME* subject = X509_NAME_new();
    const bool ok = subject &&
                    X509_NAME_add_entry_by_txt(subject, "CN", MBSTRING_UTF8,
                                               reinterpret_cast<const unsigned char*>(common_name.c_str()),
                                               -1, -1, 0) == 1 &&
                    X509_set_subject_name(cert.get(), subject) == 1;
    X509_NAME_free(subject);
    return ok ? std::move(cert) : nullptr;
}

Owned<EVP_PKEY> NewKey() { return Owned<EVP_PKEY>(EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "P-256")); }

}  // namespace

bool GenerateCa(const std::filesystem::path& dir, const std::string& common_name, int days, std::string* error) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    Owned<EVP_PKEY> key = NewKey();
    if (!key) return Fail(error, "cannot generate CA key");
    Owned<X509> cert = NewCertificate(key.get(), common_name, days);
    if (!cert || X509_set_issuer_name(cert.get(), X509_get_subject_name(cert.get())) != 1 ||
        !AddExtension(cert.get(), cert.get(), NID_basic_constraints, "critical,CA:TRUE,pathlen:0") ||
        !AddExtension(cert.get(), cert.get(), NID_key_usage, "critical,keyCertSign,cRLSign") ||
        !AddExtension(cert.get(), cert.get(), NID_subject_key_identifier, "hash") ||
        X509_sign(cert.get(), key.get(), EVP_sha256()) == 0) {
        return Fail(error, "cannot build CA certificate");
    }
    return WriteKeyAndCert(dir, "ca", key.get(), cert.get(), error);
}

bool IssueCertificate(const std::filesystem::path& ca_dir, const std::filesystem::path& out_dir,
                      const IssueOptions& options, std::string* error) {
    if (options.name.empty() || options.name.find_first_of("/\\:") != std::string::npos) {
        return Fail(error, "certificate name must be non-empty and contain no path separators");
    }

    Owned<BIO> ca_key_file(BIO_new_file((ca_dir / "ca.key").string().c_str(), "r"));
    Owned<BIO> ca_cert_file(BIO_new_file((ca_dir / "ca.crt").string().c_str(), "r"));
    if (!ca_key_file || !ca_cert_file) return Fail(error, "cannot open ca.key/ca.crt in " + ca_dir.string());
    Owned<EVP_PKEY> ca_key(PEM_read_bio_PrivateKey(ca_key_file.get(), nullptr, nullptr, nullptr));
    Owned<X509> ca_cert(PEM_read_bio_X509(ca_cert_file.get(), nullptr, nullptr, nullptr));
    if (!ca_key || !ca_cert || X509_check_private_key(ca_cert.get(), ca_key.get()) != 1) {
        return Fail(error, "ca.key and ca.crt in " + ca_dir.string() + " are unreadable or do not match");
    }

    std::string san_list;
    for (const std::string& san : options.sans.empty() ? std::vector<std::string>{"DNS:" + options.name}
                                                       : options.sans) {
        if (!san_list.empty()) san_list += ",";
        san_list += san;
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    Owned<EVP_PKEY> key = NewKey();
    if (!key) return Fail(error, "cannot generate key");
    Owned<X509> cert = NewCertificate(key.get(), options.name, options.days);
    if (!cert || X509_set_issuer_name(cert.get(), X509_get_subject_name(ca_cert.get())) != 1 ||
        !AddExtension(cert.get(), ca_cert.get(), NID_basic_constraints, "critical,CA:FALSE") ||
        !AddExtension(cert.get(), ca_cert.get(), NID_key_usage, "critical,digitalSignature") ||
        !AddExtension(cert.get(), ca_cert.get(), NID_ext_key_usage,
                      options.server ? "serverAuth" : "clientAuth") ||
        !AddExtension(cert.get(), ca_cert.get(), NID_subject_alt_name, san_list.c_str()) ||
        !AddExtension(cert.get(), ca_cert.get(), NID_subject_key_identifier, "hash") ||
        !AddExtension(cert.get(), ca_cert.get(), NID_authority_key_identifier, "keyid:always") ||
        X509_sign(cert.get(), ca_key.get(), EVP_sha256()) == 0) {
        return Fail(error, "cannot build certificate (check the --san syntax, e.g. DNS:host,IP:10.0.0.5)");
    }
    return WriteKeyAndCert(out_dir, options.name, key.get(), cert.get(), error);
}

}  // namespace tuna::tunnel

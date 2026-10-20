// Round trip of the cuBLAS remoting with no GPU and no transport: the cuBLAS
// shim encodes -> tuna_cudart_call (the runtime shim's bridge) -> CudaSession
// decodes -> the mock cuBLAS (mock_cuda.cpp) computes on "device" memory ->
// outs are encoded -> the shim decodes. Results are checked against a CPU
// reference, so argument order, transposes, leading dimensions and both
// pointer modes are all exercised end to end.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "cuda_api_ids.h"
#include "mock_cuda.h"
#include "tuna/cuda/backend.h"
#include "tuna/cuda/client.h"
#include "tuna/cuda/cublas_client.h"

using namespace tuna;
using namespace tuna::cudart;
using Catch::Matchers::WithinAbs;

namespace {

class MockBackend : public CudaBackend {
public:
    void* Resolve(const char* symbol) override { return tuna_cuda_resolve(symbol); }
};

MockBackend g_backend;
std::unique_ptr<CudaSession> g_session;

Reply InProcessCall(uint32_t api_id, const std::string& args, const std::string& bulk_in, bool) {
    RpcRequest request;
    request.set_api_id(api_id);
    request.set_args(args);
    std::string bulk_out;
    RpcResponse resp = g_session->Dispatch(request, bulk_in, &bulk_out);
    Reply reply;
    reply.result = resp.result();
    reply.out_args = resp.out_args();
    reply.bulk_out = std::move(bulk_out);
    return reply;
}

struct Fixture {
    Fixture() {
        MockCudaReset();
        g_session = std::make_unique<CudaSession>(g_backend);
        ResetClientStateForTesting();
        tuna::cublas::ResetStateForTesting();
        SetCallHook(InProcessCall);
        LastError(/*clear=*/true);
        REQUIRE(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    }
    ~Fixture() {
        cublasDestroy(handle);
        SetCallHook(nullptr);
    }
    cublasHandle_t handle = nullptr;
};

template <typename T>
T* DeviceCopy(const std::vector<T>& values) {
    void* d = nullptr;
    REQUIRE(cudaMalloc(&d, values.size() * sizeof(T)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    return static_cast<T*>(d);
}

template <typename T>
std::vector<T> HostCopy(const T* d, size_t n) {
    std::vector<T> values(n);
    REQUIRE(cudaMemcpy(values.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    return values;
}

// Reference column-major gemm.
template <typename T>
std::vector<T> RefGemm(bool ta, bool tb, int m, int n, int k, T alpha, const std::vector<T>& A, int lda,
                       const std::vector<T>& B, int ldb, T beta, std::vector<T> C, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T sum{};
            for (int p = 0; p < k; ++p) {
                const T a = ta ? A[p + i * lda] : A[i + p * lda];
                const T b = tb ? B[j + p * ldb] : B[p + j * ldb];
                sum += a * b;
            }
            C[i + j * ldc] = alpha * sum + beta * C[i + j * ldc];
        }
    }
    return C;
}

std::vector<float> Iota(size_t n, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * static_cast<float>((i * 7) % 13) - 3.0f;
    return v;
}

}  // namespace

TEST_CASE_METHOD(Fixture, "cuBLAS handles, streams and versions round-trip", "[cublas]") {
    int version = 0;
    REQUIRE(cublasGetVersion(handle, &version) == CUBLAS_STATUS_SUCCESS);
    CHECK(version == 120804);  // the mock's

    cudaStream_t stream = nullptr, got = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS);  // posted
    REQUIRE(cublasGetStream(handle, &got) == CUBLAS_STATUS_SUCCESS);    // ordered after it
    CHECK(got == stream);

    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
    REQUIRE(cublasGetPointerMode(handle, &mode) == CUBLAS_STATUS_SUCCESS);
    CHECK(mode == CUBLAS_POINTER_MODE_HOST);
    CHECK(cublasSetPointerMode(handle, static_cast<cublasPointerMode_t>(7)) == CUBLAS_STATUS_INVALID_VALUE);

    REQUIRE(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) == CUBLAS_STATUS_SUCCESS);
    cublasMath_t math = CUBLAS_DEFAULT_MATH;
    REQUIRE(cublasGetMathMode(handle, &math) == CUBLAS_STATUS_SUCCESS);
    CHECK(math == CUBLAS_TF32_TENSOR_OP_MATH);
}

TEST_CASE_METHOD(Fixture, "Matrix and vector copies honor leading dimensions and strides", "[cublas]") {
    // A 3 x 4 block out of a host matrix with lda 5, into a device matrix with ldb 6.
    const int rows = 3, cols = 4, lda = 5, ldb = 6;
    std::vector<float> host(lda * cols);
    for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<float>(i);
    float* d = DeviceCopy(std::vector<float>(ldb * cols, -1.0f));
    REQUIRE(cublasSetMatrix(rows, cols, sizeof(float), host.data(), lda, d, ldb) == CUBLAS_STATUS_SUCCESS);
    const std::vector<float> device = HostCopy(d, ldb * cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < ldb; ++i) {
            CHECK(device[i + j * ldb] == (i < rows ? host[i + j * lda] : -1.0f));
        }
    }
    std::vector<float> back(4 * cols, 0.0f);  // ldb 4 on the way back
    REQUIRE(cublasGetMatrix(rows, cols, sizeof(float), d, ldb, back.data(), 4) == CUBLAS_STATUS_SUCCESS);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) CHECK(back[i + j * 4] == host[i + j * lda]);
        CHECK(back[3 + j * 4] == 0.0f);  // padding untouched
    }

    // Strided vectors: every 2nd host element into every 3rd device element.
    const std::vector<float> x = {1, 99, 2, 99, 3, 99, 4};
    float* dv = DeviceCopy(std::vector<float>(10, 0.0f));
    REQUIRE(cublasSetVector(4, sizeof(float), x.data(), 2, dv, 3) == CUBLAS_STATUS_SUCCESS);
    const std::vector<float> dv_host = HostCopy(dv, 10);
    CHECK(dv_host == std::vector<float>{1, 0, 0, 2, 0, 0, 3, 0, 0, 4});
    std::vector<float> y(4, 0.0f);
    REQUIRE(cublasGetVector(4, sizeof(float), dv, 3, y.data(), 1) == CUBLAS_STATUS_SUCCESS);
    CHECK(y == std::vector<float>{1, 2, 3, 4});

    // The async forms, on a stream.
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const std::vector<float> z = {5, 6, 7};
    REQUIRE(cublasSetVectorAsync(3, sizeof(float), z.data(), 1, dv, 1, stream) == CUBLAS_STATUS_SUCCESS);
    std::vector<float> z_back(3, 0.0f);
    REQUIRE(cublasGetVectorAsync(3, sizeof(float), dv, 1, z_back.data(), 1, stream) == CUBLAS_STATUS_SUCCESS);
    CHECK(z_back == z);

    // Invalid arguments are rejected on the client, before any round trip.
    CHECK(cublasSetMatrix(rows, cols, sizeof(float), host.data(), 2, d, ldb) == CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cublasSetVector(4, sizeof(float), x.data(), 0, dv, 1) == CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cublasGetMatrix(rows, cols, 0, d, ldb, back.data(), 4) == CUBLAS_STATUS_INVALID_VALUE);
}

TEST_CASE_METHOD(Fixture, "Sgemm and Dgemm with transposes match a CPU reference", "[cublas]") {
    const int m = 5, n = 4, k = 3;
    for (const bool ta : {false, true}) {
        for (const bool tb : {false, true}) {
            const int lda = (ta ? k : m) + 1, ldb = (tb ? n : k) + 2, ldc = m + 1;
            const std::vector<float> A = Iota(lda * (ta ? m : k), 0.5f);
            const std::vector<float> B = Iota(ldb * (tb ? k : n), 0.25f);
            const std::vector<float> C = Iota(ldc * n, 1.0f);
            const float alpha = 1.5f, beta = -0.5f;
            float *dA = DeviceCopy(A), *dB = DeviceCopy(B), *dC = DeviceCopy(C);
            REQUIRE(cublasSgemm(handle, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
                                &alpha, dA, lda, dB, ldb, &beta, dC, ldc) == CUBLAS_STATUS_SUCCESS);
            const std::vector<float> got = HostCopy(dC, C.size());
            const std::vector<float> want = RefGemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            for (size_t i = 0; i < want.size(); ++i) CHECK_THAT(got[i], WithinAbs(want[i], 1e-4));
        }
    }

    std::vector<double> A(6), B(6), C(4, 1.0);
    for (int i = 0; i < 6; ++i) A[i] = B[i] = 0.5 * i;
    double *dA = DeviceCopy(A), *dB = DeviceCopy(B), *dC = DeviceCopy(C);
    const double alpha = 2.0, beta = 1.0;
    REQUIRE(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 3, &alpha, dA, 2, dB, 3, &beta, dC, 2) ==
            CUBLAS_STATUS_SUCCESS);
    const std::vector<double> got = HostCopy(dC, 4);
    const std::vector<double> want = RefGemm(false, false, 2, 2, 3, alpha, A, 2, B, 3, beta, C, 2);
    for (size_t i = 0; i < 4; ++i) CHECK_THAT(got[i], WithinAbs(want[i], 1e-12));
}

TEST_CASE_METHOD(Fixture, "Strided-batched gemm and GemmEx", "[cublas]") {
    const int m = 3, n = 2, k = 4, batch = 3;
    const long long sa = m * k, sb = k * n, sc = m * n;
    const std::vector<float> A = Iota(sa * batch, 0.5f), B = Iota(sb * batch, 0.25f);
    const std::vector<float> C(sc * batch, 0.0f);
    float *dA = DeviceCopy(A), *dB = DeviceCopy(B), *dC = DeviceCopy(C);
    const float alpha = 1.0f, beta = 0.0f;
    REQUIRE(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, sa, dB, k, sb, &beta,
                                      dC, m, sc, batch) == CUBLAS_STATUS_SUCCESS);
    const std::vector<float> got = HostCopy(dC, C.size());
    for (int b = 0; b < batch; ++b) {
        const std::vector<float> Ab(A.begin() + b * sa, A.begin() + (b + 1) * sa);
        const std::vector<float> Bb(B.begin() + b * sb, B.begin() + (b + 1) * sb);
        const std::vector<float> want = RefGemm(false, false, m, n, k, alpha, Ab, m, Bb, k, beta,
                                                std::vector<float>(sc, 0.0f), m);
        for (int i = 0; i < sc; ++i) CHECK_THAT(got[b * sc + i], WithinAbs(want[i], 1e-4));
    }

    // GemmEx: alpha/beta are `const void*`, sized by the compute type.
    float* dC2 = DeviceCopy(std::vector<float>(sc, 0.0f));
    REQUIRE(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_32F, m, dB, CUDA_R_32F, k, &beta,
                         dC2, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);
    const std::vector<float> got2 = HostCopy(dC2, sc);
    for (int i = 0; i < sc; ++i) CHECK_THAT(got2[i], WithinAbs(got[i], 1e-6));
    CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_32F, m, dB, CUDA_R_32F, k, &beta,
                       dC2, CUDA_R_32F, m, static_cast<cublasComputeType_t>(12345),
                       CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_INVALID_VALUE);

    CHECK(tuna::cublas::ComputeScalarSize(CUBLAS_COMPUTE_16F, CUDA_R_16F) == 2);
    CHECK(tuna::cublas::ComputeScalarSize(CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_16BF) == 4);
    CHECK(tuna::cublas::ComputeScalarSize(CUBLAS_COMPUTE_64F, CUDA_C_64F) == 16);
}

TEST_CASE_METHOD(Fixture, "Level-1 results come back in host pointer mode", "[cublas]") {
    const std::vector<float> x = {1, 2, 3, 4}, y = {4, 3, 2, 1};
    float *dx = DeviceCopy(x), *dy = DeviceCopy(y);
    float dot = 0, nrm = 0;
    REQUIRE(cublasSdot(handle, 4, dx, 1, dy, 1, &dot) == CUBLAS_STATUS_SUCCESS);
    CHECK(dot == 20.0f);
    REQUIRE(cublasSnrm2(handle, 4, dx, 1, &nrm) == CUBLAS_STATUS_SUCCESS);
    CHECK_THAT(nrm, WithinAbs(std::sqrt(30.0), 1e-5));

    const float alpha = 2.0f;
    REQUIRE(cublasSaxpy(handle, 4, &alpha, dx, 1, dy, 1) == CUBLAS_STATUS_SUCCESS);  // y += 2x
    REQUIRE(cublasSscal(handle, 2, &alpha, dy, 2) == CUBLAS_STATUS_SUCCESS);        // every 2nd *= 2
    CHECK(HostCopy(dy, 4) == std::vector<float>{12, 7, 16, 9});
}

TEST_CASE_METHOD(Fixture, "Device pointer mode sends scalars as device addresses", "[cublas]") {
    REQUIRE(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS);
    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    REQUIRE(cublasGetPointerMode(handle, &mode) == CUBLAS_STATUS_SUCCESS);
    CHECK(mode == CUBLAS_POINTER_MODE_DEVICE);

    // alpha = 3, beta = 0 live in device memory; so does the dot product.
    float* scalars = DeviceCopy(std::vector<float>{3.0f, 0.0f, 0.0f});
    const std::vector<float> A = {1, 2, 3, 4}, B = {1, 0, 0, 1};  // 2x2, B = identity
    float *dA = DeviceCopy(A), *dB = DeviceCopy(B), *dC = DeviceCopy(std::vector<float>(4, 9.0f));
    REQUIRE(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, scalars, dA, 2, dB, 2, scalars + 1, dC, 2) ==
            CUBLAS_STATUS_SUCCESS);
    CHECK(HostCopy(dC, 4) == std::vector<float>{3, 6, 9, 12});

    REQUIRE(cublasSdot(handle, 4, dA, 1, dA, 1, scalars + 2) == CUBLAS_STATUS_SUCCESS);
    CHECK(HostCopy(scalars, 3)[2] == 30.0f);

    // Back to host mode: the same call takes host scalars again.
    REQUIRE(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) == CUBLAS_STATUS_SUCCESS);
    float dot = 0;
    REQUIRE(cublasSdot(handle, 4, dA, 1, dA, 1, &dot) == CUBLAS_STATUS_SUCCESS);
    CHECK(dot == 30.0f);
}

TEST_CASE_METHOD(Fixture, "A failed posted cuBLAS call surfaces at the next synchronization", "[cublas]") {
    const float alpha = 1.0f, beta = 0.0f;
    float* d = DeviceCopy(std::vector<float>(16, 0.0f));
    // lda < m: cuBLAS rejects it on the server; the posted call already returned.
    CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, &alpha, d, 1, d, 4, &beta, d, 4) ==
          CUBLAS_STATUS_SUCCESS);
    CHECK(cudaDeviceSynchronize() == cudaErrorInvalidValue);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);  // reported once
}

TEST_CASE_METHOD(Fixture, "cuBLAS status strings and unsupported exports", "[cublas]") {
    CHECK(std::string(cublasGetStatusName(CUBLAS_STATUS_INVALID_VALUE)) == "CUBLAS_STATUS_INVALID_VALUE");
    CHECK(std::string(cublasGetStatusString(CUBLAS_STATUS_NOT_INITIALIZED)) == "the library was not initialized");
    CHECK(std::string(cublasGetStatusName(static_cast<cublasStatus_t>(42))) == "CUBLAS_STATUS_UNKNOWN");
    CHECK(cublasGetCudartVersion() == CUDART_VERSION);

    // Exported (so programs load), not remoted yet.
    const cuComplex one{1.0f, 0.0f};
    CHECK(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, nullptr, 1, nullptr, 1, &one, nullptr, 1) ==
          CUBLAS_STATUS_NOT_SUPPORTED);
}

TEST_CASE_METHOD(Fixture, "Malformed cuBLAS requests are rejected, not read out of bounds", "[cublas]") {
    // A truncated synchronous request: cuBLAS's own INVALID_VALUE.
    ArgWriter truncated;
    truncated.put_u64(reinterpret_cast<uint64_t>(handle));
    Reply r = InProcessCall(kApi_cublasSdot_v2, truncated.str(), {}, false);
    CHECK(r.result == CUBLAS_STATUS_INVALID_VALUE);

    // An over-wide host scalar is refused (the server keeps them in 16 bytes).
    ArgWriter wide;
    wide.put_u64(reinterpret_cast<uint64_t>(handle));
    const int n = 1;
    wide.put(&n, 4);
    wide.put_u32(kScalarOnHost);
    wide.put_string(std::string(64, 'x'));
    r = InProcessCall(kApi_cublasSscal_v2, wide.str(), {}, false);
    CHECK(r.result == CUBLAS_STATUS_INVALID_VALUE);  // not flagged as posted, so cuBLAS's code

    // A matrix copy claiming 2^30 x 2^30 x 8 bytes is refused before any allocation.
    ArgWriter huge;
    const int big = 1 << 30, elem = 8;
    huge.put(&big, 4);
    huge.put(&big, 4);
    huge.put(&elem, 4);
    huge.put_u64(0x1000);
    huge.put(&big, 4);
    r = InProcessCall(kApi_cublasGetMatrix, huge.str(), {}, false);
    CHECK(r.result == CUBLAS_STATUS_INVALID_VALUE);
    CHECK(r.bulk_out.empty());

    // Handles are pointers into the worker's memory: one the session did not
    // create is refused, never passed to cuBLAS.
    ArgWriter forged;
    forged.put_u64(0xdeadbeef000ull);
    r = InProcessCall(kApi_cublasGetVersion_v2, forged.str(), {}, false);
    CHECK(r.result == CUBLAS_STATUS_NOT_INITIALIZED);
    r = InProcessCall(kApi_cublasDestroy_v2, forged.str(), {}, false);
    CHECK(r.result == CUBLAS_STATUS_NOT_INITIALIZED);

    // A scalar must agree with the handle's pointer mode (host here): a
    // "device" alpha would make cuBLAS read a client-chosen host address.
    float* d = DeviceCopy(std::vector<float>(4, 1.0f));
    ArgWriter lying;
    lying.put_u64(reinterpret_cast<uint64_t>(handle));
    const int four_elems = 4, inc = 1;
    lying.put(&four_elems, 4);
    PutDeviceScalar(lying, reinterpret_cast<void*>(0x12345678));
    lying.put_u64(reinterpret_cast<uint64_t>(d));
    lying.put(&inc, 4);
    r = InProcessCall(kApi_cublasSscal_v2, lying.str(), {}, false);
    CHECK(r.result == cudaErrorInvalidValue);  // a posted call's code: cuBLAS never ran
    CHECK(HostCopy(d, 4) == std::vector<float>(4, 1.0f));

    // A host->device payload whose size does not match the claimed matrix.
    ArgWriter mismatch;
    const int two = 2, four = 4;
    mismatch.put(&two, 4);
    mismatch.put(&two, 4);
    mismatch.put(&four, 4);
    mismatch.put_u64(0x1000);
    mismatch.put(&two, 4);
    r = InProcessCall(kApi_cublasSetMatrix, mismatch.str(), std::string(3, 'x'), false);
    CHECK(r.result == CUBLAS_STATUS_INVALID_VALUE);
}

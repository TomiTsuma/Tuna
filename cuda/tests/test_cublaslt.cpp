// Round trip of the cuBLASLt remoting with no GPU and no transport: the
// cuBLASLt shim encodes -> tuna_cudart_call -> CudaSession -> the mock
// cuBLASLt (mock_cuda.cpp) computes on "device" memory. The main case replays
// what PyTorch's addmm (gemm_and_bias) does, and checks both the result and
// that the whole GEMM costs one blocking round trip.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "cuda_api_ids.h"
#include "mock_cuda.h"
#include "tuna/cuda/backend.h"
#include "tuna/cuda/client.h"
#include "tuna/cuda/cublas_client.h"
#include "tuna/cuda/cublaslt_client.h"

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
int g_blocking_calls = 0;

Reply InProcessCall(uint32_t api_id, const std::string& args, const std::string& bulk_in, bool async) {
    if (!async) ++g_blocking_calls;
    RpcRequest request;
    request.set_api_id(api_id);
    request.set_args(args);
    if (async) request.set_flags(tunnel::kFlagAsync);
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
        tuna::cublaslt::ResetStateForTesting();
        SetCallHook(InProcessCall);
        LastError(/*clear=*/true);
        REQUIRE(cublasCreate(&blas) == CUBLAS_STATUS_SUCCESS);
        lt = reinterpret_cast<cublasLtHandle_t>(blas);  // what PyTorch does
    }
    ~Fixture() {
        cublasDestroy(blas);
        SetCallHook(nullptr);
    }
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t lt = nullptr;
};

float* DeviceCopy(const std::vector<float>& values) {
    void* d = nullptr;
    REQUIRE(cudaMalloc(&d, values.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    return static_cast<float*>(d);
}

std::vector<float> HostCopy(const float* d, size_t n) {
    std::vector<float> values(n);
    REQUIRE(cudaMemcpy(values.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    return values;
}

std::vector<float> Sequence(size_t n, float scale, float offset) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = offset + scale * static_cast<float>((i * 7) % 13);
    return v;
}

template <typename T>
void Set(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr, T value) {
    REQUIRE(cublasLtMatmulDescSetAttribute(desc, attr, &value, sizeof(value)) == CUBLAS_STATUS_SUCCESS);
}

// One GEMM the way PyTorch's gemm_and_bias issues it:
//   D (m x n) = op(A) * B + bias, with A stored k x m (transposed) and B k x n.
struct BiasGemm {
    int m = 5, n = 3, k = 4;
    std::vector<float> a = Sequence(static_cast<size_t>(k) * m, 0.5f, -1.0f);  // k x m
    std::vector<float> b = Sequence(static_cast<size_t>(k) * n, 0.25f, 0.5f);  // k x n
    std::vector<float> bias = Sequence(static_cast<size_t>(m), 1.0f, -3.0f);

    std::vector<float> Reference(bool relu) const {
        std::vector<float> d(static_cast<size_t>(m) * n);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                float sum = 0;
                for (int p = 0; p < k; ++p) sum += a[p + i * k] * b[p + j * k];
                sum += bias[i];
                d[i + j * m] = relu ? std::max(sum, 0.0f) : sum;
            }
        }
        return d;
    }
};

}  // namespace

TEST_CASE_METHOD(Fixture, "cublasLt: an addmm (bias epilogue) costs one round trip", "[cuda][cublaslt]") {
    BiasGemm g;
    float* A = DeviceCopy(g.a);
    float* B = DeviceCopy(g.b);
    float* bias = DeviceCopy(g.bias);
    float* D = DeviceCopy(std::vector<float>(static_cast<size_t>(g.m) * g.n, 0.0f));
    void* workspace = nullptr;
    REQUIRE(cudaMalloc(&workspace, 1 << 20) == cudaSuccess);

    const int before = g_blocking_calls;
    cublasLtMatmulDesc_t desc = nullptr;
    REQUIRE(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS);
    Set(desc, CUBLASLT_MATMUL_DESC_TRANSA, int32_t{CUBLAS_OP_T});
    Set(desc, CUBLASLT_MATMUL_DESC_TRANSB, int32_t{CUBLAS_OP_N});
    Set(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, int32_t{CUBLASLT_EPILOGUE_BIAS});
    Set(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, static_cast<void*>(bias));
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
    REQUIRE(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, g.k, g.m, g.k) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, g.k, g.n, g.k) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, g.m, g.n, g.m) == CUBLAS_STATUS_SUCCESS);
    cublasLtMatmulPreference_t pref = nullptr;
    REQUIRE(cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS);
    const uint64_t workspace_size = 1 << 20;
    REQUIRE(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
                                                 sizeof(workspace_size)) == CUBLAS_STATUS_SUCCESS);

    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    REQUIRE(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref, 1, &result, &count) ==
            CUBLAS_STATUS_SUCCESS);
    REQUIRE(count == 1);
    REQUIRE(result.state == CUBLAS_STATUS_SUCCESS);

    const float alpha = 1.0f, beta = 0.0f;
    REQUIRE(cublasLtMatmul(lt, desc, &alpha, A, la, B, lb, &beta, D, ld, D, ld, &result.algo, workspace,
                           workspace_size, nullptr) == CUBLAS_STATUS_SUCCESS);
    for (auto layout : {la, lb, ld}) REQUIRE(cublasLtMatrixLayoutDestroy(layout) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cublasLtMatmulPreferenceDestroy(pref) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cublasLtMatmulDescDestroy(desc) == CUBLAS_STATUS_SUCCESS);
    // 13 calls, of which only the heuristic query waited for the server.
    CHECK(g_blocking_calls - before == 1);

    const std::vector<float> d = HostCopy(D, static_cast<size_t>(g.m) * g.n);
    const std::vector<float> reference = g.Reference(/*relu=*/false);
    for (size_t i = 0; i < d.size(); ++i) CHECK_THAT(d[i], WithinAbs(reference[i], 1e-5));
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(MockCudaGetStats().blas_calls == 1);
}

TEST_CASE_METHOD(Fixture, "cublasLt: device pointer mode, ReLU epilogue, own handle, no algo", "[cuda][cublaslt]") {
    BiasGemm g;
    float* A = DeviceCopy(g.a);
    float* B = DeviceCopy(g.b);
    float* bias = DeviceCopy(g.bias);
    float* D = DeviceCopy(std::vector<float>(static_cast<size_t>(g.m) * g.n, 0.0f));
    float* scalars = DeviceCopy({1.0f, 0.0f});  // alpha, beta on the device

    cublasLtHandle_t own = nullptr;
    REQUIRE(cublasLtCreate(&own) == CUBLAS_STATUS_SUCCESS);
    cublasLtMatmulDesc_t desc = nullptr;
    REQUIRE(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS);
    Set(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, int32_t{CUBLASLT_POINTER_MODE_DEVICE});
    Set(desc, CUBLASLT_MATMUL_DESC_TRANSA, int32_t{CUBLAS_OP_T});
    Set(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, int32_t{CUBLASLT_EPILOGUE_RELU_BIAS});
    Set(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, static_cast<void*>(bias));
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
    REQUIRE(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, g.k, g.m, g.k) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, g.k, g.n, g.k) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, g.m, g.n, g.m) == CUBLAS_STATUS_SUCCESS);
    // No algo: cuBLASLt picks one itself.
    REQUIRE(cublasLtMatmul(own, desc, scalars, A, la, B, lb, scalars + 1, D, ld, D, ld, nullptr, nullptr, 0,
                           nullptr) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const std::vector<float> d = HostCopy(D, static_cast<size_t>(g.m) * g.n);
    const std::vector<float> reference = g.Reference(/*relu=*/true);
    for (size_t i = 0; i < d.size(); ++i) CHECK_THAT(d[i], WithinAbs(reference[i], 1e-5));
    CHECK(cublasLtDestroy(own) == CUBLAS_STATUS_SUCCESS);
    CHECK(cublasLtDestroy(own) == CUBLAS_STATUS_NOT_INITIALIZED);  // the server forgot it
}

TEST_CASE_METHOD(Fixture, "cublasLt: the server refuses what it did not hand out", "[cuda][cublaslt]") {
    BiasGemm g;
    float* A = DeviceCopy(g.a);
    float* D = DeviceCopy(std::vector<float>(static_cast<size_t>(g.m) * g.n, 0.0f));
    cublasLtMatmulDesc_t desc = nullptr;
    REQUIRE(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS);
    cublasLtMatrixLayout_t layout = nullptr;
    REQUIRE(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, g.m, g.m, g.m) == CUBLAS_STATUS_SUCCESS);
    cublasLtMatmulPreference_t pref = nullptr;
    REQUIRE(cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS);
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;

    SECTION("an unknown handle") {
        auto bogus = reinterpret_cast<cublasLtHandle_t>(0x1234);
        CHECK(cublasLtMatmulAlgoGetHeuristic(bogus, desc, layout, layout, layout, layout, pref, 1, &result,
                                             &count) == CUBLAS_STATUS_NOT_INITIALIZED);
    }
    SECTION("an object of the wrong kind") {
        auto layout_as_desc = reinterpret_cast<cublasLtMatmulDesc_t>(layout);
        CHECK(cublasLtMatmulAlgoGetHeuristic(lt, layout_as_desc, layout, layout, layout, layout, pref, 1, &result,
                                             &count) == CUBLAS_STATUS_INVALID_VALUE);
    }
    SECTION("an algo the heuristic never returned") {
        cublasLtMatmulAlgo_t forged{};
        forged.data[0] = 0x7A1607A1ull;  // the mock's tag: only the server's allowlist stops it
        const float alpha = 1.0f, beta = 0.0f;
        REQUIRE(cublasLtMatmul(lt, desc, &alpha, A, layout, A, layout, &beta, D, layout, D, layout, &forged, nullptr,
                               0, nullptr) == CUBLAS_STATUS_SUCCESS);  // posted
        CHECK(cudaDeviceSynchronize() == cudaErrorInvalidValue);       // refused on the server
        CHECK(MockCudaGetStats().blas_calls == 0);
    }
    SECTION("a scalar encoded against the pointer mode") {
        // The client thinks device mode; the server's descriptor says host.
        RpcRequest request;
        request.set_api_id(kApi_cublasLtMatmul);
        request.set_flags(tunnel::kFlagAsync);
        ArgWriter in;
        in.put_u64(reinterpret_cast<uint64_t>(blas));
        in.put_u64(reinterpret_cast<uint64_t>(desc));
        PutDeviceScalar(in, A);
        for (int i = 0; i < 4; ++i) in.put_u64(i % 2 ? reinterpret_cast<uint64_t>(layout) : reinterpret_cast<uint64_t>(A));
        PutDeviceScalar(in, A);
        for (int i = 0; i < 4; ++i) in.put_u64(i % 2 ? reinterpret_cast<uint64_t>(layout) : reinterpret_cast<uint64_t>(D));
        in.put_u32(0);
        for (int i = 0; i < 3; ++i) in.put_u64(0);
        request.set_args(in.str());
        std::string bulk_out;
        CHECK(g_session->Dispatch(request, {}, &bulk_out).result() == cudaErrorInvalidValue);
    }
    SECTION("a destroyed descriptor") {
        REQUIRE(cublasLtMatmulDescDestroy(desc) == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasLtMatmulAlgoGetHeuristic(lt, desc, layout, layout, layout, layout, pref, 1, &result, &count) ==
              CUBLAS_STATUS_INVALID_VALUE);
        desc = nullptr;
    }
    SECTION("attribute values over the size limit, or none") {
        std::vector<char> big(tuna::cublaslt::kMaxAttributeBytes + 1);
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, big.data(), big.size()) ==
              CUBLAS_STATUS_INVALID_VALUE);
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, nullptr, 4) ==
              CUBLAS_STATUS_INVALID_VALUE);
    }
    if (desc) cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(layout);
    cublasLtMatmulPreferenceDestroy(pref);
}

TEST_CASE("cublasLt: local queries need no server", "[cuda][cublaslt]") {
    CHECK(cublasLtGetVersion() == CUBLAS_VERSION);
    int major = 0;
    CHECK(cublasLtGetProperty(MAJOR_VERSION, &major) == CUBLAS_STATUS_SUCCESS);
    CHECK(major == CUBLAS_VER_MAJOR);
    CHECK(std::string(cublasLtGetStatusName(CUBLAS_STATUS_INVALID_VALUE)) == "CUBLAS_STATUS_INVALID_VALUE");
}

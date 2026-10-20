// Round trip of the cuDNN remoting with no GPU and no transport: the cuDNN
// shim encodes -> tuna_cudart_call -> CudaSession -> the mock cuDNN
// (mock_cuda.cpp) computes on "device" memory. The cases replay what
// PyTorch does (a convolution through the backend API, as the cuDNN frontend
// builds it; batch norm through the legacy API) and check the results, and
// that the steady-state calls cost no round trip.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <cudnn.h>

#include "cuda_api_ids.h"
#include "mock_cuda.h"
#include "tuna/cuda/backend.h"
#include "tuna/cuda/client.h"
#include "tuna/cuda/cudnn_client.h"

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
int g_posted_calls = 0;

Reply InProcessCall(uint32_t api_id, const std::string& args, const std::string& bulk_in, bool async) {
    ++(async ? g_posted_calls : g_blocking_calls);
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
        tuna::cudnn::ResetStateForTesting();
        SetCallHook(InProcessCall);
        LastError(/*clear=*/true);
        REQUIRE(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS);
    }
    ~Fixture() {
        cudnnDestroy(handle);
        SetCallHook(nullptr);
    }
    cudnnHandle_t handle = nullptr;
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

// ------------------------------------------------- backend API helpers

using Desc = cudnnBackendDescriptor_t;

Desc Create(cudnnBackendDescriptorType_t type) {
    Desc d = nullptr;
    REQUIRE(cudnnBackendCreateDescriptor(type, &d) == CUDNN_STATUS_SUCCESS);
    return d;
}

template <typename T>
void Set(Desc d, cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, std::vector<T> values) {
    REQUIRE(cudnnBackendSetAttribute(d, name, type, static_cast<int64_t>(values.size()), values.data()) ==
            CUDNN_STATUS_SUCCESS);
}

void Finalize(Desc d) { REQUIRE(cudnnBackendFinalize(d) == CUDNN_STATUS_SUCCESS); }

std::vector<int64_t> Packed(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (size_t i = dims.size() - 1; i > 0; --i) strides[i - 1] = strides[i] * dims[i];
    return strides;
}

Desc Tensor(int64_t uid, const std::vector<int64_t>& dims) {
    Desc t = Create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    Set(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, std::vector<cudnnDataType_t>{CUDNN_DATA_FLOAT});
    Set(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dims);
    Set(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, Packed(dims));
    Set(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, std::vector<int64_t>{uid});
    Set(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, std::vector<int64_t>{16});
    Finalize(t);
    return t;
}

// A 3x3 convolution (pad 1, stride 1), N=1, C=2, H=W=5, K=3, with its
// execution plan built the way the cuDNN frontend builds one.
struct Conv {
    static constexpr int N = 1, C = 2, H = 5, W = 5, K = 3, R = 3, S = 3;
    std::vector<float> x = Sequence(size_t{N} * C * H * W, 0.25f, -1.0f);
    std::vector<float> w = Sequence(size_t{K} * C * R * S, 0.5f, -1.5f);
    Desc tx = nullptr, tw = nullptr, ty = nullptr, conv = nullptr, op = nullptr, graph = nullptr, heur = nullptr;
    Desc cfgs[2] = {};
    Desc plan = nullptr;

    void Build(cudnnHandle_t handle) {
        tx = Tensor(1, {N, C, H, W});
        tw = Tensor(2, {K, C, R, S});
        ty = Tensor(3, {N, K, H, W});
        conv = Create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        Set(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, std::vector<cudnnDataType_t>{CUDNN_DATA_FLOAT});
        Set(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE,
            std::vector<cudnnConvolutionMode_t>{CUDNN_CROSS_CORRELATION});
        Set(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, std::vector<int64_t>{2});
        Set(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, std::vector<int64_t>{1, 1});
        Set(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, std::vector<int64_t>{1, 1});
        Set(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, std::vector<int64_t>{1, 1});
        Set(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, std::vector<int64_t>{1, 1});
        Finalize(conv);

        op = Create(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        Set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, std::vector<Desc>{tx});
        Set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, std::vector<Desc>{tw});
        Set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, std::vector<Desc>{ty});
        Set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR,
            std::vector<Desc>{conv});
        Set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, std::vector<float>{1.0f});
        Set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, std::vector<float>{0.0f});
        Finalize(op);

        graph = Create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        Set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, std::vector<cudnnHandle_t>{handle});
        Set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, std::vector<Desc>{op});
        Finalize(graph);

        heur = Create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        Set(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, std::vector<Desc>{graph});
        Set(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, std::vector<cudnnBackendHeurMode_t>{CUDNN_HEUR_MODE_INSTANT});
        Finalize(heur);

        // The heuristic's results land in engine configs the caller created.
        int64_t count = 0;
        REQUIRE(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0,
                                         &count, nullptr) == CUDNN_STATUS_SUCCESS);
        REQUIRE(count == 2);
        for (Desc& cfg : cfgs) cfg = Create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        REQUIRE(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 2,
                                         &count, cfgs) == CUDNN_STATUS_SUCCESS);

        plan = Create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        Set(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, std::vector<cudnnHandle_t>{handle});
        Set(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, std::vector<Desc>{cfgs[0]});
        Finalize(plan);
    }

    void Destroy() {
        for (Desc d : {plan, cfgs[0], cfgs[1], heur, graph, op, conv, tx, tw, ty}) {
            if (d) cudnnBackendDestroyDescriptor(d);
        }
    }

    std::vector<float> Reference() const {
        std::vector<float> y(size_t{N} * K * H * W, 0.0f);
        for (int k = 0; k < K; ++k)
            for (int p = 0; p < H; ++p)
                for (int q = 0; q < W; ++q) {
                    double sum = 0;
                    for (int c = 0; c < C; ++c)
                        for (int r = 0; r < R; ++r)
                            for (int s = 0; s < S; ++s) {
                                const int ih = p - 1 + r, iw = q - 1 + s;
                                if (ih < 0 || iw < 0 || ih >= H || iw >= W) continue;
                                sum += x[(c * H + ih) * W + iw] * w[((k * C + c) * R + r) * S + s];
                            }
                    y[(k * H + p) * W + q] = static_cast<float>(sum);
                }
        return y;
    }
};

// One execution the way PyTorch's run_conv_plan issues it: a fresh variant
// pack (unique ids, data pointers, workspace), finalize, execute, destroy.
cudnnStatus_t RunPlan(cudnnHandle_t handle, Desc plan, std::vector<void*> pointers, void* workspace) {
    Desc pack = Create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    Set(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, std::vector<int64_t>{1, 2, 3});
    Set(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, pointers);
    Set(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, std::vector<void*>{workspace});
    Finalize(pack);
    const cudnnStatus_t status = cudnnBackendExecute(handle, plan, pack);
    cudnnBackendDestroyDescriptor(pack);
    return status;
}

// ----------------------------------------------------------- batch norm

cudnnTensorDescriptor_t LegacyTensor(std::vector<int> dims) {
    cudnnTensorDescriptor_t t = nullptr;
    REQUIRE(cudnnCreateTensorDescriptor(&t) == CUDNN_STATUS_SUCCESS);
    std::vector<int> strides(dims.size(), 1);
    for (size_t i = dims.size() - 1; i > 0; --i) strides[i - 1] = strides[i] * dims[i];
    REQUIRE(cudnnSetTensorNdDescriptor(t, CUDNN_DATA_FLOAT, static_cast<int>(dims.size()), dims.data(),
                                       strides.data()) == CUDNN_STATUS_SUCCESS);
    return t;
}

// Spatial batch norm of a 2x3x2x2 batch, computed on the host.
struct BatchNorm {
    static constexpr int N = 2, C = 3, HW = 4;
    static constexpr double kEps = 1e-5;
    std::vector<float> x = Sequence(size_t{N} * C * HW, 0.5f, -2.0f);
    std::vector<float> dy = Sequence(size_t{N} * C * HW, 0.25f, -1.0f);
    std::vector<float> scale = {1.0f, 0.5f, 2.0f};
    std::vector<float> bias = {0.0f, -1.0f, 0.5f};

    size_t At(int n, int c, int i) const { return (static_cast<size_t>(n) * C + c) * HW + i; }
    double Mean(int c) const {
        double s = 0;
        for (int n = 0; n < N; ++n)
            for (int i = 0; i < HW; ++i) s += x[At(n, c, i)];
        return s / (N * HW);
    }
    double Inv(int c) const {
        const double m = Mean(c);
        double s = 0;
        for (int n = 0; n < N; ++n)
            for (int i = 0; i < HW; ++i) s += (x[At(n, c, i)] - m) * (x[At(n, c, i)] - m);
        return 1.0 / std::sqrt(s / (N * HW) + kEps);
    }
    std::vector<float> Forward() const {
        std::vector<float> y(x.size());
        for (int c = 0; c < C; ++c)
            for (int n = 0; n < N; ++n)
                for (int i = 0; i < HW; ++i) {
                    y[At(n, c, i)] = static_cast<float>(scale[c] * (x[At(n, c, i)] - Mean(c)) * Inv(c) + bias[c]);
                }
        return y;
    }
    // dx, then dscale, then dbias.
    void Backward(std::vector<float>* dx, std::vector<float>* dscale, std::vector<float>* dbias) const {
        dx->assign(x.size(), 0.0f);
        dscale->assign(C, 0.0f);
        dbias->assign(C, 0.0f);
        const double m = N * HW;
        for (int c = 0; c < C; ++c) {
            double gsum = 0, gxhat = 0;
            for (int n = 0; n < N; ++n)
                for (int i = 0; i < HW; ++i) {
                    gsum += dy[At(n, c, i)];
                    gxhat += dy[At(n, c, i)] * (x[At(n, c, i)] - Mean(c)) * Inv(c);
                }
            for (int n = 0; n < N; ++n)
                for (int i = 0; i < HW; ++i) {
                    const double xhat = (x[At(n, c, i)] - Mean(c)) * Inv(c);
                    (*dx)[At(n, c, i)] = static_cast<float>(scale[c] * Inv(c) / m * (m * dy[At(n, c, i)] - gsum - xhat * gxhat));
                }
            (*dscale)[c] = static_cast<float>(gxhat);
            (*dbias)[c] = static_cast<float>(gsum);
        }
    }
};

void RequireNear(const std::vector<float>& got, const std::vector<float>& want) {
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) CHECK_THAT(got[i], WithinAbs(want[i], 1e-4));
}

}  // namespace

TEST_CASE_METHOD(Fixture, "cudnn: a backend-API convolution, and steady-state executions cost no round trip",
                 "[cuda][cudnn]") {
    Conv c;
    c.Build(handle);
    float* x = DeviceCopy(c.x);
    float* w = DeviceCopy(c.w);
    float* y = DeviceCopy(std::vector<float>(c.Reference().size(), 7.0f));
    void* workspace = nullptr;
    REQUIRE(cudaMalloc(&workspace, 1024) == cudaSuccess);

    // A descriptor the server filled is a real copy: its own attributes read back.
    Desc engine = Create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    int64_t count = 0;
    REQUIRE(cudnnBackendGetAttribute(c.cfgs[1], CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count,
                                     &engine) == CUDNN_STATUS_SUCCESS);
    int64_t index = -1;
    REQUIRE(cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &count, &index) ==
            CUDNN_STATUS_SUCCESS);
    CHECK(count == 1);
    CHECK(index == 1);
    int64_t workspace_size = 0;
    REQUIRE(cudnnBackendGetAttribute(c.plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &count,
                                     &workspace_size) == CUDNN_STATUS_SUCCESS);
    CHECK(workspace_size == 256);

    const int before = g_blocking_calls;
    for (int i = 0; i < 3; ++i) {
        REQUIRE(cudnnSetStream(handle, nullptr) == CUDNN_STATUS_SUCCESS);  // PyTorch does, before each op
        REQUIRE(RunPlan(handle, c.plan, {x, w, y}, workspace) == CUDNN_STATUS_SUCCESS);
    }
    CHECK(g_blocking_calls == before);  // variant packs, finalizes and executes are all posted
    CHECK(MockCudaGetStats().dnn_executes == 3);
    RequireNear(HostCopy(y, c.Reference().size()), c.Reference());
    CHECK(cudaDeviceSynchronize() == cudaSuccess);  // nothing failed on the way

    cudnnBackendDestroyDescriptor(engine);
    c.Destroy();
}

TEST_CASE_METHOD(Fixture, "cudnn: batch norm forward and backward, and cached size queries", "[cuda][cudnn]") {
    BatchNorm b;
    cudnnTensorDescriptor_t xd = LegacyTensor({b.N, b.C, 2, 2});
    cudnnTensorDescriptor_t bnd = LegacyTensor({1, b.C, 1, 1});
    const auto mode = CUDNN_BATCHNORM_SPATIAL;
    const auto ops = CUDNN_BATCHNORM_OPS_BN;

    size_t fwd_ws = 0, bwd_ws = 0, reserve = 0;
    const int before = g_blocking_calls;
    REQUIRE(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, ops, xd, nullptr, xd, bnd, nullptr,
                                                                     &fwd_ws) == CUDNN_STATUS_SUCCESS);
    REQUIRE(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, ops, nullptr, xd, &reserve) ==
            CUDNN_STATUS_SUCCESS);
    REQUIRE(cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, ops, xd, xd, xd, nullptr, xd, bnd, nullptr,
                                                              &bwd_ws) == CUDNN_STATUS_SUCCESS);
    CHECK(fwd_ws == 32 * b.C);
    CHECK(reserve == 16 * b.C);
    CHECK(bwd_ws == 48 * b.C);
    CHECK(g_blocking_calls == before + 3);

    // The next layer's (and step's) descriptors are new objects with the same
    // contents: answered by the client, as PyTorch asks every call.
    cudnnTensorDescriptor_t xd2 = LegacyTensor({b.N, b.C, 2, 2});
    size_t again = 0;
    REQUIRE(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, ops, xd2, nullptr, xd2, bnd, nullptr,
                                                                     &again) == CUDNN_STATUS_SUCCESS);
    CHECK(again == fwd_ws);
    CHECK(g_blocking_calls == before + 3);
    CHECK(MockCudaGetStats().dnn_size_queries == 3);
    // A different shape is a different answer.
    cudnnTensorDescriptor_t wide = LegacyTensor({b.N, 2 * b.C, 2, 2});
    REQUIRE(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, ops, nullptr, wide, &again) ==
            CUDNN_STATUS_SUCCESS);
    CHECK(again == 16 * 2 * b.C);
    CHECK(g_blocking_calls == before + 4);

    float* x = DeviceCopy(b.x);
    float* dy = DeviceCopy(b.dy);
    float* y = DeviceCopy(std::vector<float>(b.x.size(), 0.0f));
    float* dx = DeviceCopy(std::vector<float>(b.x.size(), 0.0f));
    float* scale = DeviceCopy(b.scale);
    float* bias = DeviceCopy(b.bias);
    float* rmean = DeviceCopy(std::vector<float>(b.C, 0.0f));
    float* rvar = DeviceCopy(std::vector<float>(b.C, 1.0f));
    float* smean = DeviceCopy(std::vector<float>(b.C, 0.0f));
    float* sinv = DeviceCopy(std::vector<float>(b.C, 0.0f));
    float* dscale = DeviceCopy(std::vector<float>(b.C, 0.0f));
    float* dbias = DeviceCopy(std::vector<float>(b.C, 0.0f));
    void *ws = nullptr, *rs = nullptr;
    REQUIRE(cudaMalloc(&ws, std::max(fwd_ws, bwd_ws)) == cudaSuccess);
    REQUIRE(cudaMalloc(&rs, reserve) == cudaSuccess);

    const float one = 1.0f, zero = 0.0f;
    const int blocking = g_blocking_calls;
    REQUIRE(cudnnBatchNormalizationForwardTrainingEx(handle, mode, ops, &one, &zero, xd, x, nullptr, nullptr, xd, y,
                                                     bnd, scale, bias, 0.1, rmean, rvar, BatchNorm::kEps, smean, sinv,
                                                     nullptr, ws, fwd_ws, rs, reserve) == CUDNN_STATUS_SUCCESS);
    REQUIRE(cudnnBatchNormalizationBackwardEx(handle, mode, ops, &one, &zero, &one, &zero, xd, x, xd, y, xd, dy,
                                              nullptr, nullptr, xd, dx, bnd, scale, bias, dscale, dbias,
                                              BatchNorm::kEps, smean, sinv, nullptr, ws, bwd_ws, rs,
                                              reserve) == CUDNN_STATUS_SUCCESS);
    CHECK(g_blocking_calls == blocking);  // both posted
    RequireNear(HostCopy(y, b.x.size()), b.Forward());
    std::vector<float> want_dx, want_dscale, want_dbias;
    b.Backward(&want_dx, &want_dscale, &want_dbias);
    RequireNear(HostCopy(dx, b.x.size()), want_dx);
    RequireNear(HostCopy(dscale, b.C), want_dscale);
    RequireNear(HostCopy(dbias, b.C), want_dbias);
    std::vector<float> want_mean(b.C);
    for (int c = 0; c < b.C; ++c) want_mean[c] = static_cast<float>(0.1 * b.Mean(c));
    RequireNear(HostCopy(rmean, b.C), want_mean);

    // Inference with the batch statistics reproduces the training output.
    std::vector<float> mean(b.C), var(b.C);
    for (int c = 0; c < b.C; ++c) {
        mean[c] = static_cast<float>(b.Mean(c));
        var[c] = static_cast<float>(1.0 / (b.Inv(c) * b.Inv(c)) - BatchNorm::kEps);
    }
    float* dmean = DeviceCopy(mean);
    float* dvar = DeviceCopy(var);
    float* y2 = DeviceCopy(std::vector<float>(b.x.size(), 0.0f));
    REQUIRE(cudnnBatchNormalizationForwardInference(handle, mode, &one, &zero, xd, x, xd, y2, bnd, scale, bias, dmean,
                                                    dvar, BatchNorm::kEps) == CUDNN_STATUS_SUCCESS);
    RequireNear(HostCopy(y2, b.x.size()), b.Forward());
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    for (auto t : {xd, xd2, bnd, wide}) cudnnDestroyTensorDescriptor(t);
}

TEST_CASE_METHOD(Fixture, "cudnn: the server refuses what it did not hand out", "[cuda][cudnn]") {
    Desc tensor = Create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    Desc op = Create(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    Desc graph = Create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);

    SECTION("a forged descriptor element") {
        auto forged = reinterpret_cast<Desc>(0x1234);
        REQUIRE(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1, &forged) == CUDNN_STATUS_SUCCESS);  // posted
        CHECK(cudaDeviceSynchronize() == cudaErrorInvalidValue);                // refused on the server
    }
    SECTION("a legacy tensor descriptor used as a backend descriptor") {
        cudnnTensorDescriptor_t legacy = LegacyTensor({1, 1, 1, 1});
        const int64_t uid = 1;
        REQUIRE(cudnnBackendSetAttribute(reinterpret_cast<Desc>(legacy), CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64,
                                         1, &uid) == CUDNN_STATUS_SUCCESS);
        CHECK(cudaDeviceSynchronize() == cudaErrorInvalidValue);
        cudnnDestroyTensorDescriptor(legacy);
    }
    SECTION("a handle the server never created") {
        auto bogus = reinterpret_cast<cudnnHandle_t>(0x5555);
        REQUIRE(cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &bogus) ==
                CUDNN_STATUS_SUCCESS);
        CHECK(cudaDeviceSynchronize() == cudaErrorInvalidValue);
    }
    SECTION("device addresses outside a variant pack") {
        void* address = reinterpret_cast<void*>(0x7f0000100000ull);
        REQUIRE(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC, CUDNN_TYPE_VOID_PTR, 1,
                                         &address) == CUDNN_STATUS_SUCCESS);
        CHECK(cudaDeviceSynchronize() == cudaErrorNotSupported);
    }
    SECTION("by-value tensors are refused before anything is sent") {
        const int posted = g_posted_calls, blocking = g_blocking_calls;
        const bool by_value = true;
        CHECK(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &by_value) ==
              CUDNN_STATUS_NOT_SUPPORTED);
        CHECK(g_posted_calls == posted);
        CHECK(g_blocking_calls == blocking);
        // ...and by the server, for a client that sends one anyway.
        RpcRequest request;
        request.set_api_id(kApi_cudnnBackendSetAttribute);
        ArgWriter in;
        in.put_u64(reinterpret_cast<uint64_t>(tensor));
        in.put_u32(CUDNN_ATTR_TENSOR_IS_BY_VALUE);
        in.put_u32(CUDNN_TYPE_BOOLEAN);
        in.put_u64(1);
        in.put(&by_value, 1);
        request.set_args(in.str());
        std::string bulk_out;
        CHECK(g_session->Dispatch(request, {}, &bulk_out).result() == CUDNN_STATUS_NOT_SUPPORTED);
    }
    SECTION("an attribute type Tuna does not know") {
        const int64_t value = 0;
        CHECK(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, static_cast<cudnnBackendAttributeType_t>(99),
                                       1, &value) == CUDNN_STATUS_NOT_SUPPORTED);
    }
    SECTION("output descriptors the client never created") {
        Conv c;
        c.Build(handle);
        auto forged = reinterpret_cast<Desc>(0x1234);
        int64_t count = 0;
        CHECK(cudnnBackendGetAttribute(c.heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count,
                                       &forged) == CUDNN_STATUS_BAD_PARAM);
        c.Destroy();
    }
    SECTION("an activation descriptor") {
        cudnnTensorDescriptor_t xd = LegacyTensor({1, 2, 2, 2});
        size_t size = 0;
        CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
                  handle, CUDNN_BATCHNORM_SPATIAL, CUDNN_BATCHNORM_OPS_BN,
                  reinterpret_cast<cudnnActivationDescriptor_t>(0x99), xd, &size) == CUDNN_STATUS_NOT_SUPPORTED);
        cudnnDestroyTensorDescriptor(xd);
    }
    SECTION("a destroyed descriptor") {
        REQUIRE(cudnnBackendDestroyDescriptor(graph) == CUDNN_STATUS_SUCCESS);
        CHECK(cudnnBackendFinalize(graph) == CUDNN_STATUS_BAD_PARAM);
        graph = nullptr;
    }
    SECTION("a short wire buffer") {
        RpcRequest request;
        request.set_api_id(kApi_cudnnBackendGetAttribute);
        request.set_args(std::string(5, '\0'));
        std::string bulk_out;
        CHECK(g_session->Dispatch(request, {}, &bulk_out).result() == CUDNN_STATUS_BAD_PARAM);
    }
    SECTION("the server's error message reaches cudnnGetLastErrorString") {
        CHECK(cudnnBackendFinalize(graph) == CUDNN_STATUS_BAD_PARAM);  // no handle, no operations
        char message[128];
        cudnnGetLastErrorString(message, sizeof(message));
        CHECK(std::string(message) == "mock: finalize: required attributes missing");
    }
    for (Desc d : {tensor, op, graph}) {
        if (d) cudnnBackendDestroyDescriptor(d);
    }
}

TEST_CASE_METHOD(Fixture, "cudnn: what the client answers itself", "[cuda][cudnn]") {
    const int before = g_blocking_calls;
    CHECK(cudnnGetVersion() == 91002);  // the server's library, asked once
    CHECK(cudnnGetVersion() == 91002);
    int major = 0, minor = 0, patch = 0;
    CHECK(cudnnGetProperty(MAJOR_VERSION, &major) == CUDNN_STATUS_SUCCESS);
    CHECK(cudnnGetProperty(MINOR_VERSION, &minor) == CUDNN_STATUS_SUCCESS);
    CHECK(cudnnGetProperty(PATCH_LEVEL, &patch) == CUDNN_STATUS_SUCCESS);
    CHECK((major == 9 && minor == 10 && patch == 2));
    CHECK(g_blocking_calls == before + 1);

    auto stream = reinterpret_cast<cudaStream_t>(0x42);
    REQUIRE(cudnnSetStream(handle, stream) == CUDNN_STATUS_SUCCESS);
    cudaStream_t got = nullptr;
    CHECK(cudnnGetStream(handle, &got) == CUDNN_STATUS_SUCCESS);
    CHECK(got == stream);

    cudnnTensorDescriptor_t t = LegacyTensor({2, 3, 4, 5});
    cudnnDataType_t type{};
    int nb = 0, dims[8] = {}, strides[8] = {};
    CHECK(cudnnGetTensorNdDescriptor(t, 8, &type, &nb, dims, strides) == CUDNN_STATUS_SUCCESS);
    CHECK((type == CUDNN_DATA_FLOAT && nb == 4 && dims[1] == 3 && strides[0] == 60 && strides[3] == 1));
    cudnnDestroyTensorDescriptor(t);
    CHECK(g_blocking_calls == before + 1);

    CHECK(std::string(cudnnGetErrorString(CUDNN_STATUS_BAD_PARAM)) == "CUDNN_STATUS_BAD_PARAM");
    CHECK(std::string(cudnnGetErrorString(CUDNN_STATUS_NOT_SUPPORTED_SHAPE)) == "CUDNN_STATUS_NOT_SUPPORTED_SHAPE");
    CHECK(cudnnOpsVersionCheck() == CUDNN_STATUS_SUCCESS);
}

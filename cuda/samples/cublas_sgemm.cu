// Plan milestone 3 (cuBLAS): an ordinary cuBLAS program, built with
// `nvcc -cudart shared ... -lcublas`, that must give the same answers natively
// and through Tuna (where libcublas.so.12 and libcudart.so.12 are Tuna's shims
// and cuBLAS itself runs on the server).
//
//   * SGEMM 1024 x 1024 (cublasSetMatrix upload, simpleCUBLAS style), checked
//     against a CPU reference on sampled columns, then timed (GFlop/s);
//   * mixed precision GemmEx: fp16 inputs, fp32 output and compute;
//   * strided-batched DGEMM;
//   * level 1: axpy, dot (host and device pointer mode), nrm2;
//   * a kernel launch and cuBLAS calls interleaved on one stream (ordering).

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        cudaError_t e_ = (call);                                                                      \
        if (e_ != cudaSuccess) {                                                                      \
            std::printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_));       \
            std::exit(1);                                                                             \
        }                                                                                             \
    } while (0)

#define CHECK_CUBLAS(call)                                                                            \
    do {                                                                                              \
        cublasStatus_t s_ = (call);                                                                   \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                            \
            std::printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #call, cublasGetStatusString(s_));    \
            std::exit(1);                                                                             \
        }                                                                                             \
    } while (0)

static int g_failures = 0;

static void Expect(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_failures;
}

__global__ void AddOne(float* data, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1.0f;
}

static float Value(int i, int salt) { return static_cast<float>((i * 131 + salt * 17) % 97) / 97.0f - 0.5f; }

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    int version = 0;
    CHECK_CUBLAS(cublasGetVersion(handle, &version));
    std::printf("cuBLAS version %d\n", version);
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    // ---- SGEMM 1024 x 1024, C = alpha * A * B + beta * C
    const int n = 1024;
    const size_t elems = static_cast<size_t>(n) * n;
    std::vector<float> A(elems), B(elems), C(elems);
    for (size_t i = 0; i < elems; ++i) {
        A[i] = Value(static_cast<int>(i), 1);
        B[i] = Value(static_cast<int>(i), 2);
        C[i] = Value(static_cast<int>(i), 3);
    }
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, elems * sizeof(float)));
    CHECK_CUBLAS(cublasSetMatrix(n, n, sizeof(float), A.data(), n, dA, n));
    CHECK_CUBLAS(cublasSetMatrix(n, n, sizeof(float), B.data(), n, dB, n));
    CHECK_CUBLAS(cublasSetMatrixAsync(n, n, sizeof(float), C.data(), n, dC, n, stream));
    const float alpha = 1.25f, beta = 0.5f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n));
    std::vector<float> got(elems);
    CHECK_CUBLAS(cublasGetMatrixAsync(n, n, sizeof(float), dC, n, got.data(), n, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double max_err = 0;
    for (int j = 0; j < n; j += 97) {  // sampled columns, every row
        for (int i = 0; i < n; ++i) {
            double sum = 0;
            for (int p = 0; p < n; ++p) sum += double{A[i + static_cast<size_t>(p) * n]} * B[p + static_cast<size_t>(j) * n];
            const double want = alpha * sum + beta * C[i + static_cast<size_t>(j) * n];
            max_err = std::fmax(max_err, std::fabs(want - got[i + static_cast<size_t>(j) * n]));
        }
    }
    std::printf("sgemm %dx%d max abs error vs CPU: %.2e\n", n, n, max_err);
    Expect(max_err < 1e-2, "sgemm matches the CPU reference");  // TF32 off by default: fp32 accuracy

    // Timed: 50 SGEMMs queued back to back on the stream.
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    const int iterations = 50;
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int it = 0; it < iterations; ++it) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    const double gflops = 2.0 * n * n * n * iterations / (ms / 1e3) / 1e9;
    std::printf("sgemm: %d x %dx%d in %.2f ms = %.0f GFlop/s\n", iterations, n, n, ms, gflops);

    // ---- mixed precision: fp16 A, B; fp32 C and compute
    const int m = 256;
    std::vector<__half> Ah(m * m), Bh(m * m);
    std::vector<float> Af(m * m), Bf(m * m);
    for (int i = 0; i < m * m; ++i) {
        Ah[i] = __float2half(Value(i, 4));
        Bh[i] = __float2half(Value(i, 5));
        Af[i] = __half2float(Ah[i]);
        Bf[i] = __half2float(Bh[i]);
    }
    __half *dAh, *dBh;
    float* dCf;
    CHECK_CUDA(cudaMalloc(&dAh, m * m * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dBh, m * m * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dCf, m * m * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dAh, Ah.data(), m * m * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBh, Bh.data(), m * m * sizeof(__half), cudaMemcpyHostToDevice));
    const float one = 1.0f, zero = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, m, m, &one, dAh, CUDA_R_16F, m, dBh, CUDA_R_16F, m,
                              &zero, dCf, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    std::vector<float> Cf(m * m);
    CHECK_CUDA(cudaMemcpyAsync(Cf.data(), dCf, m * m * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double ex_err = 0;
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0;
            for (int p = 0; p < m; ++p) sum += double{Af[p + i * m]} * Bf[p + j * m];  // A transposed
            ex_err = std::fmax(ex_err, std::fabs(sum - Cf[i + j * m]));
        }
    }
    std::printf("gemmEx fp16 x fp16 -> fp32 (%dx%d) max abs error: %.2e\n", m, m, ex_err);
    Expect(ex_err < 1e-2, "mixed-precision GemmEx matches the reference");

    // ---- strided-batched DGEMM: 8 independent 32x32 products
    const int d = 32, batch = 8;
    const long long stride = static_cast<long long>(d) * d;
    std::vector<double> Ad(stride * batch), Bd(stride * batch), Cd(stride * batch, 0.0);
    for (size_t i = 0; i < Ad.size(); ++i) {
        Ad[i] = Value(static_cast<int>(i), 6);
        Bd[i] = Value(static_cast<int>(i), 7);
    }
    double *dAd, *dBd, *dCd;
    CHECK_CUDA(cudaMalloc(&dAd, Ad.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dBd, Bd.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dCd, Cd.size() * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dAd, Ad.data(), Ad.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBd, Bd.data(), Bd.size() * sizeof(double), cudaMemcpyHostToDevice));
    const double done = 1.0, dzero = 0.0;
    CHECK_CUBLAS(cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, d, d, &done, dAd, d, stride, dBd, d,
                                           stride, &dzero, dCd, d, stride, batch));
    CHECK_CUDA(cudaMemcpyAsync(Cd.data(), dCd, Cd.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double batch_err = 0;
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < d; ++j) {
            for (int i = 0; i < d; ++i) {
                double sum = 0;
                for (int p = 0; p < d; ++p) sum += Ad[b * stride + i + p * d] * Bd[b * stride + p + j * d];
                batch_err = std::fmax(batch_err, std::fabs(sum - Cd[b * stride + i + j * d]));
            }
        }
    }
    Expect(batch_err < 1e-12, "strided-batched dgemm matches the reference");

    // ---- level 1, and a kernel launch interleaved with cuBLAS on one stream
    const int len = 1 << 16;
    std::vector<float> x(len), y(len);
    double dot_ref = 0, nrm_ref = 0;
    for (int i = 0; i < len; ++i) {
        x[i] = Value(i, 8);
        y[i] = Value(i, 9);
    }
    float *dx, *dy, *dResult;
    CHECK_CUDA(cudaMalloc(&dx, len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dResult, sizeof(float)));
    CHECK_CUBLAS(cublasSetVector(len, sizeof(float), x.data(), 1, dx, 1));
    CHECK_CUBLAS(cublasSetVector(len, sizeof(float), y.data(), 1, dy, 1));
    AddOne<<<(len + 255) / 256, 256, 0, stream>>>(dy, len);  // y += 1 (kernel)
    const float two = 2.0f;
    CHECK_CUBLAS(cublasSaxpy(handle, len, &two, dx, 1, dy, 1));  // y += 2x (cuBLAS, after the kernel)
    for (int i = 0; i < len; ++i) {
        y[i] = y[i] + 1.0f + 2.0f * x[i];
        dot_ref += double{x[i]} * y[i];
        nrm_ref += double{x[i]} * x[i];
    }
    float dot = 0;
    CHECK_CUBLAS(cublasSdot(handle, len, dx, 1, dy, 1, &dot));
    Expect(std::fabs(dot - dot_ref) < 1e-2 * (1 + std::fabs(dot_ref)), "kernel + axpy + dot in stream order");

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS(cublasSnrm2(handle, len, dx, 1, dResult));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    float nrm = 0;
    CHECK_CUDA(cudaMemcpyAsync(&nrm, dResult, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    Expect(std::fabs(nrm - std::sqrt(nrm_ref)) < 1e-3 * std::sqrt(nrm_ref), "nrm2 into device memory");

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dAh));
    CHECK_CUDA(cudaFree(dBh));
    CHECK_CUDA(cudaFree(dCf));
    CHECK_CUDA(cudaFree(dAd));
    CHECK_CUDA(cudaFree(dBd));
    CHECK_CUDA(cudaFree(dCd));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dResult));
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}

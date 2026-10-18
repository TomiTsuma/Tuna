// A cuBLAS program linked exactly as an application links cuBLAS (-lcublas
// -lcudart), run by test_cuda_e2e.py against Tuna's shims: libcublas.so.12 +
// libcudart.so.12 -> tuna_server -> worker -> mock CUDA + cuBLAS. Prints PASS
// when every check holds.
//
//   e2e_cublas_client                     the checks below, against $TUNA_SERVER
//   e2e_cublas_client --expect-no-server  cuBLAS reports "not initialized"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

bool Ok(cublasStatus_t s, const char* call) {
    if (s == CUBLAS_STATUS_SUCCESS) return true;
    std::printf("FAIL: %s -> %s\n", call, cublasGetStatusName(s));
    ++failures;
    return false;
}

bool Ok(cudaError_t e, const char* call) {
    if (e == cudaSuccess) return true;
    std::printf("FAIL: %s -> %s\n", call, cudaGetErrorName(e));
    ++failures;
    return false;
}

int NoServer() {
    cublasHandle_t handle = nullptr;
    const cublasStatus_t s = cublasCreate(&handle);
    Check(s == CUBLAS_STATUS_NOT_INITIALIZED,
          std::string("without a server, cublasCreate reports ") + cublasGetStatusName(s));
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--expect-no-server") == 0) return NoServer();

    cublasHandle_t handle = nullptr;
    if (!Ok(cublasCreate(&handle), "cublasCreate")) return 1;
    cudaStream_t stream = nullptr;
    Ok(cudaStreamCreate(&stream), "cudaStreamCreate");
    Ok(cublasSetStream(handle, stream), "cublasSetStream");

    // C = A * B for 64 x 64 matrices, uploaded with cublasSetMatrix (the
    // simpleCUBLAS pattern) and checked against a CPU reference.
    const int n = 64;
    std::vector<float> A(n * n), B(n * n), C(n * n, 0.0f), ref(n * n, 0.0f);
    for (int i = 0; i < n * n; ++i) {
        A[i] = static_cast<float>((i * 37) % 11) / 4.0f;
        B[i] = static_cast<float>((i * 53) % 7) / 2.0f;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            float sum = 0;
            for (int p = 0; p < n; ++p) sum += A[i + p * n] * B[p + j * n];
            ref[i + j * n] = sum;
        }
    }
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    Ok(cudaMalloc(reinterpret_cast<void**>(&dA), n * n * sizeof(float)), "cudaMalloc A");
    Ok(cudaMalloc(reinterpret_cast<void**>(&dB), n * n * sizeof(float)), "cudaMalloc B");
    Ok(cudaMalloc(reinterpret_cast<void**>(&dC), n * n * sizeof(float)), "cudaMalloc C");
    Ok(cublasSetMatrix(n, n, sizeof(float), A.data(), n, dA, n), "cublasSetMatrix A");
    Ok(cublasSetMatrixAsync(n, n, sizeof(float), B.data(), n, dB, n, stream), "cublasSetMatrixAsync B");
    const float alpha = 1.0f, beta = 0.0f;
    Ok(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n), "cublasSgemm");
    Ok(cublasGetMatrixAsync(n, n, sizeof(float), dC, n, C.data(), n, stream), "cublasGetMatrixAsync C");
    bool correct = true;
    for (int i = 0; i < n * n && correct; ++i) correct = std::fabs(C[i] - ref[i]) < 1e-3f * (1.0f + std::fabs(ref[i]));
    Check(correct, "sgemm 64x64 over the network matches the CPU reference");

    // A dot product written to device memory (device pointer mode), read back
    // with cudaMemcpy.
    float* dResult = nullptr;
    Ok(cudaMalloc(reinterpret_cast<void**>(&dResult), sizeof(float)), "cudaMalloc result");
    Ok(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
    Ok(cublasSdot(handle, n, dA, 1, dA, 1, dResult), "cublasSdot (device result)");
    Ok(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    float dot = 0, expected = 0;
    for (int i = 0; i < n; ++i) expected += A[i] * A[i];
    Ok(cudaMemcpy(&dot, dResult, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy result");
    Check(std::fabs(dot - expected) < 1e-3f, "device-pointer-mode dot product");

    // A failing posted gemm is reported at the next synchronizing call.
    Ok(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, 1, dB, n, &beta, dC, n), "bad sgemm (queued)");
    const cudaError_t sync = cudaDeviceSynchronize();
    Check(sync == cudaErrorInvalidValue, std::string("bad sgemm reported at sync: ") + cudaGetErrorName(sync));

    // A 24 MiB matrix (3072 x 2048 floats) rides the bulk channel both ways.
    const int rows = 3072, cols = 2048;
    std::vector<float> big(static_cast<size_t>(rows) * cols), back(big.size(), 0.0f);
    for (size_t i = 0; i < big.size(); ++i) big[i] = static_cast<float>(i % 1000) * 0.5f;
    float* dBig = nullptr;
    Ok(cudaMalloc(reinterpret_cast<void**>(&dBig), big.size() * sizeof(float)), "cudaMalloc big");
    Ok(cublasSetMatrix(rows, cols, sizeof(float), big.data(), rows, dBig, rows), "cublasSetMatrix 24 MiB");
    Ok(cublasGetMatrix(rows, cols, sizeof(float), dBig, rows, back.data(), rows), "cublasGetMatrix 24 MiB");
    Check(big == back, "24 MiB matrix round trip");

    Ok(cublasDestroy(handle), "cublasDestroy");
    Ok(cudaFree(dA), "cudaFree");
    Ok(cudaFree(dB), "cudaFree");
    Ok(cudaFree(dC), "cudaFree");
    Ok(cudaFree(dResult), "cudaFree");
    Ok(cudaFree(dBig), "cudaFree");

    std::printf("%s\n", failures == 0 ? "PASS" : "FAILED");
    return failures == 0 ? 0 : 1;
}

// Drives the REAL GPU through Tuna's generated CUDA dispatch, in-process (no
// network): the client stubs encode -> CudaDispatch decodes -> the real
// libcudart runs on the GPU -> results marshal back. This is the Phase 3a
// real-GPU milestone, minus the transport (Phase 2 already proved that).
//
// It is not a CI unit test (it needs an NVIDIA GPU); it runs in the nightly
// GPU lane and by hand. Exit 0 only if every check passes.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "tuna/cuda/backend_real.h"
#include "tuna/cuda/client.h"
#include "cuda_api_ids.h"

using namespace tuna;
using namespace tuna::cudart;

namespace {
CudaSession* g_session = nullptr;

// In-process stand-in for the transport: run the real dispatcher.
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

int fail(const char* what, cudaError_t e) {
    std::fprintf(stderr, "FAIL %s: %d\n", what, static_cast<int>(e));
    return 1;
}
}  // namespace

int main() {
    std::string error;
    auto backend = OpenRealCuda({}, &error);
    if (!backend) {
        std::fprintf(stderr, "cannot open libcudart: %s\n", error.c_str());
        return 1;
    }
    CudaSession session(*backend);
    g_session = &session;
    SetCallHook(InProcessCall);

    int count = 0;
    if (cudaError_t e = cudaGetDeviceCount(&count); e != cudaSuccess || count < 1) return fail("GetDeviceCount", e);

    cudaDeviceProp prop;
    if (cudaError_t e = cudaGetDeviceProperties(&prop, 0); e != cudaSuccess) return fail("GetDeviceProperties", e);
    std::printf("device 0: %s (cc %d.%d, %zu MiB) via Tuna dispatch\n", prop.name, prop.major, prop.minor,
                static_cast<size_t>(prop.totalGlobalMem >> 20));
    if (prop.name[0] == '\0') return fail("empty device name", cudaSuccess);

    // Allocate real device memory, round-trip a buffer through it.
    void* d = nullptr;
    const size_t n = 1 << 20;  // 1 MiB, large enough to ride the bulk channel path
    if (cudaError_t e = cudaMalloc(&d, n); e != cudaSuccess || !d) return fail("Malloc", e);

    std::vector<unsigned char> host_in(n), host_out(n, 0);
    for (size_t i = 0; i < n; ++i) host_in[i] = static_cast<unsigned char>((i * 2654435761u) >> 24);
    if (cudaError_t e = cudaMemcpy(d, host_in.data(), n, cudaMemcpyHostToDevice); e != cudaSuccess)
        return fail("Memcpy H2D", e);
    if (cudaError_t e = cudaMemcpy(host_out.data(), d, n, cudaMemcpyDeviceToHost); e != cudaSuccess)
        return fail("Memcpy D2H", e);
    if (host_out != host_in) return fail("round-trip mismatch", cudaSuccess);

    size_t free_before = 0, total = 0, free_after = 0;
    cudaMemGetInfo(&free_before, &total);
    if (cudaError_t e = cudaFree(d); e != cudaSuccess) return fail("Free", e);
    cudaMemGetInfo(&free_after, &total);
    std::printf("1 MiB device round-trip OK; free %zu -> %zu MiB after free, total %zu MiB\n",
                free_before >> 20, free_after >> 20, total >> 20);

    std::printf("PASS: real GPU driven through Tuna's generated CUDA dispatch\n");
    return 0;
}

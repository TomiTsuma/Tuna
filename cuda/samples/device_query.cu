// deviceQuery, trimmed: the first milestone of the execution plan. Run on a
// client with no NVIDIA GPU, it prints the *server's* GPU.
#include <cstdio>

#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        std::printf("cudaGetDeviceCount: %s\n", cudaGetErrorString(e));
        return 1;
    }
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("CUDA driver %d.%d, runtime %d.%d, %d device(s)\n", driver / 1000, (driver % 100) / 10,
                runtime / 1000, (runtime % 100) / 10, count);
    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess) return 1;
        int clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, d);
        std::printf("Device %d: \"%s\"\n", d, p.name);
        std::printf("  compute capability      %d.%d\n", p.major, p.minor);
        std::printf("  global memory           %zu MiB\n", static_cast<size_t>(p.totalGlobalMem >> 20));
        std::printf("  multiprocessors         %d\n", p.multiProcessorCount);
        std::printf("  max threads per block   %d\n", p.maxThreadsPerBlock);
        std::printf("  shared memory per block %zu KiB\n", static_cast<size_t>(p.sharedMemPerBlock >> 10));
        std::printf("  clock rate              %d MHz\n", clock_khz / 1000);
    }
    std::printf("PASS\n");
    return count > 0 ? 0 : 1;
}

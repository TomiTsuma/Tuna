// A driver-API program linked exactly as an application links the driver
// (-lcuda -lcudart), run by test_cuda_e2e.py against Tuna's shims:
// libcuda.so.1 + libcudart.so.12 -> tuna_server -> worker -> mock CUDA. Prints
// PASS when every check holds.
//
//   e2e_driver_client                     the checks below, against $TUNA_SERVER
//   e2e_driver_client --expect-no-server  cuInit reports CUDA_ERROR_NO_DEVICE

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

bool Ok(CUresult r, const char* call) {
    if (r == CUDA_SUCCESS) return true;
    const char* name = nullptr;
    cuGetErrorName(r, &name);
    std::printf("FAIL: %s -> %s\n", call, name ? name : "?");
    ++failures;
    return false;
}

int NoServer() {
    const CUresult r = cuInit(0);
    Check(r == CUDA_ERROR_NO_DEVICE, "without a server, cuInit reports CUDA_ERROR_NO_DEVICE");
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--expect-no-server") == 0) return NoServer();

    if (!Ok(cuInit(0), "cuInit")) return 1;
    int version = 0;
    Ok(cuDriverGetVersion(&version), "cuDriverGetVersion");
    Check(version == 12080, "driver version is the server's (" + std::to_string(version) + ")");
    int count = 0;
    Ok(cuDeviceGetCount(&count), "cuDeviceGetCount");
    Check(count == 2, "the server's devices are visible");
    char name[128] = {};
    Ok(cuDeviceGetName(name, sizeof(name), 0), "cuDeviceGetName");
    Check(std::string(name) == "Tuna Mock GPU 0", std::string("device name: ") + name);

    // cuGetProcAddress resolves to the shim itself, by base name. (Compared by
    // behavior: on Windows `&cuDeviceGetCount` here is the import thunk.)
    void* fn = nullptr;
    CUdriverProcAddressQueryResult status{};
    Ok(cuGetProcAddress("cuDeviceGetCount", &fn, 12080, CU_GET_PROC_ADDRESS_DEFAULT, &status),
       "cuGetProcAddress(cuDeviceGetCount)");
    Check(fn && status == CU_GET_PROC_ADDRESS_SUCCESS, "cuGetProcAddress finds cuDeviceGetCount");
    int via_lookup = 0;
    if (fn) reinterpret_cast<CUresult (*)(int*)>(fn)(&via_lookup);
    Check(via_lookup == 2, "the looked-up entry point is the shim's (remoted to the server)");
    // The legacy 32-bit cuDeviceTotalMem must not be what a CUDA 12 caller gets.
    Ok(cuGetProcAddress("cuDeviceTotalMem", &fn, 12080, CU_GET_PROC_ADDRESS_DEFAULT, &status),
       "cuGetProcAddress(cuDeviceTotalMem)");
    size_t total = 0;
    const CUresult total_result = fn ? reinterpret_cast<CUresult (*)(size_t*, CUdevice)>(fn)(&total, 0)
                                     : CUDA_ERROR_NOT_FOUND;
    Check(total_result == CUDA_SUCCESS && total == (8ull << 30),
          "a base name resolves to its newest variant (cuDeviceTotalMem_v2)");
    Check(cuGetProcAddress("cuNoSuchFunction", &fn, 12080, CU_GET_PROC_ADDRESS_DEFAULT, &status) ==
              CUDA_ERROR_NOT_FOUND && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND && fn == nullptr,
          "an unknown name is not found");

    // The primary context: inactive until used, active after a runtime call,
    // and retained/current through the driver.
    void* device_memory = nullptr;
    if (cudaMalloc(&device_memory, 256) != cudaSuccess) Check(false, "cudaMalloc");
    unsigned flags = 0;
    int active = 0;
    Ok(cuDevicePrimaryCtxGetState(0, &flags, &active), "cuDevicePrimaryCtxGetState");
    Check(active == 1, "the runtime's allocation made the primary context active");
    CUcontext ctx = nullptr;
    Ok(cuDevicePrimaryCtxRetain(&ctx, 0), "cuDevicePrimaryCtxRetain");
    Ok(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    CUcontext current = nullptr;
    Ok(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    Check(current == ctx, "the retained primary context is current");
    Ok(cuCtxSynchronize(), "cuCtxSynchronize");
    cudaFree(device_memory);

    // Anything else in the driver ABI loads and reports "not supported".
    CUmodule module = nullptr;
    Check(cuModuleLoad(&module, "x.cubin") == CUDA_ERROR_NOT_SUPPORTED, "cuModuleLoad is not supported yet");

    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

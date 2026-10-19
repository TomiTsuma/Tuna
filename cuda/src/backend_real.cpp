// The production CudaBackend: resolves runtime symbols from the real
// libcudart, driver symbols from libcuda and cuBLAS symbols from libcublas
// (opened on first use). The server opens it inside the per-session worker,
// so each session has its own CUDA context (Phase 2c isolates the process).

#include "tuna/cuda/backend_real.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tuna::cudart {

namespace {

#ifdef _WIN32
constexpr const char* kDefaultRuntime = "cudart64_12.dll";
constexpr const char* kDefaultDriver = "nvcuda.dll";
constexpr const char* kDefaultCublas = "cublas64_12.dll";
void* OpenLibrary(const std::string& path) { return LoadLibraryA(path.c_str()); }
void* Symbol(void* handle, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
}
void CloseLibrary(void* handle) { FreeLibrary(static_cast<HMODULE>(handle)); }
std::string LastError() { return "LoadLibrary failed (" + std::to_string(GetLastError()) + ")"; }
#else
constexpr const char* kDefaultRuntime = "libcudart.so.12";
constexpr const char* kDefaultDriver = "libcuda.so.1";
constexpr const char* kDefaultCublas = "libcublas.so.12";
// RTLD_GLOBAL so libcudart's own dependency on libcuda resolves to the same copy.
void* OpenLibrary(const std::string& path) { return dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL); }
void* Symbol(void* handle, const char* name) { return dlsym(handle, name); }
void CloseLibrary(void* handle) { dlclose(handle); }
std::string LastError() {
    const char* e = dlerror();
    return e ? e : "unknown dlopen error";
}
#endif

using ResolveHook = void* (*)(const char*);

class RealCudaBackend : public CudaBackend {
public:
    RealCudaBackend(void* runtime, void* driver, std::string cublas_path)
        : runtime_(runtime), driver_(driver), cublas_path_(std::move(cublas_path)) {
        hook_ = reinterpret_cast<ResolveHook>(Symbol(runtime_, "tuna_cuda_resolve"));
    }
    ~RealCudaBackend() override {
        // Unloading CUDA while the process still runs is a known source of
        // crashes at exit; the worker exits right after its session anyway.
    }

    void* Resolve(const char* symbol) override {
        // Cache: a dispatcher resolves the same handful of symbols constantly.
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(symbol);
        if (it != cache_.end()) return it->second;
        void* fn = nullptr;
        if (hook_) {
            fn = hook_(symbol);
        } else if (std::strncmp(symbol, "cublas", 6) == 0) {
            if (void* cublas = Cublas()) fn = Symbol(cublas, symbol);
        } else if (std::strncmp(symbol, "cuda", 4) == 0) {
            fn = Symbol(runtime_, symbol);
        } else if (driver_) {
            fn = Symbol(driver_, symbol);
        }
        cache_.emplace(symbol, fn);
        return fn;
    }

private:
    // libcublas, opened on the first cuBLAS call (nullptr if unavailable; the
    // calls then report not supported). Called with mutex_ held.
    void* Cublas() {
        if (cublas_tried_) return cublas_;
        cublas_tried_ = true;
        void* handle = OpenLibrary(cublas_path_);
        if (!handle) {
            std::fprintf(stderr, "tuna: cuBLAS unavailable: %s: %s\n", cublas_path_.c_str(), LastError().c_str());
        } else if (Symbol(handle, "tuna_cublas_shim")) {
            std::fprintf(stderr, "tuna: %s is Tuna's client shim, not NVIDIA's cuBLAS; fix the server's library path\n",
                         cublas_path_.c_str());
            CloseLibrary(handle);
        } else {
            cublas_ = handle;
        }
        return cublas_;
    }

    void* runtime_;
    void* driver_;
    std::string cublas_path_;
    void* cublas_ = nullptr;
    bool cublas_tried_ = false;
    ResolveHook hook_ = nullptr;
    std::mutex mutex_;
    std::unordered_map<std::string, void*> cache_;
};

}  // namespace

std::unique_ptr<CudaBackend> OpenRealCuda(const CudaLibraries& libraries, std::string* error) {
    const std::string runtime_path = libraries.runtime.empty() ? kDefaultRuntime : libraries.runtime;
    void* runtime = OpenLibrary(runtime_path);
    if (!runtime) {
        if (error) *error = runtime_path + ": " + LastError();
        return nullptr;
    }
    if (Symbol(runtime, "tuna_cudart_shim")) {
        if (error) *error = runtime_path + " is Tuna's client shim, not NVIDIA's runtime; fix the server's library path";
        CloseLibrary(runtime);
        return nullptr;
    }
    void* driver = nullptr;
    if (!Symbol(runtime, "tuna_cuda_resolve")) {
        const std::string driver_path = libraries.driver.empty() ? kDefaultDriver : libraries.driver;
        driver = OpenLibrary(driver_path);
        if (!driver) {
            if (error) *error = driver_path + ": " + LastError();
            return nullptr;
        }
        if (Symbol(driver, "tuna_cuda_driver_shim")) {
            if (error) *error = driver_path + " is Tuna's client shim, not NVIDIA's driver; fix the server's library path";
            CloseLibrary(driver);
            return nullptr;
        }
    }
    return std::make_unique<RealCudaBackend>(runtime, driver,
                                             libraries.cublas.empty() ? kDefaultCublas : libraries.cublas);
}

}  // namespace tuna::cudart

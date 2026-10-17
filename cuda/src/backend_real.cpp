// The production CudaBackend: resolves runtime symbols from the real
// libcudart. The server dlopen/LoadLibrary's it inside the per-session worker,
// so each session has its own CUDA context (Phase 2c isolates the process).

#include "tuna/cuda/backend_real.h"

#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tuna::cudart {

namespace {

class RealCudaBackend : public CudaBackend {
public:
    explicit RealCudaBackend(void* handle) : handle_(handle) {}
    ~RealCudaBackend() override {
#ifdef _WIN32
        if (handle_) FreeLibrary(static_cast<HMODULE>(handle_));
#else
        if (handle_) dlclose(handle_);
#endif
    }

    void* Resolve(const char* symbol) override {
        // Cache: a dispatcher resolves the same handful of symbols constantly.
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(symbol);
        if (it != cache_.end()) return it->second;
#ifdef _WIN32
        void* fn = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), symbol));
#else
        void* fn = dlsym(handle_, symbol);
#endif
        cache_.emplace(symbol, fn);
        return fn;
    }

private:
    void* handle_;
    std::mutex mutex_;
    std::unordered_map<std::string, void*> cache_;
};

}  // namespace

std::unique_ptr<CudaBackend> OpenRealCuda(const char* library, std::string* error) {
#ifdef _WIN32
    void* handle = LoadLibraryA(library ? library : "cudart64_12.dll");
    if (!handle && error) *error = "LoadLibrary failed (" + std::to_string(GetLastError()) + ")";
#else
    // RTLD_GLOBAL so libcudart's own dependency on libcuda resolves.
    void* handle = dlopen(library ? library : "libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!handle && error) *error = dlerror();
#endif
    if (!handle) return nullptr;
    return std::make_unique<RealCudaBackend>(handle);
}

}  // namespace tuna::cudart

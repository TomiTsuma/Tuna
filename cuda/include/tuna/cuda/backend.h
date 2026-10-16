#pragma once

#include <string>

#include "tuna/cuda/marshal.h"
#include "tuna/tunnel/rpc.h"
#include "tuna_frames.pb.h"

namespace tuna::cudart {

// The real CUDA runtime, as the server dispatcher sees it. The production
// backend dlopen/LoadLibrary's libcudart and returns dlsym addresses; tests
// supply a mock whose Resolve() returns pointers to stand-in functions.
class CudaBackend {
public:
    virtual ~CudaBackend() = default;
    // The real function for `symbol`, or nullptr if this backend lacks it.
    virtual void* Resolve(const char* symbol) = 0;
};

// Executes one remoted runtime call. Reads request.args() (and `bulk_in` for
// host->device copies), calls the real function through `backend`, and returns
// the response (with `bulk_out` set for device->host copies). Never throws:
// a malformed request becomes cudaErrorInvalidValue.
RpcResponse CudaDispatch(const RpcRequest& request, const std::string& bulk_in, std::string* bulk_out,
                         CudaBackend& backend);

// True if `api_id` is a CUDA runtime call this build knows (not the echo id
// and not an unannotated function).
bool IsCudaApi(uint32_t api_id);

}  // namespace tuna::cudart

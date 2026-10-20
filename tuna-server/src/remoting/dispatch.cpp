#include "remoting/dispatch.h"

#include <memory>

#include "tuna/tunnel/session.h"

#ifdef TUNA_SERVER_CUDA
#include "tuna/cuda/backend.h"
#include "tuna/cuda/backend_real.h"
#endif

namespace tuna {

namespace {

constexpr int32_t kCudaErrorNotSupported = 801;
[[maybe_unused]] constexpr int32_t kCudaErrorNoDevice = 100;

#ifdef TUNA_SERVER_CUDA
// One session's CUDA state. Opening the real stack is deferred to the first
// CUDA call: sessions that never touch the GPU (tuna_ping, health checks)
// never create a CUDA context.
struct CudaState {
    bool tried = false;
    std::unique_ptr<cudart::CudaBackend> backend;
    std::unique_ptr<cudart::CudaSession> session;
};
#endif

}  // namespace

Hello ServerHello() {
    Hello hello;
    hello.set_protocol_version(tunnel::kProtocolVersion);
    return hello;
}

tunnel::Dispatcher MakeDispatcher(std::shared_ptr<spdlog::logger> log, const ServerCudaConfig& cuda) {
#ifdef TUNA_SERVER_CUDA
    auto state = std::make_shared<CudaState>();
#endif
    return [=](tunnel::ServerCall& call) {
        RpcResponse response;
        const uint32_t api_id = call.request.api_id();
        if (api_id == tunnel::kApiEcho) {
            response.set_out_args(call.request.args());
            call.bulk_out = std::move(call.bulk_in);  // bulk echo, for tuna_ping --bulk-mb
            return response;
        }
#ifdef TUNA_SERVER_CUDA
        if (cuda.enabled && cudart::IsCudaApi(api_id)) {
            if (!state->tried) {
                state->tried = true;
                std::string error;
                state->backend = cudart::OpenRealCuda(
                    {cuda.runtime_library, cuda.driver_library, cuda.cublas_library, cuda.cudnn_library}, &error);
                if (state->backend) {
                    state->session = std::make_unique<cudart::CudaSession>(*state->backend);
                    log->info("CUDA backend opened ({})",
                              cuda.runtime_library.empty() ? "default libraries" : cuda.runtime_library);
                } else {
                    log->error("cannot open the CUDA stack: {}", error);
                }
            }
            if (!state->session) {
                response.set_result(kCudaErrorNoDevice);
                return response;
            }
            response = state->session->Dispatch(call.request, call.bulk_in, &call.bulk_out);
            if (response.result() != 0 && log->should_log(spdlog::level::debug)) {
                log->debug("{} -> error {}", cudart::CudaApiName(api_id), response.result());
            }
            return response;
        }
#else
        (void)cuda;
        (void)log;
#endif
        response.set_result(kCudaErrorNotSupported);
        return response;
    };
}

}  // namespace tuna

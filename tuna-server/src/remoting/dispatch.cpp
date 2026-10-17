#include "remoting/dispatch.h"

#include "tuna/tunnel/session.h"

namespace tuna {

namespace {

constexpr int32_t kCudaErrorNotSupported = 801;

}  // namespace

Hello ServerHello() {
    Hello hello;
    hello.set_protocol_version(tunnel::kProtocolVersion);
    return hello;
}

tunnel::Dispatcher MakeDispatcher(std::shared_ptr<spdlog::logger>) {
    return [](tunnel::ServerCall& call) {
        RpcResponse response;
        if (call.request.api_id() == tunnel::kApiEcho) {
            response.set_out_args(call.request.args());
            call.bulk_out = std::move(call.bulk_in);  // bulk echo, for tuna_ping --bulk-mb
        } else {
            response.set_result(kCudaErrorNotSupported);
        }
        return response;
    };
}

}  // namespace tuna

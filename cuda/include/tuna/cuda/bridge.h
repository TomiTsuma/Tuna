#pragma once

// The C bridge Tuna's replacement libcudart exports for Tuna's other library
// shims (libcublas today; cuBLASLt/cuDNN later). A process has one session to
// its GPU server, owned by the runtime shim: library calls must travel down
// that same session, in order with the runtime calls, because they use the
// same server-side context, device pointers and streams.
//
// A plain C ABI (no std::string across the boundary), so the libraries do not
// have to share a C++ runtime with each other.

#include <cstddef>
#include <cstdint>

extern "C" {

// How tuna_cudart_call sends a request.
enum TunaCallMode : uint32_t {
    kTunaCall = 0,             // blocking call; the reply carries the result
    kTunaCallSynchronize = 1,  // blocking, and reports a latched async failure instead
    kTunaPost = 2,             // fire-and-forget; a failure surfaces at the next synchronizing call
};

// The reply to a blocking call. The buffers belong to the calling thread and
// stay valid until its next tuna_cudart_call.
struct TunaCallReply {
    int32_t result;           // the remote function's status, or the connection error below
    int32_t transport_error;  // nonzero: the session was lost mid-call
    const char* out_args;
    size_t out_args_size;
    const char* bulk_out;
    size_t bulk_out_size;
};

// Sends one marshalled call (args encoded as in marshal.h). Connects on first
// use. Without a server, result is 100 (cudaErrorNoDevice). For kTunaPost the
// reply is only filled with the connection status (0 when queued).
// Returns `reply->result`.
int32_t tuna_cudart_call(uint32_t api_id, const char* args, size_t args_size, const char* bulk_in,
                         size_t bulk_in_size, uint32_t mode, TunaCallReply* reply);

}  // extern "C"

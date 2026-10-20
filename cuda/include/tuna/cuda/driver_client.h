#pragma once

#include <cstdint>

#include "tuna/cuda/marshal.h"

// Tuna's replacement CUDA driver library (Phase 3e): libcuda.so.1 /
// nvcuda.dll. Driver calls are marshalled like runtime calls and sent through
// the runtime shim's session (tuna_cudart_call, tuna/cuda/bridge.h), so they
// run on the server in the same worker, context and order as the app's
// runtime and cuBLAS calls.
//
// This header includes no CUDA header: the generated not-supported stubs are
// compiled without prototypes (see driver_unsupported.cpp). Status codes are
// plain ints here; CUresult and cudaError_t share the values used below.

#if defined(_WIN32)
#define TUNA_CUDA_DRIVER_EXPORT extern "C" __declspec(dllexport)
#else
#define TUNA_CUDA_DRIVER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace tuna::cudrv {

// CUresult values the shim produces itself.
constexpr int kSuccess = 0;
constexpr int kInvalidValue = 1;
constexpr int kNoDevice = 100;        // no server configured / reachable
constexpr int kInvalidDevice = 101;
constexpr int kInvalidContext = 201;
constexpr int kNotFound = 500;
constexpr int kNotSupported = 801;
constexpr int kUnknown = 999;         // connection lost mid-session, malformed reply

// A blocking call through the runtime shim's session. `synchronizing` reports
// a latched failure of an earlier fire-and-forget call instead of this call's
// own result, as CUDA reports asynchronous errors.
cudart::Reply Call(uint32_t api_id, const cudart::ArgWriter& in, bool synchronizing = false);

// Fire-and-forget (batched, in order with every other call).
int Post(uint32_t api_id, const cudart::ArgWriter& in);

// The CUresult for a reply: a lost session is CUDA_ERROR_UNKNOWN; "no server"
// is already CUDA_ERROR_NO_DEVICE (the runtime's code has the same value).
int Finish(const cudart::Reply& reply);

// Generated stubs for libcuda exports Tuna does not remote: log once (with
// TUNA_LOG=1) and return CUDA_ERROR_NOT_SUPPORTED.
int Unsupported(const char* name);

// Drops the client's cached driver state (tests, between in-process sessions).
void ResetStateForTesting();

}  // namespace tuna::cudrv

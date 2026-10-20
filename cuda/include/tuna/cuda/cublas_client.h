#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "tuna/cuda/marshal.h"

// Tuna's replacement libcublas (Phase 3d). Every cuBLAS call is marshalled
// like a runtime call and sent through the runtime shim's session
// (tuna_cudart_call, tuna/cuda/bridge.h), so it runs on the server in the same
// CUDA context as the app's cudaMalloc'd buffers and streams.
//
// This header includes no CUDA header: the generated not-supported stubs are
// compiled without prototypes (see cublas_unsupported.cpp). Status codes are
// therefore plain ints here; the generated stubs cast them.

#if defined(_WIN32)
#define TUNA_CUBLAS_EXPORT extern "C" __declspec(dllexport)
#else
#define TUNA_CUBLAS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace tuna::cublas {

// cublasStatus_t values the shim produces itself.
constexpr int kStatusSuccess = 0;
constexpr int kStatusNotInitialized = 1;
constexpr int kStatusInvalidValue = 7;
constexpr int kStatusInternalError = 14;
constexpr int kStatusNotSupported = 15;

// A blocking call through the runtime shim's session.
cudart::Reply Call(uint32_t api_id, const cudart::ArgWriter& in, const std::string& bulk_in = {},
                   bool synchronizing = false);

// Fire-and-forget (batched, in order with every other call). Returns
// kStatusSuccess once queued; a server-side failure surfaces as a cudaError_t
// at the app's next synchronizing CUDA call.
int Post(uint32_t api_id, const cudart::ArgWriter& in, const std::string& bulk_in = {});

// The cublasStatus_t for a reply. Connection-level codes from the runtime
// side (no server, an old server without cuBLAS) become cuBLAS codes.
int Finish(const cudart::Reply& reply);

// Scalars passed by pointer follow the handle's pointer mode (tracked from
// cublasSetPointerMode_v2): host mode copies `size` bytes of the value, device
// mode sends the address. False if `value` is null or `size` is not a valid
// scalar width.
bool PutScalar(cudart::ArgWriter& in, const void* handle, const void* value, size_t size);

// An OUT scalar (cublasSdot's result): in device mode its address is sent and
// the server writes it on the device; in host mode the reply carries it.
// Returns true for device mode.
bool PutResult(cudart::ArgWriter& in, const void* handle, void* result);

// Width of alpha/beta for the cublas*Ex functions, which take them as
// `const void*` typed by the compute type, complex when C is (cuBLAS docs,
// "cublasGemmEx"). 0 for an unknown compute type (the call then fails
// INVALID_VALUE).
size_t ComputeScalarSize(int compute_type, int c_type);

// Generated stubs for libcublas exports Tuna does not remote: log once (with
// TUNA_LOG=1) and return CUBLAS_STATUS_NOT_SUPPORTED.
int Unsupported(const char* name);

// Drops the client's per-handle state (tests, between in-process sessions).
void ResetStateForTesting();

}  // namespace tuna::cublas

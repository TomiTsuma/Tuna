#pragma once

#include <cstdint>

// Tuna's replacement libcublasLt (Phase 3f). Its calls are marshalled like
// the runtime's and sent through the runtime shim's session
// (tuna_cudart_call, tuna/cuda/bridge.h), in order with the app's runtime and
// cuBLAS calls; see cuda/src/cublaslt_client.cpp for the descriptor ids.
//
// This header includes no CUDA header: the generated not-supported stubs are
// compiled without prototypes (see cublaslt_unsupported.cpp).

#if defined(_WIN32)
#define TUNA_CUBLASLT_EXPORT extern "C" __declspec(dllexport)
#else
#define TUNA_CUBLASLT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace tuna::cublaslt {

// Largest attribute value a descriptor setter forwards. cuBLASLt's attributes
// are scalars, pointers and small enums; this only bounds the request.
constexpr uint64_t kMaxAttributeBytes = 256;

// Most heuristic results one query returns (cuBLASLt itself returns a few).
constexpr int kMaxHeuristicResults = 64;

// Generated stubs for libcublasLt exports Tuna does not remote: log once (with
// TUNA_LOG=1) and return CUBLAS_STATUS_NOT_SUPPORTED.
int Unsupported(const char* name);

// Drops the client's descriptor state (tests, between in-process sessions).
void ResetStateForTesting();

}  // namespace tuna::cublaslt

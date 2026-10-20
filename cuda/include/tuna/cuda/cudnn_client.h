#pragma once

#include <cstddef>
#include <cstdint>

// Tuna's replacement libcudnn (Phase 3g). Its calls are marshalled like the
// runtime's and sent through the runtime shim's session (tuna_cudart_call,
// tuna/cuda/bridge.h), in order with the app's runtime, cuBLAS and cuBLASLt
// calls; see cuda/src/cudnn_client.cpp for the descriptor ids.
//
// This header includes no cuDNN header: the generated not-supported stubs are
// compiled without prototypes (see cudnn_unsupported.cpp).

#if defined(_WIN32)
#define TUNA_CUDNN_EXPORT extern "C" __declspec(dllexport)
#else
#define TUNA_CUDNN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace tuna::cudnn {

// Most elements one backend attribute call carries (set or get). cuDNN's
// attributes are small arrays (tensor dims, a graph's operations, the
// engine configs a heuristic returns); this only bounds the request.
constexpr int64_t kMaxAttributeElements = 65536;
// Most bytes of attribute values one call carries (CHAR attributes, such as
// an execution plan's JSON, are the large ones).
constexpr uint64_t kMaxAttributeBytes = 1u << 20;
// Longest cudnnGetLastErrorString message the server returns.
constexpr uint64_t kMaxErrorStringBytes = 64u << 10;
// Distinct batch-norm workspace/reserve size queries the client remembers.
constexpr size_t kMaxCachedSizes = 4096;

// cudnnBackendAttributeType_t values whose elements are pointers, which
// cross the wire as u64 and get special treatment on the server.
constexpr uint32_t kTypeHandle = 0;       // CUDNN_TYPE_HANDLE: a cudnnHandle_t
constexpr uint32_t kTypeVoidPtr = 6;      // CUDNN_TYPE_VOID_PTR: a device address
constexpr uint32_t kTypeDescriptor = 15;  // CUDNN_TYPE_BACKEND_DESCRIPTOR

// Bytes one backend attribute element takes in memory (and on the wire), or 0
// for an attribute type Tuna does not know (refused, fail closed). Shared by
// client and server so both agree on every length. Numbered as in cuDNN 9's
// cudnnBackendAttributeType_t; enums are C ints, BOOLEAN is a C++ bool.
inline int64_t BackendElementSize(uint32_t attribute_type) {
    switch (attribute_type) {
    case kTypeHandle:
    case 3:  // INT64
    case 5:  // DOUBLE
    case kTypeVoidPtr:
    case kTypeDescriptor: return 8;
    case 2:   // BOOLEAN
    case 24:  // CHAR
        return 1;
    case 26: return 16;  // FRACTION: {int64 numerator, denominator}
    case 1:              // DATA_TYPE
    case 4:              // FLOAT
    case 7: case 8: case 9: case 10: case 11: case 12: case 13: case 14:  // CONVOLUTION_MODE .. POINTWISE_MODE
    case 16: case 17: case 18: case 19: case 20: case 21: case 22:        // GENSTATS_MODE .. PADDING_MODE
    case 23:                                                              // INT32
    case 25: case 27: case 28: case 29:  // SIGNAL_MODE, NORM_MODE, NORM_FWD_PHASE, RNG_DISTRIBUTION
        return 4;
    default: return 0;
    }
}

// Generated stubs for libcudnn exports Tuna does not remote: log once (with
// TUNA_LOG=1) and return CUDNN_STATUS_NOT_SUPPORTED.
int Unsupported(const char* name);

// Drops the client's descriptor state and caches (tests, between in-process
// sessions).
void ResetStateForTesting();

}  // namespace tuna::cudnn

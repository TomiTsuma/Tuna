// Every libcudart export Tuna does not implement yet, as a stub returning
// cudaErrorNotSupported (801). Exporting the full ABI means any program that
// links the runtime loads against the shim; only the calls it actually makes
// need to be remoted. Generated from tools/codegen/cudart12_exports.txt.
//
// This file deliberately includes no CUDA header: the stubs are declared
// without parameters, which the C calling convention allows (arguments are
// ignored), and every one of them returns cudaError_t (an int).

#include "tuna/cuda/client.h"

#include "cuda_unsupported_stubs.inc"

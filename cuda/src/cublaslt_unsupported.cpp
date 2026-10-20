// Every libcublasLt export Tuna does not implement yet, as a stub returning
// CUBLAS_STATUS_NOT_SUPPORTED (15). Exporting the full ABI means any program
// or library linked against cuBLASLt loads against the shim; only the calls
// it actually makes need to be remoted. Generated from
// tools/codegen/cublasLt12_exports.txt.
//
// Like client_unsupported.cpp, this file includes no CUDA header: the stubs
// are declared without parameters (the C calling convention ignores unused
// arguments).

#include "tuna/cuda/cublaslt_client.h"

#include "cublaslt_unsupported_stubs.inc"

// Every libcublas export Tuna does not implement yet, as a stub returning
// CUBLAS_STATUS_NOT_SUPPORTED (15). Exporting the full ABI means any program
// linked against cuBLAS loads against the shim; only the calls it actually
// makes need to be remoted. Generated from tools/codegen/cublas12_exports.txt.
//
// Like client_unsupported.cpp, this file includes no cuBLAS header: the stubs
// are declared without parameters (the C calling convention ignores unused
// arguments). Legacy-API functions that return void or a value other than a
// status are among them; calling one is unsupported either way.

#include "tuna/cuda/cublas_client.h"

#include "cublas_unsupported_stubs.inc"

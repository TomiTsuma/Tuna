// Every libcuda export Tuna does not implement yet, as a stub returning
// CUDA_ERROR_NOT_SUPPORTED (801). Exporting the full ABI means any program or
// library linked against the driver loads against the shim; only the calls it
// actually makes need to be remoted. Generated from
// tools/codegen/cuda12_driver_exports.txt.
//
// Like client_unsupported.cpp, this file includes no CUDA header: the stubs
// are declared without parameters (the C calling convention ignores unused
// arguments), and every driver function returns CUresult (an int).

#include "tuna/cuda/driver_client.h"

#include "driver_unsupported_stubs.inc"

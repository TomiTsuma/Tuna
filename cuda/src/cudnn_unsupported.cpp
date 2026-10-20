// Every libcudnn export Tuna does not implement yet, as a stub returning
// CUDNN_STATUS_NOT_SUPPORTED (3000). Exporting the full ABI means any program
// or library linked against cuDNN loads against the shim; only the calls it
// actually makes need to be remoted. Generated from
// tools/codegen/cudnn9_exports.txt.
//
// Like client_unsupported.cpp, this file includes no CUDA header: the stubs
// are declared without parameters (the C calling convention ignores unused
// arguments).

#include "tuna/cuda/cudnn_client.h"

#include "cudnn_unsupported_stubs.inc"

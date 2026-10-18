#pragma once

// Shared between the client shim's translation units; not exported.

#include <cstddef>
#include <cstdint>

namespace tuna::cudart {

// Device allocations the client knows about (from cudaMalloc/cudaMallocPitch
// and symbol addresses), so cudaMemcpyDefault can tell device pointers (server
// addresses) from the client's own host memory.
void TrackDeviceRange(const void* base, size_t size);
void UntrackDeviceRange(const void* base);
bool IsDevicePointer(const void* ptr);

// Per-file pieces of ResetClientStateForTesting().
void ResetRuntimeState();
void ResetLaunchState();

}  // namespace tuna::cudart

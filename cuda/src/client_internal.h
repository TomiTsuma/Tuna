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

// What CUDA checks synchronously at launch: block and grid dimensions within
// the current device's limits (from its cached properties). 0, or
// cudaErrorInvalidConfiguration. Execution errors stay asynchronous.
int32_t ValidateLaunch(uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y,
                       uint32_t block_z);

// Per-file pieces of ResetClientStateForTesting().
void ResetRuntimeState();
void ResetLaunchState();

}  // namespace tuna::cudart

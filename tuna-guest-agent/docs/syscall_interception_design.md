# Tuna Guest Agent – Syscall Interception Design

## 1. Overview

The **Syscall Interception Layer** captures selected system and high-level API calls made by user applications (e.g., Photoshop, Blender, Python apps) and routes them to Tuna Server for remote execution. The interception layer aims for **transparency** (apps behave unchanged), **stability**, and **minimal added latency** while enabling remote GPU/CPU offload and transparent access to local files/devices.

This document describes the interception strategy, lifecycle, serialization model, failure handling, security controls, and — importantly — the complete set of **high-level APIs** that the Hooks will target.

---

## 2. Goals and Principles

* **Invisible to apps:** Interception must preserve ABI/semantics and error codes.
* **Safe & minimal kernel footprint:** Prefer user-space hooks; use kernel hooks only when necessary.
* **Selective:** Intercept only APIs that enable offload (compute, GPU, heavy I/O, device access).
* **Performant:** Trampolines, zero-copy buffers and batching to minimize overhead.
* **Auditable & controllable:** All intercepted operations logged and subject to user policy.

---

## 3. Interception Strategy (high level)

Tuna uses a hybrid strategy:

1. **User-space API hooking** (primary): inject hooks into target processes for high-level APIs using MinHook / Detours. This covers most graphics, compute and file APIs used by applications.
2. **Optional kernel-mode monitoring** (fallback/advanced): a signed WDK driver for cases where APIs are bypassed (rare). Kernel hooks are limited to observing and mediating operations and always hand off heavy work to user-mode daemon.
3. **Decorator / runtime hooks**: language/runtime-level integration (e.g., the `@tuna_remote` decorator for Python) to catch high-level function calls.

User-space hooking is the default because it is safer, easier to maintain, and avoids system instability that comes with indiscriminate kernel patching.

---

## 4. Architectural Layers (where hooks live)

```
[Application]
  └─ high-level libs (Photoshop UXP, Python runtime)
     └─ Graphics/Compute APIs (Direct3D / OpenGL / Vulkan / CUDA / OpenCL)
        └─ Win32 / CRT / POSIX-like I/O APIs
           └─ Hook Manager (user-space trampolines)
              └─ Tunnel Manager → Tuna Server
```

---

## 5. Full list — High-Level APIs to be intercepted

**Note:** This is the canonical list of API entry points the Hook Manager will support initially. Grouped by category for clarity. Targets are chosen to cover compute (GPU/accelerated), large I/O, memory mapping and device control. Where multiple variants exist (e.g., ANSI/Unicode, runtime wrappers) both variants are included.

### 5.1 File & Filesystem I/O

* Win32 API:

  * `CreateFileW`, `CreateFileA`
  * `ReadFile`, `WriteFile`
  * `ReadFileEx`, `WriteFileEx`
  * `SetFilePointerEx`, `SetEndOfFile`
  * `GetFileSizeEx`, `GetFileAttributesEx`
  * `FlushFileBuffers`
  * `LockFile`, `LockFileEx`, `UnlockFile`, `UnlockFileEx`
  * `DeleteFileW`, `MoveFileW`
  * `GetFileAttributesW`, `SetFileAttributesW`
  * `CreateFileMappingW`, `MapViewOfFile`, `UnmapViewOfFile`
  * `DeviceIoControl` (for specific device IOCTLs)
* NT Native APIs (when needed / observed):

  * `NtCreateFile`, `NtReadFile`, `NtWriteFile`, `NtQueryInformationFile`
* C Runtime / POSIX wrappers:

  * `_open`, `_read`, `_write`, `_close`
  * `fopen` / `fread` / `fwrite` / `fclose` (CRT)
  * `open`, `read`, `write` (POSIX-style on ported apps)
* High-level frameworks:

  * .NET `System.IO.File`, `FileStream.Read`, `FileStream.Write` (hooked via CLR profiler or interop when necessary)

### 5.2 Memory & Mapping APIs

* Memory allocation / mapping:

  * `VirtualAlloc`, `VirtualFree`, `VirtualProtect`
  * `CreateFileMappingW`, `MapViewOfFile`, `UnmapViewOfFile`
  * `NtMapViewOfSection`, `NtUnmapViewOfSection`
* Memory-copy helpers relevant to large buffers:

  * `RtlCopyMemory`, `memcpy` (intercepting large memcpy patterns is optional/heuristic)

### 5.3 Graphics APIs (Desktop & GPU)

* Direct3D / DXGI:

  * `D3D11CreateDevice`, `D3D11CreateDeviceAndSwapChain`
  * `ID3D11Device::Create*` family (textures, buffers)
  * `ID3D11DeviceContext::Map/Unmap`, `UpdateSubresource`, `Draw`, `DrawIndexed`
  * `IDXGISwapChain::Present`, `IDXGISwapChain::Present1`
  * D3D12 entrypoints (via D3D12CreateDevice and command queue submission)
* DirectX (older):

  * `Direct3DCreate9`, `IDirect3DDevice9::Present`
* OpenGL:

  * `glDrawElements`, `glDrawArrays`
  * `glBufferData`, `glBufferSubData`, `glTexImage2D`, `glTexSubImage2D`
  * `wglMakeCurrent`, context creation points
* Vulkan:

  * `vkCreateDevice`, `vkQueueSubmit`, `vkCmdDraw`, `vkCmdDispatch`
  * `vkMapMemory`, `vkUnmapMemory`, `vkCmdCopyBuffer`
* DXGI / Swapchain hooks (for frame capture/encode):

  * `IDXGIOutputDuplication` flows where available
* GPU vendor APIs (examples):

  * `NvAPI_*` hooks where applicable
  * AMD Radeon API entrypoints if used by applications

### 5.4 GPU Compute APIs (CUDA / OpenCL / ROCm)

* CUDA (Runtime & Driver APIs):

  * `cudaMalloc`, `cudaFree`
  * `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemcpy2D`
  * `cudaLaunchKernel` / `cuLaunchKernel`
  * `cuModuleLoad`, `cuModuleGetFunction`, memory management calls
  * `cudaStreamCreate`, `cudaStreamSynchronize`
* OpenCL:

  * `clCreateContext`, `clCreateCommandQueue` / `clCreateCommandQueueWithProperties`
  * `clEnqueueNDRangeKernel`, `clEnqueueReadBuffer`, `clEnqueueWriteBuffer`
  * `clFinish`, `clFlush`
* ROCm (if targeted later): analogous entrypoints

### 5.5 Compute / Math Libraries (high-level)

* Common compute-heavy library entrypoints (hook or provide integration adapters):

  * BLAS/LAPACK entrypoints if linked directly (e.g., `cblas_dgemm`)
  * TensorFlow/PyTorch ops via runtime hooks or dedicated integration (prefer runtime decorator approach)

### 5.6 Network & Sockets (selective)

* Low-level Winsock:

  * `socket`, `connect`, `accept`
  * `send`, `recv`, `WSASend`, `WSARecv`
  * `sendto`, `recvfrom`
* Note: Network calls are intercepted only when needed for application consistency (e.g., local server calls), otherwise leave to normal network stack.

### 5.7 Device & USB

* USB/device control transfers:

  * `WinUSB` / `SetupDi*` enumeration flows (when proxying USB devices)
  * `DeviceIoControl` for device-specific ioctls relevant to graphics tablets, cameras, or storage devices

### 5.8 Process / Thread / Execution

* Process and thread creation where relevant to execution context:

  * `CreateProcessW`, `CreateProcessA`, `CreateRemoteThread` (for context diagnosis or to ensure correct injection)
  * Synchronization primitives: `WaitForSingleObject`, `WaitForMultipleObjects`, `CreateMutex`, `OpenMutex` may be observed for ordering semantics

### 5.9 High-level Language Runtimes & Decorators

* Python:

  * `@tuna_remote` decorator interception (via runtime wrapper)
  * `pickle` serialization hooks for large objects (opt-in)
* Java/.NET:

  * Provide runtime integration points (CLR profiler API for .NET `FileStream`, Java JVMTI agents) if required later

---

## 6. Hooking Lifecycle & Process Scope

* **Registration**: on daemon start, Hook Manager loads `hook_targets.json` and registers trampolines for whitelisted processes (process whitelist/blacklist enforced).
* **Injection**: when a target process starts, the agent injects hook DLL into that process (via CreateRemoteThread / AppInit in controlled fashion) only if consented.
* **Activation**: hooks are active for the process lifetime or until daemon requests unhook/uninject.
* **Fallback**: if a hook fails (hook conflict or crash risk), Hook Manager disables that hook for the process and logs/alerts.

---

## 7. Context Object & Serialization

For every intercepted call we build a **Context Object** containing:

* Call metadata: process name, PID, thread ID, timestamp, function name.
* Parameters: copies of primitive params and references to large buffers (with memory ranges).
* Memory handles: descriptors or mdls for large buffers (zero-copy where possible).
* Semantics flags: sync/async, blocking behavior, required return semantics.

**Serialization**

* Use Protocol Buffers (protobuf) for structured fields and a compact wire format.
* Large binary payloads use chunked transfer with checksums (xxHash or SHA256).
* All payloads encrypted via TLS (QUIC) tunnel.

---

## 8. Result Reinjection & Semantics Preservation

* Results returned from server are validated and injected back into the process:

  * For primitive return values: direct substitution.
  * For buffer results: memory mapping into process address space (via WriteProcessMemory / mapped MDL path).
  * For asynchronous operations: complete/trigger callbacks as the native API expects.
* Preserve error codes: set `GetLastError()` appropriately on failure paths.

---

## 9. Performance Optimizations

* **Batching:** coalesce many small operations (stat metadata calls) into a single round-trip.
* **Prefetch & read-ahead:** for sequential reads, request future chunks proactively.
* **Zero-copy path:** use MDLs and shared memory for large buffer transfer to avoid extra copies between kernel/user/tunnel.
* **Adaptive policy:** small, latency-sensitive calls (e.g., tiny reads) may run locally to maintain responsiveness; bigger ops are offloaded.

---

## 10. Security & Integrity

* All hookable modules and trampolines are code-signed; agent verifies integrity at load.
* Mutual TLS + server certificate validation for the tunnel.
* Per-call authorization: each intercepted request is checked against local policy (user consent, ACL, process whitelist).
* Tamper logs: append-only audit records of all intercepted calls (md5/sha256 of payloads where allowed).

---

## 11. Debugging, Diagnostics and Safe Modes

* **Dry-run mode:** hook but do not forward — useful for validating semantics.
* **Verbose logging:** detailed per-call logs written to `C:\ProgramData\Tuna\logs\hooks.log` with rotation.
* **Safe mode:** emergency unhook and switch to local-only execution if instability detected.
* **Testing harness:** unit tests for trampoline correctness, integration tests with sample GPU workloads.

---

## 12. Failure Modes & Recovery

* **Server unreachable:** retry policy with exponential backoff; fallback to local execution for that call type.
* **Serialization error:** retry once, then fallback; log and mark hook disabled if repeated failures.
* **Hook conflict or crash in target process:** unhook and mark process as unsupported; notify user via GUI.
* **Kernel driver fault (if used):** immediate disable kernel hook paths and switch to user-space only.

---

## 13. Implementation Roadmap & Checklist

**Phase 1 – Foundations**

* [ ] Finalize `hook_targets.json` (canonical API list).
* [ ] Implement Hook Manager with MinHook for user-space APIs.
* [ ] Implement Context Object protobuf schema and basic tunnel forwarding.

**Phase 2 – Core API Support**

* [ ] File I/O APIs (Win32 & CRT wrappers).
* [ ] Memory mapping APIs.
* [ ] Basic Direct3D / DXGI hooks (Present, CreateDevice).
* [ ] CUDA / OpenCL kernel entrypoint hooks (cudaLaunchKernel, clEnqueueNDRangeKernel).

**Phase 3 – Robustness & Performance**

* [ ] Zero-copy MDL path for large buffer transfers.
* [ ] Batching, prefetch, and adaptive local/remote policy.
* [ ] Integration tests with Photoshop and CUDA workloads.

**Phase 4 – Optional Kernel Extension**

* [ ] WDK driver for low-level syscall coverage (signed and restricted).
* [ ] Driver verifier testing and broad OS compatibility checks.

**Phase 5 – Production Harden**

* [ ] Full security audit, code signing, telemetry, and rollback.
* [ ] Extended hooks for Vulkan, advanced D3D12 flows, and runtime language integrations.

---

**End of Document**

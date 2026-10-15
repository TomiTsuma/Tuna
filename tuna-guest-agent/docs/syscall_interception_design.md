# Windows API Forwarding and Interception Scope

## Current implementation boundary

The initial supported application is a purpose-built Windows sample app that calls the Tuna client library's `RemoteSumOfSquares` API. The function deliberately marks a supported remote-work boundary in application code. It sends only a versioned, bounded list of integers through the gRPC client to the Linux server.

This is **not** transparent interception of arbitrary applications and does not currently hook Win32, NT, Direct3D, Vulkan, CUDA, OpenCL, filesystem, process, memory, or network APIs. Existing design notes listing broad hook targets are future research, not implemented behavior or compatibility promises.

## Why the first boundary is explicit

An arbitrary Windows API call cannot generally be replayed on Linux by serializing its function name and arguments. APIs depend on operating-system object handles, thread-local state, callbacks, device/runtime versions, pointer ownership, asynchronous completion behavior, and ABI-specific semantics. GPU APIs also depend on driver/runtime state and device-specific command execution. Passing raw pointers, process memory, or arbitrary executable code to the server is not a safe or portable contract.

The initial custom API makes data, result types, limits, authorization, error behavior, and version compatibility explicit. Further APIs should be added as independently specified adapters with a clear supported application/runtime matrix.

## Current API contract

- The API is declared by the generated `tuna.v1.WorkloadService` client stub and wrapped by `tuna::client::WorkloadClient`.
- `RemoteSumOfSquares` is deterministic, bounded to 4,096 values, checks 64-bit overflow, and returns request/version metadata.
- The sample client requires mTLS credentials and uses a finite deadline.
- Remote failures are surfaced; they are not replaced with locally generated success results.

## Future interception gate

Do not add process injection, a kernel driver, or general API hooks until all of the following exist:

1. A precise supported API/application/runtime/OS matrix and measurable semantic-compatibility tests.
2. A threat model covering injection, privilege boundaries, handles, memory buffers, callbacks, malformed server replies, and compromised/unauthorized servers.
3. User/administrator consent and an explicit per-process/per-workload policy.
4. A safe local fallback or fail-closed rule defined for each operation, plus emergency unhook/recovery and rollback.
5. A protocol adapter that never exposes raw local pointers, arbitrary code execution, or unrestricted local filesystem/device access.
6. Integration, fault-injection, performance, and security tests before enabling the hook by default.

## GPU-specific work

The intended initial server hardware is an NVIDIA RTX 3060 12 GB, but the current sample service does not use it. Any CUDA or graphics offload must specify exact supported Windows API/runtime and Linux driver/CUDA versions, data-transfer semantics, device allocation and isolation, multi-client quotas, failure behavior, and application compatibility. A `cudaLaunchKernel`-style call cannot be considered supported merely because its name appears in an API list.

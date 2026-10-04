# Tuna Client/Server Architecture

## Product boundary

The Windows client and Linux server are separate platform-specific programs that share a versioned protobuf/gRPC contract. The Windows desktop GUI is a management surface; it must not be treated as the transport or execution engine. The Linux server hosts explicitly supported workloads and must not receive arbitrary Windows process memory or system calls.

The first milestone is deliberately narrow: a Windows sample application calls either the `RemoteSumOfSquares` connectivity smoke test or the explicit `RemoteMatrixMultiply` GPU API. The client authenticates to the Linux server with mutual TLS. The server validates certificate identity and request bounds; matrix multiplication executes on CUDA device 0 with no CPU fallback. This is API-level forwarding for purpose-built sample requests, not interception of an unmodified third-party application.

## Current component boundaries

```text
Windows 10/11 x64                         Ubuntu 22.04/24.04 x64
+----------------------------+            +----------------------------+
| Sample application         |            | gRPC service               |
|   custom Tuna API          |-- mTLS ---->|   SAN authorization        |
|   gRPC client library      |  HTTP/2     |   input limits / workload  |
|   Qt GUI prototype         |            |   checked result handling  |
+----------------------------+            +----------------------------+
             \________ shared protocol/tuna/v1/workload.proto ________/
```

The sample service uses administrator-managed certificates. The client validates the server trust chain and DNS name. The server validates client certificates and permits the configured subject alternative name. TLS 1.3 is the production minimum; actual minimum-version enforcement remains a release gate for the selected gRPC/TLS build.

## Source layout

- `client/`: Windows C++ gRPC API and sample command-line application.
- `src/gui/`: Qt UI for per-user settings, the bounded sample workload, and in-session activity. It launches the RPC CLI asynchronously; no persistent session, process forwarding, or live telemetry is implemented.
- `protocol/tuna/v1/`: versioned protobuf contract shared by client and server.
- `tuna-server/`: independently configured Linux CMake service and unit tests.
- `docs/`: product plan, protocol, security, interception limitations, and lifecycle design.

## First workload semantics

`RemoteSumOfSquares` accepts at most 4,096 unsigned 64-bit values and returns their checked sum of squares. `RemoteMatrixMultiply` accepts two finite row-major float32 matrices of dimension 128–512 and returns their product from the GPU. The service targets CUDA 12.8 and compute capability 8.6 (RTX 3060), serializes device use by rejecting concurrent requests, and refuses to start in a CUDA-enabled build without a compatible GPU. CPU-only builds remain useful for protocol development but return `UNAVAILABLE` for matrix multiplication and do not satisfy the GPU release requirement.

The sample CLI creates deterministic test matrices; this is not an application offload SDK or transparent acceleration path. GPU performance and production multi-tenant isolation have not yet been qualified.

## Future components — not yet implemented

- GUI-to-client configuration/status integration and a least-privilege local IPC contract.
- Windows background service, installer, signing, upgrade, and recovery.
- Linux account provisioning, production packaging/update automation, firewall, workload quotas, logs/metrics, and deployment automation. A systemd unit template and basic CMake install rules now exist.
- Certificate enrollment/rotation/revocation and identity administration.
- Explicit additional workload adapters and workload-level resource isolation.
- Physical RTX 3060 validation for numerical correctness, capacity, performance, thermal behavior, driver compatibility, and CUDA fault recovery.
- Any process injection or OS/graphics/file/network API interception. Such work requires a separate threat model, supported-app matrix, compatibility criteria, and safe rollback design.

## Compatibility and operations

The `tuna.v1` protobuf package is the compatibility boundary. Additive optional fields can be added within v1; semantic or incompatible changes require a new major package/service version. Both sides must generate code from the checked-in schema and run contract tests. Deployment, protocol compatibility, certificate changes, server restarts, and client/server version skew must be tested before broad workload support.

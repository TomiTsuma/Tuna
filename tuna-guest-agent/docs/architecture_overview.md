# Tuna Client/Server Architecture

## Product boundary

The Windows client and Linux server are separate platform-specific programs that share a versioned protobuf/gRPC contract. The Windows desktop GUI is a management surface; it must not be treated as the transport or execution engine. The Linux server hosts explicitly supported workloads and must not receive arbitrary Windows process memory or system calls.

The first milestone is deliberately narrow: a Windows sample application calls a custom `RemoteSumOfSquares` API. The client library authenticates to the server with mutual TLS and forwards the bounded request. The Linux service validates the certificate identity and request, computes the result, and returns it. This is API-level forwarding for a purpose-built sample, not interception of an unmodified third-party application.

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
- `src/gui/`: Qt UI. Settings are stored per user and the sample workload runs the RPC CLI asynchronously; no persistent session is maintained.
- `protocol/tuna/v1/`: versioned protobuf contract shared by client and server.
- `tuna-server/`: independently configured Linux CMake service and unit tests.
- `docs/`: product plan, protocol, security, interception limitations, and lifecycle design.

## First workload semantics

`RemoteSumOfSquares` accepts a non-empty list of at most 4,096 unsigned 64-bit values and returns the checked sum of their squares. Unsupported versions, invalid IDs, oversized batches, unauthorized client identities, and arithmetic overflow fail with explicit gRPC status codes. The service does not persist request data. The sample client has a finite RPC deadline and does not fall back to local computation when remote work fails.

The example workload is CPU-only. The intended initial server has an NVIDIA RTX 3060 12 GB, but no CUDA code, GPU scheduling, or GPU isolation is implemented by this slice.

## Future components — not yet implemented

- GUI-to-client configuration/status integration and a least-privilege local IPC contract.
- Windows background service, installer, signing, upgrade, and recovery.
- Linux account provisioning, production packaging/update automation, firewall, workload quotas, logs/metrics, and deployment automation. A systemd unit template and basic CMake install rules now exist.
- Certificate enrollment/rotation/revocation and identity administration.
- Explicit additional workload adapters and workload-level resource isolation.
- GPU adapters for a specified CUDA/runtime/hardware matrix.
- Any process injection or OS/graphics/file/network API interception. Such work requires a separate threat model, supported-app matrix, compatibility criteria, and safe rollback design.

## Compatibility and operations

The `tuna.v1` protobuf package is the compatibility boundary. Additive optional fields can be added within v1; semantic or incompatible changes require a new major package/service version. Both sides must generate code from the checked-in schema and run contract tests. Deployment, protocol compatibility, certificate changes, server restarts, and client/server version skew must be tested before broad workload support.

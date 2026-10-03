# Tuna Client/Server RPC Contract

## Status and scope

This is the authoritative transport decision for the first vertical slice: **gRPC over TLS with mutual certificate authentication**, using Protocol Buffers. It replaces the earlier speculative QUIC/TTP frame design for the MVP. The transport is an RPC channel; the custom application API is the only workload boundary supported initially.

The first RPC is `RemoteSumOfSquares`, used by the sample Windows application. It accepts a bounded list of unsigned 64-bit integers and returns the checked sum of their squares. It is a protocol smoke-test workload, not a GPU benchmark or general compute-execution interface. The first implementation does not intercept arbitrary Win32, graphics, filesystem, network, or third-party application APIs.

## Platform and identity

- Client: Windows 10/11 x64.
- Server: Ubuntu 22.04/24.04 x64. The intended deployment host has an NVIDIA RTX 3060 12 GB, but this CPU-only RPC does not use the GPU.
- Both peers authenticate certificates issued by administrator-managed trust roots. The server requires and verifies a client certificate; the client verifies the server chain and hostname.
- Production policy requires TLS 1.3 or newer. The gRPC credential setup must be tested against the actual gRPC/TLS build to prove that older protocol versions are rejected before release; capability to negotiate TLS 1.3 alone is not proof of a TLS 1.3 minimum.
- Certificate issuance, rotation, revocation, and recovery are administrator responsibilities in the MVP and must be documented before production deployment.

## Contract and limits

The versioned schema lives in `protocol/tuna/v1/workload.proto`. The initial request carries a protocol version, caller-generated request ID, and at most 4,096 values. The server rejects empty IDs, unsupported versions, oversized batches, and any arithmetic overflow. The response echoes protocol version and request ID with the result. The RPC is deterministic and safe to retry, but automatic retries are not enabled by the initial sample client.

The schema deliberately omits generic syscall payloads, arbitrary code, GPU command buffers, file access, remote process execution, compression, session resumption, and custom frame checksums. The server and client cap received gRPC messages at 64 KiB and sent responses at 1 KiB; the defined request fits within those bounds. Protobuf/gRPC framing and TLS provide the transport encoding and confidentiality/integrity; application-level authorization and resource controls remain necessary.

## Connection/error behavior

- Use a configured server DNS name and port; never disable certificate or hostname verification.
- The client fails closed if the CA, client certificate, or private key is absent, unreadable, invalid, or mismatched.
- RPC deadlines bound waiting time. A transport or server failure is surfaced as an error, not a locally computed success.
- No server-side workload data is persisted. Logs must not include submitted values or private key material.
- A valid client certificate chains to the configured client CA. Production deployments must use a dedicated client CA or add a documented identity allowlist; trust of a general-purpose CA is not sufficient authorization.

## Compatibility policy

The protobuf package path includes the major version (`tuna.v1`). Additive optional fields are compatible within v1. Changes to semantics, required fields, units, or result interpretation require a new major package/service version. Client and server build/test jobs must compile the same checked-in schema and exercise shared conformance cases.

## Production gates

Before release, add tests proving TLS 1.3 minimum enforcement, hostname validation, client-certificate rejection, authorization policy, deadlines, bounded input, arithmetic overflow handling, and behavior across server restart/network loss. Add measured latency/throughput targets when the workload and deployment environment are finalized.

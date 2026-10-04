# Tuna Client/Server RPC Contract

## Status and scope

This is the authoritative transport decision for the first vertical slice: **gRPC over TLS with mutual certificate authentication**, using Protocol Buffers. It replaces the earlier speculative QUIC/TTP frame design for the MVP. The transport is an RPC channel; the custom application API is the only workload boundary supported initially.

The API contains `RemoteSumOfSquares`, a bounded CPU connectivity smoke test, and `RemoteMatrixMultiply`, a bounded CUDA workload. The GPU RPC accepts two row-major float32 square matrices with dimensions 128–512 and returns their product. This is an explicit custom API, not a general compute-execution interface. The implementation does not intercept arbitrary Win32, graphics, filesystem, network, or third-party application APIs.

## Platform and identity

- Client: Windows 10/11 x64.
- Server: Ubuntu 22.04/24.04 x64. The first-release GPU baseline is an NVIDIA RTX 3060 12 GB (compute capability 8.6) with CUDA Toolkit 12.8 and a compatible NVIDIA driver. GPU workload verification must run on the target hardware; hosted CI currently does not include a GPU.
- Both peers authenticate certificates issued by administrator-managed trust roots. The server requires and verifies a client certificate; the client verifies the server chain and hostname.
- Production policy requires TLS 1.3 or newer. The gRPC credential setup must be tested against the actual gRPC/TLS build to prove that older protocol versions are rejected before release; capability to negotiate TLS 1.3 alone is not proof of a TLS 1.3 minimum.
- Certificate issuance, rotation, revocation, and recovery are administrator responsibilities in the MVP and must be documented before production deployment.

## Contract and limits

The versioned schema lives in `protocol/tuna/v1/workload.proto`. The sum-of-squares request carries a protocol version, caller-generated request ID, and at most 4,096 values. The server rejects empty IDs, unsupported versions, oversized batches, and arithmetic overflow. The matrix request carries a protocol version, request ID, dimension, and exactly two `N×N` finite float32 matrices where `128 ≤ N ≤ 512`. The server computes on CUDA device 0, rejects simultaneous GPU submissions with `RESOURCE_EXHAUSTED`, and does not fall back to CPU. The response echoes the version, ID, and dimension with exactly `N×N` finite result values. Automatic retries are not enabled.

The schema deliberately omits generic syscall payloads, arbitrary code, raw GPU command buffers, file access, remote process execution, compression, session resumption, and custom frame checksums. Client and server send/receive messages are capped at 4 MiB to accommodate the bounded matrix payload and response. Protobuf/gRPC framing and TLS provide transport encoding and confidentiality/integrity; application-level authorization and resource controls remain necessary.

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

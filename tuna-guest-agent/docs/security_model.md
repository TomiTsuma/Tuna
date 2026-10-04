# Tuna Client/Server Security Model

## Scope and trust boundaries

Tuna consists of a Windows 10/11 x64 client and an Ubuntu 22.04/24.04 x64 server. The API provides a bounded CPU connectivity smoke test and a bounded CUDA matrix-multiplication request. It does not inject into arbitrary processes, intercept general Windows APIs, execute user-supplied code, proxy files, or accept raw GPU command buffers.

The trust boundaries are:

1. The sample application to the client library running as the logged-in user.
2. The Windows client to the Linux server over the network.
3. The Linux RPC endpoint to its workload implementation and host resources.
4. The administrator-managed certificate and configuration files to the client/server processes.

Treat all network requests as untrusted, even when they arrive over an authenticated channel.

## Authentication and transport

- Use gRPC TLS credentials with mutual certificate authentication. The server requires a client certificate chaining to its configured client trust root and matches an explicitly configured SAN value (for a DNS SAN, the value omits the `DNS:` type prefix); the client validates the server chain and endpoint hostname.
- Production policy is TLS 1.3 or newer. The initial gRPC implementation must be verified to enforce that minimum rather than merely support TLS 1.3; do not release if the selected gRPC/TLS build can negotiate an older version.
- Keep server and client trust roots dedicated to Tuna. Restrict which client identities are authorized; a certificate that chains successfully is not automatically authorized for every deployment.
- Private keys must be provisioned out of band, readable only by the intended account, excluded from source control and logs, and rotated/revoked through an administrator procedure. Do not accept a certificate path as proof of safe permissions; validate deployment ACLs.
- Never add an insecure transport fallback, skip hostname verification, or silently continue after credential errors.

## Request validation and resource limits

- The sum-of-squares smoke test accepts at most 4,096 unsigned 64-bit values. The matrix API accepts only dimensions 128–512 and exactly two finite float32 square matrices. Reject unsupported versions, empty IDs, malformed lengths, and non-finite inputs/results.
- Client/server gRPC message limits are 4 MiB to accommodate the bounded matrix input and result. Service-side validation also bounds dimensions and permits only one active GPU request; extra concurrent GPU work is rejected.
- The CUDA build requires compute capability 8.6+ and uses GPU device 0 without a CPU fallback. The systemd unit grants only the selected NVIDIA device nodes, but device access and behavior need validation on real Ubuntu hosts and target hardware.
- Do not persist request values or results on the server. Operational logs may include request IDs and outcome/duration, but not payload values, certificates, private keys, or other secrets.
- No arbitrary command, executable, file path, pointer, process ID, or client memory reference is accepted or deserialized.

## Host isolation and operations

- Run the Linux service as a dedicated unprivileged account; expose only the required listener and certificate files.
- Restrict inbound firewall access to the required RPC port and intended client networks.
- Do not install kernel hooks or drivers for the sample API.
- Keep credentials and production configuration out of tests and repository history. Use disposable test certificates and rotate them after test exposure.
- Publish certificate creation/rotation/revocation, deployment, backup (if later needed), incident response, and data-retention procedures before production.

## Known gaps / release blockers

- TLS minimum-version enforcement and hostname-verification behavior need executable integration tests with the chosen gRPC/TLS dependency build.
- Client identity allowlisting, certificate revocation/rotation, and key-protection procedures are not yet implemented.
- Server concurrency quotas, service sandboxing, packaging, firewall guidance, signed Windows installer/update flow, and independent security review remain required.
- No general syscall/GPU interception is implemented or represented as safe by this model. The explicit matrix API is the only GPU workload.

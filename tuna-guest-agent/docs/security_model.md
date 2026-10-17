# Tuna Client/Server Security Model

This document covers two tracks. Everything up to [Track B](#track-b--cuda-driver-api-remoting-draft) describes **Track A**, the explicit-workload MVP that is implemented today. Track B is a draft for a design that is not built yet. It does not relax any Track A rule.

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

## Track B — CUDA driver-API remoting (draft)

*Status: partly implemented. Phase 2 (2026-10-16) delivered the transport, identity and process-isolation controls marked **[implemented]** below. Every other control is still a requirement for later phases. Track B ships only after Phase 6 (security hardening) is complete.*

### What changes compared with Track A

Track A accepts only bounded, typed requests. Track B makes the client's CUDA driver API calls run on the server: `cuModuleLoadData` and its siblings upload **client-supplied GPU code** (PTX, cubin, fatbins), and `cuLaunchKernel` runs it. **An authenticated Track B client can therefore run arbitrary code on the server's GPU, and can read and write any device memory its own context can reach.** That is the purpose of the feature. Every control below assumes this capability and contains it. None of them try to prevent it.

### Trust model

| Party | Trusted for | Not trusted for |
|---|---|---|
| Server operator | Issuing client identities, choosing quotas, running the host | n/a |
| Authenticated client | Using the GPU inside its own session and quota | Anything outside its own session: other sessions' memory, the host OS, the listener, files, the network |
| Network | Nothing | Everything. All traffic is mutually authenticated and encrypted. |
| Client-side launcher/shim (`tuna run`, `tuna_nvcuda.dll`) | Acting for the local user in the one process it was launched into | Elevated privileges. It is never installed system-wide. |

### Transport and identity

- **[implemented] TLS 1.3 with mutual authentication**, using the same rules as Track A. OpenSSL is configured with a TLS 1.3 minimum and no session tickets, and peer verification is required. A client must chain to the configured CA **and** carry a DNS/URI SAN that exactly matches an entry in `allowed_client_names`. An empty allowlist is a config error. The client verifies the server chain plus the server's host name or IP. EKU is enforced (serverAuth/clientAuth). Because TLS provides the encryption, the v1 frame protocol carries a plain `payload` field and adds no second encryption layer.
  - Tests reject: an untrusted CA, a trusted but non-allowlisted identity, a client presenting no certificate, a TLS 1.2-only client, a server-name mismatch, and a silent peer (handshake timeout).
- **[implemented] No insecure fallback.** Without a `tls` config section the server is plaintext, and it **refuses to bind any non-loopback address**. Plaintext exists only for local development and tests.
- **[implemented] Certificate tooling:** `tuna_server --gen-ca`, `--issue-client-cert`, `--issue-server-cert`. EC P-256 keys, a CA restricted to `pathlen:0`, 397-day leaf validity by default. It never overwrites a file, and on POSIX it creates keys with mode 0600. On Windows, key files inherit their directory's ACL, so operators must keep PKI directories private. Private keys never need to leave the machine they are generated on.
- **Bulk socket authorization [implemented].** A session's second (bulk) connection is routed by a random 128-bit session token. The token is delivered only inside the TLS-protected `Hello`, but it is sent in plaintext in the bulk connection's preamble. The token is therefore treated as a *routing hint, not a credential*. The worker accepts the bulk connection only if it passes mTLS as the same client identity as the session.
- The first frame in each direction is `Hello`. The server reports its protocol and driver version, and rejects incompatible clients before it creates a CUDA context.

### Isolation: one worker process per session

- **[implemented]** The listener only accepts connections and spawns a **separate worker process for each session** (`tuna_server --worker`, same binary), handing over the raw socket. The listener never creates a CUDA context, never parses `RpcRequest.args`, and does not even run TLS. It reads only a fixed 24-byte routing preamble.
  - *Deviation from the original draft, which had the listener authenticate before the handoff:* userspace TLS state cannot be moved between processes (OpenSSL cannot export a live session, and Windows has no kTLS). So the worker performs the handshake, which is sshd's pre-authentication privilege-separation model.
  - Containment for that: `max_sessions` bounds concurrent workers, including unauthenticated ones; at most 64 connections may be pending their preamble; preamble and handshake each time out after 5 s.
- **[implemented]** Workers never outlive the listener. The listener's end of each worker's IPC channel closes when the listener exits for any reason, and the worker then ends its session and exits (with a hard exit after 10 s if a call is stuck). On Windows, workers also run in a kill-on-close Job object. Spawned workers inherit only their IPC handles: Winsock sockets are inheritable by default, so the listening socket is excluded explicitly, and POSIX descriptors are close-on-exec.
- Each worker runs as a low-privilege service account with no write access outside a per-session scratch directory. It is sandboxed (Linux: systemd sandboxing, seccomp, device allowlist; Windows: restricted token / AppContainer, to be evaluated). It has no outbound network access.
- **[implemented, tested]** A crash or hang in a worker (for example from a malicious kernel, a driver fault, or a decoder bug) ends **only that session**. The listener and other sessions continue. The integration test kills one worker mid-session and checks that a concurrent session and new connections are unaffected.
- GPUs are assigned to **one session at a time** in v1. Concurrent tenants on one GPU are out of scope until MPS/MIG-based isolation has been evaluated.

### What the server will and will not execute

| Accepted | Rejected |
|---|---|
| Generated CUDA driver API calls listed in the codegen allowlist (`annotations.yaml`) | Any `api_id` not in the allowlist. The server returns `CUDA_ERROR_NOT_SUPPORTED` and logs the name. |
| GPU code images (PTX/cubin/fatbin) uploaded via module/library load calls | Host-side executables, scripts, shared libraries, or file paths |
| Opaque handles (`CUcontext`, `CUdeviceptr`, …) that the server issued to **this session** | Handles from other sessions, forged handles, or raw host pointers. Every handle is checked against the session's own handle table. |
| `cuMemAllocManaged` and host-mapped memory | Not supported in v1 (`CUDA_ERROR_NOT_SUPPORTED`) |

### Data handling

- When a session ends, the worker frees all of its allocations, destroys its contexts, and exits. Phase 6 must verify that the next session cannot read leftover device memory. If context teardown proves insufficient, freed allocations are scrubbed explicitly (for example with `cuMemsetD8`).
- The module cache, which avoids re-uploading identical images, is keyed by content hash and **scoped to a single client identity**, so one tenant cannot detect whether another tenant has uploaded a given image.
- Logs record session metadata: identity, start and end times, API call counts, and the hashes of loaded modules. They never record memory contents or kernel arguments.

### Resource limits (per client identity, set by the operator)

Device memory cap, concurrent sessions, session wall-clock time, request rate, and bulk-transfer bytes. When a limit is exceeded the server fails the CUDA call (`CUDA_ERROR_OUT_OF_MEMORY` or `CUDA_ERROR_LAUNCH_TIMEOUT`, as appropriate) and does not queue the work without bound.

### Attack surface and verification

- **Primary:** the worker's decoder for `RpcRequest.args` and the bulk channel. Both are generated code. Each generated function gets marshalling tests, and the decoders are fuzzed with libFuzzer (Phase 6).
- **Secondary:** the closed-source NVIDIA driver, when given hostile but well-formed calls. This is mitigated by the worker sandbox and by keeping drivers patched. It cannot be eliminated.
- **Client side:** injection is launch-scoped (`tuna run <app>`), so no service injects into arbitrary processes. Binaries are code-signed before any release.

### Release blockers for Track B

Phase 6 must deliver all of the following: a signed threat-model review, fuzzing in CI, a verified memory scrub between sessions, enforced worker sandboxing, quota enforcement tests, certificate revocation, and an audit log.

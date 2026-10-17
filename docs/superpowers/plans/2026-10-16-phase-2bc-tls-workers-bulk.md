# Phase 2b + 2c — TLS/mTLS, per-session workers, bulk channel (Track B)

**Parent plan:** `docs/plans/2026-09-23-tuna-execution-plan.md` §5 Phase 2.
**Previous slice:** `2026-10-15-phase-2a-transport-rpc.md`.
**Branch:** `phase-2-transport`.

| Slice | Contents | Exit test |
|---|---|---|
| **2b** | TLS 1.3 + mTLS `Transport`; CA and certificate tooling in `tuna_server`; non-loopback listen allowed only with TLS | Handshake succeeds with a valid identity. It is rejected for an untrusted CA, an identity that is not allowlisted, a client with no certificate, and a TLS 1.2-only peer. |
| **2c** | One worker process per session with socket handoff; a separate bulk socket per session; client connect with retry and backoff | Concurrent sessions run in distinct worker processes. Killing a worker leaves the listener and the other sessions running, and a client then reconnects. Large payloads (> the 16 MiB frame cap) round-trip through the bulk channel. |

## 2b decisions

1. **OpenSSL 3.0+ on every platform.** It is one code path for the Windows client and the Linux server. It is the system library on Ubuntu 22.04/24.04 (3.0.x), and Track A's gRPC build already depends on it. Schannel would need a second implementation of the same security-critical code. The Windows MinGW build links the Shining Light "OpenSSL-Win64" install. `cmake/TunaTunnel.cmake` finds it automatically.
2. **TLS runs over memory BIOs, not a socket BIO.** OpenSSL does not support `SSL_read` and `SSL_write` on one `SSL*` from two threads at once, and our transport has one reader thread plus any number of senders. `TlsTransport` therefore does the socket I/O itself:
   - `ssl_mutex_` guards the `SSL*`. It is held only while encrypting or decrypting, never during socket I/O.
   - `write_mutex_` serializes ciphertext output. The rule is: *whoever drains the outgoing BIO must hold `write_mutex_` and send what it drained before releasing it.* That keeps TLS records in sequence order. The reader never holds a lock while it blocks in `recv()`, so two peers that both send large frames cannot deadlock each other.
3. **TLS 1.3 only, with no session tickets** (`SSL_CTX_set_num_tickets(0)`). Tickets would be the only post-handshake messages, and sessions are short-lived and per-process anyway.
4. **Identity follows Track A's convention.** The server requires a client certificate that chains to the configured CA, and requires one of its DNS or URI SAN values to *exactly* match an entry in `allowed_client_names`. The client verifies the server chain and the server name, which defaults to the host it dialled, using OpenSSL's hostname/IP verification. EKU is enforced through the default SSL purposes (serverAuth for the server, clientAuth for the client).
5. **The certificate tooling is built into `tuna_server`** and uses EC P-256 keys:
   - `--gen-ca DIR` writes `ca.key` and `ca.crt`.
   - `--issue-client-cert NAME --ca-dir DIR --out DIR` writes a cert with SAN `DNS:NAME` and EKU clientAuth.
   - `--issue-server-cert NAME --ca-dir DIR --out DIR [--san DNS:x,IP:y]` writes a cert with EKU serverAuth.

   The tools refuse to overwrite existing files. On POSIX they create key files with mode 0600. Keys are generated on the machine that will use them. The CA key stays in the operator's CA directory.
6. **Config:** an optional `"tls": {"ca_cert", "cert", "key", "allowed_client_names": [...]}`. Without it the server stays plaintext and loopback-only, as in 2a, which is for development and tests. With it, every connection must pass mTLS, and any listen address is allowed.

## 2c decisions

1. **The worker does the TLS handshake (the sshd pre-auth model).** The plan said "authenticate in the listener, then hand the authenticated socket over". With userspace TLS that cannot work: OpenSSL cannot export live session state to another process. Linux kTLS could, but Windows has no equivalent. So the listener hands over the raw TCP socket, and the worker performs TLS, mTLS and `Hello`. This is the same model sshd uses for pre-authentication privilege separation.

   The consequence is that the listener now parses **nothing** beyond a fixed 24-byte preamble, which shrinks its attack surface. Pre-auth worker spawning is bounded by `max_sessions` (default 8), and connections over the limit are closed immediately. The handshake has a 5 s timeout.
2. **Connection preamble** (plaintext, before TLS): `"TUNA"` + `u32 kind` (1 = new session, 2 = attach bulk socket) + a 16-byte session token, which is zero for kind 1. The listener uses it only to route the connection. The session token is random. The listener generates it, gives it to the worker, and the worker sends it to the client inside the TLS-protected `Hello`. An eavesdropper who sees a bulk preamble's token still cannot attach to the session, because the worker requires the bulk connection to pass mTLS **with the same client identity** as the main connection.
3. **Socket handoff uses one IPC channel per worker**, and that channel stays open for the worker's whole life:
   - **POSIX:** `fork` + `exec` with one end of a `socketpair` on fd 3. Each handoff is a message with `SCM_RIGHTS`.
   - **Windows:** the worker's stdin pipe. Each handoff writes the `WSAPROTOCOL_INFOW` produced by `WSADuplicateSocketW(socket, worker_pid)`.

   The first handoff is the session socket; later ones are bulk sockets. **Closing the IPC channel tells the worker to stop**, which gives a graceful shutdown on both platforms. On Windows, workers also sit in a kill-on-close Job object. On Linux they get `PR_SET_PDEATHSIG`. Either way, a crashed listener never leaves orphaned GPU sessions behind.
4. **The worker is the same binary** (`tuna_server --worker`, internal). That means one artifact to deploy and sign. The plan's separate `tuna-worker.exe` can be split out later if the sandbox profiles need it.
5. **`worker_mode: "thread"`** is kept as a config option for debugging (sessions run on threads in the listener, as in 2a). The default is `"process"`.
6. **Bulk channel:** the proto gains `BulkChunk {bulk_ref, offset, total_size, data}` (FrameType `IO`, channel 2), sent in 1 MiB chunks.
   - `Call()` and `Post()` accept a bulk input; `CallResult` carries a bulk output.
   - The client uses the call's `call_id` as the request `bulk_ref`. The server uses it for the reply's bulk payload. The two directions use separate stores, so the ids cannot collide.
   - Chunks are sent before the `RpcRequest`/`RpcResponse` that references them. On a single transport, ordering alone guarantees completeness. With a separate bulk socket, the receiver waits for the transfer to complete, with a timeout.
   - Limits: 4 GiB per transfer and 1 GiB of incomplete transfers per session. Violations are protocol errors.
   - When the server offers a bulk socket and the client opens it, chunks use that socket; otherwise they share the main connection.
7. **Reconnect:** `ConnectSession()` retries the initial connection with exponential backoff (100 ms doubling to 5 s, up to a deadline). A CUDA session's state lives in its worker, so after the worker dies "reconnect" means a *new* session. Resuming a session across a network blip (`Hello.resume_token`) needs the worker to outlive its socket. That is deferred: the preamble's kind field and the token routing already leave room for a `kind = 3 resume`.

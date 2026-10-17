# Phase 2b + 2c — mTLS, per-session worker processes, bulk channel (Track B)

**Date:** 2026-10-16
**Branch:** `phase-2-transport`
**Author:** Thomas Tsuma (with Claude)
**Plan:** `docs/superpowers/plans/2026-10-16-phase-2bc-tls-workers-bulk.md`

## Why

Phase 2a made the wire runtime work on loopback. This completes Phase 2:

- **2b** makes the runtime safe to expose to a network: mutual TLS 1.3, plus the operator tooling to issue identities.
- **2c** applies the isolation and data-path decisions from the execution plan (§2.2, lessons 4 and 7):
  - one worker process per session, so a crashing kernel or decoder bug takes down only its own session;
  - a separate bulk socket, so a 500 MB memcpy never sits in front of small RPCs.

Phase 3 builds the CUDA shim and dispatcher directly on top of these pieces.

## Changes

### `tunnel/` (shared by client and server)
- **`tls.h/.cpp`: `TlsContext` + `TlsTransport`.** OpenSSL 3.0+, TLS 1.3 minimum, peer verification required, no session tickets.
  - The server enforces an exact SAN allowlist; the client verifies the server's host name or IP.
  - The transport uses **memory BIOs** and does its own socket I/O. That way one reader thread and any number of senders can share a connection, which OpenSSL does not allow on a single `SSL*`.
  - Two locks: one guards the `SSL*` and is never held across I/O, and one orders ciphertext output. Neither is held while blocking in `recv()`, so peers that both send large frames cannot deadlock.
- **`pki.h/.cpp`:** EC P-256 CA and leaf issuance. It refuses to overwrite files and creates POSIX keys with mode 0600. It works on OpenSSL 3.0 (Ubuntu) and 4.0 (Windows dev box): `X509_get_subject_name` became `const` in 4.0, so names are built separately.
- **`client.h/.cpp`:**
  - The 24-byte connection preamble (`TUNA`, kind, session token).
  - `ConnectSession()`: TCP connect with exponential backoff (100 ms doubling to 5 s; only the connect is retried, never a failed authentication), then TLS, then `Hello`, then attaching the bulk socket when the server offers one.
- **`rpc`: bulk channel.**
  - `Call()` and `Post()` accept a bulk payload, and `CallResult` carries the bulk reply.
  - `BulkStore` reassembles 1 MiB `BulkChunk` frames that may arrive on either socket, in any order. Limits: 4 GiB per transfer and 4 GiB pending.
  - Dispatchers now take a `ServerCall&` (request, `bulk_in`, `bulk_out`).
  - Async calls never return bulk data: no caller would ever collect it.
- **`tcp_socket`:**
  - `ReceiveExact()` with a timeout. It never over-reads, so the bytes after the preamble stay for the worker's TLS.
  - `Release()`/`native()` for handoff.
  - Close-on-exec everywhere on POSIX (`accept4(SOCK_CLOEXEC)` on Linux).
- **`proto/tuna_frames.proto`** (additive): `Hello.session_token`, `bulk_socket_available`, `worker_pid`, and `message BulkChunk`.
- **Tools:** `tuna_ping` gains mTLS flags, `--bulk-mb`, `--hold-ms`, `--connect-deadline-ms` and `--no-bulk-socket`. `tuna_rpc_bench` gains `--tls` and `--bulk-mb`.
- **CMake:** `find_package(OpenSSL 3.0 REQUIRED)`. Under MinGW it automatically points `FindOpenSSL` at the Shining Light install's MSVC import libraries, which link fine for OpenSSL's C API.

### `tuna-server`
- **Listener (`remoting_server.cpp`)**, rewritten.
  - It accepts a connection and reads the preamble on a short-lived per-connection thread; at most 64 connections may be pending.
  - It routes the socket: a new session goes to a new worker; an attach-bulk connection goes to the worker that holds the token.
  - It enforces `max_sessions`, and stops all workers in parallel on shutdown.
  - It never runs TLS.
- **`worker_process.cpp`: spawn and socket handoff.**
  - POSIX: `fork`/`exec`, with a `socketpair` on fd 3 and `SCM_RIGHTS`.
  - Windows: `CreateProcessW` with an explicit inherit list (Winsock sockets are inheritable by default). `WSADuplicateSocketW` over the worker's stdin, a separate ack pipe, and a kill-on-close Job object.
  - Every handoff is acknowledged. On Windows the listener's copy of the socket stays open until the worker has created its own descriptor.
- **`worker_session.cpp`:** TLS, identity check and `Hello` (carrying the session token, `bulk_socket_available` and `worker_pid`), then `RpcServer::Serve()`. The bulk socket is accepted only with the same client identity. Started as `tuna_server --worker` (internal).
- **`dispatch.cpp`:** the seam for Phase 3's CUDA dispatcher. It serves only the echo API today.
- **`main.cpp`:** `--worker`, `--gen-ca`, `--issue-client-cert` and `--issue-server-cert`.
- **Config:** `tls {ca_cert, cert, key, allowed_client_names}`, `max_sessions` (default 8), and `worker_mode` (`process` by default, or `thread` for debugging). A non-loopback `listen_address` is refused unless `tls` is set.
- `-Wall -Wextra` now applies to `tuna_server` itself. It builds with zero warnings in our sources on MinGW 13.1 and GCC 13.3.

### CI and docs
- CI: `libssl-dev` (Linux), `openssl` in `vcpkg.json` (Windows client), and a Chocolatey OpenSSL install for the MSVC server-core job.
- `tuna-server/README.md`: the certificate workflow, the config reference, and how a connection is served.
- `security_model.md`: controls marked **[implemented]**. The deviation (the worker, not the listener, runs TLS) is recorded with its rationale.

## Bugs found and fixed during verification

1. **TLS data stranded after the handshake (Linux only, intermittent).** A TLS 1.3 client's first application record usually arrives in the same `recv()` as its Finished message. `Handshake()` buffered it in the memory BIO, but `Receive()` blocked on the socket for *new* bytes, so the `Hello` waited until the 5 s timeout.
   - Fix: `Receive()` decrypts buffered input before blocking.
   - The regression test (`Data sent right behind the handshake…`) fails at iteration 25 against the old code and passes with the fix.
2. **`PR_SET_PDEATHSIG` killed every Linux worker right after `Hello`.** The signal fires when the parent *thread* exits, and the listener forks from short-lived routing threads.
   - Fix: removed it. A worker now exits on EOF of its IPC channel, which the kernel closes whenever the listener dies, with a 10 s hard-exit backstop.
3. **Bulk attach race.** The client attaches its bulk socket as soon as it has the server's `Hello`, which can be before the worker starts serving. The worker refused it, and the session silently fell back to the main socket.
   - Fix: `AttachBulk` waits, up to the handshake timeout, for the session to start serving.
4. **`RpcServer` touched the caller's bulk transport after `Serve()` returned** (a use-after-free that showed up in the tests as an intermittent segfault).
   - Fix: `Serve()` now closes the transport, joins its reader and forgets it before returning.

## Verification (2026-10-16)

| Platform | Result |
|---|---|
| Windows 11, MinGW 13.1, OpenSSL 4.0.1, Track B | server 49/49 and guest agent 48/48, each `--repeat until-fail:5` |
| WSL Ubuntu 24.04, GCC 13.3, OpenSSL 3.0.13, full superbuild (Track A + B, gRPC) | 64/64 `--repeat until-fail:20` (1,280 passes); end-to-end test 25/25 standalone |

The end-to-end test (`tuna_foreground_session`, real binaries) covers:
- plaintext refusing non-loopback addresses;
- 3 concurrent sessions in 3 distinct worker PIDs;
- 40 MiB bulk echoes with and without the bulk socket;
- killing one worker mid-session while a second session and new clients are unaffected;
- `max_sessions` being enforced and released;
- thread mode;
- mTLS: an allowlisted client is accepted, while non-allowlisted, plaintext and certificate-less clients are rejected;
- the certificate tooling refusing to overwrite files;
- a client retrying with backoff until a late server comes up;
- a graceful stop with exit code 0.

`tuna_rpc_bench`, Windows loopback, 64-byte args:

| | Sync RTT p50 / p99 | Sync calls/s | Batched calls/s | Bulk echo (64 MiB) |
|---|---|---|---|---|
| TCP | 49.6 / 92.8 µs | 18.3k | 1.78M | 159 MiB/s per direction |
| TCP + mTLS 1.3 | 70.5 / 148.2 µs | 12.2k | 1.11M | 140 MiB/s per direction |

TLS costs about 21 µs per synchronous round trip on loopback. That is small next to a LAN round trip of 200–1000 µs.

Bulk throughput is low with or without TLS, so it is **copy-bound, not crypto-bound**: each MiB is copied about 8 times (payload → chunk → frame → wire, and back). It is good enough for Phase 3 correctness milestones. Phase 4 will rework it (zero-copy chunk framing, larger socket buffers, LZ4).

## Not included / follow-ups
- **Session resume** (`Hello.resume_token`) across a network blip. The worker would have to outlive its socket. The preamble's `kind` field leaves room for `kind = 3 resume`.
- **Worker sandboxing** (seccomp, systemd, restricted token or AppContainer), quotas, and log retention for the per-worker log files. These are Phase 6.
- CI changes are untested until the PR runs them, in particular the Chocolatey OpenSSL path in the MSVC job.

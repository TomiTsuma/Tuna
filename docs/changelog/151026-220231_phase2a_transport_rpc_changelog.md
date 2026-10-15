# Phase 2a — Framing, TCP transport, RPC runtime (Track B)

**Date:** 2026-10-15
**Branch:** `phase-2-transport` (local, stacked on `phase-1-core-plumbing`)
**Author:** Thomas Tsuma (with Claude)
**Plan:** `docs/superpowers/plans/2026-10-15-phase-2a-transport-rpc.md`

## Why

Phase 2 of the Track B plan (transport + RPC runtime) is estimated at 2–3 weeks,
so it ships as three slices. This is the first one. It provides the wire runtime
that every later phase builds on: a framed, ordered connection, the `Hello`
handshake, liveness, and RPC semantics that match CUDA's async error model.

## Changes

### New shared library `tunnel/` (built via `cmake/TunaTunnel.cmake`)
Client and server use the same code. Two copies of a wire protocol would drift.
- `frame_codec`: 4-byte little-endian length + serialized `Frame`, with a 16 MiB
  cap. An incremental decoder handles split and coalesced reads. An oversized
  length prefix or a garbage body is a protocol error: the decoder fails closed
  and never allocates beyond the cap.
- `tcp_socket`: a minimal Winsock/BSD wrapper. `TCP_NODELAY` is always on,
  `MSG_NOSIGNAL` is set on Linux, port 0 binds an ephemeral port, and
  `IsLoopbackHost()` is provided.
- `transport`: a blocking `Transport` interface (`Send(type, channel, payload)` /
  `Receive()` / `Close()`) and its `TcpTransport` implementation. The transport
  assigns `frame_id` and `timestamp_ns`, and serializes concurrent senders.
- `session`: a symmetric `ExchangeHello()` with a deadline and a
  protocol-version check, plus `Heartbeat` (3 s send interval, 10 s silence
  timeout, both configurable).
- `rpc`: `RpcClient` / `RpcServer`.
  - `Post()` queues async calls. The queue is flushed by 256 calls, 64 KiB of
    args, a 0.5 ms timer, or the next `Call()`, which rides in the same batch.
  - The server replies to async calls only when they fail. The client latches
    the first failure and returns it from the next *synchronizing* call.
  - Transport loss or a timeout returns `999` (`CUDA_ERROR_UNKNOWN`).
  - `api_id 0` is reserved for a diagnostic echo.
- Tools: `tuna_ping` (handshake + timed echoes) and `tuna_rpc_bench` (the Phase 2
  exit benchmark).

### Server
- `tuna_server --foreground` now serves sessions (`src/remoting/`). It
  handshakes, runs one thread per session, reaps finished sessions, and stops
  gracefully on SIGINT/SIGTERM/SIGBREAK. Until Phase 3 it serves the echo API
  only.
- The config gains `listen_address` (default `127.0.0.1`). **Non-loopback
  addresses are refused** until TLS exists (Phase 2b).

### Tests
- `tuna_tunnel_tests` (Catch2, registered once even in the superbuild) covers:
  - codec edge cases;
  - handshake success, version mismatch, and a silent-peer timeout;
  - echo;
  - post ordering before a call;
  - timer-only flush;
  - deferred async-error latching;
  - server loss;
  - heartbeat expiry;
  - loopback detection.
- `tuna_foreground_session` (Python, real binaries) covers:
  - non-loopback refusal;
  - 50 sequential and 4×25 concurrent `tuna_ping` sessions;
  - the listener surviving after sessions end;
  - a graceful stop with exit code 0.
- `tuna_rpc_bench_smoke` checks that the benchmark runs end to end.

## Bug found and fixed during verification

On Windows, `shutdown()` does **not** cancel a `recv()` blocked on the same
socket; only `closesocket()` does. Linux behaves differently, so every test
passed there. On Windows the silent-peer handshake timeout took 120 s instead of
200 ms. In practice a hung or hostile peer would have pinned a reader thread
indefinitely, which defeats the point of heartbeats and timeouts.
`TcpStream::Shutdown()` now also closes the handle on Windows. The handle is
held in an atomic, like the listener's. POSIX keeps `shutdown()` plus
close-in-destructor, because closing an fd another thread is blocked on is
unsafe there.

## Verification (2026-10-15)

| Platform | Tests (3 consecutive runs, `--repeat until-fail:3`) |
|---|---|
| Windows, MinGW 13.1, Track B | guest agent 30/30, server 28/28 |
| WSL Ubuntu 24.04, full superbuild (Track A + B, gRPC) | 43/43, 0 compiler warnings |

`tuna_rpc_bench`, loopback TCP, 64-byte args, 10k sync calls, 100k batched posts:

| | Sync RTT p50 / p99 | Sync calls/s | Batched calls/s |
|---|---|---|---|
| Windows | 49.3 / 121.2 µs | 18.6k | 1.87M |
| Linux (WSL) | 67.5 / 144.1 µs | 13.6k | 4.44M |

Batching is 100–300× faster than synchronous calls. That ratio is the reason
for the async classification in Phase 4. The sync RTT is dominated by the thread
handoff (reader thread → promise), not the kernel. Phase 4 will profile it.

## Not included (next slices)
- 2b: TLS 1.3 + mTLS behind `Transport`, and CA/cert tooling. This also removes
  the loopback-only restriction.
- 2c: per-session worker processes with socket handoff, reconnect with backoff,
  and the separate bulk socket.
- CI is unchanged: the new tests run inside the existing jobs automatically.

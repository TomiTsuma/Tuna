# Phase 2a — Framing, TCP transport, RPC runtime (Track B)

**Parent plan:** `docs/plans/2026-09-23-tuna-execution-plan.md` §5 Phase 2.
**Branch:** `phase-2-transport`, stacked on `phase-1-core-plumbing`.

Phase 2 is estimated at 2–3 weeks, so it ships as three dependency-ordered slices:

| Slice | Contents | Exit test |
|---|---|---|
| **2a (this plan)** | Length-prefixed framing, a blocking `Transport` interface, a TCP implementation, the `Hello` handshake, heartbeat, and the RPC runtime (`Call`/`Post`, batching, deferred-error latching). The server's `--foreground` mode serves an echo dispatcher on loopback. | Loopback handshake; echo RPC correctness; 10k-call p50/p99 RTT and batched calls/sec, printed by `tuna_rpc_bench` |
| 2b | TLS 1.3 + mTLS behind the same `Transport`, plus CA/cert tooling (`--gen-ca`, `--issue-client-cert`) | Handshake rejects wrong CA or wrong identity; non-loopback bind allowed only with TLS |
| 2c | One worker process per session, with socket handoff (`WSADuplicateSocket` / `SCM_RIGHTS`), reconnect with backoff, and a separate bulk socket | Kill the worker and the listener survives; the client reconnects |

## Decisions (2a)

1. **Shared code lives in `tunnel/`** at the repo root, built by `cmake/TunaTunnel.cmake` (with an include guard, the same pattern as `TunaProtobuf.cmake`). Two copies of a wire protocol would drift apart. Config and logging stay per binary as before.
2. **The wire format is a 4-byte little-endian length followed by a serialized `Frame`.** The limit is 16 MiB per frame. Larger payloads will use the bulk channel (2c). An oversized length is treated as a protocol error and closes the connection, so a hostile peer cannot make the receiver allocate without bound.
3. **The transport is blocking:** `Send(type, channel, payload)` and `Receive() → optional<Frame>`. Each endpoint owns one reader thread. That is easier to reason about and test than callbacks. The transport assigns `frame_id` (monotonic, 64-bit) and `timestamp_ns`.
4. **A GPU frame from client to server always carries an `RpcBatch`.** A single call is a batch of one, so `FrameType` needs no new discriminator. A GPU frame from server to client carries one `RpcResponse`.
5. **RPC semantics, following CUDA's async error model:**
   - `Post()` queues an `ASYNC` call and returns immediately.
   - The queue is flushed when a `Call()` is made (the sync call rides in the same batch, which preserves ordering and saves a frame), when it reaches 256 calls or 64 KiB of args, or after a 0.5 ms timer.
   - The server answers an async call **only if it fails**. The client latches the first such error.
   - A *synchronizing* `Call()` returns the latched error, if there is one, in place of its own result, and clears it.
   - If the transport is lost, every pending call fails with `999` (`CUDA_ERROR_UNKNOWN`).
6. **One batch queue per client, not per thread, for now.** It keeps a strict global order, which is a superset of per-thread order, and it is simpler. Per-thread queues are a Phase 4 optimization, once measurements show contention.
7. **Handshake:** both sides send `Hello` (CONTROL, channel 0) first, then read the peer's. A `protocol_version` mismatch closes the connection.
8. **Heartbeat:** each side sends a HEARTBEAT every 3 s and closes the transport after 10 s without receiving any frame. Both intervals are configurable so tests run quickly.
9. **Safety while there is no TLS:** the server refuses to bind a non-loopback address until 2b. Plaintext GPU RPC must never be exposed to a network.

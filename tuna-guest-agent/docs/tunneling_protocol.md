# Tunneling Protocol Specification — Tuna Guest Agent

## Overview

The **Tuna Tunneling Protocol (TTP)** is the backbone of Tuna’s secure communication layer between the **Tuna Guest Agent (client)** and the **Tuna Remote (server)**. It enables the seamless transmission of intercepted system calls, GPU/CPU data streams, and I/O between the client’s local OS and the remote GPU-enabled environment.

The protocol ensures:

* **End-to-end encryption**
* **Low latency streaming**
* **Adaptive compression**
* **Cross-platform compatibility**
* **Transparent OS integration**

---

## 1. Protocol Stack

| Layer                 | Component                        | Purpose                                            |
| --------------------- | -------------------------------- | -------------------------------------------------- |
| **Application Layer** | Tuna Guest Agent / Tuna Remote   | Encodes intercepted syscalls and data frames       |
| **Session Layer**     | TTP Session Manager              | Handles authentication, reconnection, multiplexing |
| **Transport Layer**   | QUIC / gRPC over HTTP/3          | Provides reliable, low-latency transport           |
| **Security Layer**    | TLS 1.3 with mutual certificates | Encryption, authentication, and key exchange       |
| **Network Layer**     | UDP                              | Enables fast, multiplexed packet transmission      |

---

## 2. Connection Lifecycle

### 2.1 Initialization

1. **Handshake Initiation:**
   The client (Guest Agent) sends a **ClientHello** packet over QUIC.
2. **Certificate Exchange:**
   Mutual authentication is performed using TLS 1.3 certificates.
3. **Session Negotiation:**
   Client and server negotiate compression level, encryption cipher, and session ID.
4. **Tunnel Establishment:**
   A persistent QUIC channel is created for the session.

### 2.2 Active Tunnel

Once established, the tunnel operates in full duplex mode. Each data frame is tagged with:

* **Frame Type:** `SYS`, `GPU`, `IO`, `HEARTBEAT`, `CONTROL`
* **Frame ID:** Sequential counter for tracking
* **Timestamp:** Synchronization aid
* **Payload:** Encrypted content blob

### 2.3 Termination

* Either side sends a **FIN** control frame.
* Session data flushed, keys discarded.
* QUIC session teardown follows.

---

## 3. Data Frame Structure

```
┌───────────────────────────┐
│ Frame Header              │
│  - Frame Type (1 byte)    │
│  - Frame ID (4 bytes)     │
│  - Payload Length (4 bytes)│
│  - Timestamp (8 bytes)    │
└───────────────────────────┘
┌───────────────────────────┐
│ Encrypted Payload          │
│ (Variable Length)          │
└───────────────────────────┘
```

### Frame Types

| Type        | Purpose                                       |
| ----------- | --------------------------------------------- |
| `SYS`       | Serialized system call metadata and arguments |
| `GPU`       | GPU command buffers, CUDA kernel invocations  |
| `IO`        | File/network I/O redirections                 |
| `HEARTBEAT` | Health check and latency measurement          |
| `CONTROL`   | Configuration, logs, and diagnostics          |

---

## 4. Security Architecture

### 4.1 Authentication

* **Mutual TLS (mTLS):**
  Both client and server present certificates signed by a common CA.
* **Device Binding:**
  Each client certificate is tied to a specific hardware fingerprint (TPM hash).

### 4.2 Encryption

* **TLS 1.3 AES-256-GCM** for payload encryption.
* **Ephemeral ECDH keys** for forward secrecy.
* **Nonce rotation** every 10 minutes or after 100MB transferred.

### 4.3 Integrity & Replay Protection

* Frame-level **SHA-256 checksums**.
* **Monotonic frame IDs** prevent replay or out-of-order injection.

---

## 5. Multiplexing & Channel Management

Each tunnel supports multiple **logical channels** for different purposes:

* `chan_sys`: System calls
* `chan_gpu`: GPU data streams
* `chan_io`: File and socket operations
* `chan_ctrl`: Control and monitoring

Each channel is multiplexed using QUIC streams, enabling parallel, low-latency operations.

---

## 6. Compression & Optimization

| Feature                 | Description                                                         |
| ----------------------- | ------------------------------------------------------------------- |
| **Zstandard (Zstd)**    | Default compression for high-throughput syscalls                    |
| **GPU Stream Batching** | Aggregates multiple small GPU calls into one large buffer           |
| **Delta Encoding**      | For repeated syscall sequences                                      |
| **Zero-Copy Buffering** | Avoids redundant memory copying between user-space and kernel-space |

---

## 7. Keep-Alive & Reconnection

* **Heartbeat Frames:** Sent every 3s with round-trip time (RTT) data.
* **Timeout:** 10s with exponential backoff for reconnection attempts.
* **Resume Tokens:** Sessions can resume from the last acknowledged frame ID after disconnection.

---

## 8. Implementation Notes

### Client Side (Guest Agent)

* Uses an embedded QUIC client library (e.g., `msquic` or `aioquic`).
* Manages syscall interception hooks and streams data frames through TTP.
* Includes local buffer queues to prevent blocking.

### Server Side (Tuna Remote)

* QUIC listener accepts multiple client tunnels.
* Decodes incoming frames, routes them to hypervisor or GPU daemon.
* Sends responses (e.g., syscall return values) through the same channel.

---

## 9. Development Stack

| Component     | Recommendation                       |
| ------------- | ------------------------------------ |
| QUIC Library  | `aioquic` (Python) or `msquic` (C++) |
| Encryption    | OpenSSL / Rustls                     |
| Serialization | FlatBuffers / Protobuf               |
| Compression   | Zstandard                            |
| Monitoring    | Prometheus + Grafana                 |
| Logging       | Structured JSON logs via Fluent Bit  |

---

## 10. Developer Checklist

* [ ] Define protobuf schemas for all frame types
* [ ] Implement handshake & certificate exchange
* [ ] Implement frame serialization/deserialization
* [ ] Add compression + encryption pipeline
* [ ] Integrate syscall hook streams with tunnel sender
* [ ] Test reconnection, heartbeat, and session resumption
* [ ] Perform performance benchmarking (latency, throughput)
* [ ] Add logging and metrics endpoints
* [ ] Conduct full security audit before deployment

---

**End of Document**

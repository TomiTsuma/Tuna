# Tuna Guest Agent – Architecture Overview

## 1. Introduction

**Tuna Guest Agent** is the client-side component of the Tuna virtualization ecosystem. It acts as a lightweight system daemon that transparently redirects heavy computation tasks from the client’s machine to a high-performance remote server (running Tuna Server). The user experiences no disruption or latency in their normal application workflow — applications such as Adobe Photoshop, Blender, or other compute-intensive software continue to behave as if they were running locally.

At its core, Tuna Guest Agent bridges local user interactions and remote compute execution, allowing low-spec machines to leverage high-end GPUs, CPUs, and RAM available on the Tuna Server through a secure and multiplexed tunnel.

---

## 2. High-Level Architecture

```
+---------------------------------------------------------------+
|                  Client (Local Computer)                      |
|                                                               |
|  +-------------------+      +-------------------------------+ |
|  |   User Software   | ---> |   Tuna Guest Agent (Service)  | |
|  | (Photoshop, etc.) |      +-------------------------------+ |
|  |                   |        |                              |
|  |   Syscalls / API  |<-------| Hook Manager (MinHook)       |
|  +-------------------+        |                              |
|                               | Tunnel Manager (gRPC/TLS)    |
|                               | Encryption & Auth Layer      |
|                               | I/O Redirector               |
|                               | Compute Redirector           |
|                               +-------------------------------+
|                               |  GUI Frontend (Optional)     |
|                               +-------------------------------+
+-------------------------------|--------------------------------
                                |
                                v
+---------------------------------------------------------------+
|                  Tuna Server (Remote Machine)                 |
|  +----------------------------------------------------------+ |
|  |  Virtualization Layer (Hyper-V / Tuna VM Engine)         | |
|  |  Remote Compute Daemon                                   | |
|  |  GPU/CPU Resource Pool                                   | |
|  |  I/O Sync & Cache Engine                                 | |
|  |  Secure Tunnel Endpoint (TLS Multiplexed)                | |
|  +----------------------------------------------------------+ |
+---------------------------------------------------------------+
```

---

## 3. Key Components and Their Roles

### 3.1 Tuna Guest Agent Core

A background Windows Service that:

* Maintains a persistent, encrypted connection to Tuna Server.
* Monitors system events and application launches.
* Dynamically hooks selected system and API calls (e.g., GPU or disk access).
* Routes intercepted calls to the remote execution layer.

### 3.2 Syscall Interceptor (Hook Manager)

* Uses **MinHook** or **Microsoft Detours** to intercept low-level Win32 API or kernel calls.
* Forwards intercepted calls (like GPU compute or file I/O) to Tuna Server.
* Restores or bypasses hooks when no remote connection is active.

### 3.3 Tunnel Manager

* Implements secure gRPC/ZeroMQ channels over TLS.
* Multiplexes different data streams (compute, I/O, control, metrics).
* Supports encryption, authentication, and session resumption.
* Ensures data compression and latency optimization.

### 3.4 Compute Redirector

* Detects when CPU/GPU-bound tasks are invoked.
* Serializes execution context (function name, parameters, memory references).
* Sends the context to the Tuna Server for remote execution.
* Receives computed results and reinjects them into the local process space transparently.

### 3.5 I/O Redirector

* Provides transparent file I/O proxying.
* Syncs read/write operations between client and server.
* Maintains a local cache to prevent latency from affecting responsiveness.

### 3.6 Configuration Manager

* Loads and validates `defaults.json`.
* Handles certificate-based authentication and connection settings.
* Manages daemon startup and failure recovery.

### 3.7 GUI Interface (Optional)

* Built with Electron or Qt.
* Displays connection status, server usage, and logs.
* Allows user to configure server address, port, and credentials.

---

## 4. Communication Flow

1. **Initialization**

   * The Tuna Guest Agent daemon starts at system boot.
   * It reads configuration and establishes a secure tunnel to Tuna Server.

2. **Hook Registration**

   * Hooks are registered on key APIs (e.g., DirectX/OpenGL, file I/O).
   * Hook Manager monitors target processes and dynamically injects hooks.

3. **Execution Redirection**

   * When the user runs a heavy operation (e.g., applying a Photoshop filter), the relevant syscall is intercepted.
   * The call context and data are serialized and sent to Tuna Server.

4. **Remote Execution**

   * Tuna Server executes the operation using its own GPU/CPU.
   * The output is sent back via the tunnel to Tuna Guest Agent.

5. **Result Reinjection**

   * Tuna Guest Agent reinjects results into the calling process memory space.
   * To the user, the application behaves as if everything was done locally.

---

## 5. Security Considerations

* **TLS 1.3** encryption for all tunnel communication.
* **Mutual authentication** (client and server certificates).
* **Process sandboxing**: Tuna Guest Agent runs with limited privileges.
* **Integrity checks**: All transmitted binaries or buffers are hashed.
* **Zero-trust posture**: No filesystem-level access from server without explicit permission.

---

## 6. Performance Optimizations

* **Zero-copy buffers** between intercepted processes and tunnel stream.
* **Async batching** for small syscalls to minimize overhead.
* **Local caching** for repeated reads and unmodified resources.
* **Compression** (LZ4 or zstd) for data transfer efficiency.

---

## 7. Fault Tolerance and Recovery

* Automatic reconnection if the tunnel drops.
* Graceful degradation: fall back to local compute if server unavailable.
* Transaction rollback for I/O operations.
* Periodic heartbeats to monitor tunnel health.

---

## 8. Development Roadmap

| Phase   | Deliverable            | Key Features                            |
| ------- | ---------------------- | --------------------------------------- |
| Phase 1 | Core Daemon            | Service startup, basic TLS tunnel       |
| Phase 2 | Syscall Hooking        | Intercept GPU/IO syscalls               |
| Phase 3 | Remote Execution       | Task serialization & result reinjection |
| Phase 4 | GUI Dashboard          | User config, monitoring, and logs       |
| Phase 5 | Optimization & Caching | Latency reduction, performance tuning   |

---

## 9. Future Extensions

* Linux/macOS support using equivalent system hooks.
* Multi-server load balancing.
* Dynamic GPU allocation across multiple clients.
* Integration with containerized workloads.

---

## ✅ 10. Implementation Checklist

**Phase 1 – Setup**

* [ ] Create base project with CMake + Windows Service template.
* [ ] Define `tuna.proto` and generate gRPC stubs.
* [ ] Implement secure TLS handshake and tunnel connection.

**Phase 2 – Hooking**

* [ ] Integrate MinHook for syscall interception.
* [ ] Implement test hooks (GPU API, file read).
* [ ] Serialize and send intercepted context to mock server.

**Phase 3 – Remote Execution**

* [ ] Implement Compute Redirector and response injection.
* [ ] Validate latency and response consistency.
* [ ] Add I/O proxy and caching mechanisms.

**Phase 4 – GUI**

* [ ] Build Electron-based GUI for control and monitoring.
* [ ] Integrate GUI with service via local API bridge.

**Phase 5 – Security and Stability**

* [ ] Add certificate-based mutual authentication.
* [ ] Harden tunnel and daemon privileges.
* [ ] Implement auto-reconnect and logging.

**Phase 6 – Testing**

* [ ] Create full integration test with Photoshop or CUDA app.
* [ ] Benchmark CPU/GPU forwarding performance.
* [ ] Validate recovery from tunnel interruptions.

---

**End of Document**

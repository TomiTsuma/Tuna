# Tuna Guest Agent – Daemonization Workflow

## 1. Overview

The **Tuna Guest Agent Daemon** is the core background process responsible for maintaining continuous communication with Tuna Server, intercepting syscalls, and managing compute redirection. It operates as a **Windows Service** that runs in the background without user interaction, starting automatically at boot and maintaining persistence across sessions.

This document outlines how the daemon is initialized, managed, monitored, and gracefully terminated. It also covers its threading model, failure handling, and interaction with the system kernel and GUI layer.

---

## 2. Core Objectives

The daemonization process ensures:

* **Automatic startup** at boot with minimal overhead.
* **Continuous operation** even when no user is logged in.
* **Secure lifecycle management** (start, stop, restart).
* **Isolation** from user applications to prevent interference.
* **Graceful shutdown and recovery** in case of faults or system reboots.

---

## 3. High-Level Architecture

```
+---------------------------------------------------------+
|                 Windows Operating System                |
|                                                         |
|  +---------------------+       +----------------------+ |
|  |  Service Manager    | <----> |  Tuna Guest Daemon   | |
|  |  (SCM / Win32 API)  |       +----------------------+ |
|  |                     |              |                 |
|  +---------------------+              v                 |
|                                     Threads:            |
|   - Control Thread (service handler)                     |
|   - Tunnel Thread (gRPC connection)                      |
|   - Hook Thread (syscall monitoring)                     |
|   - IO Thread (proxy & cache sync)                       |
|   - Heartbeat Thread (health checks)                     |
|                                                         |
|  +---------------------------+                           |
|  |  Tuna GUI (Electron/Qt)   | <----> Local API Bridge   |
|  +---------------------------+                           |
+---------------------------------------------------------+
```

---

## 4. Lifecycle Stages

### 4.1 Installation

* The installer (e.g., `installer.nsi` or PowerShell script) registers Tuna Guest Agent as a **Windows Service** using the **Service Control Manager (SCM)**.
* Configuration files (`defaults.json`, certificates, and credentials) are stored in:

  ```
  C:\ProgramData\Tuna\config\
  ```
* Log files are stored in:

  ```
  C:\ProgramData\Tuna\logs\
  ```

### 4.2 Startup

1. SCM invokes the service main function (e.g., `ServiceMain()` in `service_main.cpp`).
2. Tuna initializes its subsystems in this order:

   * Configuration Loader
   * Logging Engine
   * Encryption Engine
   * Tunnel Manager
   * Hook Manager
3. After initialization, the daemon sets its state to **SERVICE_RUNNING** and spawns worker threads.

### 4.3 Runtime Operation

The daemon runs five concurrent threads:

| Thread               | Responsibility                                                                |
| -------------------- | ----------------------------------------------------------------------------- |
| **Control Thread**   | Listens for start/stop/restart events from Windows SCM.                       |
| **Tunnel Thread**    | Maintains secure TLS connection to Tuna Server; handles message multiplexing. |
| **Hook Thread**      | Monitors and intercepts API/syscalls as configured.                           |
| **IO Thread**        | Manages file system redirection and caching between client and server.        |
| **Heartbeat Thread** | Periodically sends health checks and metrics to the server.                   |

---

## 5. Communication with the GUI

* The daemon exposes a **local loopback API** (via gRPC or IPC socket).
* The Tuna GUI uses this local endpoint to:

  * Query service status.
  * Display logs and metrics.
  * Update configuration or credentials.
* All GUI actions are sandboxed — user interactions cannot directly alter core processes, only signal reconfiguration through the API.

---

## 6. Fault Tolerance and Recovery

### 6.1 Watchdog Monitoring

* A lightweight watchdog monitors the daemon’s heartbeat file (`tuna.status`).
* If no update is detected within N seconds, the watchdog requests SCM to restart the service.

### 6.2 Graceful Failure Handling

On any internal fault:

1. The daemon writes a stack trace to `logs/daemon_crash.log`.
2. Unfinished tunnel transmissions are queued for retry.
3. Hooks are automatically uninstalled to prevent instability.
4. The daemon notifies SCM with **SERVICE_STOP_PENDING** and attempts a clean restart.

### 6.3 Auto-Reconnect

* If the network tunnel drops, the Tunnel Manager retries every 5 seconds using exponential backoff.
* Once reconnected, it resynchronizes session metadata and cached operations.

---

## 7. Shutdown Sequence

When a stop signal is received (via SCM or manual command):

1. The Control Thread sets a global termination flag.
2. Hook Manager removes all active syscall hooks.
3. Tunnel Manager sends a **disconnect** message to Tuna Server.
4. IO Thread flushes any pending cache data.
5. All worker threads are joined and gracefully terminated.
6. Final logs are written, and SCM is notified of **SERVICE_STOPPED**.

---

## 8. Logging and Monitoring

* Logs are handled by `spdlog` and written to rotating log files:

  ```
  C:\ProgramData\Tuna\logs\tuna_agent.log
  ```
* Logs include:

  * Initialization messages.
  * Hook registration/unregistration.
  * Tunnel connection state changes.
  * Error traces and recovery events.
* Optionally, a Prometheus exporter can be enabled for remote telemetry.

---

## 9. Security and Permissions

* Service runs under a dedicated Windows account: `NT SERVICE\TunaAgent`.
* No direct access to user directories except where permission is explicitly granted.
* Certificates and configuration files are readable only by this account.
* All network communications use **TLS 1.3 + client certificates**.

---

## 10. Development Notes

* For debugging, Tuna can also run in **foreground mode**:

  ```bash
  tuna_guest_agent.exe --foreground --verbose
  ```
* In production, SCM handles restarts and recovery.
* Daemon supports Windows Event Viewer integration via event logging API.

---

## ✅ 11. Implementation Checklist

**Setup**

* [ ] Implement Windows Service boilerplate (`ServiceMain`, `HandlerEx`, `ReportStatus`).
* [ ] Create installer to register Tuna Agent with SCM.
* [ ] Configure log directories and permissions.

**Initialization**

* [ ] Load configuration and certificates at startup.
* [ ] Initialize Tunnel, Hook, IO, and Control subsystems.

**Runtime**

* [ ] Spawn worker threads for each subsystem.
* [ ] Maintain secure heartbeat with Tuna Server.
* [ ] Handle reconnection and tunnel health.

**Shutdown**

* [ ] Implement clean termination sequence.
* [ ] Flush logs and caches.
* [ ] Unhook syscalls safely.

**Testing**

* [ ] Test service install/start/stop via PowerShell and SCM.
* [ ] Validate watchdog and auto-restart behavior.
* [ ] Simulate network failure and recovery.
* [ ] Verify log rotation and event reporting.

---

**End of Document**

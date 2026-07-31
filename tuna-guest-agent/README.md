# 🐟 Tuna Guest Agent

## Overview

**Tuna Guest Agent** is a lightweight virtualization client that allows low-powered computers to offload compute- and GPU-intensive workloads to a remote, high-performance **Tuna Server**. It creates the illusion that the client machine has access to powerful hardware, while all heavy computation is transparently executed on the server.

The system works by intercepting key system calls and API-level operations (file, graphics, process, and network calls), routing them through a secure, multiplexed tunnel to the Tuna Server. The user continues working on their local machine as usual—launching applications like Photoshop, Blender, or games—while all compute, rendering, and data processing happen remotely.

---

## Core Features

* **Secure Multiplexed Tunneling** — QUIC + TLS 1.3-based encrypted communication with bidirectional data streams.
* **Syscall Interception** — Redirection of CPU, GPU, and I/O system calls to the server using lightweight hooks.
* **Qt GUI** — Cross-platform, native interface for managing connections, credentials, and performance metrics.
* **Daemon Mode** — Background service that initializes hooks, manages secure connections, and monitors processes.
* **Seamless Virtualization** — Offloads computation to Tuna Server without altering user workflows.
* **Modular Architecture** — Distinct subsystems for GUI, daemon, hooks, and networking.

---

## File and Folder Structure

```
tuna-guest-agent/
├── src/
│   ├── core/                       # OS-agnostic config, logging, error handling
│   │   ├── config_manager.cpp
│   │   ├── logger.cpp
│   │   ├── error_handler.cpp
│   │   └── platform_utils.cpp
│   │
│   ├── daemon/                     # Background service (Tuna Daemon)
│   │   └── win/                    # Windows Service implementation
│   │       ├── daemon_main.cpp
│   │       ├── daemon_service.cpp
│   │       ├── privilege_utils.cpp
│   │       └── watchdog.cpp
│   │
│   ├── hooks/                      # Hooking and interception logic
│   │   └── win/                    # MinHook-based Win32/DirectX/CUDA hooks
│   │       ├── hooks_init.cpp
│   │       ├── file_hooks.cpp
│   │       ├── graphics_hooks.cpp
│   │       ├── process_hooks.cpp
│   │       ├── network_hooks.cpp
│   │       ├── memory_hooks.cpp
│   │       ├── hook_handlers.cpp
│   │       └── hook_handlers.h
│   │
│   ├── tunnel/                     # Communication and transport layer
│   │   ├── quic_tunnel.cpp
│   │   ├── multiplexing_layer.cpp
│   │   ├── data_serializer.cpp
│   │   ├── handshake_protocol.cpp
│   │   └── heartbeat.cpp
│   │
│   ├── gui/                        # Qt-based graphical frontend
│   │   ├── main.cpp
│   │   ├── mainwindow.ui
│   │   ├── mainwindow.cpp
│   │   ├── settings_dialog.cpp
│   │   └── tray_icon.cpp
│   │
│   └── main.cpp                    # Unified entry point (launches GUI or daemon)
│
├── docs/
│   ├── architecture_overview.md
│   ├── daemonization_workflow.md
│   ├── syscall_interception_design.md
│   ├── tunneling_protocol.md
│   └── security_model.md
│
├── minhook/                         # Vendored as a git submodule (TsudaKageyu/minhook)
│
├── build/                          # Compiled binaries and build artifacts
│
├── tests/
│   ├── unit/                       # Unit tests (Catch2)
│   ├── integration/                # End-to-end tests (hooks + tunnel)
│   ├── createfile_hook.c           # Reference prototype for the Phase 3 CreateFileW hook
│   └── main.c                      # Reference ETW-based file-monitoring experiment (not part of the build)
│
├── CMakeLists.txt                  # Build configuration
├── LICENSE
└── README.md
```

---

## Dependencies

**Languages**

* C++20 (core agent, hooks, and networking)
* Python (optional for testing and build automation)

**Libraries & Frameworks**

* **Qt 6** — GUI and network modules
* **libcap / polkit** — privilege elevation
* **OpenSSL** — encryption layer for TLS
* **libuv / epoll** — asynchronous event loop
* **protobuf / msgpack** — data serialization
* **libhook / LD_PRELOAD / Microsoft Detours** — system call interception
* **spdlog** — logging framework
* **Catch2** — unit testing framework

---

## Build Instructions

```bash
git clone https://github.com/your-org/tuna-guest-agent.git
cd tuna-guest-agent
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

---

## Running the Tuna Guest Agent

**Start GUI Mode**

```bash
tuna-guest-agent --gui
```

**Run as Background Daemon**

```bash
sudo systemctl start tuna-guest-agent.service
```

**Check Daemon Logs**

```bash
journalctl -u tuna-guest-agent
```

---

## Security Highlights

* 🔐 End-to-end encryption using **TLS 1.3** with mutual certificate authentication.
* 🧩 Sandbox isolation for syscall hooks to prevent privilege escalation.
* 🧱 All privileged operations follow **least-privilege principles**.
* ♻️ Automatic key rotation and session renewal to prevent replay attacks.

---

## Development Roadmap

| Status | Feature                               |
| :----- | :------------------------------------ |
| ✅      | Daemonization and lifecycle control   |
| ✅      | Secure tunneling (QUIC/TLS)           |
| 🔄     | GPU command redirection               |
| 🔄     | Filesystem virtualization             |
| 🔲     | Performance optimization and caching  |
| 🔲     | Packaging for Linux / Windows / macOS |

---

## Contributing

We welcome community contributions.
To contribute:

1. Fork this repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/new-module
   ```
3. Add your changes and commit:

   ```bash
   git commit -m "Add new module"
   ```
4. Push and open a pull request.

Please ensure your code is well-documented, unit-tested, and adheres to the coding conventions in `docs/architecture_overview.md`.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Maintainers

* **Lead Developer:** Thomas Tsuma
* **Organization:** Core & Outline
* **Contact:** [coreandoutline.ai@gmail.com](mailto:coreandoutline.ai@gmail.com)

---

### ✅ Developer Setup Checklist

* [ ] Install dependencies (`Qt6`, `libhook`, `libcap`, `protobuf`, `spdlog`)
* [ ] Build with CMake
* [ ] Run daemon service
* [ ] Configure server connection in GUI
* [ ] Verify tunnel handshake with Tuna Server
* [ ] Test interception (file, network, graphics)
* [ ] Validate security and performance metrics

---

**Tuna Guest Agent** — Bringing high-end compute to every desktop, effortlessly.

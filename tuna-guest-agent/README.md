# Tuna Windows Client and Linux Server

Tuna is being developed as a **Windows client** that calls a separately built **Linux server**. This repository currently contains a Qt dashboard prototype and an early gRPC sample workload path. It does not yet provide general-purpose remote execution or transparent interception of third-party applications.

## MVP boundary

The first end-to-end API is `RemoteSumOfSquares`: a small Windows sample app sends 1–4,096 unsigned integers to an Ubuntu server and receives the checked sum of their squares. The service requires administrator-provisioned mutual TLS certificates and permits a configured client certificate SAN. It is a protocol/workload vertical slice, not a GPU benchmark.

The intended Linux deployment host is Ubuntu 22.04/24.04 x64 with an NVIDIA RTX 3060 12 GB. This first workload is CPU-only. It does not use CUDA or intercept arbitrary Windows APIs, graphics calls, files, processes, or existing applications such as Photoshop or Blender. Such support requires separate workload adapters, compatibility and security design, and validation before it can be claimed.

## Repository layout

```text
protocol/tuna/v1/workload.proto  Versioned protobuf/gRPC contract
client/                          Windows RPC client library and sample app
src/gui/                         Qt management GUI (settings plus sample RPC launcher)
tuna-server/                     Separate Linux server CMake project, tests, and systemd unit
docs/                             Architecture, protocol, security, and roadmap
```

The GUI uses the sample RPC executable for the bounded test workload; it does not maintain a persistent session. Connection settings are saved with Qt's per-user settings store. The dashboard displays the last RPC result and supports cancelling a running request, while unrelated process/GPU/throughput metrics remain unavailable.

## Dependencies

### Windows client

- CMake 3.21+
- C++20 compiler (MSVC x64 recommended for the gRPC/vcpkg build)
- Protocol Buffers and gRPC C++ packages
- Qt 6.5+ with the matching compiler kit only when building the GUI

Install gRPC and protobuf with vcpkg, then configure the client:

```powershell
vcpkg install --triplet x64-windows
cmake -S . -B build `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DTUNA_BUILD_GUI=OFF `
  -DTUNA_BUILD_SAMPLE_CLIENT=ON
cmake --build build --config Release
```

To build the GUI, install Qt 6.5+ for the same compiler and set `TUNA_BUILD_GUI=ON` and `CMAKE_PREFIX_PATH` to that Qt installation. Build the GUI and sample client in the same CMake tree; both executables are emitted under `build\bin`. Run `build\bin\tuna_gui.exe`, then configure the server host, port, CA, client certificate, and private key in Connection → Settings. The application stores credential file paths, not private key contents. Ensure the private-key file ACLs restrict access to the Windows user running Tuna. Process data, GPU, and throughput metrics remain unavailable.

For an unsigned developer bundle, build both targets with the matching Qt and vcpkg compiler toolchains, then run:

```powershell
.\scripts\package_windows.ps1 `
  -BuildDirectory build `
  -OutputDirectory dist\tuna-client `
  -QtBinDirectory C:\Qt\6.10.0\msvc2022_64\bin
```

The script copies the two executables and vcpkg runtime DLLs, then invokes Qt's `windeployqt`. It does not sign binaries, create an installer, provision certificates, or install a Windows service; do not distribute the unsigned bundle as a production release.

### Linux server

On Ubuntu 22.04/24.04 x64, install CMake, a C++20 compiler, protobuf, and gRPC development packages. For a server-only build:

```bash
sudo apt update
sudo apt install -y build-essential cmake libgrpc++-dev libprotobuf-dev \
  protobuf-compiler protobuf-compiler-grpc
cmake -S tuna-server -B build-server -DCMAKE_BUILD_TYPE=Release
cmake --build build-server --parallel
ctest --test-dir build-server --output-on-failure
```

Run these commands from the repository root. The Linux server has its own CMake project. To build both the sample client and server on Linux for the local mTLS integration test, configure the root project with the GUI disabled and server enabled:

```bash
cmake -S . -B build-all -DTUNA_BUILD_GUI=OFF \
  -DTUNA_BUILD_SAMPLE_CLIENT=ON -DTUNA_BUILD_LINUX_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-all --parallel
ctest --test-dir build-all --output-on-failure
```

The integration test creates short-lived certificates in a temporary directory and checks a valid call, rejected client CA and SAN, server-name validation, and arithmetic overflow. Windows CI also launches the sample client with `--help` to catch missing runtime dependencies.

## Local mTLS smoke test

Create disposable development certificates using a local CA. The server certificate must contain `DNS:localhost` in its subject alternative names and server-auth usage. The client certificate must contain `DNS:sample-client` and client-auth usage. Do not commit private keys or reuse development certificates in production. The Linux integration test creates these automatically. See [the protocol contract](docs/tunneling_protocol.md) for the certificate trust and identity rules.

Start the server on Linux (or locally for a development test):

```bash
./build-server/tuna_server \
  --listen 0.0.0.0:50051 \
  --cert server.crt \
  --key server.key \
  --client-ca client-ca.crt \
  --allowed-client-san sample-client
```

Run the Windows sample app with a server name present in the server certificate:

```powershell
.\build\bin\tuna_sample_app.exe `
  --server localhost:50051 `
  --ca server-ca.crt `
  --cert client.crt `
  --key client.key `
  3 4
```

Expected result for `3 4` is `sum_of_squares=25`. Use the real DNS name when connecting to a remote server and ensure it matches the server certificate. A TLS, authentication, timeout, or server error must fail the call; the client does not silently compute locally.

## Security and deployment notes

- Mutual TLS is mandatory for the sample service. The client validates the server CA/hostname; the server validates the client CA and configured SAN.
- Production policy requires TLS 1.3 or newer. The current gRPC credential code has not yet demonstrated enforcement of that minimum; verifying and enforcing it is a release blocker.
- The Linux systemd unit and CMake install rules are available, but account provisioning, firewall exposure, certificate rotation/revocation, runtime quotas, and production package/update automation remain incomplete.
- The sample server binds to loopback by default. Expose it to a network only through an explicit `--listen` value and suitable firewall rules.
- Do not send sensitive workloads to this prototype. The sample API is limited and has no persistent job storage.

## Documentation and status

- [Implementation plan](docs/implementation_plan.md) — staged completion roadmap and acceptance criteria.
- [Architecture overview](docs/architecture_overview.md) — client/server boundaries and intended evolution.
- [RPC contract](docs/tunneling_protocol.md) — gRPC, schema, authentication, and compatibility rules.
- [Security model](docs/security_model.md) — trust boundaries, controls, and remaining blockers.
- [Interception design](docs/syscall_interception_design.md) — scoped API boundary and future limitations.
- [Daemon lifecycle](docs/daemonization_workflow.md) — Windows background-service design, not yet implemented.

## Current implementation status

Implemented in this initial slice: versioned protobuf schema, Windows gRPC client library/sample CLI, Linux gRPC mTLS server, exact client SAN authorization, bounded message sizes/input, checked arithmetic, server-side unit and mTLS integration tests, a Qt action that invokes/cancels the sample client asynchronously with persisted non-secret settings, Linux CMake install rules/systemd unit, and an unsigned Windows development-bundle script. Not yet implemented: signed Windows installer/service, TLS 1.3 minimum enforcement verification, production certificate lifecycle, Linux package/update automation, production observability/resource isolation, CUDA/GPU workload, broad API interception, and full supported-OS CI results.

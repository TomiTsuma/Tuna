# Tuna Linux Sample Server

This is a separate Linux deployable for the first Tuna RPC vertical slice. It implements `tuna.v1.WorkloadService/RemoteSumOfSquares` and is currently CPU-only. The intended hardware target is NVIDIA RTX 3060 12 GB, but this service does not access the GPU.

## Supported environments

- Ubuntu 22.04 or 24.04 x64
- C++20 compiler, CMake 3.21+, gRPC C++ and Protocol Buffers development packages
- OpenSSL command line for disposable development certificates (not required at runtime)

## Build and test

From the repository root:

```bash
sudo apt update
sudo apt install -y build-essential cmake libgrpc++-dev libprotobuf-dev \
  protobuf-compiler protobuf-compiler-grpc python3 openssl
cmake -S tuna-server -B build-server -DCMAKE_BUILD_TYPE=Release
cmake --build build-server --parallel
ctest --test-dir build-server --output-on-failure
```

Install the executable, operator guide, and systemd unit under `/usr/local`:

```bash
sudo cmake --install build-server --prefix /usr/local
```

To build both client and server and run the local mTLS interoperability test:

```bash
cmake -S . -B build-all -DTUNA_BUILD_GUI=OFF \
  -DTUNA_BUILD_SAMPLE_CLIENT=ON -DTUNA_BUILD_LINUX_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-all --parallel
ctest --test-dir build-all --output-on-failure
```

## Service contract and limits

The server:

- Requires a client certificate issued by the configured client CA.
- Additionally requires one exact configured client subject alternative name (SAN) value (for example `sample-client` for a certificate DNS SAN of `DNS:sample-client`).
- Validates protocol version and request ID.
- Accepts only 1–4,096 unsigned 64-bit input values and rejects arithmetic overflow.
- Caps incoming gRPC messages at 64 KiB and outgoing messages at 1 KiB.
- Does not persist submitted values or results.
- Binds to `127.0.0.1:50051` by default. Use an explicit listener address only when network exposure is intended.

This is not a general remote code execution service. Do not expose it to untrusted networks or send sensitive workload data.

## Development-only certificates

The integration test creates ephemeral keys/certificates in a temporary directory. Do not reuse those credentials in production. Production certificates must be provisioned by an administrator from a protected CA, and the server private key must be readable only by the dedicated `tuna` service account. The mTLS SAN allowlist is configured as an exact identity value without its `DNS:` prefix; one server instance currently accepts a single configured SAN.

## Install as a systemd service

Create a dedicated account and protected certificate/configuration files after installing:

```bash
sudo useradd --system --user-group --home-dir /nonexistent --shell /usr/sbin/nologin tuna
sudo install -d -o root -g tuna -m 0750 /etc/tuna
sudo install -o root -g tuna -m 0640 server.crt /etc/tuna/server.crt
sudo install -o tuna -g tuna -m 0600 server.key /etc/tuna/server.key
sudo install -o root -g tuna -m 0640 client-ca.crt /etc/tuna/client-ca.crt
```

Create `/etc/tuna/tuna-server.env` with the server SAN identity and a deliberate listener address:

```ini
TUNA_LISTEN_ADDRESS=0.0.0.0:50051
TUNA_ALLOWED_CLIENT_SAN=sample-client
```

The environment file contains configuration, not secrets. Restrict it to root and the service group:

```bash
sudo chown root:tuna /etc/tuna/tuna-server.env
sudo chmod 0640 /etc/tuna/tuna-server.env
sudo systemctl daemon-reload
sudo systemctl enable --now tuna-server.service
sudo systemctl status tuna-server.service
sudo journalctl -u tuna-server.service
```

Open TCP/50051 only to trusted client networks using the host/cloud firewall. The supplied listener and unit file do not configure a firewall, certificate renewal, revocation, or server updates. Validate access and key permissions in the actual deployment environment before enabling the service.

## Production blockers

- TLS 1.3 minimum enforcement has not yet been proven for the selected gRPC/OpenSSL build.
- Certificate lifecycle/revocation, multi-client identity management, packaging, upgrade/rollback, and signed release artifacts remain undefined.
- Resource limits in the service unit are initial defense-in-depth, not a substitute for workload quotas or independent isolation.
- Metrics/health endpoints and independent security review remain outstanding.

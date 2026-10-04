# Windows Client Background Service — Future Design

## Status

The current Windows client is a Qt GUI that launches `tuna_sample_app` as a child process for one bounded RPC. It does not install a Windows Service, run at boot, maintain a persistent tunnel, inject into applications, or expose local IPC. The Linux Tuna Server is a separate Linux service documented in `tuna-server/README.md`; it must not be managed by Windows SCM.

This document records a future lifecycle design only. Do not describe a service as shipped until installer, upgrade, recovery, and clean-machine tests exist.

## Goals and scope

A future Windows background service may own durable client operations when a real product needs session persistence or noninteractive workloads. Its initial authority must remain limited to the explicitly supported Tuna APIs and authenticated gRPC requests. A Windows service does not imply authorization to inspect user processes or hook arbitrary APIs.

Before implementation, determine:

- Whether service mode is needed for the MVP or whether the user-launched client remains the supported mode.
- Which Windows 10/11 builds and service accounts are supported.
- Whether the service runs per-machine or per-user and how interactive GUI sessions are attached.
- Which operations require elevation; default to no elevation and least privilege.
- Who provisions and rotates machine/user client certificates and how ACLs protect private keys.

## Proposed components

1. **GUI:** Runs in the logged-in desktop session. Displays the service/client state and requests only documented operations.
2. **Windows client service (if approved):** Owns background RPC work and reads only its protected configuration and certificates.
3. **Local control channel:** A named pipe with an explicit DACL, peer/user authorization, bounded message size, versioned request schema, and per-operation authorization. Do not expose a loopback unauthenticated HTTP control port.
4. **RPC client:** Maintains TLS peer/hostname validation and client certificate authentication for the Linux server. Configuration errors fail closed.
5. **Installer/updater:** Installs signed binaries, service configuration, ACLs, and recovery policy; it never replaces credentials with defaults or weakens certificate checks.

Service and GUI should exchange status/commands, not private key contents. Do not store a private key in GUI settings or send it through local IPC.

## Lifecycle requirements

### Install and enrollment

- Install only signed, versioned artifacts through a documented installer.
- Create the service identity and grant it only required file, registry, network, and IPC access.
- Provision certificates out of band; validate ownership and access-control lists before use.
- Configure server address, CA, certificate, and key references without storing secret key material in ordinary settings.
- Provide an explicit, auditable enrollment/recovery workflow.

### Startup and runtime

- Report `START_PENDING`/`RUNNING`/`STOP_PENDING`/`STOPPED` with SCM timeouts and checkpoints.
- Validate configuration and credential files before reporting fully ready.
- Use bounded worker queues, RPC deadlines, cancellation, and concurrency/resource limits.
- Expose accurate service health, last RPC result, and sanitized diagnostics to the GUI.
- Do not log workload payloads, credentials, certificate private-key data, or sensitive server responses.

### Shutdown and recovery

- Stop accepting new work, cancel or finish in-flight operations according to the documented workload contract, and close gRPC channels.
- Ensure no hooks, child processes, mapped buffers, or temporary workload files remain; no generic hooks are part of the current client.
- Use SCM recovery only for unexpected service failure, with bounded restart attempts and operator-visible logs.
- On invalid identity, hostname mismatch, missing key, or authorization failure, fail closed; never downgrade to plaintext or report a successful local fallback.

## GUI and local IPC security

- Apply named-pipe ACLs to the intended user/group and service identity.
- Authenticate the peer using Windows identity/token information and verify that each message is allowed for that principal.
- Bound message length, parse only versioned schemas, reject unknown privileged operations, and rate-limit requests.
- Never accept arbitrary command lines, process IDs/pointers, executable paths, or file paths for server-side access.
- Keep status/log output sanitized; avoid exposing certificate paths to unauthorized local users.

## Packaging and verification gates

- MSI/MSIX or another selected installer must support install, repair, signed upgrade, rollback, and uninstall without deleting user data implicitly.
- Test install/start/stop/restart/uninstall on clean Windows 10 and 11 x64 VMs and across user logoff/reboot.
- Test service-account ACLs, forged local IPC requests, malformed/oversized messages, certificate expiry/revocation, server outage, RPC timeout/cancel, upgrade interruption, and recovery.
- The GUI must show the real SCM/service state; no hardcoded “connected,” fake metrics, or successful fallback.
- Obtain a security review before enabling service auto-start or privileged operations.

# Tuna Client/Server Completion Plan

## Purpose and scope

This plan turns the current Tuna Guest Agent concept and GUI prototype into a secure, testable product with a **Windows client** and a **Linux server**. The Linux server is being added under `tuna-server/` in this repository as a distinct build/deployable, while remaining suitable for extraction into its own repository later.

The target experience is that a Windows user can configure and observe a Tuna connection, submit supported work to a Linux Tuna Server, and receive results without compromising the Windows machine or server. “Complete” must mean a clearly defined set of supported operations; it must not imply that arbitrary Windows applications can transparently move their CPU, GPU, system calls, or device access to Linux.

## Initial baseline (before implementation work)

- The repository is a first-commit Windows/Qt GUI prototype with simulated metrics, hardcoded process/tunnel details, and a local-only connect toggle.
- Settings are displayed but not saved or applied.
- The daemon, syscall-hook, tunneling, utility, script, and test source files are empty placeholders. There is no service, network client, configuration backend, or working test target.
- `CMakeLists.txt` defines only a GUI target and hardcodes a local Qt installation path. Several README build/run instructions describe Linux/systemd behavior and do not match this Windows client.
- The design documents describe multiple incompatible or alternative choices for transport, serialization, hooks, and UI. `security_model.md`, `LICENSE`, and several placeholder assets are empty.
- No Linux Tuna Server code or client/server contract is present here.

Accordingly, the roadmap begins with scope and protocol decisions rather than treating the documentation’s aspirational feature claims as implemented requirements.

## Implementation progress

- **Scope/protocol decision:** Recorded the selected platform split, custom sample API boundary, and gRPC/mTLS transport. Runtime tests showed gRPC exposes the DNS SAN value without the `DNS:` prefix; server identity configuration now uses that exact value. The user selected GPU offload as a first-release requirement, a separate matrix-multiplication API while retaining sum-of-squares as a smoke test, and an RTX 3060/CUDA 12.8 initial qualification target.
- **Vertical-slice source:** Added a shared v1 schema, Windows C++ client/sample CLI, Linux server, exact client-SAN authorization, bounded input and overflow checks, unit tests, and a disposable-certificate integration test.
- **GPU vertical-slice work:** Added the bounded `RemoteMatrixMultiply` schema and client/server implementation, a CUDA 12.8 build path targeting compute capability 8.6, strict no-CPU-fallback behavior, startup GPU checks, and single-active-request admission control. Added the GUI action and systemd device allowlist. Hardware correctness/performance testing remains outstanding; CPU-only builds are developer/contract-test builds, not release artifacts.
- **GUI/configuration:** Added per-user non-secret settings persistence, field/file validation, and asynchronous invocation/cancellation of the sample RPC client. The frontend is focused on the supported workload and in-session activity; mock connection states, random metrics, process-count placeholders, and nonfunctional diagnostics tabs have been removed. The GUI makes no persistent-session or telemetry claims.
- **Linux operations:** Added a hardened systemd unit template and server deployment/operator guide, including dedicated-account and certificate-file guidance.
- **Packaging foundation:** Added CMake install rules for the Linux server, operator docs and systemd unit, plus an unsigned Windows GUI/client bundling script that stages Qt and vcpkg runtime dependencies.
- **Repository cleanup:** Removed tracked empty subsystem/build/test placeholders, the invalid empty mockup asset, and machine-specific Qt Creator configuration; filled the MIT license to match the README.
- **Documentation/build:** Replaced conflicting protocol and architecture claims, populated the security model, added platform-specific build instructions, and removed the hardcoded Qt install path from CMake.
- **Validation:** The Qt GUI builds with Qt 6.10/MinGW. Ubuntu 24.04 WSL CPU-only builds compile client/server and pass server unit tests plus mTLS integration, including explicit rejection of GPU work by a non-CUDA server. The current WSL host exposes an RTX 4050 6 GB with a CUDA-capable driver, but has no CUDA compiler/toolkit and does not match the approved RTX 3060 12 GB baseline; the CUDA kernel, GPU CTest, and target-device behavior remain unverified. systemd syntax validates with a temporary executable path, but NVIDIA device access still needs a real service-host test. Ubuntu 22.04 WSL is unavailable because its registered VHDX is missing; hosted Ubuntu 22.04 CI has not been observed. TLS 1.3 minimum enforcement remains a release gate. Windows service and full release stages are incomplete.

## Product and architecture constraints

1. **Platform split:** Build, package, run, and test the client on supported Windows versions; build, package, run, and test the server on supported Linux distributions. Specify exact minimum versions and architectures before implementation.
2. **Explicit supported-work contract:** Select the first workload and its request/result semantics. Begin with a narrow, opt-in job or SDK integration and a sample application. Do not promise general remote execution of arbitrary Windows syscalls or applications.
3. **Protocol contract:** Choose one authoritative transport, framing, serialization, version-negotiation, streaming, timeout, retry, and error model. Keep GUI-to-client control separate from client-to-server protocol.
4. **Security before real workloads:** Specify identity provisioning, server-name validation, credential storage, authorization, revocation, least privilege, logging/privacy, and update/signing policy before sending user workload data.
5. **Separate deployables:** The Windows GUI/client and Linux server are distinct products with distinct build and release pipelines. Share protocol definitions and compatibility tests, not OS-specific binaries.
6. **Measured claims:** Show “connected,” performance metrics, process state, and offload status only when backed by live client/server data. Label unavailable or simulated data clearly during development.

## Agreed MVP decisions

- **Platforms:** Windows 10/11 x64 client; Ubuntu 22.04/24.04 x64 server.
- **First workload:** retain bounded sum-of-squares as a CPU connectivity smoke test; provide GPU offload through a separate bounded matrix-multiplication API. It demonstrates forwarding at an explicit API boundary only; it does not intercept arbitrary third-party applications or operating-system calls.
- **Transport:** gRPC over TLS, with TLS 1.3 as the production minimum.
- **Enrollment:** administrator-provisioned client certificates and mutual TLS; no account/control service in the MVP.
- **Server GPU baseline:** NVIDIA RTX 3060 12 GB, compute capability 8.6, CUDA Toolkit 12.8, compatible official driver, Ubuntu 22.04/24.04 x64. Physical-hardware qualification is a first-release blocker; GPU operation is not verified by the available CPU-only CI runners.

## Decisions still to resolve

These choices affect the architecture and must be recorded in the relevant design documents before dependent implementation begins:

- Which concrete Windows 10/11 releases and Ubuntu point releases/kernel versions are included in the support matrix?
- Define the qualified CUDA/driver maintenance policy, CUDA fault/recovery behavior, and numerical/performance release thresholds. The initial target is CUDA 12.8 GA with NVIDIA Linux driver >=570.26 and CUDA device 0; GPU sharing/isolation and performance targets are not yet established.
- Which measured latency, throughput, availability, and resource-isolation targets are release gates?
- Where may submitted workload data be stored, for how long, and how can the operator/client request its deletion?
- What certificate issuance, rotation, revocation, and recovery procedures will operators use? The MVP assumes administrator-managed certificates but does not yet define their lifecycle.

Record the answers in architecture, protocol, security, and deployment documentation. Do not silently assume answers from the current mock UI or aspirational docs.

## Work plan

### Stage 0 — Define product scope and acceptance contract

**Work**

- Agree the first supported workflow, client/server roles, supported platforms, and out-of-scope behavior.
- Define user, administrator, and operator workflows: enrollment, connect/disconnect, submit work, inspect status, recover from errors, upgrade, and uninstall.
- Select measurable service and performance targets and data-retention/privacy requirements.
- Create a requirements-to-test matrix. Separate MVP requirements from later roadmap items.
- Identify the Linux server repository/project and its maintainers; create or link its implementation plan if needed.

**Exit criteria**

- A signed-off MVP statement names the first workload and explicitly lists unsupported workloads/platforms.
- The product, performance, compatibility, and security acceptance criteria are measurable.
- Cross-repository ownership and a versioned client/server contract have an owner.

### Stage 1 — Reconcile design, repository layout, and build foundations

**Work**

- Update the README and design docs to consistently describe a Windows client and Linux server; remove unsupported Linux-client/systemd instructions from the Windows client guide.
- Establish clear client source boundaries (GUI, client core/service, transport, workload adapters, utilities) and a separately buildable Linux server layout in its repository.
- Replace the machine-specific Qt path with portable CMake configuration and documented dependency discovery. Define separate targets for GUI, core, service/helper components, and tests as appropriate.
- Add reproducible Windows and Linux developer setup, formatting/static-analysis settings, dependency version policy, and CI build jobs.
- Remove or clearly label empty placeholders and generated local IDE files; make the repository tree and build instructions truthful.
- Fill in the security-model document and resolve the empty/misrepresented license and asset placeholders.

**Exit criteria**

- A clean checkout can configure and build the intended Windows client on a documented Windows environment.
- The server repository can configure and build on a documented Linux environment.
- CI builds both targets and runs at least a small automated test set.
- README, source tree, dependencies, and build commands agree.

### Stage 2 — Specify and verify the client/server contract

**Work**

- Publish a versioned protocol schema and compatibility policy covering identity, session setup, requests, results, errors, cancellation, limits, heartbeats, and shutdown.
- Specify authentication and authorization, TLS configuration, peer-name validation, replay protection, resource limits, and safe handling of malformed messages.
- Decide which data is control metadata versus workload payload; define size bounds, streaming/chunking, checksums where needed, and compression policy.
- Create shared contract fixtures and protocol conformance tests usable by both Windows and Linux implementations.
- Document connection state transitions, timeout/retry behavior, idempotency, and what happens when either side disconnects mid-operation.

**Exit criteria**

- The Windows client and Linux server can independently consume the same versioned contract.
- Tests reject incompatible versions, malformed/oversized requests, unauthorized identities, and invalid state transitions.
- The protocol document has one selected transport/serialization path rather than unresolved alternatives.

### Stage 3 — Deliver the secure vertical slice

**Work**

- Implement a minimal Linux server endpoint that authenticates a client, accepts the selected MVP workload, returns a result, and emits structured operational logs.
- Implement the matching Windows client core for connect, submit, wait/cancel, receive result, and disconnect.
- Use real certificate/identity validation and explicit error reporting; do not ship mock credentials or bypass validation for convenience.
- Add a local development/test mode with test identities and a deterministic sample workload, isolated from production credentials.
- Add end-to-end tests that build/run the client and server on their target operating systems, including rejection of invalid certificates and recovery from server/network interruption.

**Exit criteria**

- A Windows client completes the sample workload against a Linux server over the selected secure transport.
- Negative-path tests verify that authentication, timeout, cancellation, and server errors fail visibly and safely.
- The GUI remains clearly identified as disconnected/error/working/complete based on actual core state.

### Stage 4 — Productionize client lifecycle and management UI

**Work**

- Implement durable, validated configuration with explicit handling for secret material; persist non-secret settings separately from credentials.
- Add a Windows service or other approved background-process model only after deciding its account, install, upgrade, recovery, and interactive-user communication model.
- Implement a narrowly permissioned local IPC/API between GUI and client core; authenticate/authorize local control requests and validate all inputs.
- Connect UI settings, connect/disconnect, status, logs, and metrics to real client state. Remove generated/random success signals and hardcoded processes/session details.
- Implement clear states, progress, cancellation, reconnect policy, and actionable user-facing errors.
- Provide installer, upgrade/rollback, uninstall, service recovery, signing, and clean-data-removal behavior appropriate to the selected Windows support matrix.

**Exit criteria**

- A non-developer can install, enroll/configure, connect, run the MVP workload, inspect status, disconnect, upgrade, and uninstall on a clean supported Windows machine.
- UI status and metrics are sourced from the running client/server or explicitly shown as unavailable.
- Service and GUI restart/disconnect behavior is tested without corrupting in-flight work or leaving hooks/resources active.

### Stage 5 — Harden and operate the Linux server

**Work**

- Run the server under a dedicated unprivileged identity with least-privilege filesystem, network, and device access.
- Define workload isolation, per-client quotas, concurrency/backpressure, time/resource limits, cancellation, and cleanup of temporary data.
- Add managed configuration, secret/certificate loading, structured logs, health/readiness checks, metrics, and operator-facing diagnostics.
- Implement Linux packaging/deployment and upgrades for the explicitly supported distribution(s), including service supervision and safe rollback.
- Document firewall/ports, certificate enrollment/rotation, backups if applicable, incident response, and data retention.

**Exit criteria**

- A server operator can deploy, configure, observe, upgrade, and recover the server using documented procedures.
- Isolation and resource-limit tests demonstrate that a client cannot access another client’s data or exceed its assigned limits.
- Server shutdown/restart cleans up sessions and temporary data safely.

### Stage 6 — Add workload support incrementally

**Work**

- Implement each workload as a versioned, testable adapter with explicit input/output types and documented platform/runtime requirements.
- Begin with the Stage 0 MVP adapter and sample client; add further adapters only after compatibility and security review.
- If GPU offload is approved, validate end-to-end device/runtime compatibility and isolation on named hardware before advertising support.
- Treat transparent application interception as a separate research/product phase: define a supported API surface, process consent/policy, ABI/semantic guarantees, failure fallback, and a limited application compatibility matrix before implementing hooks.
- Add workload-specific benchmarks and correctness tests against local execution where meaningful.

**Exit criteria**

- Every advertised workload has conformance, failure, security-boundary, and performance tests.
- Unsupported applications and APIs fail safely and are documented; they are not implied to be remotely virtualized.
- No hook or driver is introduced without a threat model, compatibility plan, rollback path, and explicit user/administrator policy.

### Stage 7 — Release readiness and ongoing maintenance

**Work**

- Conduct an independent security review of client, protocol, server, packaging, and update paths; resolve findings before production release.
- Test supported Windows/Linux versions, clean installation, upgrade/downgrade policy, certificate rotation/revocation, network partitions, resource exhaustion, and recovery.
- Establish signed release artifacts, dependency vulnerability monitoring, SBOM/provenance, release notes, and a vulnerability-reporting process.
- Publish administrator, end-user, developer, protocol, and troubleshooting guides with known limitations and support policy.
- Define version support, protocol deprecation, incident response, telemetry opt-in/retention, and release ownership.

**Exit criteria**

- All release-blocking acceptance criteria pass in CI and on representative clean machines.
- Security review findings are resolved or explicitly accepted by the responsible owner with documented mitigation.
- Signed client and server releases can be installed and operated using published documentation.

## Cross-cutting validation

Maintain automated coverage throughout the stages:

- **Builds:** clean Windows client and Linux server builds using pinned/documented dependencies.
- **Unit tests:** configuration validation, state machines, serialization, policy checks, error mapping, and workload adapters.
- **Contract tests:** shared schema fixtures, protocol-version compatibility, malformed input, size limits, and state transitions.
- **Integration tests:** Windows-to-Linux authenticated request/result flow, timeout, cancellation, disconnect/reconnect, and server restart.
- **Security tests:** invalid/expired/revoked identities, unauthorized requests, malformed payloads, resource exhaustion, local IPC authorization, and isolation boundaries.
- **Quality gates:** static analysis, formatting, dependency/security scanning, installer checks, and end-to-end smoke tests.
- **Performance:** workload-specific baseline and targets under realistic network conditions; report measured values rather than simulated UI data.

## Completion definition

The project is ready for its declared initial release only when the chosen MVP workflow works end to end between a supported Windows client and supported Linux server; identity, transport, workload, lifecycle, and failure behavior are documented and tested; both products can be installed and operated from clean environments; security and release gates are satisfied; and all advertised capabilities accurately reflect implemented behavior. Features outside that declared scope remain explicitly marked future work.

## Working order and dependencies

Stages 0–2 are prerequisites for production implementation. Stage 3 proves the client/server architecture with the smallest complete workflow. Stages 4 and 5 productionize each platform and may proceed in parallel after the protocol contract is stable. Stage 6 expands workloads only after the vertical slice and security boundaries exist. Stage 7 is required before a production release. Revisit earlier decisions when evidence from implementation, testing, or deployment changes the assumptions.

[^plan-change]: This plan can be changed in case of future important information being realized by me or by an AI agent.

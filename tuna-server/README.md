# Tuna Server

Remote compute daemon for the Tuna Guest Agent. Receives redirected file
I/O and GPU/CPU compute calls over the Tuna Tunneling Protocol and
executes them on this machine's own hardware.

**Status:** scaffold only — see
`../docs/plans/2026-07-31-tuna-guest-and-server-completion-design.md`
for the full design and phased roadmap.

## Build

```bash
cmake -B build -S .
cmake --build build
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

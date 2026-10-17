# nvcuda_trace

Pass-through `nvcuda.dll` that forwards every driver API export to the real
`System32\nvcuda.dll` and records:
- calls per export (made through the DLL's exports),
- every `cuGetProcAddress` request (symbol, version, flags),
- which `cuGetExportTable` UUIDs are requested (tables are returned untouched).

It is the coverage tool for growing Tuna's remoted API set (Phase 3a).

```bash
python gen_trace.py C:/Windows/System32/nvcuda.dll gen
gcc -O2 -shared -Igen -o bin/nvcuda.dll trace.c gen/thunks.S gen/exports.def   # MinGW
```

Load it before the app initializes CUDA (`os.add_dll_directory("bin")` or
`LoadLibrary("bin/nvcuda.dll")`; a `PATH` entry is not enough). The report is
written at exit to `%TUNA_TRACE_LOG%` (default `%TEMP%\tuna_nvcuda_trace.txt`).
Findings: `docs/plans/2026-10-16-phase3-spike-findings.md`.

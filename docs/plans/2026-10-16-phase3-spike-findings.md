# Phase 3 spike findings: how cudart reaches the driver (2026-10-16)

**Plan step:** execution plan §8 step 3 / Phase 3b ("investigate `cuGetExportTable` in week 1").
**Setup:**
- Windows 11, RTX 4050 Laptop, driver 571.96 (CUDA 12.8).
- PyTorch 2.6.0+cu118, which bundles cudart 11.8, cuBLAS and cuDNN DLLs.
- Tool: `tools/nvcuda_trace/`, a pass-through `nvcuda.dll` that forwards all 694 exports to the real System32 driver and counts calls.

## Findings

1. **Loading the shim.** cudart loads `nvcuda.dll` with restricted search flags:

   | Method | Shim loaded? |
   |---|---|
   | Shim directory first on `PATH` | **No** |
   | `os.add_dll_directory(dir)` / `AddDllDirectory` | **Yes** |
   | `LoadLibrary` of the shim before CUDA initializes | **Yes** (the loader returns the already loaded module) |

   So `tuna run` needs no code-patching hooks. It only has to pre-load the shim, or add its directory, before the app touches CUDA.

2. **Entry points come from `cuGetProcAddress`, not imports.** cudart's first act is `cuGetProcAddress("cuGetProcAddress", 11030)`. It then resolves about 400 `(symbol, version, flags)` pairs through that pointer, for example `("cuMemAlloc", 3020)` and `("cuMemcpy", 7000, flags=2)` (per-thread default stream). The pointers the driver returns are **not** its exported addresses.

   The shim's `cuGetProcAddress` must therefore resolve by name + version + flags. The version table can be generated from CUDA's `cudaTypedefs.h` (`PFN_<symbol>_v<version>` typedefs).

3. **cudart 11.8 requests 6 undocumented export tables** (`cuGetExportTable`, UUIDs below), once per DLL that statically links cudart. These tables are **internal to NVIDIA's runtime/driver pair and integrity-checked**: returning anything other than the driver's own table makes cudart fail with `cudaErrorIntegrityCheck` (103). Tuna will **not** try to emulate or bypass them.

   ```
   6bd5fb6c5bf4e74a8987d93912fd9df9  a094798c2e742e7493f20800200c0a66
   42d85a8123f6cb478298f6e78a3aecdc  c693336e1121df11a8c368f355d89593
   d4082055bde6704b8d34ba123c66e1f2  21318c60971432488ca641ff7324c8f2
   ```

## Consequence for Phase 3

A pure driver-API shim, which is the plan's first choice, **cannot serve NVIDIA's cudart, and therefore cannot serve PyTorch**, without reimplementing integrity-checked internals. This is the risk in plan §7, and it has materialized. The documented-boundary options are:

| Option | What it supports | Cost / risk |
|---|---|---|
| **A. Driver-API shim only** | Apps that use the driver API directly (custom runtimes, NVRTC/PTX workflows, cuda-python's driver bindings, CuPy's driver paths). Plan milestones 1–3 can be met with driver-API samples. | Smallest. It does not reach PyTorch, which is plan milestones 4–5. |
| **B. Runtime-API shim** (replace `cudart64_*.dll`; the rCUDA/Cricket model) | PyTorch on Windows, which loads cudart as a DLL, plus anything else linking cudart dynamically | A much larger API surface (runtime + kernel registration via `__cudaRegisterFatBinary`), and cuBLAS/cuDNN also need shims or remote execution. It only works where cudart is a separate DLL. |
| **C. Framework-level remoting** (for example a PyTorch device backend, or Track A's explicit API) | PyTorch workloads | It is not transparent; it is a different product. |

Recommendation: build **A** now. It is the shared foundation: codegen, marshalling, handle tables, module upload, the launch path and the server dispatcher are all needed by B as well. In parallel, decide whether B is worth its cost before committing to the PyTorch milestones.

#!/usr/bin/env python3
"""Generates the sources of a tracing pass-through libcudnn.so.9 (Linux).

Every export of the real libcudnn becomes a two-instruction x86-64 thunk:
count the call, then jump to the real function. The backend (graph) API
calls are wrapped in C instead (trace.c), so the report can say which
descriptor types, attributes and finalizes an application uses.

Usage: gen_trace.py REAL_LIBCUDNN OUT_DIR
Writes OUT_DIR/thunks.S, OUT_DIR/names.inc and OUT_DIR/libcudnn.map.
"""

import subprocess
import sys
from pathlib import Path

# Exported by trace.c instead of a plain thunk.
WRAPPED = {
    "cudnnBackendCreateDescriptor",
    "cudnnBackendDestroyDescriptor",
    "cudnnBackendSetAttribute",
    "cudnnBackendGetAttribute",
    "cudnnBackendFinalize",
    "cudnnBackendExecute",
}


def elf_exports(path):
    out = subprocess.run(["nm", "-D", "--defined-only", path], capture_output=True, text=True, check=True).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in "TW" and parts[2].startswith("cudnn"):
            names.add(parts[2].split("@")[0])
    return sorted(names)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    names = elf_exports(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "names.inc", "w") as f:
        f.write(f"#define TUNA_EXPORT_COUNT {len(names)}\n")
        f.write("static const char* const tuna_names[TUNA_EXPORT_COUNT] = {\n")
        f.writelines(f'    "{n}",\n' for n in names)
        f.write("};\n")
        for i, n in enumerate(names):
            if n in WRAPPED:
                f.write(f"#define TUNA_INDEX_{n} {i}\n")

    with open(out / "thunks.S", "w") as f:
        f.write(".text\n")
        for i, n in enumerate(names):
            if n in WRAPPED:
                continue
            f.write(f".globl {n}\n.type {n},@function\n{n}:\n")
            f.write(f"    lock incq tuna_counts+{8 * i}(%rip)\n")
            f.write(f"    jmp *tuna_real+{8 * i}(%rip)\n")
        f.write('.section .note.GNU-stack,"",@progbits\n')

    with open(out / "libcudnn.map", "w") as f:
        f.write("libcudnn.so.9 {\n  global:\n    cudnn*;\n  local:\n    *;\n};\n")
    print(f"{len(names)} exports, {len(WRAPPED & set(names))} wrapped")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generates Tuna's CUDA runtime-API remoting stubs from the real headers.

Parses cuda_runtime_api.h with libclang, cross-references annotations.yaml for
pointer semantics, and emits (into OUT_DIR):

  cuda_api_ids.h          enum TunaCudaApi + name table (stable ids)
  cuda_client_stubs.inc   client shim: one C++ function body per API
  cuda_server_dispatch.inc server: switch(api_id){ decode -> real call -> encode }
  cuda_coverage.inc        the set of generated names (for logging unknowns)

The flat little-endian arg encoding is the one in the wire plan (§4): too hot
for protobuf-per-call. Handles (device pointers) cross as uint64; the server
owns the real values. Functions without an annotation become stubs that return
cudaErrorNotSupported and log their name, so coverage grows by use.

Usage: gen_cuda.py CUDA_INCLUDE_DIR annotations.yaml OUT_DIR
"""

import sys
from pathlib import Path

import clang.cindex as ci
import yaml

# cudaError_t values we emit directly (no CUDA headers in this file).
CUDA_SUCCESS = 0
CUDA_ERROR_NOT_SUPPORTED = 3  # cudaErrorInitializationError-adjacent; see note below
# NOTE: 3 is cudaErrorInitializationError in older runtimes; cudaErrorNotSupported
# is 801. We use 801 to match the driver-API convention used elsewhere in Tuna.
NOT_SUPPORTED = 801
TRANSPORT_ERROR = 999


def parse(header, include_dir):
    index = ci.Index.create()
    tu = index.parse(str(header), args=[
        "-x", "c", "-I", str(include_dir),
        "-D__CUDACC__", "-D__cudart_builtin__=", "-DCUDARTAPI=", "-D__host__=", "-D__device__=",
    ])
    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if errors:
        for d in errors[:10]:
            print(f"  parse error: {d.spelling}", file=sys.stderr)
        raise SystemExit(f"{header}: {len(errors)} parse error(s)")
    functions = {}
    for c in tu.cursor.walk_preorder():
        if c.kind != ci.CursorKind.FUNCTION_DECL or not c.spelling.startswith("cuda"):
            continue
        if c.spelling in functions:  # keep the first declaration
            continue
        functions[c.spelling] = {
            "result": c.result_type.spelling,
            "args": [(a.type.spelling, a.spelling, a.type) for a in c.get_arguments()],
        }
    return functions


def width_of(clang_type):
    size = clang_type.get_size()
    if size <= 0:
        raise SystemExit(f"cannot size type '{clang_type.spelling}'")
    return size


class Field:
    """One marshalled parameter."""
    def __init__(self, name, role, ctype, width):
        self.name = name
        self.role = role      # in_scalar | in_handle | out_scalar | out_handle | out_struct
        self.ctype = ctype    # C type spelling (e.g. "int *", "size_t")
        self.width = width    # bytes on the wire


def classify(func, annotation):
    fields = []
    arg_by_name = {name: (ctype, t) for ctype, name, t in func["args"]}
    for name, role in annotation.items():
        if name not in arg_by_name:
            raise SystemExit(f"annotation names unknown parameter '{name}'")
        ctype, t = arg_by_name[name]
        if role in ("in_handle", "out_handle"):
            width = 8
        elif role == "in_scalar":
            width = width_of(t)
        elif role in ("out_scalar", "out_struct"):
            width = width_of(t.get_pointee())
        else:
            raise SystemExit(f"unknown role '{role}' for {name}")
        fields.append(Field(name, role, ctype, width))
    # Keep declaration order so the encoding is stable and readable.
    order = {name: i for i, (_, name, _) in enumerate(func["args"])}
    fields.sort(key=lambda f: order[f.name])
    return fields


def real_signature(func):
    args = ", ".join(ctype for ctype, _, _ in func["args"]) or "void"
    return f"{func['result']} (*)({args})"


def emit_client_stub(name, api_id, func, fields):
    ins = [f for f in fields if f.role.startswith("in")]
    outs = [f for f in fields if f.role.startswith("out")]
    params = ", ".join(f"{ctype} {aname}" for ctype, aname, _ in func["args"]) or "void"
    lines = [f"{func['result']} {name}({params}) {{",
             "    tuna::cudart::ArgWriter in;"]
    for f in ins:
        if f.role == "in_handle":
            lines.append(f"    in.put_u64(reinterpret_cast<uint64_t>({f.name}));")
        else:
            lines.append(f"    in.put(&{f.name}, {f.width});")
    lines.append(f"    tuna::cudart::Reply reply = tuna::cudart::Call({api_id}, in);")
    lines.append("    if (reply.transport_error) return static_cast<cudaError_t>(999);")
    if outs:
        lines.append("    tuna::cudart::ArgReader out(reply.out_args);")
        for f in outs:
            if f.role == "out_handle":
                lines.append(f"    *{f.name} = reinterpret_cast<void*>(out.get_u64());")
            else:
                lines.append(f"    if ({f.name}) out.get({f.name}, {f.width});")
    lines.append("    return static_cast<cudaError_t>(reply.result);")
    lines.append("}")
    return "\n".join(lines)


def emit_dispatch_case(name, api_id, func, fields):
    ins = [f for f in fields if f.role.startswith("in")]
    outs = [f for f in fields if f.role.startswith("out")]
    lines = [f"    case {api_id}: {{  // {name}",
             "        tuna::cudart::ArgReader in(request.args());"]
    call_args = []
    for ctype, aname, t in func["args"]:
        field = next((f for f in fields if f.name == aname), None)
        if field is None:
            lines.append(f"        resp.set_result(801); break;  // unannotated param {aname}")
            break
        if field.role == "in_handle":
            lines.append(f"        void* {aname} = reinterpret_cast<void*>(in.get_u64());")
            call_args.append(aname)
        elif field.role == "in_scalar":
            base = ctype
            lines.append(f"        {base} {aname}{{}}; in.get(&{aname}, {field.width});")
            call_args.append(aname)
        elif field.role == "out_handle":
            lines.append(f"        void* {aname} = nullptr;")
            call_args.append(f"&{aname}")
        elif field.role in ("out_scalar", "out_struct"):
            pointee = ctype.rstrip(" *")
            lines.append(f"        {pointee} {aname}{{}};")
            call_args.append(f"&{aname}")
    else:
        sig = real_signature(func)
        lines.append(f'        auto fn = reinterpret_cast<{sig}>(backend.Resolve("{name}"));')
        lines.append("        if (!fn) { resp.set_result(801); break; }")
        lines.append(f"        resp.set_result(fn({', '.join(call_args)}));")
        lines.append("        tuna::cudart::ArgWriter out;")
        for f in outs:
            if f.role == "out_handle":
                lines.append(f"        out.put_u64(reinterpret_cast<uint64_t>({f.name}));")
            else:
                lines.append(f"        out.put(&{f.name}, {f.width});")
        lines.append("        resp.set_out_args(out.str());")
        lines.append("        break;")
    lines.append("    }")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    include_dir, annotations_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    functions = parse(include_dir / "cuda_runtime_api.h", include_dir)
    annotations = yaml.safe_load(annotations_path.read_text())
    simple = annotations.get("functions", {}) or {}
    special = annotations.get("special", {}) or {}

    # Stable ids: 0 is the diagnostic echo; CUDA apis start at 1, sorted so the
    # assignment does not shift when annotations are reordered.
    names = sorted(set(simple) | set(special))
    api_id = {name: i + 1 for i, name in enumerate(names)}

    missing = [n for n in simple if n not in functions]
    if missing:
        raise SystemExit(f"annotated functions not found in header: {missing}")

    ids_h = ["// Generated by gen_cuda.py. Do not edit.", "#pragma once", "#include <cstdint>",
             "namespace tuna::cudart {", "enum TunaCudaApi : uint32_t {"]
    for name in names:
        ids_h.append(f"    kApi_{name} = {api_id[name]},")
    ids_h += ["};", f"constexpr uint32_t kFirstCudaApi = 1;",
              f"constexpr uint32_t kLastCudaApi = {len(names)};", "}  // namespace tuna::cudart"]
    (out_dir / "cuda_api_ids.h").write_text("\n".join(ids_h) + "\n")

    client = ["// Generated by gen_cuda.py. Do not edit."]
    dispatch = ["// Generated by gen_cuda.py. Do not edit.", "    switch (request.api_id()) {"]
    for name in names:
        if name in special:
            client.append(f"// {name}: hand-written (special:{special[name]}), see cuda_special.cpp")
            dispatch.append(f"    // {name}: hand-written (special:{special[name]})")
            continue
        fields = classify(functions[name], simple[name])
        client.append(emit_client_stub(name, f"kApi_{name}", functions[name], fields))
        dispatch.append(emit_dispatch_case(name, f"kApi_{name}", functions[name], fields))
    dispatch += ["    default: resp.set_result(801); break;", "    }"]
    (out_dir / "cuda_client_stubs.inc").write_text("\n\n".join(client) + "\n")
    (out_dir / "cuda_server_dispatch.inc").write_text("\n".join(dispatch) + "\n")

    coverage = ["// Generated by gen_cuda.py. Do not edit.",
                "static const char* const kCudaApiNames[] = {", '    "echo",']
    for name in names:
        coverage.append(f'    "{name}",')
    coverage += ["};"]
    (out_dir / "cuda_coverage.inc").write_text("\n".join(coverage) + "\n")

    print(f"generated {len(names)} apis ({len(simple)} simple, {len(special)} special) "
          f"from {len(functions)} runtime functions")


if __name__ == "__main__":
    main()

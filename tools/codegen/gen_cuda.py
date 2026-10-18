#!/usr/bin/env python3
"""Generates Tuna's CUDA runtime-API remoting stubs from the real headers.

Parses cuda_runtime_api.h with libclang, cross-references annotations.yaml for
pointer semantics, and emits (into OUT_DIR):

  cuda_api_ids.h              enum TunaCudaApi (stable ids from api_ids.yaml)
  cuda_client_stubs.inc       client shim: one C++ function per annotated API
  cuda_server_dispatch.inc    server: switch(api_id){ decode -> real call -> encode }
  cuda_unsupported_stubs.inc  every other libcudart export, returning
                              cudaErrorNotSupported (801), so any program loads
  cuda_error_strings.inc      cudaGetErrorName/String tables (from driver_types.h)
  cuda_coverage.inc           id -> name table (for logging)

The flat little-endian arg encoding is the one in the wire plan (§4): too hot
for protobuf-per-call. Handles (device pointers, streams, events) cross as
uint64; the server owns the real values and the client never dereferences them.

api_ids.yaml is an append-only registry: an id, once assigned, never changes,
so a newer client and an older server still agree on every call they share.
New names get the next free id and the registry is rewritten.

Usage: gen_cuda.py CUDA_INCLUDE_DIR annotations.yaml OUT_DIR
       (api_ids.yaml and cudart12_exports.txt are read next to annotations.yaml)
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import clang.cindex as ci
import yaml

NOT_SUPPORTED = 801
ROLES = ("in_scalar", "in_handle", "in_struct", "out_scalar", "out_handle", "out_struct")


def system_include_args():
    """pip's libclang does not know where libc headers live on Linux; borrow
    gcc's. (On Windows libclang finds the MSVC/SDK headers by itself.)"""
    if sys.platform == "win32" or not shutil.which("gcc"):
        return []
    gcc_include = subprocess.run(["gcc", "-print-file-name=include"], capture_output=True,
                                 text=True).stdout.strip()
    return ["-isystem", gcc_include, "-isystem", "/usr/include"]


def parse_tu(include_dir):
    index = ci.Index.create()
    tu = index.parse(str(include_dir / "cuda_runtime_api.h"), args=system_include_args() + [
        "-x", "c", "-I", str(include_dir),
        "-D__CUDACC__", "-D__cudart_builtin__=", "-DCUDARTAPI=", "-D__host__=", "-D__device__=",
    ], options=ci.TranslationUnit.PARSE_INCLUDE_BRIEF_COMMENTS_IN_CODE_COMPLETION)
    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if errors:
        for d in errors[:10]:
            print(f"  parse error: {d.spelling}", file=sys.stderr)
        raise SystemExit(f"cuda_runtime_api.h: {len(errors)} parse error(s)")
    return tu


def collect_functions(tu):
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


def collect_errors(tu):
    """(value, name, description) for every cudaError enumerator."""
    errors = {}
    for c in tu.cursor.walk_preorder():
        if c.kind == ci.CursorKind.ENUM_DECL and c.spelling == "cudaError":
            for e in c.get_children():
                if e.kind != ci.CursorKind.ENUM_CONSTANT_DECL or e.enum_value in errors:
                    continue
                text = (e.brief_comment or e.raw_comment or e.spelling).strip()
                text = re.sub(r"^/\*\*?<?|\*/$", "", text).strip()
                text = re.sub(r"(^|\s)\*(?=\s)", " ", text)  # doc-comment line markers
                text = text.replace(r"\deprecated","Deprecated:").replace("::", "")
                text = re.sub(r"\s+", " ", text).strip()
                errors[e.enum_value] = (e.spelling, text)
    if not errors:
        raise SystemExit("cudaError enum not found")
    return sorted((v, n, d) for v, (n, d) in errors.items())


def width_of(clang_type):
    size = clang_type.get_size()
    if size <= 0:
        raise SystemExit(f"cannot size type '{clang_type.spelling}'")
    return size


class Field:
    """One marshalled parameter."""
    def __init__(self, name, role, ctype, pointee, width):
        self.name = name
        self.role = role        # one of ROLES
        self.ctype = ctype      # C type spelling of the parameter
        self.pointee = pointee  # for pointer roles: the pointed-to type spelling
        self.width = width      # bytes on the wire


def classify(name, func, annotation):
    fields = []
    arg_by_name = {aname: (ctype, t) for ctype, aname, t in func["args"]}
    for pname, role in annotation.items():
        if pname.startswith("_"):
            continue  # function flags (_async, _sync)
        if pname not in arg_by_name:
            raise SystemExit(f"{name}: annotation names unknown parameter '{pname}'")
        if role not in ROLES:
            raise SystemExit(f"{name}: unknown role '{role}' for {pname}")
        ctype, t = arg_by_name[pname]
        pointee = None
        if role in ("in_handle", "in_scalar"):
            width = 8 if role == "in_handle" else width_of(t)
        elif role == "out_handle":
            width, pointee = 8, t.get_pointee().spelling
        else:  # out_scalar, out_struct, in_struct: a pointer to a fixed-size value
            pointee_type = t.get_pointee()
            width, pointee = width_of(pointee_type), pointee_type.spelling
            if role == "in_struct":
                pointee = re.sub(r"^const\s+", "", pointee)
        fields.append(Field(pname, role, ctype, pointee, width))
    unmarshalled = set(arg_by_name) - {f.name for f in fields}
    if unmarshalled:
        raise SystemExit(f"{name}: parameters without an annotation: {sorted(unmarshalled)}")
    if annotation.get("_async") and any(f.role.startswith("out") for f in fields):
        raise SystemExit(f"{name}: an _async function cannot have OUT parameters")
    order = {aname: i for i, (_, aname, _) in enumerate(func["args"])}
    fields.sort(key=lambda f: order[f.name])
    return fields


def real_signature(func):
    args = ", ".join(ctype for ctype, _, _ in func["args"]) or "void"
    return f"{func['result']} (*)({args})"


def emit_client_stub(name, api, func, fields, flags):
    params = ", ".join(f"{ctype} {aname}" for ctype, aname, _ in func["args"]) or "void"
    # _wrap: the client exports a hand-written function of this name (caching,
    # allocation tracking) that calls this generated one.
    head = f"static {func['result']} tuna_remote_{name}" if flags.get("_wrap") else         f"TUNA_CUDART_EXPORT {func['result']} {name}"
    lines = [f"{head}({params}) {{",
             "    tuna::cudart::ArgWriter in;"]
    for f in fields:
        if f.role == "in_handle":
            lines.append(f"    in.put_u64(reinterpret_cast<uint64_t>({f.name}));")
        elif f.role == "in_scalar":
            lines.append(f"    in.put(&{f.name}, {f.width});")
        elif f.role == "in_struct":
            lines.append(f"    if (!{f.name}) return static_cast<cudaError_t>(tuna::cudart::Fail(cudaErrorInvalidValue));")
            lines.append(f"    in.put({f.name}, {f.width});")
    if flags.get("_async"):
        lines.append(f"    return static_cast<cudaError_t>(tuna::cudart::Post({api}, in));")
        lines.append("}")
        return "\n".join(lines)
    sync = "true" if flags.get("_sync") else "false"
    lines.append(f"    tuna::cudart::Reply reply = tuna::cudart::Call({api}, in, {{}}, {sync});")
    outs = [f for f in fields if f.role.startswith("out")]
    if outs:
        lines.append("    if (!reply.transport_error && reply.result == 0) {")
        lines.append("        try {")
        lines.append("            tuna::cudart::ArgReader out(reply.out_args);")
        for f in outs:
            if f.role == "out_handle":
                lines.append(f"            const uint64_t {f.name}_h = out.get_u64();")
                lines.append(f"            if ({f.name}) *{f.name} = reinterpret_cast<{f.pointee}>({f.name}_h);")
            else:
                lines.append(f"            if ({f.name}) out.get({f.name}, {f.width}); else out.skip({f.width});")
        lines.append("        } catch (const std::exception&) {")
        lines.append("            return static_cast<cudaError_t>(tuna::cudart::Fail(cudaErrorUnknown));  // malformed reply")
        lines.append("        }")
        lines.append("    }")
    lines.append("    return static_cast<cudaError_t>(tuna::cudart::Finish(reply));")
    lines.append("}")
    return "\n".join(lines)


def emit_dispatch_case(name, api, func, fields):
    lines = [f"    case {api}: {{  // {name}",
             "        tuna::cudart::ArgReader in(request.args());"]
    call_args = []
    outs = []
    for ctype, aname, _ in func["args"]:
        f = next(f for f in fields if f.name == aname)
        if f.role == "in_handle":
            lines.append(f"        {ctype} {aname} = reinterpret_cast<{ctype}>(in.get_u64());")
            call_args.append(aname)
        elif f.role == "in_scalar":
            lines.append(f"        {ctype} {aname}{{}};")
            lines.append(f"        in.get(&{aname}, {f.width});")
            call_args.append(aname)
        elif f.role == "in_struct":
            lines.append(f"        {f.pointee} {aname}{{}};")
            lines.append(f"        in.get(&{aname}, {f.width});")
            call_args.append(f"&{aname}")
        else:  # out_*
            lines.append(f"        {f.pointee} {aname}{{}};")
            call_args.append(f"&{aname}")
            outs.append(f)
    lines.append("        if (in.remaining() != 0) { resp.set_result(kCudaErrorInvalidValue); break; }")
    lines.append(f'        auto fn = reinterpret_cast<{real_signature(func)}>(backend_.Resolve("{name}"));')
    lines.append("        if (!fn) { resp.set_result(kCudaErrorNotSupported); break; }")
    lines.append(f"        resp.set_result(fn({', '.join(call_args)}));")
    if outs:
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


def load_registry(path):
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    ids = list(data.values())
    if len(ids) != len(set(ids)) or 0 in ids:
        raise SystemExit(f"{path}: ids must be unique and non-zero (0 is the echo)")
    return data


def save_registry(path, registry):
    lines = ["# Append-only api_id registry for remoted CUDA runtime calls.",
             "# gen_cuda.py adds new names at the end. Never renumber or reuse an id:",
             "# clients and servers of different versions agree on calls by id.",
             "# 0 is reserved for the tunnel's echo call."]
    for name, api_id in sorted(registry.items(), key=lambda kv: kv[1]):
        lines.append(f"{name}: {api_id}")
    path.write_text("\n".join(lines) + "\n")


def c_string(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    include_dir, annotations_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    codegen_dir = annotations_path.parent

    tu = parse_tu(include_dir)
    functions = collect_functions(tu)
    annotations = yaml.safe_load(annotations_path.read_text())
    simple = annotations.get("functions", {}) or {}
    special = annotations.get("special", {}) or {}   # remoted, marshalled by hand
    local = set(annotations.get("local", []) or [])  # implemented in the client only

    missing = [n for n in simple if n not in functions]
    if missing:
        raise SystemExit(f"annotated functions not found in header: {missing}")
    overlap = (set(simple) & set(special)) | (set(simple) & local) | (set(special) & local)
    if overlap:
        raise SystemExit(f"functions listed more than once: {sorted(overlap)}")

    registry_path = codegen_dir / "api_ids.yaml"
    registry = load_registry(registry_path)
    remoted = sorted(set(simple) | set(special))
    next_id = max(registry.values(), default=0) + 1
    for name in remoted:
        if name not in registry:
            registry[name] = next_id
            next_id += 1
    save_registry(registry_path, registry)

    ids_h = ["// Generated by gen_cuda.py. Do not edit.", "#pragma once", "#include <cstdint>",
             "namespace tuna::cudart {", "enum TunaCudaApi : uint32_t {"]
    for name, api_id in sorted(registry.items(), key=lambda kv: kv[1]):
        ids_h.append(f"    kApi_{name} = {api_id},")
    ids_h += ["};", "constexpr uint32_t kFirstCudaApi = 1;",
              f"constexpr uint32_t kLastCudaApi = {max(registry.values())};",
              "}  // namespace tuna::cudart"]
    (out_dir / "cuda_api_ids.h").write_text("\n".join(ids_h) + "\n")

    client = ["// Generated by gen_cuda.py. Do not edit."]
    dispatch = ["// Generated by gen_cuda.py. Do not edit.", "    switch (request.api_id()) {"]
    for name in sorted(simple):
        fields = classify(name, functions[name], simple[name])
        client.append(emit_client_stub(name, f"kApi_{name}", functions[name], fields, simple[name]))
        dispatch.append(emit_dispatch_case(name, f"kApi_{name}", functions[name], fields))
    dispatch += ["    default: resp.set_result(kCudaErrorNotSupported); break;", "    }"]
    (out_dir / "cuda_client_stubs.inc").write_text("\n\n".join(client) + "\n")
    (out_dir / "cuda_server_dispatch.inc").write_text("\n".join(dispatch) + "\n")

    # Every other libcudart export becomes a stub, so a program that links any
    # runtime symbol still loads; calling one reports cudaErrorNotSupported.
    # These are emitted without prototypes (their TU includes no CUDA header):
    # the C ABI ignores unused arguments, and all of them return cudaError_t.
    exports = (codegen_dir / "cudart12_exports.txt").read_text().split()
    implemented = set(simple) | set(special) | local
    unsupported = [n for n in exports if n not in implemented]
    stubs = ["// Generated by gen_cuda.py. Do not edit."]
    for name in unsupported:
        stubs.append(f'TUNA_CUDART_EXPORT int {name}() {{ return tuna::cudart::Unsupported("{name}"); }}')
    (out_dir / "cuda_unsupported_stubs.inc").write_text("\n".join(stubs) + "\n")

    errors = collect_errors(tu)
    err = ["// Generated by gen_cuda.py. Do not edit.",
           "struct CudaErrorText { int value; const char* name; const char* text; };",
           "static const CudaErrorText kCudaErrorTexts[] = {"]
    for value, ename, text in errors:
        err.append(f"    {{{value}, {c_string(ename)}, {c_string(text)}}},")
    err.append("};")
    (out_dir / "cuda_error_strings.inc").write_text("\n".join(err) + "\n")

    coverage = ["// Generated by gen_cuda.py. Do not edit.",
                "struct CudaApiName { uint32_t id; const char* name; };",
                "static const CudaApiName kCudaApiNames[] = {"]
    for name, api_id in sorted(registry.items(), key=lambda kv: kv[1]):
        coverage.append(f'    {{{api_id}, "{name}"}},')
    coverage += ["};"]
    (out_dir / "cuda_coverage.inc").write_text("\n".join(coverage) + "\n")

    print(f"generated {len(remoted)} remoted apis ({len(simple)} simple, {len(special)} special), "
          f"{len(local)} client-local, {len(unsupported)} unsupported stubs, {len(errors)} error codes; "
          f"{len(functions)} runtime functions in the header")


if __name__ == "__main__":
    main()

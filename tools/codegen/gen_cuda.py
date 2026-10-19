#!/usr/bin/env python3
"""Generates Tuna's CUDA runtime-API (and cuBLAS) remoting stubs from the real headers.

Parses cuda_runtime_api.h with libclang, cross-references annotations.yaml for
pointer semantics, and emits (into OUT_DIR):

  cuda_api_ids.h              enum TunaCudaApi (stable ids from api_ids.yaml)
  cuda_client_stubs.inc       client shim: one C++ function per annotated API
  cuda_server_dispatch.inc    server: switch(api_id){ decode -> real call -> encode }
  cuda_unsupported_stubs.inc  every other libcudart export, returning
                              cudaErrorNotSupported (801), so any program loads
  cuda_error_strings.inc      cudaGetErrorName/String tables (from driver_types.h)
  cuda_coverage.inc           id -> name table (for logging)

With --cublas, it also parses cublas_v2.h with annotations_cublas.yaml (next to
annotations.yaml) and emits the replacement libcublas's pieces:

  cublas_client_stubs.inc       client: one exported function per annotated API
  cublas_server_dispatch.inc    server: decode -> real cuBLAS call -> encode
  cublas_unsupported_stubs.inc  every other libcublas export, returning
                                CUBLAS_STATUS_NOT_SUPPORTED
  cublaslt_unsupported_stubs.inc  every libcublasLt export not remoted by hand
                                (annotations_cublaslt.yaml, cublasLt12_exports.txt)

Always, it also parses cuda.h with annotations_driver.yaml and emits the
replacement libcuda's (driver API's) pieces:

  driver_client_stubs.inc       client: one exported function per annotated API
  driver_server_dispatch.inc    server: decode -> real driver call -> encode
  driver_unsupported_stubs.inc  every other libcuda export (cuda12_driver_exports.txt),
                                returning CUDA_ERROR_NOT_SUPPORTED
  driver_error_strings.inc      cuGetErrorName/String tables (from cuda.h)

The flat little-endian arg encoding is the one in the wire plan (§4): too hot
for protobuf-per-call. Handles (device pointers, streams, events) cross as
uint64; the server owns the real values and the client never dereferences them.

api_ids.yaml is an append-only registry: an id, once assigned, never changes,
so a newer client and an older server still agree on every call they share.
New names get the next free id and the registry is rewritten.

Usage: gen_cuda.py CUDA_INCLUDE_DIR annotations.yaml OUT_DIR [--cublas CUBLAS_INCLUDE_DIR]
       (api_ids.yaml, cudart12_exports.txt, annotations_cublas.yaml,
       cublas12_exports.txt, annotations_driver.yaml and
       cuda12_driver_exports.txt are read next to annotations.yaml)
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import clang.cindex as ci
import yaml

NOT_SUPPORTED = 801
ROLES = ("in_scalar", "in_handle", "in_struct", "out_scalar", "out_handle", "out_struct",
         "in_scalar_ref", "out_scalar_ref")


def system_include_args():
    """pip's libclang does not know where libc headers live on Linux; borrow
    gcc's. (On Windows libclang finds the MSVC/SDK headers by itself.)"""
    if sys.platform == "win32" or not shutil.which("gcc"):
        return []
    gcc_include = subprocess.run(["gcc", "-print-file-name=include"], capture_output=True,
                                 text=True).stdout.strip()
    return ["-isystem", gcc_include, "-isystem", "/usr/include"]


def parse_tu(header, include_dirs, extra_args=()):
    index = ci.Index.create()
    includes = [arg for d in include_dirs for arg in ("-I", str(d))]
    tu = index.parse(str(header), args=system_include_args() + ["-x", "c"] + includes + list(extra_args) + [
        "-D__CUDACC__", "-D__cudart_builtin__=", "-DCUDARTAPI=", "-D__host__=", "-D__device__=",
    ], options=ci.TranslationUnit.PARSE_INCLUDE_BRIEF_COMMENTS_IN_CODE_COMPLETION)
    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if errors:
        for d in errors[:10]:
            print(f"  parse error: {d.spelling}", file=sys.stderr)
        raise SystemExit(f"{header.name}: {len(errors)} parse error(s)")
    return tu


ARRAY_KINDS = (ci.TypeKind.INCOMPLETEARRAY, ci.TypeKind.CONSTANTARRAY, ci.TypeKind.VARIABLEARRAY)


def param_spelling(t):
    """A parameter's type as it can be written before its name: an array
    parameter (cuBLAS's 'const float* const Aarray[]') is the pointer it decays to."""
    if t.kind in ARRAY_KINDS:
        return t.element_type.spelling + ' *'
    return t.spelling


def collect_functions(tu, prefix):
    functions = {}
    for c in tu.cursor.walk_preorder():
        if c.kind != ci.CursorKind.FUNCTION_DECL or not c.spelling.startswith(prefix):
            continue
        if c.spelling in functions:  # keep the first declaration
            continue
        functions[c.spelling] = {
            "result": c.result_type.spelling,
            "args": [(param_spelling(a.type), a.spelling, a.type) for a in c.get_arguments()],
        }
    return functions


def collect_errors(tu, enum_name="cudaError"):
    """(value, name, description) for every enumerator of the runtime's cudaError
    (or, with enum_name="cudaError_enum", the driver's CUresult)."""
    errors = {}
    for c in tu.cursor.walk_preorder():
        if c.kind == ci.CursorKind.ENUM_DECL and c.spelling == enum_name:
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
        raise SystemExit(f"{enum_name} enum not found")
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
        self.width = width      # bytes on the wire (a C++ expression for sized scalar refs)


def parse_role(text):
    """'in_scalar_ref(computeType)' -> ('in_scalar_ref', 'computeType')."""
    match = re.fullmatch(r"(\w+)(?:\((.+)\))?", text.strip())
    if not match:
        raise SystemExit(f"malformed role '{text}'")
    return match.group(1), match.group(2)


def classify(name, func, annotation):
    fields = []
    arg_by_name = {aname: (ctype, t) for ctype, aname, t in func["args"]}
    for pname, role_text in annotation.items():
        if pname.startswith("_"):
            continue  # function flags (_async, _sync)
        if pname not in arg_by_name:
            raise SystemExit(f"{name}: annotation names unknown parameter '{pname}'")
        role, size_expr = parse_role(role_text)
        if role not in ROLES:
            raise SystemExit(f"{name}: unknown role '{role}' for {pname}")
        if size_expr and role != "in_scalar_ref":
            raise SystemExit(f"{name}: only in_scalar_ref takes a size expression ({pname})")
        ctype, t = arg_by_name[pname]
        pointee = None
        if role in ("in_handle", "in_scalar"):
            width = 8 if role == "in_handle" else width_of(t)
        elif role == "out_handle":
            width, pointee = 8, t.get_pointee().spelling
        elif role in ("in_scalar_ref", "out_scalar_ref"):
            if "handle" not in arg_by_name:
                raise SystemExit(f"{name}: {role} needs a `handle` parameter (its pointer mode)")
            pointee_type = t.get_pointee()
            pointee = re.sub(r"^const\s+", "", pointee_type.spelling)
            if size_expr:
                width = f"static_cast<size_t>({size_expr})"
            elif pointee_type.get_size() > 0:
                width = width_of(pointee_type)
            else:
                raise SystemExit(f"{name}: {pname} points to '{pointee}'; give its size: in_scalar_ref(expr)")
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


# How the runtime and driver client stubs differ: the export macro, the status
# type and where calls go. The runtime records failures as the thread's last
# error (cudaGetLastError); the driver API has no last error, and its calls go
# through the runtime shim's session (tuna_cudart_call), like cuBLAS's.
RUNTIME_CLIENT = {
    "export": "TUNA_CUDART_EXPORT {result}",
    "status": "cudaError_t",
    "invalid": "static_cast<cudaError_t>(tuna::cudart::Fail(cudaErrorInvalidValue))",
    "malformed": "static_cast<cudaError_t>(tuna::cudart::Fail(cudaErrorUnknown))",
    "post": "tuna::cudart::Post",
    "call": "tuna::cudart::Call({api}, in, {{}}, {sync})",
    "finish": "tuna::cudart::Finish",
}
DRIVER_CLIENT = {
    "export": "TUNA_CUDA_DRIVER_EXPORT {result} CUDAAPI",
    "status": "CUresult",
    "invalid": "CUDA_ERROR_INVALID_VALUE",
    "malformed": "CUDA_ERROR_UNKNOWN",
    "post": "tuna::cudrv::Post",
    "call": "tuna::cudrv::Call({api}, in, {sync})",
    "finish": "tuna::cudrv::Finish",
}


def emit_client_stub(name, api, func, fields, flags, lib=RUNTIME_CLIENT):
    params = ", ".join(f"{ctype} {aname}" for ctype, aname, _ in func["args"]) or "void"
    status = lib["status"]
    # _wrap: the client exports a hand-written function of this name (caching,
    # allocation tracking) that calls this generated one.
    head = f"static {func['result']} tuna_remote_{name}" if flags.get("_wrap") else \
        f"{lib['export'].format(result=func['result'])} {name}"
    lines = [f"{head}({params}) {{",
             "    tuna::cudart::ArgWriter in;"]
    for f in fields:
        if f.role == "in_handle":
            lines.append(f"    in.put_u64(reinterpret_cast<uint64_t>({f.name}));")
        elif f.role == "in_scalar":
            lines.append(f"    in.put(&{f.name}, {f.width});")
        elif f.role == "in_struct":
            lines.append(f"    if (!{f.name}) return {lib['invalid']};")
            lines.append(f"    in.put({f.name}, {f.width});")
    if flags.get("_async"):
        lines.append(f"    return static_cast<{status}>({lib['post']}({api}, in));")
        lines.append("}")
        return "\n".join(lines)
    sync = "true" if flags.get("_sync") else "false"
    lines.append(f"    tuna::cudart::Reply reply = {lib['call'].format(api=api, sync=sync)};")
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
        lines.append(f"            return {lib['malformed']};  // malformed reply")
        lines.append("        }")
        lines.append("    }")
    lines.append(f"    return static_cast<{status}>({lib['finish']}(reply));")
    lines.append("}")
    return "\n".join(lines)


def emit_dispatch_case(name, api, func, fields, prologue=()):
    lines = [f"    case {api}: {{  // {name}",
             *prologue,
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


# ------------------------------------------------------------------- cuBLAS
#
# The replacement libcublas reaches the server through the cudart shim's
# session (tuna_cudart_call), so its stubs differ from the runtime's only in
# status codes and in the two scalar-reference roles.

def emit_cublas_client_stub(name, api, func, fields, flags):
    params = ", ".join(f"{ctype} {aname}" for ctype, aname, _ in func["args"]) or "void"
    head = f"static cublasStatus_t CUBLASWINAPI tuna_remote_{name}" if flags.get("_wrap") else \
        f"TUNA_CUBLAS_EXPORT cublasStatus_t CUBLASWINAPI {name}"
    lines = [f"{head}({params}) {{",
             "    tuna::cudart::ArgWriter in;"]
    for f in fields:
        if f.role == "in_handle":
            lines.append(f"    in.put_u64(reinterpret_cast<uint64_t>({f.name}));")
        elif f.role == "in_scalar":
            lines.append(f"    in.put(&{f.name}, {f.width});")
        elif f.role == "in_struct":
            lines.append(f"    if (!{f.name}) return CUBLAS_STATUS_INVALID_VALUE;")
            lines.append(f"    in.put({f.name}, {f.width});")
        elif f.role == "in_scalar_ref":
            lines.append(f"    if (!tuna::cublas::PutScalar(in, handle, {f.name}, {f.width})) "
                         "return CUBLAS_STATUS_INVALID_VALUE;")
        elif f.role == "out_scalar_ref":
            lines.append(f"    if (!{f.name}) return CUBLAS_STATUS_INVALID_VALUE;")
            lines.append(f"    const bool {f.name}_on_device = tuna::cublas::PutResult(in, handle, {f.name});")
    if flags.get("_async"):
        lines.append(f"    return static_cast<cublasStatus_t>(tuna::cublas::Post({api}, in));")
        lines.append("}")
        return "\n".join(lines)
    lines.append(f"    tuna::cudart::Reply reply = tuna::cublas::Call({api}, in);")
    outs = [f for f in fields if f.role.startswith("out")]
    if outs:
        lines.append("    if (!reply.transport_error && reply.result == 0) {")
        lines.append("        try {")
        lines.append("            tuna::cudart::ArgReader out(reply.out_args);")
        for f in outs:
            if f.role == "out_handle":
                lines.append(f"            const uint64_t {f.name}_h = out.get_u64();")
                lines.append(f"            if ({f.name}) *{f.name} = reinterpret_cast<{f.pointee}>({f.name}_h);")
            elif f.role == "out_scalar_ref":
                lines.append(f"            if (!{f.name}_on_device) out.get({f.name}, {f.width});")
            else:
                lines.append(f"            if ({f.name}) out.get({f.name}, {f.width}); else out.skip({f.width});")
        lines.append("        } catch (const std::exception&) {")
        lines.append("            return CUBLAS_STATUS_INTERNAL_ERROR;  // malformed reply")
        lines.append("        }")
        lines.append("    }")
    lines.append("    return static_cast<cublasStatus_t>(tuna::cublas::Finish(reply));")
    lines.append("}")
    return "\n".join(lines)


def emit_cublas_dispatch_case(name, api, func, fields, flags):
    # A posted (_async) call's failure is latched by the client and reported by
    # the next synchronizing CUDA call, so it must be a cudaError_t; a
    # synchronous call answers with the cublasStatus_t itself.
    asynchronous = bool(flags.get("_async"))
    invalid = "kCudaErrorInvalidValue" if asynchronous else "CUBLAS_STATUS_INVALID_VALUE"
    unsupported = "kCudaErrorNotSupported" if asynchronous else "CUBLAS_STATUS_NOT_SUPPORTED"
    not_initialized = "CublasAsyncError(CUBLAS_STATUS_NOT_INITIALIZED)" if asynchronous else \
        "CUBLAS_STATUS_NOT_INITIALIZED"
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
        elif f.role in ("in_scalar_ref", "out_scalar_ref"):
            lines.append(f"        tuna::cudart::ScalarRef {aname}_ref(in);")
            call_args.append(f"static_cast<{ctype}>({aname}_ref.ptr())")
            if f.role == "out_scalar_ref":
                outs.append(f)
        else:  # out_*
            lines.append(f"        {f.pointee} {aname}{{}};")
            call_args.append(f"&{aname}")
            outs.append(f)
    lines.append(f"        if (in.remaining() != 0) {{ resp.set_result({invalid}); break; }}")
    # A cublasHandle_t is a host pointer inside this process: only handles this
    # session created are passed to cuBLAS. A scalar must also agree with the
    # handle's pointer mode, or cuBLAS would dereference a client-chosen
    # address as host memory (or our host storage as device memory).
    if any(f.name == "handle" and f.role == "in_handle" for f in fields):
        lines.append("        const auto known = cublas_handles_.find(handle);")
        lines.append(f"        if (known == cublas_handles_.end()) {{ resp.set_result({not_initialized}); break; }}")
        for f in fields:
            if f.role in ("in_scalar_ref", "out_scalar_ref"):
                lines.append(f"        if ({f.name}_ref.on_device() != known->second) "
                             f"{{ resp.set_result({invalid}); break; }}")
    lines.append(f'        auto fn = reinterpret_cast<{real_signature(func)}>(backend_.Resolve("{name}"));')
    lines.append(f"        if (!fn) {{ resp.set_result({unsupported}); break; }}")
    lines.append(f"        const cublasStatus_t status = fn({', '.join(call_args)});")
    lines.append("        resp.set_result(" + ("CublasAsyncError(status)" if asynchronous else "status") + ");")
    if outs:
        lines.append("        tuna::cudart::ArgWriter out;")
        for f in outs:
            if f.role == "out_handle":
                lines.append(f"        out.put_u64(reinterpret_cast<uint64_t>({f.name}));")
            elif f.role == "out_scalar_ref":
                lines.append(f"        if (!{f.name}_ref.on_device()) out.put({f.name}_ref.ptr(), {f.width});")
            else:
                lines.append(f"        out.put(&{f.name}, {f.width});")
        lines.append("        resp.set_out_args(out.str());")
    lines.append("        break;")
    lines.append("    }")
    return "\n".join(lines)


def load_annotations(path):
    annotations = yaml.safe_load(path.read_text())
    simple = annotations.get("functions", {}) or {}
    special = annotations.get("special", {}) or {}   # remoted, marshalled by hand
    local = set(annotations.get("local", []) or [])  # implemented in the client only
    overlap = (set(simple) & set(special)) | (set(simple) & local) | (set(special) & local)
    if overlap:
        raise SystemExit(f"{path.name}: functions listed more than once: {sorted(overlap)}")
    return simple, special, local


def generate_cublas(cublas_include, cuda_include, codegen_dir, out_dir):
    """Emits the libcublas pieces; returns (remoted names, summary line)."""
    # cuComplex.h calls sqrtf/fabs without including <math.h> when built as C.
    tu = parse_tu(cublas_include / "cublas_v2.h", [cublas_include, cuda_include], ["-include", "math.h"])
    functions = collect_functions(tu, "cublas")
    simple, special, local = load_annotations(codegen_dir / "annotations_cublas.yaml")
    missing = [n for n in list(simple) + list(special) + sorted(local) if n not in functions]
    if missing:
        raise SystemExit(f"cuBLAS functions not found in cublas_api.h: {missing}")

    def emit(registry):
        client = ["// Generated by gen_cuda.py. Do not edit."]
        dispatch = ["// Generated by gen_cuda.py. Do not edit.", "    switch (request.api_id()) {"]
        for name in sorted(simple):
            fields = classify(name, functions[name], simple[name])
            client.append(emit_cublas_client_stub(name, f"kApi_{name}", functions[name], fields, simple[name]))
            if not simple[name].get("_server_by_hand"):  # handle bookkeeping: dispatch_cublas.cpp
                dispatch.append(emit_cublas_dispatch_case(name, f"kApi_{name}", functions[name], fields,
                                                          simple[name]))
        dispatch += ["    default: return false;", "    }", "    return true;"]
        (out_dir / "cublas_client_stubs.inc").write_text("\n\n".join(client) + "\n")
        (out_dir / "cublas_server_dispatch.inc").write_text("\n".join(dispatch) + "\n")

        exports = (codegen_dir / "cublas12_exports.txt").read_text().split()
        implemented = set(simple) | set(special) | local
        unsupported = [n for n in exports if n not in implemented]
        stubs = ["// Generated by gen_cuda.py. Do not edit."]
        for name in unsupported:
            stubs.append(f'TUNA_CUBLAS_EXPORT int {name}() {{ return tuna::cublas::Unsupported("{name}"); }}')
        (out_dir / "cublas_unsupported_stubs.inc").write_text("\n".join(stubs) + "\n")
        return (f"cuBLAS: {len(simple) + len(special)} remoted ({len(simple)} simple, {len(special)} special), "
                f"{len(local)} client-local, {len(unsupported)} unsupported stubs; "
                f"{len(functions)} functions in the header")

    return sorted(set(simple) | set(special)), emit


def generate_cublaslt(cublas_include, cuda_include, codegen_dir, out_dir):
    """The libcublasLt pieces: its calls are all hand-written (descriptor
    objects named by the client), so this only checks the names against
    cublasLt.h and emits the not-supported stubs."""
    tu = parse_tu(cublas_include / "cublasLt.h", [cublas_include, cuda_include], ["-include", "math.h"])
    functions = collect_functions(tu, "cublasLt")
    simple, special, local = load_annotations(codegen_dir / "annotations_cublaslt.yaml")
    if simple:
        raise SystemExit("annotations_cublaslt.yaml: cuBLASLt calls are hand-written; list them under special:")
    missing = [n for n in list(special) + sorted(local) if n not in functions]
    if missing:
        raise SystemExit(f"cuBLASLt functions not found in cublasLt.h: {missing}")

    def emit(registry):
        del registry  # ids come from cuda_api_ids.h
        exports = (codegen_dir / "cublasLt12_exports.txt").read_text().split()
        implemented = set(special) | local
        unsupported = [n for n in exports if n not in implemented]
        stubs = ["// Generated by gen_cuda.py. Do not edit."]
        for name in unsupported:
            stubs.append(f'TUNA_CUBLASLT_EXPORT int {name}() {{ return tuna::cublaslt::Unsupported("{name}"); }}')
        (out_dir / "cublaslt_unsupported_stubs.inc").write_text("\n".join(stubs) + "\n")
        return (f"cuBLASLt: {len(special)} remoted (by hand), {len(local)} client-local, "
                f"{len(unsupported)} unsupported stubs; {len(functions)} functions in the header")

    return sorted(special), emit


# --------------------------------------------------------------- driver API
#
# The replacement libcuda.so.1 / nvcuda.dll (Phase 3e). Applications rarely
# call the driver API themselves, but frameworks do (PyTorch checks the primary
# context per op), and NVIDIA's own libraries reach the driver through it. Its
# calls go through the runtime shim's session like cuBLAS's.

# Driver handles that may cross the wire: their values are addresses the GPU
# (not the worker's CPU) dereferences, or a stream the server issued. Contexts,
# modules and functions are host pointers in the worker, so the generated
# driver calls never take them; the shim keeps its own tokens for those.
DRIVER_WIRE_HANDLES = ("CUdeviceptr", "CUstream")


def generate_driver(cuda_include, codegen_dir, out_dir):
    """Emits the libcuda pieces; returns (remoted names, emit(registry) -> summary)."""
    tu = parse_tu(cuda_include / "cuda.h", [cuda_include])
    functions = collect_functions(tu, "cu")
    simple, special, local = load_annotations(codegen_dir / "annotations_driver.yaml")
    missing = [n for n in list(simple) + list(special) + sorted(local) if n not in functions]
    if missing:
        raise SystemExit(f"driver functions not found in cuda.h: {missing}")
    for name, annotation in simple.items():
        for pname, role in annotation.items():
            if role != "in_handle":
                continue
            ctype = next(c for c, a, _ in functions[name]["args"] if a == pname)
            if ctype not in DRIVER_WIRE_HANDLES:
                raise SystemExit(f"{name}: '{ctype} {pname}' is a host pointer on the server; "
                                 "driver calls may only pass " + ", ".join(DRIVER_WIRE_HANDLES))

    def emit(registry):
        del registry  # ids come from cuda_api_ids.h
        client = ["// Generated by gen_cuda.py. Do not edit."]
        dispatch = ["// Generated by gen_cuda.py. Do not edit.", "    switch (request.api_id()) {"]
        # The driver must be initialized in the worker before any driver call
        # (the runtime does it lazily, and may not have run yet).
        prologue = ["        if (int32_t e = DriverInit()) { resp.set_result(e); break; }"]
        for name in sorted(simple):
            fields = classify(name, functions[name], simple[name])
            client.append(emit_client_stub(name, f"kApi_{name}", functions[name], fields, simple[name],
                                           DRIVER_CLIENT))
            dispatch.append(emit_dispatch_case(name, f"kApi_{name}", functions[name], fields, prologue))
        dispatch += ["    default: return false;", "    }", "    return true;"]
        (out_dir / "driver_client_stubs.inc").write_text("\n\n".join(client) + "\n")
        (out_dir / "driver_server_dispatch.inc").write_text("\n".join(dispatch) + "\n")

        exports = (codegen_dir / "cuda12_driver_exports.txt").read_text().split()
        implemented = set(simple) | set(special) | local
        unsupported = [n for n in exports if n not in implemented]
        stubs = ["// Generated by gen_cuda.py. Do not edit."]
        for name in unsupported:
            stubs.append(f'TUNA_CUDA_DRIVER_EXPORT int {name}() {{ return tuna::cudrv::Unsupported("{name}"); }}')
        (out_dir / "driver_unsupported_stubs.inc").write_text("\n".join(stubs) + "\n")

        errors = collect_errors(tu, "cudaError_enum")
        err = ["// Generated by gen_cuda.py. Do not edit.",
               "struct CuErrorText { int value; const char* name; const char* text; };",
               "static const CuErrorText kCuErrorTexts[] = {"]
        for value, ename, text in errors:
            err.append(f"    {{{value}, {c_string(ename)}, {c_string(text)}}},")
        err.append("};")
        (out_dir / "driver_error_strings.inc").write_text("\n".join(err) + "\n")
        return (f"driver: {len(simple) + len(special)} remoted ({len(simple)} simple, {len(special)} special), "
                f"{len(local)} client-local, {len(unsupported)} unsupported stubs, {len(errors)} error codes; "
                f"{len(functions)} functions in cuda.h")

    return sorted(set(simple) | set(special)), emit


def load_registry(path):
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    ids = list(data.values())
    if len(ids) != len(set(ids)) or 0 in ids:
        raise SystemExit(f"{path}: ids must be unique and non-zero (0 is the echo)")
    return data


def save_registry(path, registry):
    lines = ["# Append-only api_id registry for remoted CUDA runtime (and library) calls.",
             "# gen_cuda.py adds new names at the end. Never renumber or reuse an id:",
             "# clients and servers of different versions agree on calls by id.",
             "# 0 is reserved for the tunnel's echo call."]
    for name, api_id in sorted(registry.items(), key=lambda kv: kv[1]):
        lines.append(f"{name}: {api_id}")
    path.write_text("\n".join(lines) + "\n")


def c_string(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    args = sys.argv[1:]
    cublas_include = None
    if "--cublas" in args:
        i = args.index("--cublas")
        if i + 1 >= len(args):
            raise SystemExit(__doc__)
        cublas_include = Path(args[i + 1])
        del args[i:i + 2]
    if len(args) != 3:
        raise SystemExit(__doc__)
    include_dir, annotations_path, out_dir = Path(args[0]), Path(args[1]), Path(args[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    codegen_dir = annotations_path.parent

    tu = parse_tu(include_dir / "cuda_runtime_api.h", [include_dir])
    functions = collect_functions(tu, "cuda")
    simple, special, local = load_annotations(annotations_path)

    missing = [n for n in simple if n not in functions]
    if missing:
        raise SystemExit(f"annotated functions not found in header: {missing}")

    cublas_remoted, emit_cublas = [], None
    cublaslt_remoted, emit_cublaslt = [], None
    if cublas_include:
        cublas_remoted, emit_cublas = generate_cublas(cublas_include, include_dir, codegen_dir, out_dir)
        cublaslt_remoted, emit_cublaslt = generate_cublaslt(cublas_include, include_dir, codegen_dir, out_dir)
    # The driver header ships with every toolkit (and the cuda_cudart
    # redistributable), so the driver shim is always generated.
    driver_remoted, emit_driver = generate_driver(include_dir, codegen_dir, out_dir)

    # The runtime's names first, then cuBLAS's, then the driver's: one id
    # space, so the server routes every Tuna call by id alone.
    registry_path = codegen_dir / "api_ids.yaml"
    registry = load_registry(registry_path)
    remoted = sorted(set(simple) | set(special))
    next_id = max(registry.values(), default=0) + 1
    for name in remoted + cublas_remoted + driver_remoted + cublaslt_remoted:
        if name not in registry:
            registry[name] = next_id
            next_id += 1
    save_registry(registry_path, registry)
    cublas_summary = emit_cublas(registry) if emit_cublas else None
    cublaslt_summary = emit_cublaslt(registry) if emit_cublaslt else None
    driver_summary = emit_driver(registry)

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
    if cublas_summary:
        print(cublas_summary)
    if cublaslt_summary:
        print(cublaslt_summary)
    print(driver_summary)


if __name__ == "__main__":
    main()

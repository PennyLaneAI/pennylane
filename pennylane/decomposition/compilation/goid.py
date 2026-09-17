import numpy as np

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.ops import Adjoint, Controlled

from .uid import calculate_uid

type GOID = str

SPECIAL_CASES = {"MultiRZ", "PauliRot", "PCPhase", "GlobalPhase"}


def _unwrap(op, is_adjoint=False, n_ctrls=0):
    if not isinstance(op, (Adjoint, Controlled)):
        return op, is_adjoint, n_ctrls
    if isinstance(op, Adjoint):
        return _unwrap(op.base, not is_adjoint, n_ctrls)
    # is controlled
    return _unwrap(op.base, is_adjoint, n_ctrls + len(op.control_wires))


def is_custom_op(op: Operator2) -> bool:
    if op.compilable_argnames or op.static_argnames or op.hybrid_argnames:
        return False
    if op.wire_argnames != ("wires",):
        return False
    return all(
        math.shape(v) == () and math.get_dtype_name(v) == "float64"
        for v in op.dynamic_args.values()
    )


_DTYPE_NAME_MAP = {
    "int8": "i8",
    "int32": "i32",
    "int64": "i64",
    "float32": "f32",
    "float64": "f64",
    "complex128": "complex<f64>",
    "complex64": "complex<f32>",
    "bool": "bool",
}


def _format_arg(arg):
    if not hasattr(arg, "shape"):
        arg = math.asarray(arg)
    dtype = _DTYPE_NAME_MAP[np.dtype(arg.dtype).name]
    if arg.shape:
        return f"tensor<{"x".join(str(s) for s in arg.shape)}x{dtype}>"
    return f"tensor<{dtype}>"


def _format_compilable_arg(arg):
    if isinstance(arg, str):
        return arg
    if arg is None:
        return "none"
    if arg is True:
        return "true"
    if arg is False:
        return "false"
    raise NotImplementedError


def _format_dynamic_params(op):
    """Format the dynamic-parameter group of a GraphOpID."""
    if op.name in SPECIAL_CASES:
        return {arg: "f64" for arg in op.dynamic_argnames}
    if is_custom_op(op):
        return {str(i): "f64" for i, _ in enumerate(op.dynamic_argnames)}
    return {arg: _format_arg(v) for arg, v in op.dynamic_args.items()}


def graph_op_id(op: Operator2) -> GOID:
    """Build a canonical frontend GraphOpID from its identity components."""
    op, is_adjoint, n_ctrls = _unwrap(op)

    name = f"Adjoint({op.name})" if is_adjoint else op.name
    if n_ctrls:
        prefix = "C" if n_ctrls == 1 else f"{n_ctrls}C"
        name = f"{prefix}({name})"

    dynamic_id = ",".join(f"{k}:{v}" for k, v in _format_dynamic_params(op).items())
    wire_id = ",".join(
        f"{n}:{len(w)}" for n, w in op.wire_args.items() if n not in op.hybrid_argnames
    )
    compilable_id = ", ".join(
        f"{k} = {_format_compilable_arg(val)}" for k, val in op.compilable_args.items()
    )
    uid = calculate_uid(op)
    formatted_tail = f"[{uid}]" if uid else ""
    return f"{name}{{{dynamic_id}}}{{{wire_id}}}{{{compilable_id}}}{formatted_tail}"

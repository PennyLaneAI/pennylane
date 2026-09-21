# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines a function mapping a function to its graph op id used to represent it's
compile-time information in mlir.
"""

from functools import singledispatch

import numpy as np

from pennylane import math
from pennylane.core.operator import Operator2

from .uid import calculate_uid
from .unwrap import unwrap

type GOID = str

SPECIAL_CASES = {"MultiRZ", "PauliRot", "PCPhase", "GlobalPhase"}


def is_custom_op(op: Operator2) -> bool:
    """Whether or not the operator lowered to a considered a "custom op" or not.

    Note that the special cases need to be checked independently.
    """
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


_COMPILABLE_ARG_MAP = {
    None: "none",
    True: "true",
    False: "false",
}


@singledispatch
def _format_compilable_arg(arg):
    if arg in _COMPILABLE_ARG_MAP:
        return _COMPILABLE_ARG_MAP[arg]
    raise NotImplementedError


@_format_compilable_arg.register(str)
def _handle_str(arg: str):
    return arg


@_format_compilable_arg.register(int)
def _handle_int(arg: int):
    if not -(2**63) < arg < 2**63:
        raise ValueError("only ints between -2**63 and 2**63 are compilable.")
    return f"{arg} : {"si64" if arg < 0 else "i64"}"


@_format_compilable_arg.register(float)
def _handle_float(arg: float):
    return f"{arg:e} : f64"


@_format_compilable_arg.register
def _handle_tuple(arg: list | tuple):
    return f"[{", ".join(_format_compilable_arg(a) for a in arg)}]"


@_format_compilable_arg.register
def _handle_dict(arg: dict):
    assert all(isinstance(k, str) for k in arg)
    f_contents = (
        f"{_format_compilable_arg(k)} = {_format_compilable_arg(v)}" for k, v in arg.items()
    )
    return f"{{{", ".join(f_contents)}}}"


def _format_dynamic_params(op):
    """Format the dynamic-parameter group of a GraphOpID."""
    if op.name in SPECIAL_CASES:
        return {arg: "f64" for arg in op.dynamic_argnames}
    if is_custom_op(op):
        return {str(i): "f64" for i, _ in enumerate(op.dynamic_argnames)}
    return {arg: _format_arg(v) for arg, v in op.dynamic_args.items()}


def graph_op_id(op: Operator2) -> GOID:
    """Build a canonical frontend GraphOpID from its identity components."""
    op, is_adjoint, n_ctrls = unwrap(op)

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


def graph_op_id_to_operator(goid: GOID) -> Operator2:
    split = goid.split("{")

    name = split[0]
    name_split = name.split("(")

    raise NotImplementedError

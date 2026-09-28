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
"""Defines generate_uid, a utility for getting a lowerable rep for non-lowerable operators"""

import hashlib
from functools import singledispatch
from typing import Any

from jax.tree import flatten

from pennylane import math
from pennylane.typing import AbstractWires
from pennylane.wires import Wires

from .operator2 import Operator2


def _unwrap(op, is_adjoint=False, n_ctrls=0):
    from pennylane.ops import Adjoint, Controlled  # pylint: disable=import-outside-toplevel

    if not isinstance(op, (Adjoint, Controlled)):
        return op, is_adjoint, n_ctrls
    if isinstance(op, Adjoint):
        return _unwrap(op.base, not is_adjoint, n_ctrls)
    return _unwrap(op.base, is_adjoint, n_ctrls + len(op.control_wires))


@singledispatch
def _serialize_static(val: Any, name: str | None):
    """Create a reduced representation of a value that can be used to easily
    create a UID for it.

    The reduced representation will be a tuple with the following format:

    .. code-block::

        (name, type, hashable_reduction)
    """
    # For arbitrary opaque data that may be unhashable, just use the id
    return (name, type(val), id(val))


# pylint: disable=unused-argument
@_serialize_static.register(type(None))
def _serialize_none(val, name):
    return (name, type(None), None)


@_serialize_static.register(bool)
def _serialize_bool(val, name):
    return (name, bool, val)


@_serialize_static.register(int)
def _serialize_int(val, name):
    return (name, int, val)


@_serialize_static.register(float)
def _serialize_float(val, name):
    return (name, float, repr(val))


@_serialize_static.register(complex)
def _serialize_complex(val, name):
    return (name, complex, (repr(val.real), repr(val.imag)))


@_serialize_static.register(str)
def _serialize_str(val, name):
    return (name, str, val)


@_serialize_static.register(list)
def _serialize_list(val, name):
    return (name, list, tuple(_serialize_static(item, None) for item in val))


@_serialize_static.register(tuple)
def _serialize_tuple(val, name):
    return (name, tuple, tuple(_serialize_static(item, None) for item in val))


@_serialize_static.register(dict)
def _serialize_dict(val, name):
    return (
        name,
        dict,
        frozenset((_serialize_static(k, None), _serialize_static(v, None)) for k, v in val.items()),
    )


@_serialize_static.register(set | frozenset)
def _serialize_set(val, name):
    return (name, type(val), frozenset(_serialize_static(item, None) for item in val))


def _is_wires_like(val: Any) -> bool:
    """Whether ``val`` is a (possibly abstract) container of wires."""
    return isinstance(val, (Wires, AbstractWires))


def _leaf_aval(value: Any) -> Any:
    """Reduce a pytree leaf to a hashable aval for UID generation. Wire containers (leaves for
    which :func:`_is_wires_like` is ``True``) are reduced to their length, since wire *labels*
    never affect an operator's UID, only how many wires there are. Everything else is reduced
    to its ``(shape, dtype name)``.
    """
    if _is_wires_like(value):
        return len(value)
    return (math.shape(value), math.get_dtype_name(value))


def generate_uid(op: Operator2, *, adjoint: bool = False, n_ctrls: int = 0) -> int | None:
    """Generate a unique identifier (UID) that distinguishes ``op`` from other operators of the
    same type based on the concrete values of its non-compilable static and hybrid arguments.

    Such a UID is needed to represent operators with non-compilable static or hybrid data at
    the compiler level, since their static/hybrid data cannot always be represented concretely
    in the compiler's intermediate representation (IR).

    Two operators of the same type only receive the same UID if they also agree on:

    * the shapes and dtypes of their dynamic arguments,
    * the number of wires of each (non-hybrid) wire argument,
    * the PyTree structure, shapes/dtypes, and wire-counts of their hybrid arguments, and
    * their (hashable) static arguments.

    In particular, the UID does not depend on the concrete *values* of dynamic arguments, nor
    on concrete wire *labels*.

    Args:
        op (Operator2): the operator to generate a UID for. ``op`` may be abstract, e.g. one of
            its wire arguments may hold an :class:`~.AbstractWires` value instead of concrete
            :class:`~.Wires`.
        adjoint (bool): whether ``op`` is adjointed. Default is ``False``. ``op`` may also
            be an Adjoint operator, which will compose with this.
        n_ctrls (int): the number of controls wrapping ``op``. Default is ``0``. ``op``
            may also be a ``Controlled`` operator, which will compose with this.

    Returns:
        int | None: the generated UID if the op has static or compilable argnames. None if
            a uid is not needed.

    Any operator without ``static_argnames`` or ``compilable_argnames`` will return ``None``:

    >>> print(generate_uid(qp.X(0)))
    None

    Otherwise, the UID that depends on the the static information.

    >>> op = qp.Select([qp.X(0), qp.Y(0)], 1)
    >>> generate_uid(op)
    557601774904406859
    >>> op2 = qp.Select([qp.X(1), qp.Y(2)], 3)
    >>> generate_uid(op2)
    557601774904406859

    ``adjoint`` and ``n_ctrls`` keywords compose with the operator itself:

    >>> generate_uid(op, adjoint=True)
    439243942176638131
    >>> generate_uid(qp.adjoint(op))
    439243942176638131

    This UID is consistent across processes.

    """
    op, adjoint, n_ctrls = _unwrap(op, adjoint, n_ctrls)
    op_cls = type(op)

    if not op.static_argnames and not op.hybrid_argnames:
        print(op, "is noen?")
        print(op.static_argnames, op.hybrid_argnames)
        return None

    dynamic_avals = tuple(_leaf_aval(val) for val in op.dynamic_args.values())

    wire_lens = tuple(
        len(op.arguments[name]) for name in op.wire_argnames if name not in op.hybrid_argnames
    )

    hybrid_trees = []
    hybrid_avals = []
    for name in op.hybrid_argnames:
        leaves, tree = flatten(op.arguments[name], is_leaf=_is_wires_like)
        hybrid_trees.append(tree)

        if name in op.wire_argnames:
            hybrid_avals.append(sum(len(l) for l in leaves))
        else:
            hybrid_avals.append(tuple(_leaf_aval(l) for l in leaves))

    reduced_static_args = tuple(
        _serialize_static(val, name) for name, val in op.static_args.items()
    )

    reduced = [
        op_cls,
        ("dynamic", dynamic_avals),
        ("wires", wire_lens),
        ("hybrid", tuple(hybrid_trees), tuple(hybrid_avals)),
        ("static", reduced_static_args),
        ("adjoint", adjoint),
        ("n_ctrls", n_ctrls),
    ]

    encoded_bytes = str(reduced).encode("utf-8")
    sha_hash = hashlib.sha256(encoded_bytes).hexdigest()

    # hexdigest() returns the hexadecimal hash in string format
    # Take 16 hexadecimals, since UID on Operator op is I64Attr, which is a 64-bit unsigned
    return int("0" + sha_hash[:15], 16)

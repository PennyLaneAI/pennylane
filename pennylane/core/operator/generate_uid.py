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


@singledispatch
def _serialize(val: Any):
    """Create a serialized representation of a value that can be used to easily
    create a UID for it.
    """
    return str(val)


@_serialize.register(list | tuple)
def _serialize_sequence(val):
    return tuple(_serialize(item) for item in val)


@_serialize.register(dict)
def _serialize_dict(val):
    serialized = ((str(_serialize(k)), _serialize(v)) for k, v in val.items())
    return tuple(sorted(serialized, key=lambda item: item[0]))


@_serialize.register(set | frozenset)
def _serialize_set(val):
    return tuple(sorted(str(_serialize(v)) for v in val))


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


def generate_uid(op: Operator2) -> int | None:
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

    Returns:
        int | None: the generated UID if the op has static or compilable argnames. None if
            a uid is not needed.

    Any operator without ``static_argnames`` or ``compilable_argnames`` will return ``None``:

    >>> print(generate_uid(qp.X(0)))
    None

    Otherwise, the an integer that depends on the the static and hybrid arguments is returned.

    >>> op = qp.Select([qp.X(0), qp.Y(0)], 1)
    >>> generate_uid(op)
    8998383583439072221
    >>> op2 = qp.Select([qp.X(1), qp.Y(2)], 3)
    >>> generate_uid(op2)
    8998383583439072221

    Note that it does not depend on the static information from the dynamic, wire, or compilable arguments.

    >>> op3 = qp.Select([qp.X(0), qp.Y(0)], (2, 3))
    >>> generate_uid(op3)
    8998383583439072221

    """
    if not op.static_argnames and not op.hybrid_argnames:
        return None

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
        (name, type(val), _serialize(val)) for name, val in op.static_args.items()
    )

    reduced = (
        type(op),
        ("hybrid", tuple(hybrid_trees), tuple(hybrid_avals)),
        ("static", reduced_static_args),
    )

    encoded_bytes = str(reduced).encode("utf-8")
    sha_hash = hashlib.sha256(encoded_bytes).hexdigest()

    # hexdigest() returns the hexadecimal hash in string format
    # Take 16 hexadecimals, since UID on Operator op is I64Attr, which is a 64-bit unsigned
    return int(sha_hash[:16], 16) >> 1

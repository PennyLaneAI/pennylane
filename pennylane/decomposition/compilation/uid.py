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
"""
UID generation logic for compiling operators with non-compilable data.
"""

import hashlib
from functools import singledispatch
from typing import Any

import numpy as np
from cachetools import LRUCache

from pennylane import math
from pennylane.core.operator import Operator2, abstractify
from pennylane.pytrees import flatten

from .unwrap import unwrap

type UID = int

UID_CACHE: LRUCache[Operator2, UID] = LRUCache[Operator2, UID](maxsize=1000)


def _handle_array(arr):
    if not hasattr(arr, "shape"):
        arr = math.asarray(arr)
    return (arr.shape, np.dtype(arr.dtype).name)


def _handle_hybrid(op):
    trees = []
    avals = []
    for val in op.hybrid_args.values():
        leaves, tree = flatten(val)
        avals += [_handle_array(l) for l in leaves]
        trees.append(tree)
    return tuple(trees), tuple(avals)


@singledispatch
def _serialize_static(val: Any, name: str | None):
    """Create a reduced representation of a value that can be used to easily
    create a UID for it.

    The reduced representation will be a tuple with the following format:

    .. code-block:: python

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


def calculate_uid(op: Operator2) -> UID | None:
    """Calculates a unique representation of operators with non-lowerable components.

    Args:
        op (Operator2): an operator

    Returns:
        int | None: None indicates an operator that is fully lowerable. int is the unique identifier

    >>> print(calculate_uid(qp.X(0)))
    None
    >>> op = qp.Select([qp.X(0), qp.Y(1), qp.Z(0), qp.H(1)], (2,3))
    >>> calculate_uid(op)
    140033415976329661
    >>> op2 = qp.Select([qp.X(3), qp.Y(4), qp.Z(3), qp.H(4)], (0,1))
    >>> calculate_uid(op2)
    140033415976329661

    """
    if not op.static_argnames and not op.hybrid_argnames:
        return None
    aop = abstractify(op)
    if aop in UID_CACHE:
        return UID_CACHE[aop]

    aop, is_adjoint, n_ctrls = unwrap(aop)

    # Flat dynamic arguments
    dynamic_avals = tuple((val.shape, val.dtype.name) for val in aop.dynamic_args)
    wire_lens = tuple[int, ...](
        len(wires) for n, wires in aop.wire_args.items() if n not in aop.hybrid_argnames
    )
    reduced_static_args = [
        _serialize_static(val, name) for name, val in aop.compilable_args.items()
    ]
    reduced_static_args += [_serialize_static(val, name) for name, val in aop.static_args.items()]

    reduced = [type(aop)]
    reduced.append(("dynamic", dynamic_avals))
    reduced.append(("wires", wire_lens))
    reduced.append(("hybrid", *_handle_hybrid(op)))
    reduced.append(("static", tuple(reduced_static_args)))
    reduced.append(("adjoint", is_adjoint))
    reduced.append(("n_ctrls", n_ctrls))

    encoded_bytes = str(reduced).encode("utf-8")
    sha_hash = hashlib.sha256(encoded_bytes).hexdigest()

    # hexdigest() returns the hexadecimal hash in string format
    # Take 16 hexadecimals, since UID on Operator op is I64Attr, which is a 64-bit unsigned
    uid = int("0" + sha_hash[:15], 16)
    UID_CACHE[aop] = uid
    return uid


def op_for_uid(uid: UID) -> Operator2 | None:
    """Return the operator for a uid if it exists in the cache."""
    for aop, target_uid in UID_CACHE.items():
        if uid == target_uid:
            return aop
    return None

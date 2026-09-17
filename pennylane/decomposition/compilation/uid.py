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

from pennylane.core import Operator2
from pennylane.ops import Adjoint, Controlled
from pennylane.pytrees import flatten


def _unwrap(op, is_adjoint=False, n_ctrls=0):
    if not isinstance(op, (Adjoint, Controlled)):
        return op, is_adjoint, n_ctrls
    if isinstance(op, Adjoint):
        return _unwrap(op.base, not is_adjoint, n_ctrls)
    # is controlled
    return _unwrap(op.base, is_adjoint, n_ctrls + len(op.control_wires))


def _handle_hybrid(op):
    trees = []
    avals = []
    for val in op.hybrid_args.values():
        leaves, tree = flatten(val)
        avals += [(l.shape, np.dtype(l.dtype).name) for l in leaves]
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


def calculate_uid(op: Operator2) -> int | None:
    if not op.static_argnames and not op.hybrid_argnames:
        return None

    op, is_adjoint, n_ctrls = _unwrap(op)

    # Flat dynamic arguments
    dynamic_avals = tuple((val.shape, val.dtype.name) for val in op.dynamic_args)
    wire_lens = tuple[int, ...](
        len(wires) for n, wires in op.wire_args.items() if n not in op.hybrid_argnames
    )
    reduced_static_args = [_serialize_static(val, name) for name, val in op.compilable_args.items()]
    reduced_static_args += [_serialize_static(val, name) for name, val in op.static_args.items()]

    reduced = [type(op)]
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
    return int("0" + sha_hash[:15], 16)

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
"""Utilities for operators."""

from __future__ import annotations

from functools import singledispatch
from numbers import Number
from typing import TYPE_CHECKING

from pennylane import math
from pennylane.core.queuing import QueuingManager
from pennylane.pytrees import flatten, leaf, unflatten
from pennylane.typing import AbstractArray, AbstractWires
from pennylane.wires import Wires

if TYPE_CHECKING:
    from pennylane.decomposition.resources import CompressedResourceOp

    from .base import Operator


@singledispatch
@QueuingManager.stop_recording()
def abstractify(val) -> AbstractArray | AbstractWires | Operator | CompressedResourceOp:
    """Convert the provided object into its abstract form.

    Args:
        val: The value to convert.

    Returns:
        The abstract version of the provided value.

    **Example**

    An abstract object in the context of this function is an object that stores only
    the shape and type of any data whose concrete value would only be known at runtime.
    For example, the corresponding abstract type of an float array of length 3 is an
    ``AbstractArray`` with shape ``(3,)`` and type ``float64``:

    >>> qp.core.abstractify(np.array([0.1, 0.2, 0.3]))
    AbstractArray((3,), float64)

    Similarly, concrete operators have concrete data and wire labels:

    >>> op = qp.CRZ(0.5, wires=[0, 1])
    >>> op
    CRZ(0.5, wires=[0, 1])

    The corresponding abstract object is an instance of the same operator with its dynamic
    data and wires replaced with ``AbstractArray`` and ``AbstractWire`` instances:

    >>> qp.core.abstractify(op)
    CRZ(AbstractArray((), float64, weak_type=True), wires=AbstractWires(2))

    For operators with fixed signatures (i.e., the shape and type of every argument is
    statically known and specified in its ``arg_specs``), ``abstractify`` can be used with
    the operator type and still returns the correct abstract instance:

    >>> qp.core.abstractify(qp.CRZ)
    CRZ(AbstractArray((), float64, weak_type=True), wires=AbstractWires(2))

    Note that this currently does not work if the operator's signature is not fully fixed.
    For example, ``PauliRot`` takes an arbitrary number of wires, so this fails:

    >>> qp.core.abstractify(qp.PauliRot)
    Traceback (most recent call last):
        ...
    TypeError: 'PauliRot' must set 'arg_specs' and cover all dynamic and wire arguments with fixed abstract types to be abstractified.


    """

    # pylint: disable-next=import-outside-toplevel
    from .operator2 import Operator2

    # NOTE: Don't flatten Operator2 instances as they can be handled by their custom dispatch.
    leaves, tree = flatten(val, is_leaf=lambda x: isinstance(x, (Wires, Operator2)))
    if tree != leaf:
        abstract_leaves = tuple(abstractify(l) for l in leaves)
        return unflatten(abstract_leaves, tree)

    if isinstance(val, Number):
        return AbstractArray((), type(val))

    shape = math.shape(val)
    dtype = math.get_dtype_name(val)
    return AbstractArray(shape, dtype)


@abstractify.register(type)
def _abstractify_type(val: type) -> AbstractArray:
    """Abstractify a type."""
    if issubclass(val, Number):
        return AbstractArray((), val)
    raise NotImplementedError(f"Cannot abstractify type '{val}'")


@abstractify.register(Wires)
def _abstractify_wires(val: Wires) -> AbstractWires:
    """Abstractify wires."""
    return AbstractWires(len(val))


@abstractify.register(AbstractArray | AbstractWires)
def _abstractify_abstract_type(val: AbstractArray | AbstractWires) -> AbstractArray | AbstractWires:
    """Abstractify an abstract type, i.e., do nothing."""
    return val

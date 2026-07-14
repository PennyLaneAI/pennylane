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
from pennylane.pytrees import flatten, leaf, unflatten
from pennylane.typing import AbstractArray, AbstractWires
from pennylane.wires import Wires

if TYPE_CHECKING:
    from pennylane.decomposition.resources import CompressedResourceOp

    from .base import Operator


@singledispatch
def abstractify(val) -> AbstractArray | AbstractWires | Operator | CompressedResourceOp:
    """Convert the provided value into an abstract type."""

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

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

from functools import singledispatch

from pennylane import math
from pennylane.pytrees import flatten, leaf, unflatten
from pennylane.typing import AbstractArray
from pennylane.wires import AbstractWires, Wires


@singledispatch
def abstractify(val) -> AbstractArray:
    """Convert the provided value into an abstract type."""
    leaves, tree = flatten(val)
    if tree != leaf:
        abstract_leaves = tuple(abstractify(l) for l in leaves)
        return unflatten(abstract_leaves, tree)

    shape = math.shape(val)
    dtype = math.get_dtype_name(val)
    return AbstractArray(shape, dtype)


@abstractify.register(Wires)
def _abstractify_wires(val) -> AbstractWires:
    """Abstractify wires."""
    return AbstractWires(len(val))

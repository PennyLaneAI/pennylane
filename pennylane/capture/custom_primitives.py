# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This submodule offers custom primitives for the PennyLane capture module.
"""

from enum import Enum
from typing import Any

from jax.extend.core import Primitive


class PrimitiveType(Enum):
    """Enum to define valid set of primitive classes"""

    DEFAULT = "default"
    OPERATOR = "operator"
    MEASUREMENT = "measurement"
    HIGHER_ORDER = "higher_order"
    TRANSFORM = "transform"


def _restore_dict(obj: Any) -> dict:
    """Restore dict from hashable tuple representation.

    This is used by transforms that need to restore kwargs from hashable tuple
    representations stored in JAX primitives. The hashable representation is a
    sorted tuple of (key, value) tuples created by _make_hashable_nested().

    Args:
        obj: Tuple of (key, value) tuples representing a dict

    Returns:
        dict: Restored dictionary with recursively restored values

    Example:
        >>> _restore_dict((('a', 1), ('b', 2)))
        {'a': 1, 'b': 2}
    """
    if not isinstance(obj, tuple):
        return obj

    # Empty tuple represents empty dict
    if len(obj) == 0:
        return {}

    # Check if this tuple is actually a dict representation
    # (all elements must be 2-tuples to be key-value pairs)
    if not all(isinstance(item, tuple) and len(item) == 2 for item in obj):
        # Not a dict representation, return as-is (it's a regular tuple)
        return obj

    # Convert tuple of (key, value) tuples back to dict, recursively restoring nested values
    return {k: _restore_dict(v) for k, v in obj}


# pylint: disable=abstract-method,too-few-public-methods
class QmlPrimitive(Primitive):
    """A subclass for JAX's Primitive that differentiates between different
    classes of primitives."""

    _prim_type: PrimitiveType = PrimitiveType.DEFAULT

    @property
    def prim_type(self):
        """Value of Enum representing the primitive type to differentiate between various
        sets of PennyLane primitives."""
        return self._prim_type.value

    @prim_type.setter
    def prim_type(self, value: str | PrimitiveType):
        """Setter for QmlPrimitive.prim_type."""
        self._prim_type = PrimitiveType(value)

    def bind(self, *args, **params):
        """Bind method for QmlPrimitive.

        Note: JAX 0.7.0+ requires all parameters to be hashable. Callers are responsible
        for ensuring parameters are hashable (e.g., passing tuples instead of lists/dicts/slices).
        """
        return super().bind(*args, **params)

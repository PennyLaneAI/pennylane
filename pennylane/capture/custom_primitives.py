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

from qpjax.extend.core import Primitive


class PrimitiveType(Enum):
    """Enum to define valid set of primitive classes"""

    DEFAULT = "default"
    OPERATOR = "operator"
    MEASUREMENT = "measurement"
    HIGHER_ORDER = "higher_order"
    TRANSFORM = "transform"


def _make_hashable(obj: Any) -> Any:
    """Convert potentially unhashable objects to hashable equivalents for JAX 0.7.0+.

    JAX 0.7.0 requires all primitive parameters to be hashable. This helper converts
    common unhashable types (list, dict, slice) to hashable tuples.

    Args:
        obj: Object to potentially convert to hashable form

    Returns:
        Hashable version of the object
    """
    if isinstance(obj, slice):
        return (obj.start, obj.stop, obj.step)
    if isinstance(obj, list):
        return tuple(_make_hashable(item) for item in obj)
    if isinstance(obj, dict):
        return tuple((k, _make_hashable(v)) for k, v in obj.items())

    return obj


# pylint: disable=abstract-method,too-few-public-methods
class QmlPrimitive(Primitive):
    """A subclass for JAX's Primitive that differentiates between different
    classes of primitives and automatically makes parameters hashable for JAX 0.7.0+."""

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
        """Bind with automatic parameter hashability conversion for JAX 0.7.0+.

        Overrides the parent bind method to automatically convert unhashable parameters
        (like lists, dicts, and slices) to hashable tuples, which is required by JAX 0.7.0+.
        """
        # Convert all parameters to hashable forms
        hashable_params = {k: _make_hashable(v) for k, v in params.items()}
        return super().bind(*args, **hashable_params)

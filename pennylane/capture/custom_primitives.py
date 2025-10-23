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


def _make_hashable(obj: Any) -> Any:
    """Convert potentially unhashable objects to hashable equivalents for JAX 0.7.0+.

    JAX 0.7.0 requires all primitive parameters to be hashable. This helper converts
    common unhashable types (list, dict, slice) to hashable tuples.

    Args:
        obj: Object to potentially convert to hashable form

    Returns:
        Hashable version of the object
    """
    # Import here to avoid circular dependency
    import jax
    import numpy as np

    if isinstance(obj, slice):
        return (obj.start, obj.stop, obj.step)

    # Check if obj is a JAX tracer - these are already hashable, don't convert
    # Must check before Array check since tracers can also be Array instances
    if isinstance(obj, jax.core.Tracer):
        return obj

    # Convert arrays (JAX and NumPy) to nested tuples for hashability
    # Must check before list check since tolist() returns lists
    if hasattr(jax, "Array") and isinstance(obj, jax.Array):
        # Recursively convert nested lists from tolist() to tuples
        return _make_hashable(obj.tolist())
    if isinstance(obj, np.ndarray):
        # Recursively convert nested lists from tolist() to tuples
        return _make_hashable(obj.tolist())
    if isinstance(obj, list):
        return tuple(_make_hashable(item) for item in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    return obj


def _restore_slice(obj: Any) -> Any:
    """Restore slice objects from hashable tuple representation.

    This is the inverse of _make_hashable for slice objects. When QmlPrimitive.bind()
    converts slices to tuples for hashability, primitive implementations need to
    convert them back.

    Args:
        obj: Object that may be a tuple representation of a slice

    Returns:
        slice object if obj is a 3-tuple, otherwise returns obj unchanged
    """
    if isinstance(obj, tuple) and len(obj) == 3:
        # Tuples representing slices have exactly 3 elements (start, stop, step)
        # Check if it's likely a slice by checking if elements look like slice components
        # (None or integers for start/stop/step)
        if all(x is None or isinstance(x, int) for x in obj):
            return slice(*obj)
    return obj


def _restore_hashable(obj: Any) -> Any:
    """Restore unhashable objects from their hashable tuple representations.

    This is the general inverse of _make_hashable. It attempts to detect and restore
    lists and dicts that were converted to tuples.

    Args:
        obj: Object that may be a tuple representation of a list or dict

    Returns:
        Restored object (dict, list, slice) or original object unchanged
    """
    if not isinstance(obj, tuple):
        return obj

    # Empty tuple likely represents empty dict in transform kwargs context
    if len(obj) == 0:
        return {}

    # Check if it's a dict representation (tuple of (key, value) tuples)
    if len(obj) > 0 and all(isinstance(item, tuple) and len(item) == 2 for item in obj):
        # Try to restore as dict
        try:
            return {k: _restore_hashable(v) for k, v in obj}
        except (TypeError, ValueError):
            # If it fails, it's probably a regular tuple, fall through
            pass

    # Check if it's a slice representation (3-tuple with None or ints)
    if len(obj) == 3 and all(x is None or isinstance(x, int) for x in obj):
        return slice(*obj)

    # Otherwise try to restore as list with recursive restoration
    return tuple(_restore_hashable(item) for item in obj)


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

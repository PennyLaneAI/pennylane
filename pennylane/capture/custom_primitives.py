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

    # Import here to avoid circular dependency and only when needed
    # pylint: disable=import-outside-toplevel,too-many-return-statements
    import jax
    import numpy as np

    # Check slice BEFORE hash() because slice is hashable in Python 3.12+ but not in 3.11
    # We need consistent behavior across Python versions for CI compatibility
    if isinstance(obj, slice):
        return (obj.start, obj.stop, obj.step)

    # Check if the object is already hashable
    try:
        hash(obj)
        return obj
    except TypeError:
        pass

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
        # Sort by str(key) to handle non-comparable keys (e.g., class types like ABCCaptureMeta)
        # This ensures consistent ordering without requiring keys to implement __lt__
        return tuple(
            sorted(((k, _make_hashable(v)) for k, v in obj.items()), key=lambda x: str(x[0]))
        )
    return obj


def _restore_slice(obj: Any) -> Any:
    """Restore slice objects from hashable tuple representation.

    This is the inverse of _make_hashable for slice objects. When QmlPrimitive.bind()
    converts slices to tuples for hashability, primitive implementations need to
    convert them back.

    Args:
        obj: Object that may be a tuple representation of a slice

    Returns:
        slice object if obj is a 3-tuple with (start, stop, step), otherwise returns obj unchanged
    """
    if isinstance(obj, tuple) and len(obj) == 3:
        # Tuples representing slices have exactly 3 elements (start, stop, step)
        # Check if it's likely a slice by checking if elements look like slice components
        # (None or integers for start/stop/step)
        if all(x is None or isinstance(x, int) for x in obj):
            return slice(*obj)
    return obj


def _restore_dict(obj: Any) -> dict:
    """Restore dict from hashable tuple representation.

    This is the inverse of _make_hashable for dicts. When QmlPrimitive.bind() converts
    dicts to sorted tuples of (key, value) pairs for hashability, this function converts
    them back to dicts.

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
    classes of primitives and automatically makes parameters hashable for JAX 0.7.0+.

    Note: With kwargs-as-inputs architecture, we still need hashable parameters for
    JAX primitives, but we no longer bake kwargs into closures using functools.partial.
    Instead, we pass kwargs directly to make_jaxpr, which eliminates closure hashability
    issues while still requiring primitive parameters to be hashable.
    """

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

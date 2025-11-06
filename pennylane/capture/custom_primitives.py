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

    **IMPORTANT**: Callers must explicitly call this function to convert parameters
    before passing them to primitive.bind(). The automatic conversion in QmlPrimitive.bind()
    has been removed to ensure primitives never contain unhashable metadata from the start.

    Args:
        obj: Object to potentially convert to hashable form

    Returns:
        Hashable version of the object

    Example:
        >>> # Convert a slice to hashable tuple before binding
        >>> args_slice = slice(0, 5)
        >>> hashable_slice = _make_hashable(args_slice)  # Returns (0, 5, None)
        >>> primitive.bind(data, args_slice=hashable_slice)

        >>> # Convert a dict to hashable tuple of tuples before binding
        >>> kwargs = {'key': 'value'}
        >>> hashable_kwargs = _make_hashable(kwargs)  # Returns (('key', 'value'),)
        >>> primitive.bind(data, kwargs=hashable_kwargs)
    """

    # First, check if the object is already hashable
    try:
        hash(obj)
        return obj
    except TypeError:
        pass

    # Import here to avoid circular dependency and only when needed
    # pylint: disable=import-outside-toplevel,too-many-return-statements
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


def _restore_list(obj: Any) -> list:
    """Restore list from hashable tuple representation.

    This is the inverse of _make_hashable for lists. When QmlPrimitive.bind() converts
    lists to tuples for hashability, this function converts them back to lists.

    Args:
        obj: Tuple representing a list

    Returns:
        list: Restored list with recursively restored elements

    Example:
        >>> _restore_list((1, 2, (3, 4)))
        [1, 2, [3, 4]]
    """
    if not isinstance(obj, tuple):
        return obj

    # Recursively restore nested tuples to lists
    return [_restore_list(item) for item in obj]


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
        """Bind method for QmlPrimitive.

        Note: JAX 0.7.0+ requires all parameters to be hashable. Callers are responsible
        for ensuring parameters are hashable (e.g., passing tuples instead of lists/dicts/slices).
        Use _make_hashable() helper function to convert unhashable objects before calling bind().
        """
        return super().bind(*args, **params)

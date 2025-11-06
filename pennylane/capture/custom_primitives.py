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

from jax.extend.core import Primitive


class PrimitiveType(Enum):
    """Enum to define valid set of primitive classes"""

    DEFAULT = "default"
    OPERATOR = "operator"
    MEASUREMENT = "measurement"
    HIGHER_ORDER = "higher_order"
    TRANSFORM = "transform"


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

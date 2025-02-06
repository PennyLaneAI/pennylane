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
from typing import Union

import jax


class PrimitiveType(Enum):
    """Enum to define valid set of primitive classes"""

    DEFAULT = "default"
    OPERATOR = "operator"
    MEASUREMENT = "measurement"
    HIGHER_ORDER = "higher_order"
    TRANSFORM = "transform"


# pylint: disable=too-few-public-methods,abstract-method
class QmlPrimitive(jax.core.Primitive):
    """A subclass for JAX's Primitive that differentiates between different
    classes of primitives."""

    _prim_type: PrimitiveType = PrimitiveType.DEFAULT

    @property
    def prim_type(self):
        """Value of Enum representing the primitive type to differentiate between various
        sets of PennyLane primitives."""
        return self._prim_type.value

    @prim_type.setter
    def prim_type(self, value: Union[str, PrimitiveType]):
        """Setter for QmlPrimitive.prim_type."""
        self._prim_type = PrimitiveType(value)


# pylint: disable=too-few-public-methods,abstract-method
class NonInterpPrimitive(QmlPrimitive):
    """A subclass to JAX's Primitive that works like a Python function
    when evaluating JVPTracers and BatchTracers."""

    def bind_with_trace(self, trace, args, params):
        """Bind the ``NonInterpPrimitive`` with a trace.

        If the trace is a ``JVPTrace``or a ``BatchTrace``, binding falls back to a standard Python function call.
        Otherwise, the bind call of JAX's standard Primitive is used."""
        if isinstance(trace, (jax.interpreters.ad.JVPTrace, jax.interpreters.batching.BatchTrace)):
            return self.impl(*args, **params)
        return super().bind_with_trace(trace, args, params)

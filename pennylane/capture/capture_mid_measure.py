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
This submodule defines the abstract classes and primitives for capturing mid-circuit measurements.
"""

from functools import lru_cache
from typing import Optional

import pennylane as qml

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


@lru_cache
def create_mid_measure_primitive() -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to an mid-circuit measurement type.

    Called when defining any :class:`~.Operator` subclass, and is used to set the
    ``Operator._primitive`` class property.

    Args:
        operator_type (type): a subclass of qml.operation.Operator

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive with the same name as the operator subclass.
        ``None`` is returned if jax is not available.

    """
    if not has_jax:
        return None

    primitive = jax.core.Primitive("mid_measure")

    @primitive.def_impl
    def _(wires, reset=False, postselect=None):
        wires = qml.wires.Wires(wires)
        if len(wires) > 1:
            raise qml.QuantumFunctionError(
                "Only a single qubit can be measured in the middle of the circuit"
            )
        # Do nothing with the MidMeasureMP. The MeasurementProcess primitive will handle that
        mp = qml.measurements.MidMeasureMP(wires=wires, reset=reset, postselect=postselect)
        return qml.measurements.MeasurementValue([mp], processing_fn=lambda v: v)

    @primitive.def_abstract_eval
    def _(wires, **_):  # pylint: disable=unused-argument
        return jax.core.ShapedArray((), jax.numpy.bool_)

    return primitive

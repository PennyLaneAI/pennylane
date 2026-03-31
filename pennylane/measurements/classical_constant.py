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
"""
This modules contains a measurement for returning a constant classical
piece of data from a QNode.
"""
from functools import lru_cache
from importlib.util import find_spec

from pennylane import math
from pennylane.typing import TensorLike

from .capture_measurements import _get_abstract_measurement
from .measurements import SampleMeasurement, StateMeasurement

has_jax = find_spec("jax") is not None


@lru_cache
def _make_primitive():
    if not has_jax:
        return None
    from pennylane.capture.custom_primitives import (  # pylint: disable=import-outside-toplevel
        QmlPrimitive,
    )

    primitive = QmlPrimitive("classical_constant")
    primitive.prim_type = "measurement"

    @primitive.def_impl
    def _impl(constant):
        return constant

    abstract_type = _get_abstract_measurement()

    @primitive.def_abstract_eval
    def _abstract_eval(constant):
        def _shape_and_dtype(**_):
            return constant.shape, constant.dtype

        return abstract_type(_shape_and_dtype)

    return primitive


class ClassicalConstant(SampleMeasurement, StateMeasurement):
    """Allows for the return of a strictly classical value from ``QNode``'s. Use of this
    measurement process promises that the value is strictly independent of the quantum
    component and does not depend on any mid circuit or terminal measurements.

    .. note::
        This is currently a developer tool for testing and debugging.
        This is currently a developer tool for testing and debugging.
        This is currently a developer tool for testing and debugging.

    Args:
        constant (TensorLike): The value that should be returned from the ``QNode``.

    .. code-block:: python

        @qml.qnode(qml.device('reference.qubit', wires=2))
        def c(x):
            return qml.measurements.ClassicalConstant(x)

    >>> c(np.array([0,1]))
    array([0, 1])

    """

    _primitive = _make_primitive()

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, constant):
        return cls._primitive.bind(constant)

    def _flatten(self):
        return (self._constant,), None

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0])

    @property
    def numeric_type(self):
        return getattr(self.constant, "dtype", type(self.constant))

    def shape(self, shots=None, num_device_wires=0):
        return math.shape(self._constant)

    def __init__(self, constant: TensorLike):
        self._constant = constant
        super().__init__()

    def __repr__(self):
        return f"ClassicalConstant({self.constant})"

    @property
    def hash(self):
        c = (
            str(id(self._constant))
            if math.is_abstract(self._constant)
            else str(math.round(self._constant, 10))
        )
        return hash(("ClassicalConstant", c))

    @property
    def constant(self) -> TensorLike:
        """The constant to be returned."""
        return self._constant

    def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
        return self._constant

    def process_counts(self, counts, wire_order):
        return self._constant

    def process_state(self, state, wire_order):
        return self._constant

    def process_density_matrix(self, density_matrix, wire_order):
        return self._constant


ClassicalConstant._obs_primitive = None  # pylint: disable=protected-access
ClassicalConstant._wires_primitive = None  # pylint: disable=protected-access
ClassicalConstant._mcm_primitive = None  # pylint: disable=protected-access

# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
"""
This module contains the qml.measure measurement.
"""
import copy
import uuid
from typing import Generic, TypeVar

import pennylane as qml
from pennylane.wires import Wires

from .measurements import MeasurementProcess, MidMeasure


def measure(wires):  # TODO: Change name to mid_measure
    """Perform a mid-circuit measurement in the computational basis on the
    supplied qubit.

    Measurement outcomes can be obtained and used to conditionally apply
    operations.

    If a device doesn't support mid-circuit measurements natively, then the
    QNode will apply the :func:`defer_measurements` transform.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def func(x, y):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            m_0 = qml.measure(1)

            qml.cond(m_0, qml.RY)(y, wires=0)
            return qml.probs(wires=[0])

    Executing this QNode:

    >>> pars = np.array([0.643, 0.246], requires_grad=True)
    >>> func(*pars)
    tensor([0.90165331, 0.09834669], requires_grad=True)

    Args:
        wires (Wires): The wire of the qubit the measurement process applies to.

    Raises:
        QuantumFunctionError: if multiple wires were specified
    """
    wire = Wires(wires)
    if len(wire) > 1:
        raise qml.QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    # Create a UUID and a map between MP and MV to support serialization
    measurement_id = str(uuid.uuid4())[:8]
    _MidMeasure(wires=wire, id=measurement_id)
    return MeasurementValue(measurement_id)


T = TypeVar("T")


class _MidMeasure(MeasurementProcess):
    """Mid-circuit measurement."""

    @property
    def return_type(self):
        return MidMeasure

    @property
    def _queue_category(self):
        return "_ops"


class MeasurementValueError(ValueError):
    """Error raised when an unknown measurement value is being used."""


class MeasurementValue(Generic[T]):
    """A class representing unknown measurement outcomes in the qubit model.

    Measurements on a single qubit in the computational basis are assumed.

    Args:
        measurement_id (str): The id of the measurement that this object depends on.
        zero_case (float): the first measurement outcome value
        one_case (float): the second measurement outcome value
    """

    __slots__ = ("_depends_on", "_zero_case", "_one_case", "_control_value")

    def __init__(
        self,
        measurement_id: str,
        zero_case: float = 0,
        one_case: float = 1,
    ):
        self._depends_on = measurement_id
        self._zero_case = zero_case
        self._one_case = one_case
        self._control_value = one_case  # By default, control on the one case

    @property
    def branches(self):
        """A dictionary representing all the possible outcomes of the MeasurementValue."""
        branch_dict = {}
        branch_dict[(0,)] = self._zero_case
        branch_dict[(1,)] = self._one_case
        return branch_dict

    def __invert__(self):
        """Return a copy of the measurement value with an inverted control
        value."""
        inverted_self = copy.copy(self)
        zero = self._zero_case
        one = self._one_case

        inverted_self._control_value = one if self._control_value == zero else zero

        return inverted_self

    def __eq__(self, control_value):
        """Allow asserting measurement values."""
        measurement_outcomes = {self._zero_case, self._one_case}

        if not isinstance(control_value, tuple(type(val) for val in measurement_outcomes)):
            raise MeasurementValueError(
                "The equality operator is used to assert measurement outcomes, but got a value "
                + f"with type {type(control_value)}."
            )

        if control_value not in measurement_outcomes:
            raise MeasurementValueError(
                "Unknown measurement value asserted; the set of possible measurement outcomes is: "
                + f"{measurement_outcomes}."
            )

        self._control_value = control_value
        return self

    @property
    def control_value(self):
        """The control value to consider for the measurement outcome."""
        return self._control_value

    @property
    def measurements(self):
        """List of all measurements this MeasurementValue depends on."""
        return [self._depends_on]

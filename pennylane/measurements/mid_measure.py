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
import uuid
from typing import Generic, TypeVar

import pennylane as qml
import pennylane.numpy as np

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
    wire = qml.wires.Wires(wires)
    if len(wire) > 1:
        raise qml.QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    # Create a UUID and a map between MP and MV to support serialization
    measurement_id = str(uuid.uuid4())[:8]
    MeasurementProcess(MidMeasure, wires=wire, id=measurement_id)
    return MeasurementValue([measurement_id], fn=lambda v: v)


T = TypeVar("T")


class MeasurementValueError(ValueError):
    """Error raised when an unknown measurement value is being used."""


class MeasurementValue(Generic[T]):
    """A class representing unknown measurement outcomes in the qubit model.

    Measurements on a single qubit in the computational basis are assumed.

    Args:
        measurement_ids (list of str): The id of the measurement that this object depends on.
        fn (callable): A transformation applied to the measurements.
    """

    def __init__(self, measurement_ids, fn):
        self.measurement_ids = measurement_ids
        self.fn = fn

    def items(self):
        """A generator representing all the possible outcomes of the MeasurementValue."""
        for i in range(2 ** len(self.measurement_ids)):
            branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurement_ids)))
            yield branch, self.fn(*branch)

    def __invert__(self):
        """Return a copy of the measurement value with an inverted control
        value."""
        return self.apply(lambda v: not v)

    def __eq__(self, other):
        if isinstance(other, MeasurementValue):
            return self.merge(other).apply(lambda v: v[0] == v[1])
        return self.apply(lambda v: v == other)

    def __add__(self, other):
        if isinstance(other, MeasurementValue):
            return self.merge(other).apply(sum)
        return self.apply(lambda v: v + other)

    def __lt__(self, other):
        if isinstance(other, MeasurementValue):
            return self.merge(other).apply(lambda v: v[0] < v[1])
        return self.apply(lambda v: v < other)

    def __gt__(self, other):
        if isinstance(other, MeasurementValue):
            return self.merge(other).apply(lambda v: v[0] > v[1])
        return self.apply(lambda v: v > other)

    def apply(self, fn):
        """Apply a post computation to this measurement"""
        return MeasurementValue(self.measurement_ids, lambda *x: fn(self.fn(*x)))

    def merge(self, other: "MeasurementValue"):
        """merge two measurement values"""

        # create a new merged list with no duplicates and in lexical ordering
        merged_measurement_ids = list(set(self.measurement_ids).union(set(other.measurement_ids)))
        merged_measurement_ids.sort()

        # create a new function that selects the correct indices for each sub function
        def merged_fn(*x):
            out_1 = self.fn(
                *(x[i] for i in [merged_measurement_ids.index(m) for m in self.measurement_ids])
            )
            out_2 = other.fn(
                *(x[i] for i in [merged_measurement_ids.index(m) for m in other.measurement_ids])
            )

            return out_1, out_2

        return MeasurementValue(merged_measurement_ids, merged_fn)

    def __getitem__(self, i):
        # branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurement_ids)).split())
        branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurement_ids)))
        return self.fn(*branch)

    def __str__(self):
        lines = []
        for i in range(2 ** (len(self.measurement_ids))):
            branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurement_ids)))
            lines.append(
                "if "
                + ",".join([f"{self.measurement_ids[j]}={branch[j]}" for j in range(len(branch))])
                + " => "
                + str(self.fn(*branch))
            )
        return "\n".join(lines)

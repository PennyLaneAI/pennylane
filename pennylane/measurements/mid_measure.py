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

import pennylane as qml
from pennylane.wires import Wires

from .measurements import MeasurementProcess, MeasurementValue, ObservableReturnTypes

MidMeasure = ObservableReturnTypes.MidMeasure
"""Enum: An enumeration which represents returning sampling the computational
basis in the middle of the circuit."""


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
    MeasurementProcess(MidMeasure, wires=wire, id=measurement_id)
    return MeasurementValue(measurement_id)

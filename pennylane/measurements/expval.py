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
This module contains the qml.expval measurement.
"""
import warnings

from pennylane.operation import Operator

from .measurements import MeasurementProcess, ObservableReturnTypes

Expectation = ObservableReturnTypes.Expectation
"""Enum: An enumeration which represents returning the expectation
value of an observable on specified wires."""


def expval(op: Operator):
    r"""Expectation value of the supplied observable.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(0))

    Executing this QNode:

    >>> circuit(0.5)
    -0.4794255386042029

    Args:
        op (Observable): a quantum observable object

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
    """
    if not op.is_hermitian:
        warnings.warn(f"{op.name} might not be hermitian.")

    return MeasurementProcess(Expectation, obs=op)

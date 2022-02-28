# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Contains the condition transform.
"""
from functools import wraps

import pennylane as qml
from pennylane.tape import QuantumTape, get_active_tape
from pennylane.operation import DecompositionUndefinedError, Operation, AnyWires
from pennylane.measurements import MeasurementValue
from pennylane.wires import Wires
from pennylane.transforms.adjoint import adjoint
from typing import Type


class Conditional(Operation):
    """
    Conditional operation wrapper class.
    """

    num_wires = AnyWires

    def __init__(
        self,
        expr: MeasurementValue[bool],
        then_op: Type[Operation],
        else_op = None: Type[Operation],
        do_queue=True,
        id=None,
    ):
        self.meas_val = expr
        self.then_op = then_op
        self.else_op = else_op
        if else_op and len(self.then_op.wires) != len(self.else_op.wires):
            raise ValueError("Number of wires doesn't match.")

        super().__init__(wires=then_op.wires, do_queue=do_queue, id=id)


def cond(measurement, operation, operation=None):
    """Create an operation that applies a version of the provided operation
    that is conditioned on a value dependent on quantum measurements.

    Args:
        measurement (MeasurementDependentValue): The measurement dependent
            value to consider.
        operation (Operation): The PennyLane operation to condition.

    Returns:
        function: A new function that applies the controlled equivalent of ``operation``. The returned
        operation takes the same input arguments as ``operation``.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        first_par = 0.1
        sec_par = 0.3

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode():
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(first_par, wires=1)

            m_1 = qml.measure(2)
            qml.cond(m_0, qml.RZ)(sec_par, wires=1)
            return qml.expval(qml.PauliZ(1))
    """

    @wraps(operation)
    def wrapper(*args, **kwargs):
        return Conditional(measurement, operation(*args, do_queue=False, **kwargs))

    return wrapper

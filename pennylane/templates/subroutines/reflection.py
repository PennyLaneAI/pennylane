# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This submodule contains the template for the Reflection operation.
"""

import pennylane as qml
from pennylane.operation import Operation
from pennylane.ops import SymbolicOp


class Reflection(SymbolicOp, Operation):
    r"""Reflection(U, alpha)
    Apply a :math:`\alpha`-reflection over the state :math:`|\Psi\rangle = U|0\rangle`.

    .. math::

        \text{Reflection}(U, \alpha) = \identity - (1 - e^{-i\alpha}) |\Psi\rangle \langle \Psi|


    Args:
        U (qml.ops.op_math.prod.Prod): the product of operations that generate the state :math:`|\Psi\rangle`.
        alpha (float): the reflection angle.

    **Example**

    The reflection :math:`\identity - 2|+\rangle \langle +|` applied to the state :math:`|1\rangle` would be as follows:

    .. code-block::

        @qml.prod
        def generator(wires):
            qml.Hadamard(wires=wires)

        U = generator(wires=0)

        dev = qml.device('default.qubit')

        @qml.qnode(dev)
        def circuit():

            # Initialize to the state |1>
            qml.qml.PauliX(wires=0)

            # Apply the reflection
            qml.Reflection(U, alpha=np.pi)

            return qml.state()

        circuit()
    """

    def __init__(self, U, alpha, reflection_wires = None, id=None):
        self.U = U
        self.alpha = alpha

        if reflection_wires is None:
            self.reflection_wires = U.wires
        else:
            self.reflection_wires = reflection_wires

        super().__init__(base=U, id=id)

    @property
    def has_matrix(self):
        return False

    @property
    def wires(self):
        return self.U.wires

    def decomposition(self):
        wires = qml.wires.Wires(self.reflection_wires)
        print(wires, "sssss")
        ops = []

        ops.append(qml.adjoint(self.U))

        if len(wires) > 1:
            print("adios")
            ops.append(qml.PauliX(wires=wires[-1]))
            ops.append(
                qml.ctrl(
                    qml.PhaseShift(-self.alpha, wires=wires[-1]),
                    control=wires[:-1],
                    control_values=[0] * (len(wires) - 1),
                )
            )
            ops.append(qml.PauliX(wires=wires[-1]))

        else:
            print("hola", self.alpha)
            ops.append(qml.PauliX(wires=wires))
            ops.append(qml.PhaseShift(-self.alpha, wires=wires))
            ops.append(qml.PauliX(wires=wires))

        ops.append(self.U)

        return ops
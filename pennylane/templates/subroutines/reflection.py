# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane.ops import SymbolicOp


class Reflection(SymbolicOp, Operation):
    r"""An operation that applies a reflection on a state :math:`|\Psi\rangle`.
    This operator is useful in algorithms such as `Amplitude Amplification <https://arxiv.org/abs/quant-ph/0005055>`__
    or `Oblivious Amplitude Amplification <https://arxiv.org/abs/1312.1414>`__.

    Given an :class:`~.Operator` :math:`U` such that :math:`|\Psi\rangle = U|0\rangle`,  and a reflection angle :math:`\alpha`,
    this template creates the operation:

    .. math::

        \text{Reflection}(U, \alpha) = -\mathbb{I} + (1 - e^{i\alpha}) |\Psi\rangle \langle \Psi|


    Args:
        U (Operator): The operator that generate the state :math:`|\Psi\rangle`.
        alpha (float): the reflection angle. Default is :math:`\pi`.
        reflection_wires (Any or Iterable[Any]): Subsystem of wires on which to reflect. The default is None and the reflection will be applied on the U wires.

    **Example**

    This example shows how to apply the reflection :math:`-\mathbb{I} + 2|+\rangle \langle +|` to the state :math:`|1\rangle`.

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
            qml.Reflection(U)

            return qml.state()

        circuit()


    .. details::
        :title: Theory

        The operator is built as follows:

        .. math::

            \text{Reflection}(U, \alpha) = -\mathbb{I} + (1 - e^{i\alpha}) |\Psi\rangle \langle \Psi| = U(-\mathbb{I} + (1 - e^{i\alpha}) |0\rangle \langle 0|)U^{\dagger}.

        The central block is obtained through a PhaseShift controlled operator.

        In the case of specifying the reflection wires,  the operator would have the following expression.

        .. math::

            U(\mathbb{I} - (1 - e^{i\alpha}) |0\rangle^{\otimes m} \langle 0|^{\otimes m}\otimes \mathbb{I}^{n-m}})U^{\dagger},

        where :math:`m` is the number of reflection wires and :math:`n` is the total number of wires.

    """

    def __init__(self, U, alpha=np.pi, reflection_wires=None, id=None):
        self.wires = U.wires

        if reflection_wires is None:
            reflection_wires = U.wires
        else:
            reflection_wires = reflection_wires

        if not set(reflection_wires).issubset(set(U.wires)):
            raise ValueError("The reflection wires must be a subset of the operation wires.")

        self._name = "Reflection"

        super().__init__(base=U, alpha=alpha, reflection_wires=reflection_wires, id=id)

    @property
    def has_matrix(self):
        return False

    # pylint:disable=arguments-differ
    @staticmethod
    def compute_decomposition(base, alpha, reflection_wires):
        wires = qml.wires.Wires(reflection_wires)

        ops = []

        ops.append(qml.GlobalPhase(np.pi))
        ops.append(qml.adjoint(base))

        if len(wires) > 1:
            ops.append(qml.PauliX(wires=wires[-1]))
            ops.append(
                qml.ctrl(
                    qml.PhaseShift(alpha, wires=wires[-1]),
                    control=wires[:-1],
                    control_values=[0] * (len(wires) - 1),
                )
            )
            ops.append(qml.PauliX(wires=wires[-1]))

        else:
            ops.append(qml.PauliX(wires=wires))
            ops.append(qml.PhaseShift(alpha, wires=wires))
            ops.append(qml.PauliX(wires=wires))

        ops.append(base)

        return ops

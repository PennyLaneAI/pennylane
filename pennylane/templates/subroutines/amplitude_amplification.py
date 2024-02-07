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
This submodule contains the template for Amplitude Amplification.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation


class AmplitudeAmplification(Operation):
    r"""AmplitudeAmplification(U, O, iters=1, fixed_point=False, aux_wire=None)

    Given a state :math:`|\Psi\rangle = \alpha |\psi\rangle + \beta|\psi^{\perp}}\rangle`, this subroutine amplifies the amplitude of the state :math:`|\psi\rangle`.

    .. math::

            \text{AmplitudeAmplification}(U, O)|\Psi\rangle \sim |\psi\rangle

    Args:
        U (qml.ops.op_math.prod.Prod): the product of operations that generate the state :math:`|\Psi\rangle`.
        O (qml.ops.op_math.prod.Prod): the oracle that flips the sign of the state :math:`|\psi\rangle` and do nothing to the state :math:`|\psi^{\perp}\rangle`.
        iters (int): the number of iterations of the amplitude amplification subroutine. Default is 1.
        fixed_point (bool): whether to use the fixed-point amplitude amplification algorithm. Default is False.
        aux_wire (int): the auxiliary wire to use for the fixed-point amplitude amplification algorithm. Default is None.

    Raises:
        ValueError: aux_wire must be specified if fixed_point == True.
        ValueError: aux_wire must be different from the wires of U.

    **Example**

    Amplification of state :math:`|2\rangle` using Grover's algorithm with 3 qubits:

    .. code-block::

        @qml.prod
        def generator(wires):
          for wire in wires:
            qml.Hadamard(wires = wire)

        U = generator(wires = range(3))
        O = qml.FlipSign(2, wires = range(3))

        dev = qml.device("default.qubit")


        @qml.qnode(dev)
        def circuit():

          generator(wires = range(3))
          qml.AmpAmp(U, O, iters = 5, fixed_point=True, aux_wire=3)

          return qml.probs(wires = range(3))

    .. code-block:: pycon

        >>> print(np.round(circuit(),3))
        [0.001 0.001 0.994 0.001 0.001 0.001 0.001 0.001]

    """

    def __init__(self, U, O, iters=1, fixed_point=False, aux_wire=None, reflection_wires = None):
        self.U = U
        self.O = O
        self.aux_wire = aux_wire
        self.fixed_point = fixed_point
        self.n_iterations = iters

        if reflection_wires is None:
            self.reflection_wires = U.wires
        else:
            self.reflection_wires = reflection_wires

        self.gamma = 0.95

        self.queue()

        if fixed_point and aux_wire is None:
            raise ValueError(f"aux_wire must be specified if fixed_point == True.")

        if fixed_point and iters % 2 != 0:
            raise ValueError(f"Number of iterations must be even if fixed_point == True.")


        if fixed_point and len(U.wires + qml.wires.Wires(aux_wire)) == len(U.wires):
            raise ValueError(f"aux_wire must be different from the wires of U.")

        if fixed_point:
            super().__init__(wires=U.wires + qml.wires.Wires(aux_wire))
        else:
            super().__init__(wires=U.wires)

    def decomposition(self):
        ops = []

        if self.fixed_point:
            alphas, betas = self.get_fixed_point_angles()

            for iter in range(self.n_iterations // 2):
                ops.append(qml.Hadamard(wires=self.aux_wire))
                ops.append(qml.ctrl(self.O, control=self.aux_wire))
                ops.append(qml.Hadamard(wires=self.aux_wire))
                ops.append(qml.PhaseShift(betas[iter], wires=self.aux_wire))
                ops.append(qml.Hadamard(wires=self.aux_wire))
                ops.append(qml.ctrl(self.O, control=self.aux_wire))
                ops.append(qml.Hadamard(wires=self.aux_wire))

                ops.append(qml.Reflection(self.U, -alphas[iter], reflection_wires=self.reflection_wires))
        else:
            for _ in range(self.n_iterations):
                ops.append(self.O)
                ops.append(qml.Reflection(self.U, np.pi, reflection_wires=self.reflection_wires))

        return ops

    def get_fixed_point_angles(self):
        """
        Returns the angles needed for the fixed-point amplitude amplification algorithm.
        """
        n_iterations = self.n_iterations

        alphas = [
            2 * np.arctan(1 / (np.tan(2 * np.pi * j / n_iterations) * np.sqrt(1 - self.gamma**2)))
            for j in range(1, n_iterations // 2 + 1)
        ]
        betas = [-alphas[-j] for j in range(1, n_iterations // 2 + 1)]
        return alphas[: n_iterations // 2], betas[: n_iterations // 2]

    def queue(self, context=qml.QueuingManager):
        for op in [self.U, self.O]:
            context.remove(op)
        context.append(self)
        return self
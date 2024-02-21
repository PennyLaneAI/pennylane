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
This submodule contains the template for Amplitude Amplification.
"""

# pylint: disable-msg=too-many-arguments
import numpy as np
from pennylane.operation import Operation
import pennylane as qml


def get_fixed_point_angles(iters):
    """
    Returns the angles needed for the fixed-point amplitude amplification algorithm.
    This is extracted from the equation (11) of the paper `fixed-point quantum search <https://arxiv.org/pdf/1409.3305.pdf>`__.
    """
    gamma = 0.95

    alphas = [
        2 * np.arctan(1 / (np.tan(2 * np.pi * j / iters) * np.sqrt(1 - gamma**2)))
        for j in range(1, iters // 2 + 1)
    ]
    betas = [-alphas[-j] for j in range(1, iters // 2 + 1)]
    return alphas[: iters // 2], betas[: iters // 2]


class AmplitudeAmplification(Operation):
    r"""Operator that carries out the Amplitude Amplification subroutine.
    Given a state :math:`|\Psi\rangle = \alpha |\psi\rangle + \beta|\psi^{\perp}}\rangle`, this subroutine amplifies the amplitude of the state :math:`|\psi\rangle`.

    .. math::

            \text{AmplitudeAmplification}(U, O)|\Psi\rangle \sim |\psi\rangle

    The main idea of this operator is based on the `amplitude amplification <https://arxiv.org/abs/quant-ph/0005055>`__ paper.
    With the corresponding flag it is possible to work with advanced techniques such as `fixed-point quantum search <https://arxiv.org/abs/quant-ph/0005055>`__.
    It also allows the use of `oblivious amplitude amplification <https://arxiv.org/abs/1312.1414>`__ by reflecting on a subset of the wires.

    Args:
        U (Operator): Operator that generate the state :math:`|\Psi\rangle`.
        O (Operator): The oracle that flips the sign of the state :math:`|\psi\rangle` and do nothing to the state :math:`|\psi^{\perp}\rangle`.
        iters (int): the number of iterations of the amplitude amplification subroutine. Default is 1.
        fixed_point (bool): whether to use the fixed-point amplitude amplification algorithm. Default is False.
        work_wire (int): the auxiliary wire to use for the fixed-point amplitude amplification algorithm. Default is None.
        reflection_wires (Wires): the wires to reflect on. Default is the wires of U.

    Raises:
        ValueError: work_wire must be specified if fixed_point == True.
        ValueError: work_wire must be different from the wires of U.

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
          qml.AmplitudeAmplification(U, O, iters = 5, fixed_point=True, work_wire=3)

          return qml.probs(wires = range(3))

    .. code-block:: pycon

        >>> print(np.round(circuit(),3))
        [0.001 0.001 0.994 0.001 0.001 0.001 0.001 0.001]

    """

    def __init__(self, U, O, iters=1, fixed_point=False, work_wire=None, reflection_wires=None):
        if reflection_wires is None:
            reflection_wires = U.wires

        self.operations = [U, O]
        self.queue()

        if fixed_point and work_wire is None:
            raise qml.wires.WireError("work_wire must be specified if fixed_point == True.")

        if fixed_point and len(U.wires + qml.wires.Wires(work_wire)) == len(U.wires):
            raise ValueError("work_wire must be different from the wires of U.")

        if fixed_point:
            wires = U.wires + qml.wires.Wires(work_wire)
        else:
            wires = U.wires

        self.hyperparameters["U"] = U
        self.hyperparameters["O"] = O
        self.hyperparameters["iters"] = iters
        self.hyperparameters["fixed_point"] = fixed_point
        self.hyperparameters["work_wire"] = work_wire
        self.hyperparameters["reflection_wires"] = qml.wires.Wires(reflection_wires)

        super().__init__(wires=wires)

    @property
    def U(self):
        """The generator operation."""
        return self.hyperparameters["U"]

    @property
    def O(self):
        """The oracle operation."""
        return self.hyperparameters["O"]

    @property
    def iters(self):
        """The number of iterations."""
        return self.hyperparameters["iters"]

    @property
    def fixed_point(self):
        """Whether to use the fixed-point amplitude amplification algorithm."""
        return self.hyperparameters["fixed_point"]

    @property
    def work_wire(self):
        """The auxiliary wire to use for the fixed-point amplitude amplification algorithm."""
        return self.hyperparameters["work_wire"]

    @property
    def reflection_wires(self):
        """The wires on which the reflection is performed."""
        return self.hyperparameters["reflection_wires"]

    # pylint:disable=arguments-differ
    @staticmethod
    def compute_decomposition(*_, U, O, iters, fixed_point, work_wire, reflection_wires, **__):
        ops = []

        if fixed_point:
            alphas, betas = get_fixed_point_angles(iters)

            for iter in range(iters // 2):
                ops.append(qml.Hadamard(wires=work_wire))
                ops.append(qml.ctrl(O, control=work_wire))
                ops.append(qml.Hadamard(wires=work_wire))
                ops.append(qml.PhaseShift(betas[iter], wires=work_wire))
                ops.append(qml.Hadamard(wires=work_wire))
                ops.append(qml.ctrl(O, control=work_wire))
                ops.append(qml.Hadamard(wires=work_wire))

                ops.append(qml.Reflection(U, -alphas[iter], reflection_wires=reflection_wires))
        else:
            for _ in range(iters):
                ops.append(O)
                ops.append(qml.Reflection(U, np.pi, reflection_wires=reflection_wires))

        return ops

    def queue(self, context=qml.QueuingManager):
        for op in self.operations:
            context.remove(op)
        context.append(self)
        return self

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


def _get_fixed_point_angles(iters, p_min):
    """
    Returns the angles needed for the fixed-point amplitude amplification algorithm.
    The angles are computed using equation (11) of `arXiv:1409.3305v2 <https://arxiv.org/abs/1409.3305>`__.
    """

    delta = np.sqrt(1 - p_min)
    gamma = np.cos(np.arccos(1 / delta, dtype=np.complex128) / iters, dtype=np.complex128) ** -1

    alphas = [
        2 * np.arctan(1 / (np.tan(2 * np.pi * j / iters) * np.sqrt(1 - gamma**2)))
        for j in range(1, iters // 2 + 1)
    ]
    betas = [-alphas[-j] for j in range(1, iters // 2 + 1)]
    return alphas[: iters // 2], betas[: iters // 2]


class AmplitudeAmplification(Operation):
    r"""Applies amplitude amplification.

    Given a state :math:`|\Psi\rangle = \alpha |\phi\rangle + \beta|\phi^{\perp}\rangle`, this
    subroutine amplifies the amplitude of the state :math:`|\phi\rangle` such that

    .. math::

            \text{A}(U, O)|\Psi\rangle \sim |\phi\rangle.

    The implementation of the algorithm is based on [`arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`__].
    The template also unlocks advanced techniques such as fixed-point quantum search
    [`arXiv:1409.3305 <https://arxiv.org/abs/1409.3305>`__] and oblivious amplitude amplification
    [`arXiv:1312.1414 <https://arxiv.org/abs/1312.1414>`__], by reflecting on a subset of wires.

    Args:
        U (Operator): the operator that prepares the state :math:`|\Psi\rangle`
        O (Operator): the oracle that flips the sign of the state :math:`|\phi\rangle` and does nothing to the state :math:`|\phi^{\perp}\rangle`
        iters (int): the number of iterations of the amplitude amplification subroutine, default is ``1``
        fixed_point (bool): whether to use the fixed-point amplitude amplification algorithm, default is ``False``
        work_wire (int): the auxiliary wire to use for the fixed-point amplitude amplification algorithm, default is ``None``
        reflection_wires (Wires): the wires to reflect on, default is the wires of ``U``
        p_min (int): the lower bound for the probability of success in fixed-point amplitude amplification, default is ``0.9``

    Raises:
        ValueError: ``work_wire`` must be specified if ``fixed_point == True``.
        ValueError: ``work_wire`` must be different from the wires of the oracle ``O``.

    **Example**

    Amplification of state :math:`|2\rangle` using Grover's algorithm with 3 qubits.
    The state :math:`|\Psi\rangle` is constructed as a uniform superposition of basis states.

    .. code-block::

        @qml.prod
        def generator(wires):
            for wire in wires:
                qml.Hadamard(wires=wire)

        U = generator(wires=range(3))
        O = qml.FlipSign(2, wires=range(3))

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():

            generator(wires=range(3))
            qml.AmplitudeAmplification(U, O, iters=5, fixed_point=True, work_wire=3)

            return qml.probs(wires=range(3))

    .. code-block:: pycon

        >>> print(np.round(circuit(),3))
        [0.013, 0.013, 0.91, 0.013, 0.013, 0.013, 0.013, 0.013]
    """

    def _flatten(self):
        data = (self.hyperparameters["U"], self.hyperparameters["O"])
        metadata = tuple(
            (key, value) for key, value in self.hyperparameters.items() if key not in ["O", "U"]
        )
        return data, metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        U, O = (data[0], data[1])
        hyperparams_dict = dict(metadata)
        return cls(U, O, **hyperparams_dict)

    def __init__(
        self, U, O, iters=1, fixed_point=False, work_wire=None, p_min=0.9, reflection_wires=None
    ):
        self._name = "AmplitudeAmplification"
        if reflection_wires is None:
            reflection_wires = U.wires

        if fixed_point and work_wire is None:
            raise qml.wires.WireError("work_wire must be specified if fixed_point == True.")

        if fixed_point and len(O.wires + qml.wires.Wires(work_wire)) == len(O.wires):
            raise ValueError("work_wire must be different from the wires of O.")

        if fixed_point:
            wires = U.wires + qml.wires.Wires(work_wire)
        else:
            wires = U.wires

        self.hyperparameters["U"] = U
        self.hyperparameters["O"] = O
        self.hyperparameters["iters"] = iters
        self.hyperparameters["fixed_point"] = fixed_point
        self.hyperparameters["work_wire"] = work_wire
        self.hyperparameters["p_min"] = p_min
        self.hyperparameters["reflection_wires"] = qml.wires.Wires(reflection_wires)

        super().__init__(wires=wires)

    # pylint:disable=arguments-differ
    @staticmethod
    def compute_decomposition(**kwargs):
        U = kwargs["U"]
        O = kwargs["O"]
        iters = kwargs["iters"]
        fixed_point = kwargs["fixed_point"]
        work_wire = kwargs["work_wire"]
        p_min = kwargs["p_min"]
        reflection_wires = kwargs["reflection_wires"]

        ops = []

        if fixed_point:
            alphas, betas = _get_fixed_point_angles(iters, p_min)

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
        for op in [self.hyperparameters["U"], self.hyperparameters["O"]]:
            context.remove(op)
        context.append(self)
        return self

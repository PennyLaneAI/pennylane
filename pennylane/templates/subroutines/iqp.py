# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Contains the IQP template.
"""
from collections import defaultdict
from functools import reduce

import numpy as np

from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    register_resources,
    resource_rep,
)
from pennylane.math import expand_matrix
from pennylane.operation import Operation
from pennylane.ops import Hadamard, MultiRZ, PauliRot, PauliX
from pennylane.typing import TensorLike


class IQP(Operation):
    r"""
    A template that builds an Instantaneous Quantum Polynomial (IQP) circuit. The gates of these circuits correspond
    to multi-qubit X rotations, whose generators are given by tensor products of Pauli X operators.

    In this `IQPOpt <https://arxiv.org/pdf/2501.04776>`__ paper and this
    `train on classical, deploy on quantum <https://arxiv.org/pdf/2503.02934>`__ paper, the authors
    present methods for analytically approximating expectation values coming from measurements made on IQP circuits.
    This allows for the classical training of the parameters of these circuits prior to deploying them to a
    quantum computer for actual computation.

    Certain computational problems such as generative machine learning and combinatorial optimization can be cast as
    a minimization over functions of these expectation values. Since these circuits are also believed to be hard to
    sample from using classical algorithms, they can potentially lead to a quantum advantage.

    Args:
        weights (list): The parameters of the IQP gates.
        num_wires (int): Number of wires in the circuit.
        pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of ``pattern`` corresponds to a
            unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
            Generators are specified by listing the qubits on which an X operator acts. For example, the ``pattern``
            ``[[[0]], [[1]], [[2]], [[3]]]`` specifies a circuit with single qubit rotations on the first four qubits, each
            with its own trainable parameter. The ``pattern`` ``[[[0],[1]], [[2],[3]]]`` correspond to a circuit with two
            trainable parameters with generators :math:`X_0+X_1` and :math:`X_2+X_3` respectively. A circuit with a
            single trainable gate with generator :math:`X_0\otimes X_1` corresponds to the ``pattern``
            ``[[[0,1]]]``.
        spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
            :math:`\frac{1}{\sqrt(2)}(|00\dots0> + |11\dots1>)` is used in place of :math:`|00\dots0>`.

    Raises:
        ValueError: when ``pattern`` and ``weights`` have a different number of elements.

    **Example:**

    Below is an example of a 2-qubit IQP circuit. At this small scale, a state vector simulation is tractable.

    .. code-block:: python

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def iqp_circuit(weights, pattern, spin_sym):
            qml.IQP(weights=weights, num_wires=2, pattern=pattern, spin_sym=spin_sym)
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    >>> iqp_circuit(weights=[0.89, 0.54], pattern=[[[0]], [[1]]], spin_sym=False)  # doctest: +SKIP
    [np.float64(-0.20768100160878344), np.float64(0.47132836417373947)]

    >>> print(qml.draw(iqp_circuit, level="device")([0.89, 0.54], [[[0]], [[1]]], False))  # doctest: +SKIP
    0: ─╭IQP─┤  <Z>
    1: ─╰IQP─┤  <Z>

    .. seealso:: `IQP tutorial <https://pennylane.ai/qml/demos/tutorial_iqp_circuit_optimization_jax#parameterized-iqp-circuits>`__
    """

    resource_keys = {"spin_sym", "pattern", "num_wires"}

    def __init__(
        self, weights, num_wires, pattern, spin_sym=False, id=None
    ):  # pylint: disable=too-many-arguments
        if len(pattern) != len(weights):
            raise ValueError(
                f"Number of gates and number of parameters for an Instantaneous Quantum Polynomial circuit must be the same, got {len(pattern)} gates and {len(weights)} weights."
            )

        if num_wires == 0:
            raise ValueError("At least one valid wire is required.")

        self._hyperparameters = {
            "spin_sym": spin_sym,
            "weights": weights,
            "pattern": pattern,
            "num_wires": num_wires,
        }
        super().__init__(wires=range(num_wires), id=id)

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_matrix(weights, num_wires, pattern, spin_sym) -> TensorLike:
        layers = []
        wires = list(range(num_wires))

        if spin_sym:
            pauli_mat = PauliRot.compute_matrix(2 * np.pi / 4, "Y" + "X" * (num_wires - 1))
            layers.append(pauli_mat)

        for par, gate in zip(weights, pattern):
            for gen in gate:
                x_mat = reduce(math.kron, [PauliX.compute_matrix() for _ in gen])
                rx_mat = math.expm(-1j * par * expand_matrix(x_mat, gen, wires))
                layers.append(rx_mat)

        return reduce(math.matmul, layers[::-1])

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def resource_params(self):
        return {
            "spin_sym": self.hyperparameters["spin_sym"],
            "pattern": self.hyperparameters["pattern"],
            "num_wires": len(self.wires),
        }


def _instantaneous_quantum_polynomial_resources(spin_sym, pattern, num_wires):
    resources = defaultdict(int)
    if spin_sym:
        resources[
            resource_rep(
                PauliRot,
                pauli_word="Y" + "X" * (num_wires - 1),
            )
        ] = 1

    resources[resource_rep(Hadamard)] = 2 * num_wires

    for gate in pattern:
        for gen in gate:
            resources[resource_rep(MultiRZ, num_wires=len(gen))] += 1

    return resources


@register_resources(_instantaneous_quantum_polynomial_resources)
def _instantaneous_quantum_polynomial_decomposition(
    wires, weights, pattern, spin_sym, **__
):  # pylint: disable=unused-argument, too-many-arguments
    num_wires = len(wires)

    if spin_sym:
        PauliRot(2 * np.pi / 4, "Y" + "X" * (num_wires - 1), wires=range(num_wires))

    for i in range(num_wires):
        Hadamard(i)

    for par, gate in zip(weights, pattern):
        for gen in gate:
            MultiRZ(2 * par, wires=gen)

    for i in range(num_wires):
        Hadamard(i)


add_decomps(IQP, _instantaneous_quantum_polynomial_decomposition)

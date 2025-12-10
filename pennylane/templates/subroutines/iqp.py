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

import numpy as np

from pennylane.decomposition import (
    add_decomps,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import Hadamard, MultiRZ, PauliRot


class IQP(Operation):
    """
    A template that builds an Instantaneous Quantum Polynomial (IQP) circuit.

    In this `IQPOpt <https://arxiv.org/pdf/2501.04776>`__ paper and this
    `train on classical, deploy on quantum <https://arxiv.org/pdf/2503.02934>`__ paper, tht authors
    present methods for analytically approximating expectation values coming from measurements following IQP circuits.
    This allows for the classical training of the parameters of these circuits prior to deploying them to a
    quantum computer for actual computation.

    Many problems can be cast to be solved via minimizing these expectations, such as showing quantum advantage
    by calculating the integrals of oscillating functions as detailed
    `here <https://strawberryfields.ai/photonics/demos/run_iqp.html>`__ (an intractable task classically), thereby
    making this a useful circuit.
    """

    resource_keys = {"spin_sym", "pattern", "num_wires"}

    def __init__(
        self, num_wires, pattern, weights, spin_sym=None, id=None
    ):  # pylint: disable=too-many-arguments
        r"""
        IQP template corresponding to a parameterized IQP circuit. Based on `IQPopt: Fast optimization of
        instantaneous quantum polynomial circuits in JAX <https://arxiv.org/pdf/2501.04776>`__.

        Args:
            num_wires (int): Number of wires in the circuit.
            pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of gates corresponds to a
                unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
                Generators are specified by listing the qubits on which an X operator acts.
            weights (list): The parameters of the IQP gates.
            spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
                :math:`\frac{1}{\sqrt(2)}(|00\dots0> + |11\dots1>)` is used in place of :math:`|00\dots0>`.

        Raises:
            Exception: when pattern and weights have a different number of elements.
        """
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
        }
        super().__init__(wires=range(num_wires), id=id)

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
    wires, weights, pattern, spin_sym
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

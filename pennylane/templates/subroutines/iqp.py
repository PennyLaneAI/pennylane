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

    resource_keys = {"spin_sym", "gates", "n_qubits", "init_gates"}

    def __init__(
        self, wires, gates, params, init_gates=None, init_coeffs=None, spin_sym=None, id=None
    ):  # pylint: disable=too-many-arguments
        """
        IQP template corresponding to a parameterized IQP circuit. Based on `IQPopt: Fast optimization of
        instantaneous quantum polynomial circuits in JAX <https://arxiv.org/pdf/2501.04776>`__.

        Args:
            gates (list[list[list[int]]]): Specification of the trainable gates. Each element of gates corresponds to a
                unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
                Generators are specified by listing the qubits on which an X operator acts.
            params (list): The parameters of the IQP gates.
            init_gates (list[list[list[int]]], optional): A specification of gates of the same form as the gates argument. The
                parameters of these gates will be defined by init_coeffs later on.
            init_coeffs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
            spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
                1/sqrt(2)(|00...0> + |11...1>) is used in place of |00...0>.

        Raises:
            Exception: when gates and params have a different number of elements.
        """
        if len(gates) != len(params):
            raise ValueError(
                f"Number of gates and number of parameters for an Instantaneous Quantum Polynomial circuit must be the same, got {len(gates)} gates and {len(params)} params."
            )

        if not isinstance(wires, int) and len(wires) == 0:
            raise ValueError("At least one valid wire is required.")

        if isinstance(wires, int):
            wires = [wires]

        self._hyperparameters = {
            "spin_sym": spin_sym,
            "params": params,
            "gates": gates,
            "init_gates": init_gates,
            "init_coeffs": init_coeffs,
        }
        super().__init__(wires=wires, id=id)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def resource_params(self):
        return {
            "spin_sym": self.hyperparameters["spin_sym"],
            "gates": self.hyperparameters["gates"],
            "n_qubits": len(self.wires),
            "init_gates": self.hyperparameters["init_gates"],
        }


def _instantaneous_quantum_polynomial_resources(spin_sym, gates, n_qubits, init_gates):
    resources = defaultdict(int)
    if spin_sym:
        resources[
            resource_rep(
                PauliRot,
                pauli_word="Y" + "X" * (n_qubits - 1),
            )
        ] = 1

    resources[resource_rep(Hadamard)] = 2 * n_qubits

    if init_gates is not None:
        for gate in init_gates:
            for gen in gate:
                resources[resource_rep(MultiRZ, num_wires=len(gen))] += 1

    for gate in gates:
        for gen in gate:
            resources[resource_rep(MultiRZ, num_wires=len(gen))] += 1

    return resources


@register_resources(_instantaneous_quantum_polynomial_resources)
def _instantaneous_quantum_polynomial_decomposition(
    wires, params, gates, init_gates, init_coeffs, spin_sym
):  # pylint: disable=unused-argument, too-many-arguments
    n_qubits = len(wires)

    if spin_sym:
        PauliRot(2 * np.pi / 4, "Y" + "X" * (n_qubits - 1), wires=range(n_qubits))

    for i in range(n_qubits):
        Hadamard(i)

    if init_gates is not None:
        for par, gate in zip(init_coeffs, init_gates):
            for gen in gate:
                MultiRZ(2 * par, wires=gen)

    for par, gate in zip(params, gates):
        for gen in gate:
            MultiRZ(2 * par, wires=gen)

    for i in range(n_qubits):
        Hadamard(i)


add_decomps(IQP, _instantaneous_quantum_polynomial_decomposition)

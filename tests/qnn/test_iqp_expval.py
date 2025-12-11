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
"""Unit tests for the math.iqp module"""
from itertools import combinations

import numpy as np
import pytest

from pennylane import IQP, device, qnode
from pennylane.math import arange
from pennylane.measurements import expval
from pennylane.ops import PauliZ
from pennylane.qnn.iqp import op_expval


def local_gates(n_qubits: int, max_weight=2):
    """
    Generates a gate list for containing all gates whose generators have Pauli weight
    less or equal than max_weight.
    :param n_qubits: The number of qubits in the gate list
    :param max_weight: maximum Pauli weight of gate generators
    :return (list[list[list[int]]]): gate list
    """
    gates = []
    for weight in arange(1, max_weight + 1):
        for gate in combinations(arange(n_qubits), weight):
            gates.append([list(gate)])
    return gates


@pytest.mark.jax
@pytest.mark.parametrize(
    (
        "gates_fn",
        "params",
        "n_qubits",
        "spin_sym",
        "sparse",
        "n_samples",
        "max_batch_samples",
        "max_batch_ops",
        "indep_estimates",
        "expected_std",
    ),
    [
        ("multi_gens", [0.54], 2, True, False, 10, 10_000, 10_000, True, [0.0]),
        ("local_gates", [0.3], 1, False, True, 10, 10_000, 10_000, False, [0.0]),
        ("multi_gens", [-0.41], 2, True, False, 10, None, None, True, [0.0]),
        ("multi_gens", [0.0], 2, True, True, 10, None, None, True, [0.0]),
    ],
)
def test_expval(
    gates_fn,
    params,
    n_qubits,
    spin_sym,
    sparse,
    n_samples,
    max_batch_samples,
    max_batch_ops,
    indep_estimates,
    expected_std,
):  # pylint: disable=too-many-arguments
    import jax
    import jax.numpy as jnp

    gates = local_gates(n_qubits, 1)
    if gates_fn == "multi_gens":
        gates = [[gates[0][0], gates[1][0]]] + gates[2:]

    circuit = IQP(
        num_wires=n_qubits,
        pattern=gates,
        weights=params,
        spin_sym=spin_sym,
    )

    ops = jnp.array([[1, 0], [0, 1]])
    key = jax.random.PRNGKey(np.random.randint(0, 99999))

    exp_val, std = op_expval(
        ops=ops,
        n_samples=n_samples,
        key=key,
        circuit=circuit,
        sparse=sparse,
        indep_estimates=indep_estimates,
        max_batch_samples=max_batch_samples,
        max_batch_ops=max_batch_ops,
    )

    dev = device("default.qubit")

    @qnode(dev)
    def iqp_circuit(
        weights, pattern, spin_sym, n_qubits, ops
    ):  # pylint: disable=too-many-arguments
        IQP(weights, n_qubits, pattern, spin_sym)

        expectation_operators = []
        for l in ops:
            for i, qubit in enumerate(l):
                if qubit == 1:
                    expectation_operators.append(expval(PauliZ(i)))

        return expectation_operators

    simulated_exp_val = jnp.array(iqp_circuit(params, gates, spin_sym, n_qubits, ops))
    assert np.allclose(exp_val, simulated_exp_val)
    assert np.allclose(std, jnp.array(expected_std))

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
from scipy.sparse import csr_matrix

from pennylane import IQP, device, qnode
from pennylane.math import arange
from pennylane.measurements import expval
from pennylane.ops import PauliZ
from pennylane.qnn.iqp import iqp_expval

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def local_gates(n_qubits: int, max_weight=2):
    """
    Generates a gate list for an IqpSimulator object containing all gates whose generators have Pauli weight
    less or equal than max_weight.
    :param n_qubits: The number of qubits in the gate list
    :param max_weight: maximum Pauli weight of gate generators
    :return (list[list[list[int]]]): gate list object for IqpSimulator
    """
    gates = []
    for weight in arange(1, max_weight + 1):
        for gate in combinations(arange(n_qubits), weight):
            gates.append([list(gate)])
    return gates


@pytest.mark.jax
@pytest.mark.parametrize(
    (
        "ops",
        "gates_fn",
        "params",
        "n_qubits",
        "spin_sym",
        "sparse",
        "n_samples",
        "max_batch_samples",
        "max_batch_ops",
        "indep_estimates",
    ),
    [
        (
            [[1, 0], [0, 1]],
            "multi_gens",
            [0.54],
            2,
            True,
            False,
            10_000,
            10_000,
            10_000,
            True,
        ),
        (
            csr_matrix([[0, 1], [1, 0]]),
            "local_gates",
            [0.3, 0.2],
            2,
            False,
            False,
            10_000,
            10_000,
            10_000,
            False,
        ),
        (
            [1],
            "local_gates",
            [0.3],
            1,
            False,
            True,
            10_000,
            10_000,
            10_000,
            False,
        ),
        (
            [[0, 1], [0, 1]],
            "local_gates",
            [0.3, 0.2],
            2,
            False,
            True,
            10_000,
            10_000,
            10_000,
            False,
        ),
        ([[1, 0], [0, 1]], "multi_gens", [-0.41], 2, True, False, 10_000, None, None, True),
        ([[1, 0], [1, 0]], "multi_gens", [0.2], 2, True, True, 10_000, None, None, True),
    ],
)
def test_expval(
    ops,
    gates_fn,
    params,
    n_qubits,
    spin_sym,
    sparse,
    n_samples,
    max_batch_samples,
    max_batch_ops,
    indep_estimates,
):  # pylint: disable=too-many-arguments
    gates = local_gates(n_qubits, 1)
    if gates_fn == "multi_gens":
        gates = [[gates[0][0], gates[1][0]]] + gates[2:]

    key = jax.random.PRNGKey(np.random.randint(0, 99999))

    if not isinstance(ops, csr_matrix):
        ops = jnp.array(ops)

    exp_val, std = iqp_expval(
        ops=ops,
        n_samples=n_samples,
        key=key,
        num_wires=n_qubits,
        pattern=gates,
        weights=params,
        spin_sym=spin_sym,
        sparse=sparse,
        indep_estimates=indep_estimates,
        max_batch_samples=max_batch_samples,
        max_batch_ops=max_batch_ops,
    )

    dev = device("default.qubit")

    @qnode(dev)
    def iqp_circuit(weights, pattern, spin_sym, n_qubits, ops):
        IQP(weights, n_qubits, pattern, spin_sym)

        expectation_operators = []
        if not isinstance(ops, csr_matrix):
            for l in ops:
                for i, qubit in enumerate(l):
                    if qubit == 1:
                        expectation_operators.append(expval(PauliZ(i)))
        else:
            rows, cols = ops.nonzero()
            for row, col in zip(rows, cols):
                if ops[row, col] == 1:
                    expectation_operators.append(expval(PauliZ(col)))

        return expectation_operators

    if len(ops.shape) == 1:
        ops = ops.reshape(1, -1)

    simulated_exp_val = jnp.array(iqp_circuit(params, gates, spin_sym, n_qubits, ops))

    for i, val in enumerate(simulated_exp_val):
        # Due to the distribution, we expect the simulated and the approximated values to be within 2 standard
        # deviations 96% of the time. We can instead check they are withing 3 standard deviations, which should
        # be True 99.8% of the time, minimizing stochastic failures.
        assert np.isclose(val, exp_val[i], atol=std[i] * 3)

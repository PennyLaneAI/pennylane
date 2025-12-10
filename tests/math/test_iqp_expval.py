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

from pennylane import IQP
from pennylane.math import arange
from pennylane.math.iqp import op_expval


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
        "expected_val",
        "expected_std",
    ),
    [
        ("multi_gens", [0], 2, False, False, 10, 10_000, 10_000, True, [1.0], [0.0]),
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
    expected_val,
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

    op = np.random.randint(0, 2, (n_qubits,))
    key = jax.random.PRNGKey(np.random.randint(0, 99999))

    expval, std = op_expval(
        ops=op,
        n_samples=n_samples,
        key=key,
        circuit=circuit,
        sparse=sparse,
        indep_estimates=indep_estimates,
        max_batch_samples=max_batch_samples,
        max_batch_ops=max_batch_ops,
    )

    assert expval == jnp.array(expected_val)
    assert std == jnp.array(expected_std)

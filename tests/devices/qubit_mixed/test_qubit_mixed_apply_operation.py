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
"""Unit tests for apply_operation in devices/qubit_mixed/apply_operation."""

from functools import reduce

import numpy as np
import pytest
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.devices.qubit_mixed import QUDIT_DIM, apply_operation

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


def get_random_mixed_state(num_qubits):
    dim = QUDIT_DIM**num_qubits

    rng = np.random.default_rng(seed=4774)
    basis = unitary_group(dim=dim, seed=584545).rvs()
    schmidt_weights = rng.dirichlet(np.ones(dim), size=1).astype(complex)[0]
    mixed_state = np.zeros((dim, dim)).astype(complex)
    for i in range(dim):
        mixed_state += schmidt_weights[i] * np.outer(np.conj(basis[i]), basis[i])

    return mixed_state.reshape([QUDIT_DIM] * (2 * num_qubits))


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestOperation:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations (not channels) are applied correctly."""

    three_qubit_state = get_random_mixed_state(3)

    unbroadcasted_ops = [
        qml.Hadamard(wires=0),
        qml.RX(np.pi / 3, wires=0),
        qml.RY(2 * np.pi / 3, wires=1),
        qml.RZ(np.pi / 6, wires=2),
    ]
    num_qubits = 3
    num_batched = 2

    @classmethod
    def get_expected_state(cls, expanded_operator, state):
        """Finds expected state after applying operator"""
        flattened_state = state.reshape((QUDIT_DIM**cls.num_qubits,) * 2)
        adjoint_matrix = np.conj(expanded_operator).T
        new_state = expanded_operator @ flattened_state @ adjoint_matrix
        return new_state.reshape([QUDIT_DIM] * (cls.num_qubits * 2))

    @classmethod
    def expand_matrices(cls, op, batch_size=0):
        """Find expanded operator matrices, since qml.matrix isn't working for qubits #4367"""
        pre_wires_identity = np.eye(QUDIT_DIM ** op.wires[0])
        post_wires_identity = np.eye(QUDIT_DIM ** ((cls.num_qubits - 1) - op.wires[-1]))
        mat = op.matrix()

        def expand_matrix(matrix):
            return reduce(np.kron, (pre_wires_identity, matrix, post_wires_identity))

        if batch_size:
            return [expand_matrix(mat[i]) for i in range(batch_size)]
        return expand_matrix(mat)

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_no_broadcasting(self, op, ml_framework, three_qubit_state):
        """Tests that unbatched operations are applied correctly to an unbatched state."""
        res = apply_operation(op, qml.math.asarray(three_qubit_state, like=ml_framework))

        expanded_operator = self.expand_matrices(op)
        expected = self.get_expected_state(expanded_operator, three_qubit_state)

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

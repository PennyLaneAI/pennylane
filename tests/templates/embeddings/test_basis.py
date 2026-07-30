# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the BasisEmbedding template.
"""

import numpy as np
import pytest

import pennylane as qp


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    wires = qp.wires.Wires((0, 1, 2))
    op = qp.BasisEmbedding(np.array([1, 1, 1]), wires=wires)
    qp.ops.functions.assert_valid(op, skip_differentiation=True, skip_capture=True)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize("features", [[1, 0, 1], [1, 1, 1], [0, 1, 0]])
    def test_expansion(self, features):
        """Checks the queue."""

        op = qp.BasisEmbedding(features, wires=range(3))
        tape = qp.tape.QuantumScript(op.decomposition())

        assert len(tape.operations) == features.count(1)
        for gate in tape.operations:
            assert gate.name == "PauliX"

    @pytest.mark.parametrize("state", [[0, 1], [1, 1], [1, 0], [0, 0]])
    def test_state(self, state):
        """Checks that the correct state is prepared."""

        n_qubits = 2
        dev = qp.device("default.qubit", wires=n_qubits)

        @qp.qnode(dev)
        def circuit(x=None):
            qp.BasisEmbedding(x, wires=range(2))
            return [qp.expval(qp.PauliZ(i)) for i in range(n_qubits)]

        res = circuit(x=state)
        expected = [1 if s == 0 else -1 for s in state]
        assert np.allclose(res, expected)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_equivalent_to_basis_state(self):
        """Tests that BasisEmbedding is an alias of BasisState."""
        assert qp.BasisEmbedding is qp.BasisState

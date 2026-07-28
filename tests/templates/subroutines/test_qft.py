# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the qft template.
"""

import numpy as np
import pytest
from gate_data import QFT

import pennylane as qp


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    op = qp.QFT(wires=(0, 1, 2))
    qp.ops.functions.assert_valid(op)


class TestQFT:
    """Tests for the qft operations"""

    def test_QFT(self):
        """Test if the QFT matrix is equal to a manually-calculated version for 3 qubits"""
        op = qp.QFT(wires=range(3))
        res = op.matrix()
        exp = QFT
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("n_qubits", range(2, 6))
    def test_QFT_compute_decomposition(self, n_qubits):
        """Test if the QFT operation is correctly decomposed"""
        decomp = qp.QFT.compute_decomposition(wires=range(n_qubits))

        dev = qp.device("default.qubit", wires=n_qubits)

        out_states = []
        for state in np.eye(2**n_qubits):
            ops = [qp.StatePrep(state, wires=range(n_qubits))] + decomp
            qs = qp.tape.QuantumScript(ops, [qp.state()])
            out_states.append(dev.execute(qs))

        reconstructed_unitary = np.array(out_states).T
        expected_unitary = qp.QFT(wires=range(n_qubits)).matrix()

        assert np.allclose(reconstructed_unitary, expected_unitary)

    @pytest.mark.parametrize("n_qubits", range(2, 6))
    def test_QFT_decomposition(self, n_qubits):
        """Test if the QFT operation is correctly decomposed"""
        op = qp.QFT(wires=range(n_qubits))
        decomp = op.decomposition()

        dev = qp.device("default.qubit", wires=n_qubits)

        out_states = []
        for state in np.eye(2**n_qubits):
            ops = [qp.StatePrep(state, wires=range(n_qubits))] + decomp
            qs = qp.tape.QuantumScript(ops, [qp.state()])
            out_states.append(dev.execute(qs))

        reconstructed_unitary = np.array(out_states).T
        expected_unitary = qp.QFT(wires=range(n_qubits)).matrix()

        assert np.allclose(reconstructed_unitary, expected_unitary)

    @pytest.mark.parametrize("n_qubits", range(2, 10))
    def test_QFT_adjoint_identity(self, n_qubits, tol):
        """Test if using the qp.adjoint transform the resulting operation is
        the inverse of QFT."""

        dev = qp.device("default.qubit", wires=n_qubits)

        @qp.qnode(dev)
        def circ(n_qubits):
            qp.adjoint(qp.QFT)(wires=range(n_qubits))
            qp.QFT(wires=range(n_qubits))
            return qp.state()

        assert np.allclose(1, circ(n_qubits)[0], tol)

        for i in range(1, n_qubits):
            assert np.allclose(0, circ(n_qubits)[i], tol)

    def test_matrix(self, tol):
        """Test that the matrix representation is correct."""

        res_static = qp.QFT.compute_matrix(2)
        res_dynamic = qp.QFT(wires=[0, 1]).matrix()
        res_reordered = qp.QFT(wires=[0, 1]).matrix([1, 0])

        expected = np.array(
            [
                [0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                [0.5 + 0.0j, 0.0 + 0.5j, -0.5 + 0.0j, -0.0 - 0.5j],
                [0.5 + 0.0j, -0.5 + 0.0j, 0.5 - 0.0j, -0.5 + 0.0j],
                [0.5 + 0.0j, -0.0 - 0.5j, -0.5 + 0.0j, 0.0 + 0.5j],
            ]
        )

        assert np.allclose(res_static, expected, atol=tol, rtol=0)
        assert np.allclose(res_dynamic, expected, atol=tol, rtol=0)

        expected_permuted = [
            [0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
            [0.5 + 0.0j, 0.5 - 0.0j, -0.5 + 0.0j, -0.5 + 0.0j],
            [0.5 + 0.0j, -0.5 + 0.0j, 0.0 + 0.5j, -0.0 - 0.5j],
            [0.5 + 0.0j, -0.5 + 0.0j, -0.0 - 0.5j, 0.0 + 0.5j],
        ]
        assert np.allclose(res_reordered, expected_permuted, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jit(self):
        import jax
        import jax.numpy as jnp

        wires = 3

        dev = qp.device("default.qubit", wires=wires)

        @qp.qnode(dev)
        def circuit_qft(basis_state):
            qp.BasisState(basis_state, wires=range(wires))
            qp.QFT(wires=range(wires))
            return qp.state()

        jit_qft = jax.jit(circuit_qft)

        res = circuit_qft(jnp.array([1.0, 0.0, 0.0]))
        res2 = jit_qft(jnp.array([1.0, 0.0, 0.0]))

        assert qp.math.allclose(res, res2)

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
Unit tests for the qubit matrix-based operations.
"""
# pylint: disable=import-outside-toplevel
from functools import reduce

import numpy as np
import pytest
from gate_data import H, I, S, T, X, Z
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix
from scipy.stats import unitary_group

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.ops.op_math.decompositions.unitary_decompositions import _compute_udv
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform, fractional_matrix_power
from pennylane.wires import Wires


class TestQubitUnitaryCSR:
    """Tests for using csr_matrix in QubitUnitary."""

    def test_compute_matrix_blocked(self):
        """Test that when csr_matrix is used, the compute_matrix method works correctly."""
        U = csr_matrix(I)
        with pytest.raises(
            qp.operation.MatrixUndefinedError,
            match="U is sparse matrix",
        ):
            qp.QubitUnitary.compute_matrix(U)

        with pytest.raises(
            qp.operation.MatrixUndefinedError,
            match="U is sparse matrix",
        ):
            op = qp.QubitUnitary(U, wires=[0])
            op.matrix()

    def test_compute_sparse_matrix(self):
        """Test that the compute_sparse_matrix method works correctly."""
        U = np.array([[0, 1], [1, 0]])
        U = csr_matrix(U)
        op = qp.QubitUnitary.compute_sparse_matrix(U)
        assert isinstance(op, csr_matrix)
        assert np.allclose(op.toarray(), U.toarray())

        # Test that the sparse matrix accepts the format parameter.
        op_csc = qp.QubitUnitary.compute_sparse_matrix(U, format="csc")
        op_lil = qp.QubitUnitary.compute_sparse_matrix(U, format="lil")
        op_coo = qp.QubitUnitary.compute_sparse_matrix(U, format="coo")
        assert isinstance(op_csc, csc_matrix)
        assert isinstance(op_lil, lil_matrix)
        assert isinstance(op_coo, coo_matrix)

    def test_generic_sparse_convert_to_csr(self):
        """Test that other generic sparse matrices can be converted to csr_matrix."""
        # 4x4 Identity as a csr_matrix
        dense = np.eye(4)
        sparse = coo_matrix(dense)
        op = qp.QubitUnitary(sparse, wires=[0, 1])
        assert isinstance(op.sparse_matrix(), csr_matrix)
        sparse = csc_matrix(dense)
        op = qp.QubitUnitary(sparse, wires=[0, 1])
        assert isinstance(op.sparse_matrix(), csr_matrix)

    @pytest.mark.parametrize(
        "dense",
        [H, I, S, T, X, Z],
    )
    def test_csr_matrix_init_success(self, dense):
        """Test that a valid 2-wire csr_matrix can be instantiated, covering necessary single-qubit gates."""
        # 4x4 Identity as a csr_matrix
        sparse = csr_matrix(dense)
        op = qp.QubitUnitary(sparse, wires=[0])
        assert isinstance(op.sparse_matrix(), csr_matrix)  # Should still be sparse
        with pytest.raises(
            qp.operation.MatrixUndefinedError,
            match="U is sparse matrix",
        ):
            assert qp.math.allclose(op.matrix(), dense)

    def test_csr_matrix_shape_mismatch(self):
        """Test that shape mismatch with csr_matrix raises an error."""
        dense = np.eye(2)  # Only 2x2
        sparse = csr_matrix(dense)
        with pytest.raises(ValueError, match="Input unitary must be of shape"):
            qp.QubitUnitary(sparse, wires=[0, 1])

    def test_csr_matrix_unitary_check_fail(self):
        """Test that unitary_check warns if the matrix may not be unitary."""
        dense = np.array([[1, 0], [0, 0.5]])  # Not a unitary
        sparse = csr_matrix(dense)
        with pytest.warns(UserWarning, match="may not be unitary"):
            qp.QubitUnitary(sparse, wires=0, unitary_check=True)

    def test_csr_matrix_pow_integer(self):
        """Test that QubitUnitary.pow() works for integer exponents with csr_matrix."""
        dense = np.eye(4)
        sparse = csr_matrix(dense)
        op = qp.QubitUnitary(sparse, wires=[0, 1])
        powered_ops = op.pow(2)
        assert len(powered_ops) == 1

        powered_op = powered_ops[0]
        assert isinstance(powered_op, qp.QubitUnitary)
        assert isinstance(powered_op.sparse_matrix(), csr_matrix)
        # The resulting matrix should still be the identity
        final_mat = powered_op.sparse_matrix()
        # If it's still sparse, compare .toarray()
        if isinstance(final_mat, csr_matrix):
            final_mat = final_mat.toarray()
        assert qp.math.allclose(final_mat, dense)

    @pytest.mark.parametrize(
        "dense",
        [
            np.eye(4),
            np.array(
                [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
            ),  # sample permutation matrix
        ],
    )
    def test_csr_matrix_adjoint(self, dense):
        """Test that QubitUnitary.adjoint() works with csr_matrix, matching dense result."""
        sparse = csr_matrix(dense)
        op = qp.QubitUnitary(sparse, wires=[0, 1])
        adj_op = op.adjoint()

        assert isinstance(adj_op, qp.QubitUnitary)
        assert isinstance(adj_op.sparse_matrix(), csr_matrix)

        final_mat = adj_op.sparse_matrix()
        # Compare with dense representation if still sparse
        if isinstance(final_mat, csr_matrix):
            final_mat = final_mat.toarray()

        # For real/complex conjugate transpose, if dense is unitary, final_mat == dense^\dagger
        expected = dense.conjugate().T
        assert qp.math.allclose(final_mat, expected)

    def test_csr_matrix_adjoint_large(self):
        """Construct a large sparse matrix (e.g. 2^20 dimension) but only store minimal elements."""
        N = 20
        dim = 2**N

        # For demonstration, let's just store a single 1 on the diagonal
        row_indices = [12345]  # some arbitrary index < dim
        col_indices = [12345]
        data = [1.0]

        sparse_large = csr_matrix((data, (row_indices, col_indices)), shape=(dim, dim))
        with pytest.warns(UserWarning, match="may not be unitary"):
            op = qp.QubitUnitary(sparse_large, wires=range(N), unitary_check=True)
        adj_op = op.adjoint()

        assert isinstance(adj_op, qp.QubitUnitary)
        assert isinstance(adj_op.sparse_matrix(), csr_matrix)

        # The single element should remain 1 at [12345,12345] after conjugate transpose
        final_mat = adj_op.sparse_matrix()
        assert final_mat[12345, 12345] == 1.0

    def test_csr_matrix_decomposition(self):
        """Test that QubitUnitary.decomposition() works with csr_matrix."""
        # 4x4 Identity as a csr_matrix
        U = csr_matrix(unitary_group.rvs(4))
        op = qp.QubitUnitary(U, wires=[0, 1])
        assert not op.has_decomposition
        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

        # 2x2 Identity as a csr_matrix
        mat = csr_matrix(unitary_group.rvs(2))
        op = qp.QubitUnitary(mat, wires=[0])
        assert op.has_decomposition
        decomp = op.decomposition()
        assert len(decomp) == 4
        mat2 = qp.matrix(op.decomposition, wire_order=[0])()
        assert qp.math.allclose(mat2, mat.todense())

    def test_csr_matrix_decomposition_new(self):
        """Tests that the QubitUnitary's decomposition works with csr_matrix."""

        U = csr_matrix(unitary_group.rvs(2))
        op = qp.QubitUnitary(U, wires=[0])
        rule = qp.list_decomps(qp.QubitUnitary)[0]
        with qp.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        tape = qp.tape.QuantumScript.from_queue(q)
        actual_mat = qp.matrix(tape)
        assert qp.math.allclose(actual_mat, U.todense())

    def test_csr_matrix_pow_new(self):
        """Tests the pow decomposition of a QubitUnitary works with csr_matrix."""

        U = csr_matrix(unitary_group.rvs(2))
        op = qp.pow(qp.QubitUnitary(U, wires=[0]), 2)
        rule = qp.list_decomps("Pow(QubitUnitary)")[0]
        with qp.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        tape = qp.tape.QuantumScript.from_queue(q)
        actual_mat = qp.matrix(tape)
        assert qp.math.allclose(actual_mat, (U @ U).todense())


class TestQubitUnitary:
    """Tests for the QubitUnitary class."""

    def test_qubit_unitary_noninteger_pow(self):
        """Test QubitUnitary raised to a non-integer power raises an error."""

        U = np.array(
            [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]]
        )

        op = qp.QubitUnitary(U, wires="a")
        [pow_op] = op.pow(0.123)
        expected = fractional_matrix_power(U, 0.123)

        assert qp.math.allclose(pow_op.matrix(), expected)

    def test_qubit_unitary_noninteger_pow_broadcasted(self):
        """Test broadcasted QubitUnitary raised to a non-integer power raises an error."""

        U = np.array(
            [
                [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]],
                [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]],
            ]
        )

        op = qp.QubitUnitary(U, wires="a")

        with pytest.raises(qp.operation.PowUndefinedError):
            op.pow(0.123)

    @pytest.mark.parametrize("n", (1, 3, -1, -3))
    def test_qubit_unitary_pow(self, n):
        """Test qubit unitary raised to an integer power."""

        U = np.array(
            [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]]
        )

        op = qp.QubitUnitary(U, wires="a")
        new_ops = op.pow(n)

        assert len(new_ops) == 1
        assert new_ops[0].wires == op.wires

        mat_to_pow = qp.math.linalg.matrix_power(qp.matrix(op), n)
        new_mat = qp.matrix(new_ops[0])

        assert qp.math.allclose(mat_to_pow, new_mat)

    @pytest.mark.parametrize("n", (1, 3, -1, -3))
    def test_qubit_unitary_pow_broadcasted(self, n):
        """Test broadcasted qubit unitary raised to an integer power."""

        U = np.array(
            [
                [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]],
                [[0.4125124 + 0.0j, 0.0 - 0.91095199j], [0.0 - 0.91095199j, 0.4125124 + 0.0j]],
            ]
        )

        op = qp.QubitUnitary(U, wires="a")
        new_ops = op.pow(n)

        assert len(new_ops) == 1
        assert new_ops[0].wires == op.wires

        mat_to_pow = qp.math.linalg.matrix_power(qp.matrix(op), n)
        new_mat = qp.matrix(new_ops[0])

        assert qp.math.allclose(mat_to_pow, new_mat)

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "U,num_wires", [(H, 1), (np.kron(H, H), 2), (np.tensordot([1j, -1, 1], H, axes=0), 1)]
    )
    def test_qubit_unitary_autograd(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with autograd."""

        out = qp.QubitUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert qp.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qp.QubitUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = U.copy()
        U3[0, 0] += 0.5
        with pytest.warns(UserWarning, match="may not be unitary"):
            qp.QubitUnitary(U3, wires=range(num_wires), unitary_check=True).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qp.QubitUnitary(U, wires=range(num_wires + 1)).matrix()

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "U,num_wires", [(H, 1), (np.kron(H, H), 2), (np.tensordot([1j, -1, 1], H, axes=0), 1)]
    )
    def test_qubit_unitary_torch(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with torch."""

        import torch

        U = torch.tensor(U)
        out = qp.QubitUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, torch.Tensor)

        # verify equivalent to input state
        assert qp.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qp.QubitUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = U.detach().clone()
        U3[0, 0] += 0.5
        with pytest.warns(UserWarning, match="may not be unitary"):
            qp.QubitUnitary(U3, wires=range(num_wires), unitary_check=True).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qp.QubitUnitary(U, wires=range(num_wires + 1)).matrix()

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "U,num_wires", [(H, 1), (np.kron(H, H), 2), (np.tensordot([1j, -1, 1], H, axes=0), 1)]
    )
    def test_qubit_unitary_tf(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with tensorflow."""

        import tensorflow as tf

        U = tf.Variable(U)
        out = qp.QubitUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, tf.Variable)

        # verify equivalent to input state
        assert qp.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qp.QubitUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = tf.Variable(U + 0.5)
        with pytest.warns(UserWarning, match="may not be unitary"):
            qp.QubitUnitary(U3, wires=range(num_wires), unitary_check=True).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qp.QubitUnitary(U, wires=range(num_wires + 1)).matrix()

    @pytest.mark.tf
    def test_qubit_unitary_int_pow_tf(self):
        """Test that QubitUnitary.pow works with tf and int z values."""

        import tensorflow as tf

        mat = tf.Variable([[1, 0], [0, tf.exp(1j)]])
        expected = tf.Variable([[1, 0], [0, tf.exp(3j)]])
        [op] = qp.QubitUnitary(mat, wires=[0]).pow(3)
        assert qp.math.allclose(op.matrix(), expected)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "U,num_wires", [(H, 1), (np.kron(H, H), 2), (np.tensordot([1j, -1, 1], H, axes=0), 1)]
    )
    def test_qubit_unitary_jax(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with jax."""

        from jax import numpy as jnp

        U = jnp.array(U)
        out = qp.QubitUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, jnp.ndarray)

        # verify equivalent to input state
        assert qp.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qp.QubitUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = U + 0.5
        with pytest.warns(UserWarning, match="may not be unitary"):
            qp.QubitUnitary(U3, wires=range(num_wires), unitary_check=True).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qp.QubitUnitary(U, wires=range(num_wires + 1)).matrix()

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "U,num_wires",
        [
            (H, 1),
            (np.kron(H, H), 2),
            (np.kron(np.kron(np.kron(H, H), H), H), 4),
            (np.tensordot([1j, -1, 1], H, axes=0), 1),
        ],
    )
    def test_qubit_unitary_jax_jit(self, U, num_wires):
        """Tests that QubitUnitary works with jitting."""

        import jax
        from jax import numpy as jnp

        U = jnp.array(U)

        def mat_fn(m):
            return qp.QubitUnitary(m, wires=range(num_wires)).matrix()

        out = jax.jit(mat_fn)(U)
        assert qp.math.allclose(out, qp.QubitUnitary(U, wires=range(num_wires)).matrix())

    @pytest.mark.jax
    def test_qubit_unitary_jax_jit_decomposition(self):
        """Tests that QubitUnitary works with jitting when decomposing the operator."""

        import jax
        from jax import numpy as jnp

        matrix = jnp.array(qp.matrix(qp.QFT(wires=[0, 1, 2])))

        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def circuit(matrix):
            qp.QubitUnitary.compute_decomposition(matrix, wires=[0, 1, 2])
            return qp.state()

        state_expected = circuit(matrix)
        state_jit = jax.jit(circuit)(matrix)

        assert qp.math.allclose(state_expected, state_jit)

    @pytest.mark.parametrize(
        "U, expected_params",
        [
            (I, [0.0, 0.0, 0.0, 0.0]),
            (Z, [np.pi / 2, 0.0, np.pi / 2, -np.pi / 2]),
            (S, [np.pi / 4, 0.0, np.pi / 4, -np.pi / 4]),
            (T, [np.pi / 8, 0.0, np.pi / 8, -np.pi / 8]),
            (qp.matrix(qp.RZ(0.3, wires=0)), [0.15, 0.0, 0.15, 0.0]),
            (
                qp.matrix(qp.RZ(-0.5, wires=0)),
                [4 * np.pi - 0.25, 0.0, 4 * np.pi - 0.25, 0.0],
            ),
            (
                np.array(
                    [
                        [0, -9.831019270939975e-01 + 0.1830590094588862j],
                        [9.831019270939975e-01 + 0.1830590094588862j, 0],
                    ]
                ),
                [12.382273469673908, np.pi, 0.18409714468526372, 0.0],
            ),
            (H, [np.pi, np.pi / 2, 0.0, -np.pi / 2]),
            (X, [np.pi / 2, np.pi, 7 * np.pi / 2, -np.pi / 2]),
            (
                qp.matrix(qp.Rot(0.2, 0.5, -0.3, wires=0)),
                [0.2, 0.5, 4 * np.pi - 0.3, 0.0],
            ),
            (
                np.exp(1j * 0.02) * qp.matrix(qp.Rot(-1.0, 2.0, -3.0, wires=0)),
                [4 * np.pi - 1.0, 2.0, 4 * np.pi - 3.0, -0.02],
            ),
            # An instance of a broadcast unitary
            (
                np.exp(1j * 0.02)
                * qp.Rot(
                    np.array([1.2, 2.3]), np.array([0.12, 0.5]), np.array([0.98, 0.567]), wires=0
                ).matrix(),
                [[1.2, 2.3], [0.12, 0.5], [0.98, 0.567], [-0.02, -0.02]],
            ),
        ],
    )
    def test_qubit_unitary_decomposition(self, U, expected_params):
        """Tests that single-qubit QubitUnitary decompositions are performed."""

        expected_gates = (qp.RZ, qp.RY, qp.RZ, qp.GlobalPhase)

        decomp = qp.QubitUnitary.compute_decomposition(U, wires=0)
        decomp2 = qp.QubitUnitary(U, wires=0).decomposition()

        mat1 = qp.matrix(qp.QubitUnitary.compute_decomposition, wire_order=[0])(U, wires=0)
        assert qp.math.allclose(mat1, U)

        assert len(decomp) == 4 == len(decomp2)
        for i in range(4):
            assert isinstance(decomp[i], expected_gates[i])
            assert np.allclose(decomp[i].parameters, expected_params[i], atol=1e-7)
            assert isinstance(decomp2[i], expected_gates[i])
            assert np.allclose(decomp2[i].parameters, expected_params[i], atol=1e-7)

    @pytest.mark.parametrize(
        "U",
        [
            (qp.matrix(qp.GlobalPhase(12, wires=0) @ qp.CRX(2, wires=[1, 0]))),  # 2 cnots
            (qp.matrix(qp.CRX(2, wires=[1, 0]))),  # 2 cnots
            (qp.matrix(qp.TrotterProduct(qp.X(0) + 0.3 * qp.Y(1), time=1, n=5))),  # 0 cnots
            (
                qp.matrix(qp.TrotterProduct(qp.X(0) @ qp.Z(1) - 0.3 * qp.Y(1), time=1))
            ),  # 2 cnots
            (qp.matrix(qp.CRY(1, wires=[0, 1]))),  # 2 cnots
            (qp.matrix(qp.QFT(wires=[0, 1]))),  # 3 cnots
            (qp.matrix(qp.GlobalPhase(12, wires=0) @ qp.QFT(wires=[0, 1]))),  # 3 cnots
            (qp.matrix(qp.RZ(1, wires=0) @ qp.GroverOperator(wires=[0, 1]))),  # 1 cnot
            (qp.matrix(qp.GlobalPhase(12, wires=0) @ qp.GroverOperator(wires=[0, 1]))),  # 1 cnot
            (qp.matrix(qp.CRY(-1, wires=[0, 1]))),  # 2 cnots
            (qp.matrix(qp.SWAP(wires=[0, 1]))),  # 3 cnots
            (qp.matrix(qp.SWAP(wires=[0, 1]) @ qp.GlobalPhase(3))),  # 3 cnots
            (qp.matrix(qp.Hadamard(wires=[0]) @ qp.RX(2, wires=1))),  # 0 cnots
            (qp.matrix(qp.RX(-1, wires=[0]) @ qp.RZ(-5, wires=1))),  # 0 cnots
            (
                qp.matrix(
                    qp.MottonenStatePreparation(np.sqrt([0.25, 0.15, 0.2, 0.4]), wires=[0, 1])
                )
            ),  # 2 cnots
        ],
    )
    def test_qubit_unitary_correct_global_phase(self, U):
        """Tests that the input matrix matches with the decomposition matrix even in the global phase"""

        ops_decompostion = qp.QubitUnitary.compute_decomposition(U, wires=[0, 1])

        assert qp.math.allclose(
            U, qp.matrix(qp.prod(*ops_decompostion[::-1]), wire_order=[0, 1]), atol=1e-7
        )

    def test_operations_correctly_queued(self):
        """Tests that the operators in the decomposition are queued correctly"""

        dev = qp.device("default.qubit")

        matrix = qp.matrix(qp.QFT(wires=[0, 1]))
        ops_decompostion = qp.QubitUnitary.compute_decomposition(matrix, wires=[0, 1])

        @qp.qnode(dev)
        def circuit():
            qp.QubitUnitary.compute_decomposition(matrix, wires=[0, 1])
            return qp.state()

        tape = qp.workflow.construct_tape(circuit)()
        for op1, op2 in zip(tape.operations, ops_decompostion):
            qp.assert_equal(op1, op2)

    def test_broadcasted_two_qubit_qubit_unitary_decomposition_raises_error(self):
        """Tests that broadcasted QubitUnitary decompositions are not supported."""
        U = qp.IsingYY.compute_matrix(np.array([1.2, 2.3, 3.4]))

        with pytest.raises(DecompositionUndefinedError, match="QubitUnitary does not support"):
            qp.QubitUnitary.compute_decomposition(U, wires=[0, 1])
        with pytest.raises(DecompositionUndefinedError, match="QubitUnitary does not support"):
            qp.QubitUnitary(U, wires=[0, 1]).decomposition()

    @pytest.mark.parametrize(
        "U, wires",
        [
            (qp.matrix(qp.CRX(2, wires=[1, 0])), [0, 1]),
            (qp.matrix(qp.QFT(wires=[0, 1, 2, 3, 4])), [0, 1, 2, 3, 4]),
            (qp.matrix(qp.CRX(1, [0, 2]) @ qp.CRY(2, [1, 3])), [0, 1, 2, 3]),
            (qp.matrix(qp.GroverOperator([0, 1, 2, 3, 4, 5])), [0, 1, 2, 3, 4, 5]),
        ],
    )
    def test_correctness_decomposition(self, U, wires):
        """Tests that the decomposition is correct"""

        ops_decompostion = qp.QubitUnitary.compute_decomposition(U, wires=wires)

        assert qp.math.allclose(
            U, qp.matrix(qp.prod(*ops_decompostion[::-1]), wire_order=wires), atol=1e-7
        )

    @pytest.mark.parametrize(
        "a, b, size",
        [
            (qp.matrix(qp.RY(1, 0) @ qp.RY(2, 1)), qp.matrix(qp.RX(2, 0) @ qp.RZ(4, 1)), 4),
            (qp.matrix(qp.RY(1, 0)), qp.matrix(qp.RX(2, 0)), 2),
            (qp.matrix(qp.GroverOperator([0, 1, 2])), qp.matrix(qp.QFT([0, 1, 2])), 8),
        ],
    )
    def test_compute_udv(self, a, b, size):
        """Test the helper function `_compute_udv` used in the QubitUnitary decomposition."""

        u, d, v = _compute_udv(a, b)
        d = np.diag(d)

        initial = np.block(
            [[a, np.zeros((size, size), dtype=complex)], [np.zeros((size, size), dtype=complex), b]]
        )
        u_block = np.block(
            [[u, np.zeros((size, size), dtype=complex)], [np.zeros((size, size), dtype=complex), u]]
        )
        v_block = np.block(
            [[v, np.zeros((size, size), dtype=complex)], [np.zeros((size, size), dtype=complex), v]]
        )
        d_block = np.block(
            [
                [d, np.zeros((size, size), dtype=complex)],
                [np.zeros((size, size), dtype=complex), np.conj(d)],
            ]
        )

        assert np.allclose(initial, u_block @ d_block @ v_block)

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array(
            [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]]
        )
        res_static = qp.QubitUnitary.compute_matrix(U)
        res_dynamic = qp.QubitUnitary(U, wires=0).matrix()
        expected = U
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_matrix_representation_broadcasted(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array(
            [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]]
        )
        U = np.tensordot([1j, -1.0, (1 + 1j) / np.sqrt(2)], U, axes=0)
        res_static = qp.QubitUnitary.compute_matrix(U)
        res_dynamic = qp.QubitUnitary(U, wires=0).matrix()
        expected = U
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_controlled(self):
        """Test QubitUnitary's controlled method."""
        # pylint: disable=protected-access
        U = qp.PauliX.compute_matrix()
        base = qp.QubitUnitary(U, wires=0)

        expected = qp.ControlledQubitUnitary(U, wires=["a", 0])

        out = base._controlled("a")
        qp.assert_equal(out, expected)


class TestWalshHadamardTransform:
    """Test the helper function walsh_hadamard_transform."""

    @pytest.mark.parametrize(
        "inp, exp",
        [
            ([1, 1, 1, 1], [1, 0, 0, 0]),
            ([1, 1.5, 0.5, 1], [1, -0.25, 0.25, 0]),
            ([1, 0, -1, 2.5], [0.625, -0.625, -0.125, 1.125]),
        ],
    )
    def test_compare_analytic_results(self, inp, exp):
        """Test against hard-coded results."""
        inp = np.array(inp)
        output = _walsh_hadamard_transform(inp)
        assert qp.math.allclose(output, exp)

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("provide_n", [True, False])
    def test_compare_matrix_mult(self, n, provide_n, seed):
        """Test against matrix multiplication for a few random inputs."""
        rng = np.random.default_rng(seed)
        inp = rng.random(2**n)
        output = _walsh_hadamard_transform(inp, n=n if provide_n else None)
        h = np.array([[0.5, 0.5], [0.5, -0.5]])
        h = reduce(np.kron, [h] * n)
        exp = h @ inp
        assert qp.math.allclose(output, exp)

    def test_compare_analytic_results_broadcasted(self):
        """Test against hard-coded results."""
        inp = np.array([[1, 1, 1, 1], [1, 1.5, 0.5, 1], [1, 0, -1, 2.5]])
        exp = [[1, 0, 0, 0], [1, -0.25, 0.25, 0], [0.625, -0.625, -0.125, 1.125]]
        output = _walsh_hadamard_transform(inp)
        assert qp.math.allclose(output, exp)

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("provide_n", [True, False])
    def test_compare_matrix_mult_broadcasted(self, n, provide_n, seed):
        """Test against matrix multiplication for a few random inputs."""
        rng = np.random.default_rng(seed)
        inp = rng.random((5, 2**n))
        output = _walsh_hadamard_transform(inp, n=n if provide_n else None)
        h = np.array([[0.5, 0.5], [0.5, -0.5]])
        h = reduce(np.kron, [h] * n)
        exp = qp.math.moveaxis(qp.math.tensordot(h, inp, [[1], [1]]), 0, 1)
        assert qp.math.allclose(output, exp)


class TestDiagonalQubitUnitary:  # pylint: disable=too-many-public-methods
    """Test the DiagonalQubitUnitary operation."""

    def test_decomposition_single_qubit(self):
        """Test that a single-qubit DiagonalQubitUnitary is decomposed correctly."""
        D = np.array([1j, -1])

        decomp = qp.DiagonalQubitUnitary.compute_decomposition(D, [0])
        decomp2 = qp.DiagonalQubitUnitary(D, wires=[0]).decomposition()

        for dec in (decomp, decomp2):
            assert len(dec) == 2
            qp.assert_equal(decomp[0], qp.GlobalPhase(-3 * np.pi / 4, 0))
            qp.assert_equal(decomp[1], qp.RZ(np.pi / 2, 0))

    def test_decomposition_single_qubit_broadcasted(self):
        """Test that a broadcasted single-qubit DiagonalQubitUnitary is decomposed correctly."""
        D = np.exp(1j * np.pi * np.array([[1 / 2, 1], [1 / 8, -1 / 8], [1 / 2, -1 / 2], [1, 1]]))

        decomp = qp.DiagonalQubitUnitary.compute_decomposition(D, [0])
        decomp2 = qp.DiagonalQubitUnitary(D, wires=[0]).decomposition()

        angles = np.array([1 / 2, -1 / 4, -1, 0]) * np.pi
        global_angles = np.array([3 / 4, 0, 0, 1]) * np.pi
        for dec in (decomp, decomp2):
            assert len(dec) == 2
            qp.assert_equal(decomp[0], qp.GlobalPhase(-global_angles, 0))
            qp.assert_equal(decomp[1], qp.RZ(angles, 0))

    def test_decomposition_two_qubits(self):
        """Test that a two-qubit DiagonalQubitUnitary is decomposed correctly."""
        D = np.exp(1j * np.array([1, -1, 0.5, 1]))

        decomp = qp.DiagonalQubitUnitary.compute_decomposition(D, [0, 1])
        decomp2 = qp.DiagonalQubitUnitary(D, wires=[0, 1]).decomposition()

        angles = np.array([-2, 0.5])
        new_D = np.exp(1j * np.array([0, 3 / 4]))

        for dec in (decomp, decomp2):
            assert len(dec) == 2
            qp.assert_equal(decomp[0], qp.DiagonalQubitUnitary(new_D, wires=[0]))
            qp.assert_equal(decomp[1], qp.SelectPauliRot(angles, [0], target_wire=1))

    def test_decomposition_two_qubits_broadcasted(self):
        """Test that a broadcasted two-qubit DiagonalQubitUnitary is decomposed correctly."""
        D = np.exp(1j * np.array([[1, -1, 0.5, 1], [2.3, 1.9, 0.3, -0.9], [1.1, 0.4, -0.8, 1.2]]))

        decomp = qp.DiagonalQubitUnitary.compute_decomposition(D, [0, 1])
        decomp2 = qp.DiagonalQubitUnitary(D, wires=[0, 1]).decomposition()

        angles = np.array([[-2, 0.5], [-0.4, -1.2], [-0.7, 2.0]])
        new_D = np.exp(1j * np.array([[0, 3 / 4], [2.1, -0.3], [0.75, 0.2]]))

        for dec in (decomp, decomp2):
            assert len(dec) == 2
            qp.assert_equal(decomp[0], qp.DiagonalQubitUnitary(new_D, wires=[0]))
            qp.assert_equal(decomp[1], qp.SelectPauliRot(angles, [0], target_wire=1))

    def test_decomposition_three_qubits(self):
        """Test that a three-qubit DiagonalQubitUnitary is decomposed correctly."""
        D = np.exp(1j * np.array([1, -1, 0.5, 1, 0.2, 0.1, 0.6, 2.3]))

        decomp = qp.DiagonalQubitUnitary.compute_decomposition(D, [0, 1, 2])
        decomp2 = qp.DiagonalQubitUnitary(D, wires=[0, 1, 2]).decomposition()

        angles = np.array([-2, 0.5, -0.1, 1.7])
        new_D = np.exp(1j * np.array([0, 3 / 4, 0.15, 1.45]))
        for dec in (decomp, decomp2):
            assert len(dec) == 2
            qp.assert_equal(decomp[0], qp.DiagonalQubitUnitary(new_D, wires=[0, 1]))
            qp.assert_equal(decomp[1], qp.SelectPauliRot(angles, [0, 1], target_wire=2))

    def test_decomposition_three_qubits_broadcasted(self):
        """Test that a broadcasted three-qubit DiagonalQubitUnitary is decomposed correctly."""
        D = np.exp(
            1j
            * np.array(
                [[1, -1, 0.5, 1, 0.2, 0.1, 0.6, 2.3], [1, 0.2, 0.1, 0.6, 2.3, 0.5, 0.2, 0.1]]
            )
        )

        decomp = qp.DiagonalQubitUnitary.compute_decomposition(D, [0, 1, 2])
        decomp2 = qp.DiagonalQubitUnitary(D, wires=[0, 1, 2]).decomposition()

        angles = np.array([[-2, 0.5, -0.1, 1.7], [-0.8, 0.5, -1.8, -0.1]])
        new_D = np.exp(1j * np.array([[0, 3 / 4, 0.15, 1.45], [0.6, 0.35, 1.4, 0.15]]))
        for dec in (decomp, decomp2):
            assert len(dec) == 2
            qp.assert_equal(decomp[0], qp.DiagonalQubitUnitary(new_D, wires=[0, 1]))
            qp.assert_equal(decomp[1], qp.SelectPauliRot(angles, [0, 1], target_wire=2))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_decomposition_matrix_match(self, n, seed):
        """Test that the matrix of the decomposition matches the original matrix."""
        rng = np.random.default_rng(seed)
        D = np.exp(1j * rng.random(2**n))
        wires = list(range(n))
        decomp = qp.DiagonalQubitUnitary.compute_decomposition(D, wires)
        decomp2 = qp.DiagonalQubitUnitary(D, wires=wires).decomposition()

        orig_mat = qp.DiagonalQubitUnitary(D, wires=wires).matrix()
        decomp_mat = qp.matrix(qp.tape.QuantumScript(decomp), wire_order=wires)
        decomp_mat2 = qp.matrix(qp.tape.QuantumScript(decomp2), wire_order=wires)
        assert qp.math.allclose(orig_mat, decomp_mat)
        assert qp.math.allclose(orig_mat, decomp_mat2)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_decomposition_matrix_match_broadcasted(self, n, seed):
        """Test that the broadcasted matrix of the decomposition matches the original matrix."""
        rng = np.random.default_rng(seed)
        D = np.exp(1j * rng.random((5, 2**n)))
        wires = list(range(n))
        decomp = qp.DiagonalQubitUnitary.compute_decomposition(D, wires)
        decomp2 = qp.DiagonalQubitUnitary(D, wires=wires).decomposition()

        orig_mat = qp.DiagonalQubitUnitary(D, wires=wires).matrix()
        decomp_mat = qp.matrix(qp.tape.QuantumScript(decomp), wire_order=wires)
        decomp_mat2 = qp.matrix(qp.tape.QuantumScript(decomp2), wire_order=wires)
        assert qp.math.allclose(orig_mat, decomp_mat)
        assert qp.math.allclose(orig_mat, decomp_mat2)

    @pytest.mark.parametrize(
        "dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.complex128, np.complex64]
    )
    def test_decomposition_cast_to_complex128(self, dtype):
        """Test that the parameters of decomposed operations are of the correct dtype."""
        D = np.array([1, 1, -1, -1]).astype(dtype)
        wires = [0, 1]
        decomp1 = qp.DiagonalQubitUnitary(D, wires).decomposition()
        decomp2 = qp.DiagonalQubitUnitary.compute_decomposition(D, wires)

        r_dtype = (
            np.float64 if dtype in [np.float64, np.int64, np.int32, np.complex128] else np.float32
        )
        c_dtype = (
            np.complex128
            if dtype in [np.float64, np.int64, np.int32, np.complex128]
            else np.complex64
        )
        assert decomp1[0].data[0].dtype == c_dtype
        assert decomp2[0].data[0].dtype == c_dtype
        assert decomp1[1].data[0].dtype == r_dtype
        assert decomp2[1].data[0].dtype == r_dtype

    @pytest.mark.parametrize(
        "op",
        [
            qp.DiagonalQubitUnitary(np.array([1j, -1]), wires=[0]),
            qp.DiagonalQubitUnitary(np.exp(1j * np.array([1, -1, 0.5, 1])), wires=[0, 1]),
            qp.DiagonalQubitUnitary(
                np.exp(1j * np.array([1, -1, 0.5, 1, 0.2, 0.1, 0.6, 2.3])), wires=[0, 1, 2]
            ),
        ],
    )
    def test_decomposition_rule_new(self, op):
        """Tests the decomposition rule compatible with the graph-based interface."""
        for rule in qp.list_decomps(qp.DiagonalQubitUnitary):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        "op",
        [
            pytest.param(qp.DiagonalQubitUnitary(np.ones(4), wires=[0, 1]), id="Identity-2Q"),
            pytest.param(
                qp.DiagonalQubitUnitary(np.array([1j, 1j, 1j, 1j]), wires=[0, 1]),
                id="GlobalPhase-2Q",
            ),
            pytest.param(
                qp.DiagonalQubitUnitary(np.array([1, 1, 1, -1]), wires=[0, 1]),
                id="CZ-Gate",
            ),
            pytest.param(
                qp.DiagonalQubitUnitary(np.array([1, 1, 1, 1, 1, 1, 1, -1]), wires=[0, 1, 2]),
                id="CCZ-Gate",
            ),
            pytest.param(
                # angles are [pi, -pi]. diff is -2pi (equiv to 0), mean is 0.
                # Should decompose to a GlobalPhase and an Identity RZ.
                qp.DiagonalQubitUnitary(np.exp(1j * np.array([np.pi, -np.pi])), wires=[0]),
                id="Phase-Wrap-Around",
            ),
            pytest.param(
                qp.DiagonalQubitUnitary(np.exp(1j * np.array([1e-12, 2e-12])), wires=[0]),
                id="Small-Angle-Difference",
            ),
            pytest.param(
                qp.DiagonalQubitUnitary(
                    # d0 is just below the negative real axis, d1 is on it.
                    # Normalizing to ensure they remain unitary.
                    np.array([(-1 - 1e-9j) / np.abs(-1 - 1e-9j), -1 + 0j]),
                    wires=[0],
                ),
                id="Angle-Branch-Cut-Boundary",
            ),
        ],
    )
    def test_decomposition_rule_edge_cases(self, op):
        """Tests the decomposition rule for various edge cases."""
        for rule in qp.list_decomps(qp.DiagonalQubitUnitary):
            _test_decomposition_rule(op, rule)

    def test_controlled(self):
        """Test that the correct controlled operation is created when controlling a qp.DiagonalQubitUnitary."""
        # pylint: disable=protected-access
        D = np.array([1j, 1, 1, -1, -1j, 1j, 1, -1])
        op = qp.DiagonalQubitUnitary(D, wires=[1, 2, 3])
        with qp.queuing.AnnotatedQueue() as q:
            op._controlled(control=0)
        tape = qp.tape.QuantumScript.from_queue(q)
        mat = qp.matrix(tape, wire_order=[0, 1, 2, 3])
        assert qp.math.allclose(
            mat, qp.math.diag(qp.math.append(qp.math.ones(8, dtype=complex), D))
        )

    def test_controlled_broadcasted(self):
        """Test that the correct controlled operation is created when
        controlling a qp.DiagonalQubitUnitary with a broadcasted diagonal."""
        # pylint: disable=protected-access
        D = np.array([[1j, 1, -1j, 1], [1, -1, 1j, -1]])
        op = qp.DiagonalQubitUnitary(D, wires=[1, 2])
        with qp.queuing.AnnotatedQueue() as q:
            op._controlled(control=0)
        tape = qp.tape.QuantumScript.from_queue(q)
        mat = qp.matrix(tape, wire_order=[0, 1, 2])
        expected = np.array(
            [np.diag([1, 1, 1, 1, 1j, 1, -1j, 1]), np.diag([1, 1, 1, 1, 1, -1, 1j, -1])]
        )
        assert qp.math.allclose(mat, expected)

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        diag = np.array([1, -1])
        res_static = qp.DiagonalQubitUnitary.compute_matrix(diag)
        res_dynamic = qp.DiagonalQubitUnitary(diag, wires=0).matrix()
        expected = np.array([[1, 0], [0, -1]])
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_matrix_representation_broadcasted(self, tol):
        """Test that the matrix representation is defined correctly for a broadcasted diagonal."""
        diag = np.array([[1, -1], [1j, -1], [-1j, -1]])
        res_static = qp.DiagonalQubitUnitary.compute_matrix(diag)
        res_dynamic = qp.DiagonalQubitUnitary(diag, wires=0).matrix()
        expected = np.array([[[1, 0], [0, -1]], [[1j, 0], [0, -1]], [[-1j, 0], [0, -1]]])
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    @pytest.mark.parametrize("n", (2, -1, 0.12345))
    @pytest.mark.parametrize("diag", ([1.0, -1.0], np.array([1.0, -1.0])))
    def test_pow(self, n, diag):
        """Test pow method returns expected results."""
        op = qp.DiagonalQubitUnitary(diag, wires="b")
        pow_ops = op.pow(n)
        assert len(pow_ops) == 1

        for x_op, x_pow in zip(op.data[0], pow_ops[0].data[0]):
            assert (x_op + 0.0j) ** n == x_pow

    @pytest.mark.parametrize("n", (2, -1, 0.12345))
    @pytest.mark.parametrize(
        "diag", ([[1.0, -1.0]] * 5, np.array([[1.0, -1j], [1j, 1j], [-1j, 1]]))
    )
    def test_pow_broadcasted(self, n, diag):
        """Test pow method returns expected results for broadcasted diagonals."""
        op = qp.DiagonalQubitUnitary(diag, wires="b")
        pow_ops = op.pow(n)
        assert len(pow_ops) == 1

        qp.math.allclose(np.array(op.data[0], dtype=complex) ** n, pow_ops[0].data[0])

    @pytest.mark.parametrize("D", [[1, 2], [[0.2, 1.0, -1.0], [1.0, -1j, 1j]]])
    def test_error_matrix_not_unitary(self, D):
        """Tests that error is raised if diagonal by `compute_matrix` does not lead to a unitary"""
        with pytest.raises(ValueError, match="Operator must be unitary"):
            qp.DiagonalQubitUnitary.compute_matrix(np.array(D))
        with pytest.raises(ValueError, match="Operator must be unitary"):
            qp.DiagonalQubitUnitary(np.array(D), wires=1).matrix()

    @pytest.mark.parametrize("D", [[1, 2], [[0.2, 1.0, -1.0], [1.0, -1j, 1j]]])
    def test_error_eigvals_not_unitary(self, D):
        """Tests that error is raised if diagonal by `compute_matrix` does not lead to a unitary"""
        with pytest.raises(ValueError, match="Operator must be unitary"):
            qp.DiagonalQubitUnitary.compute_eigvals(np.array(D))
        with pytest.raises(ValueError, match="Operator must be unitary"):
            qp.DiagonalQubitUnitary(np.array(D), wires=0).eigvals()

    @pytest.mark.jax
    def test_jax_jit(self):
        """Test that the diagonal matrix unitary operation works
        within a QNode that uses the JAX JIT"""
        import jax

        jnp = jax.numpy

        dev = qp.device("default.qubit", wires=1)

        @jax.jit
        @qp.qnode(dev)
        def circuit(x):
            diag = jnp.exp(1j * x * jnp.array([1, -1]) / 2)
            qp.Hadamard(wires=0)
            qp.DiagonalQubitUnitary(diag, wires=0)
            return qp.expval(qp.PauliX(0))

        x = 0.654
        grad = jax.grad(circuit)(x)
        expected = -jnp.sin(x)
        assert np.allclose(grad, expected)

    @pytest.mark.jax
    def test_jax_jit_broadcasted(self):
        """Test that the diagonal matrix unitary operation works
        within a QNode that uses the JAX JIT and broadcasting"""
        import jax

        jnp = jax.numpy

        dev = qp.device("default.qubit", wires=1)

        @jax.jit
        @qp.qnode(dev)
        def circuit(x):
            diag = jnp.exp(1j * jnp.outer(x, jnp.array([1, -1])) / 2)
            qp.Hadamard(wires=0)
            qp.DiagonalQubitUnitary(diag, wires=0)
            return qp.expval(qp.PauliX(0))

        x = jnp.array([0.654, 0.321])
        jac = jax.jacobian(circuit)(x)
        expected = jnp.diag(-jnp.sin(x))
        assert np.allclose(jac, expected)

    @pytest.mark.tf
    @pytest.mark.slow  # test takes 12 seconds due to tf.function
    def test_tf_function(self):
        """Test that the diagonal matrix unitary operation works
        within a QNode that uses TensorFlow autograph"""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=1)

        @tf.function
        @qp.qnode(dev)
        def circuit(x):
            x = tf.cast(x, tf.complex128)
            diag = tf.math.exp(1j * x * tf.constant([1.0 + 0j, -1.0 + 0j]) / 2)
            qp.Hadamard(wires=0)
            qp.DiagonalQubitUnitary(diag, wires=0)
            return qp.expval(qp.PauliX(0))

        x = tf.Variable(0.452)

        with tf.GradientTape() as tape:
            loss = circuit(x)

        grad = tape.gradient(loss, x)
        expected = -tf.math.sin(x)  # pylint: disable=invalid-unary-operand-type
        assert np.allclose(grad, expected)


labels = [X, X, [1, 1]]
ops = [
    qp.QubitUnitary(X, wires=0),
    qp.ControlledQubitUnitary(X, wires=[0, 1]),
    qp.DiagonalQubitUnitary([1, 1], wires=0),
]


class TestUnitaryLabels:
    """Test the label of matrix operations."""

    @pytest.mark.parametrize("op", ops)
    def test_no_cache(self, op):
        """Test labels work without a provided cache."""
        assert op.label() == "U"

    @pytest.mark.parametrize("op", ops)
    def test_matrices_not_in_cache(self, op):
        """Test provided cache doesn't have a 'matrices' keyword."""
        assert op.label(cache={}) == "U"

    @pytest.mark.parametrize("op", ops)
    def test_cache_matrices_not_list(self, op):
        """Test 'matrices' key pair is not a list."""
        assert op.label(cache={"matrices": 0}) == "U"

    @pytest.mark.parametrize("mat, op", zip(labels, ops))
    def test_empty_cache_list(self, mat, op):
        """Test matrices list is provided, but empty. Operation should have `0` label and matrix
        should be added to cache."""
        cache = {"matrices": []}
        assert op.label(cache=cache) == "U\n(M0)"
        assert qp.math.allclose(cache["matrices"][0], mat)

    @pytest.mark.parametrize("mat, op", zip(labels, ops))
    def test_something_in_cache_list(self, mat, op):
        """If something exists in the matrix list, but parameter is not in the list, then parameter
        added to list and label given number of its position."""
        cache = {"matrices": [Z]}
        assert op.label(cache=cache) == "U\n(M1)"

        assert len(cache["matrices"]) == 2
        assert qp.math.allclose(cache["matrices"][1], mat)

    @pytest.mark.parametrize("mat, op", zip(labels, ops))
    def test_matrix_already_in_cache_list(self, mat, op):
        """If the parameter already exists in the matrix cache, then the label uses that index and the
        matrix cache is unchanged."""
        cache = {"matrices": [Z, mat, S]}
        assert op.label(cache=cache) == "U\n(M1)"

        assert len(cache["matrices"]) == 3


class TestBlockEncode:
    """Test the BlockEncode operation."""

    @pytest.mark.parametrize(
        ("input_matrix", "issparse"),
        [
            (np.array([[1, 0], [0, 1]]), False),
            (X, False),
            (csr_matrix([[1, 0], [0, 1]]), True),
        ],
    )
    def test_property(self, input_matrix, issparse):
        """Test that BlockEncode has the correct properties."""
        op = qp.BlockEncode(input_matrix, wires=range(2))
        assert op.has_matrix != issparse
        assert op.has_sparse_matrix == issparse
        if issparse:
            # Test if there's correct error out
            with pytest.raises(qp.operation.MatrixUndefinedError):
                op.matrix()
        else:
            with pytest.raises(qp.operation.SparseMatrixUndefinedError):
                op.sparse_matrix()

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "input_matrix",
        [
            np.array([[1, 0], [0, 1]]),
            X,
            csr_matrix([[1, 0], [0, 1]]),
        ],
    )
    def test_applicable(self, input_matrix):
        """Integration Test that BlockEncode can be applied onto a state."""
        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            qp.BlockEncode(input_matrix, wires=range(2))
            return qp.state()

        state = circuit()
        assert state.shape == (2**2,)

    @pytest.mark.parametrize(
        ("input_matrix", "wires", "expected_hyperparameters"),
        [
            (1, 1, {"norm": 1, "subspace": (1, 1, 2)}),
            ([1], 1, {"norm": 1, "subspace": (1, 1, 2)}),
            ([[1]], 1, {"norm": 1, "subspace": (1, 1, 2)}),
            ([1, 0], [0, 1], {"norm": 1, "subspace": (1, 2, 4)}),
            (pnp.array(1), [1], {"norm": 1, "subspace": (1, 1, 2)}),
            (pnp.array([1]), 1, {"norm": 1, "subspace": (1, 1, 2)}),
            (pnp.array([[1]]), 1, {"norm": 1, "subspace": (1, 1, 2)}),
            ([[1, 0], [0, 1]], [0, 1], {"norm": 1.0, "subspace": (2, 2, 4)}),
            (pnp.array([[1, 0], [0, 1]]), range(2), {"norm": 1.0, "subspace": (2, 2, 4)}),
            (pnp.identity(3), ["a", "b", "c"], {"norm": 1.0, "subspace": (3, 3, 8)}),
        ],
    )
    def test_accepts_various_types(self, input_matrix, wires, expected_hyperparameters):
        """Test that BlockEncode outputs expected attributes for various input matrix types."""
        op = qp.BlockEncode(input_matrix, wires)
        assert np.allclose(op.parameters, input_matrix)
        assert op.hyperparameters["norm"] == expected_hyperparameters["norm"]
        assert op.hyperparameters["subspace"] == expected_hyperparameters["subspace"]

    @pytest.mark.parametrize(
        ("input_matrix", "wires"),
        [(1, 1), (1, 2), (1, [1]), (1, range(2)), (np.identity(2), ["a", "b"])],
    )
    def test_varied_wires(self, input_matrix, wires):
        """Test that BlockEncode wires are stored correctly for various wire input types."""
        assert qp.BlockEncode(input_matrix, wires).wires == Wires(wires)

    @pytest.mark.parametrize(
        ("input_matrix", "wires", "msg"),
        [
            (
                [[0, 1], [1, 0]],
                1,
                r"Block encoding a \(2 x 2\) matrix requires a Hilbert space of size"
                r" at least \(4 x 4\). Cannot be embedded in a 1 qubit system.",
            ),
        ],
    )
    def test_error_raised_invalid_hilbert_space(self, input_matrix, wires, msg):
        """Test the correct error is raised when inputting an invalid number of wires."""
        with pytest.raises(ValueError, match=msg):
            qp.BlockEncode(input_matrix, wires)

    @pytest.mark.parametrize(
        ("input_matrix", "wires", "output_matrix"),
        [
            (1, 0, [[1, 0], [0, -1]]),
            (0.3, 0, [[0.3, 0.9539392], [0.9539392, -0.3]]),
            (
                0.1,
                range(2),
                [[0.1, 0.99498744, 0, 0], [0.99498744, -0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ),
            (
                [[0.1, 0.2], [0.3, 0.4]],
                range(2),
                [
                    [0.1, 0.2, 0.97283788, -0.05988708],
                    [0.3, 0.4, -0.05988708, 0.86395228],
                    [0.94561648, -0.07621992, -0.1, -0.3],
                    [-0.07621992, 0.89117368, -0.2, -0.4],
                ],
            ),
            (
                [[0.1, 0.2, 0.3], [0.3, 0.4, 0.2], [0.1, 0.2, 0.3]],
                range(3),
                [
                    [
                        [0.1, 0.2, 0.3, 0.91808609, -0.1020198, -0.08191391, 0.0, 0.0],
                        [0.3, 0.4, 0.2, -0.1020198, 0.83017102, -0.1020198, 0.0, 0.0],
                        [0.1, 0.2, 0.3, -0.08191391, -0.1020198, 0.91808609, 0.0, 0.0],
                        [0.93589192, -0.09400608, -0.07258899, -0.1, -0.3, -0.1, 0.0, 0.0],
                        [-0.09400608, 0.85841586, -0.11952016, -0.2, -0.4, -0.2, 0.0, 0.0],
                        [-0.07258899, -0.11952016, 0.87203542, -0.3, -0.2, -0.3, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                ],
            ),
        ],
    )
    def test_correct_output_matrix(self, input_matrix, wires, output_matrix):
        """Test that BlockEncode outputs the correct matrix."""
        assert np.allclose(qp.matrix(qp.BlockEncode(input_matrix, wires)), output_matrix)

    @pytest.mark.parametrize(
        ("input_matrix", "wires"),
        [
            (1, 0),
            (0.3, 0),
            (np.array([[0.1, 0.2], [0.3, 0.4]]), range(2)),
            (np.array([[0.1, 0.2, 0.3]]), range(2)),
            (np.array([[0.1], [0.2], [0.3]]), range(2)),
            (np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), range(3)),
            (np.array([[1, 2], [3, 4]]), range(2)),
        ],
    )
    def test_unitary(self, input_matrix, wires):
        """Test that BlockEncode matrices are unitary."""
        mat = qp.matrix(qp.BlockEncode(input_matrix, wires))
        assert np.allclose(np.eye(len(mat)), mat.dot(mat.T.conj()))

    @pytest.mark.tf
    @pytest.mark.parametrize(
        ("input_matrix", "wires", "output_matrix"),
        [
            (1.0, 0, [[1, 0], [0, -1]]),
            (0.3, 0, [[0.3, 0.9539392], [0.9539392, -0.3]]),
            (
                [[0.1, 0.2], [0.3, 0.4]],
                range(2),
                [
                    [0.1, 0.2, 0.97283788, -0.05988708],
                    [0.3, 0.4, -0.05988708, 0.86395228],
                    [0.94561648, -0.07621992, -0.1, -0.3],
                    [-0.07621992, 0.89117368, -0.2, -0.4],
                ],
            ),
            (
                0.1,
                range(2),
                [[0.1, 0.99498744, 0, 0], [0.99498744, -0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ),
        ],
    )
    def test_blockencode_tf(self, input_matrix, wires, output_matrix):
        """Test that the BlockEncode operator matrix is correct for tf."""
        import tensorflow as tf

        input_matrix = tf.Variable(input_matrix)

        op = qp.BlockEncode(input_matrix, wires)
        assert np.allclose(qp.matrix(op), output_matrix)
        assert qp.math.get_interface(qp.matrix(op)) == "tensorflow"

    @pytest.mark.torch
    @pytest.mark.parametrize(
        ("input_matrix", "wires", "output_matrix"),
        [
            (1, 0, [[1, 0], [0, -1]]),
            (0.3, 0, [[0.3, 0.9539392], [0.9539392, -0.3]]),
            (
                [[0.1, 0.2], [0.3, 0.4]],
                range(2),
                [
                    [0.1, 0.2, 0.97283788, -0.05988708],
                    [0.3, 0.4, -0.05988708, 0.86395228],
                    [0.94561648, -0.07621992, -0.1, -0.3],
                    [-0.07621992, 0.89117368, -0.2, -0.4],
                ],
            ),
            (
                0.1,
                range(2),
                [[0.1, 0.99498744, 0, 0], [0.99498744, -0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ),
        ],
    )
    def test_blockencode_torch(self, input_matrix, wires, output_matrix):
        """Test that the BlockEncode operator matrix is correct for torch."""
        import torch

        input_matrix = torch.tensor(input_matrix)
        op = qp.BlockEncode(input_matrix, wires)
        assert np.allclose(qp.matrix(op), output_matrix)
        assert qp.math.get_interface(qp.matrix(op)) == "torch"

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("input_matrix", "wires", "output_matrix"),
        [
            (1, 0, [[1, 0], [0, -1]]),
            ([0.3], 0, [[0.3, 0.9539392], [0.9539392, -0.3]]),
            (
                [[0.1, 0.2], [0.3, 0.4]],
                range(2),
                [
                    [0.1, 0.2, 0.97283788, -0.05988708],
                    [0.3, 0.4, -0.05988708, 0.86395228],
                    [0.94561648, -0.07621992, -0.1, -0.3],
                    [-0.07621992, 0.89117368, -0.2, -0.4],
                ],
            ),
            (
                0.1,
                range(2),
                [[0.1, 0.99498744, 0, 0], [0.99498744, -0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ),
        ],
    )
    def test_blockencode_jax(self, input_matrix, wires, output_matrix):
        """Test that the BlockEncode operator matrix is correct for jax."""
        import jax
        import jax.numpy as jnp

        input_matrix = jnp.array(input_matrix)
        op = qp.BlockEncode(input_matrix, wires)
        assert np.allclose(qp.matrix(op), output_matrix)
        assert qp.math.get_interface(qp.matrix(op)) == "jax"

        # Test jitting behaviour as well.
        @jax.jit
        def f(A):
            op = qp.BlockEncode(A, wires)
            return qp.matrix(op)

        assert np.allclose(f(input_matrix), output_matrix)

    @pytest.mark.parametrize("method", ["backprop"])
    @pytest.mark.parametrize(
        (
            "wires",
            "input_matrix",
            "expected_result",
        ),  # expected_results calculated manually
        [
            (range(1), pnp.array(0.3), 4 * 0.3),
            (range(2), pnp.diag([0.2, 0.3]), 4 * pnp.diag([0.2, 0])),
        ],
    )
    def test_blockencode_grad(self, method, wires, input_matrix, expected_result):
        """Test that BlockEncode is differentiable."""
        dev = qp.device("default.qubit", wires=wires)

        @qp.qnode(dev, diff_method=method)
        def circuit(input_matrix):
            qp.BlockEncode(input_matrix, wires=wires)
            return qp.expval(qp.PauliZ(wires=0))

        assert np.allclose(qp.grad(circuit)(input_matrix), expected_result)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        (
            "wires",
            "input_matrix",
            "expected_result",
        ),  # expected_results calculated manually
        [
            (range(1), pnp.array(0.3), 4 * 0.3),
            (range(2), pnp.diag([0.2, 0.3]), 4 * pnp.diag([0.2, 0])),
        ],
    )
    def test_blockencode_grad_jax(self, wires, input_matrix, expected_result):
        """Test that block encode is differentiable when using jax."""
        import jax
        import jax.numpy as jnp

        input_matrix = jnp.array(input_matrix)
        expected_result = jnp.array(expected_result)

        dev = qp.device("default.qubit", wires=wires)

        @qp.qnode(dev)
        def circuit(input_matrix):
            qp.BlockEncode(input_matrix, wires=wires)
            return qp.expval(qp.PauliZ(wires=0))

        grad = jax.grad(circuit, argnums=0)(input_matrix)
        assert np.allclose(grad, expected_result)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        ("wires", "input_matrix", "expected_result"),  # expected_results calculated manually
        [
            (range(1), pnp.array(0.3), 4 * 0.3),
            (range(2), pnp.diag([0.2, 0.3]), 4 * pnp.diag([0.2, 0])),
        ],
    )
    def test_blockencode_grad_tf(self, wires, input_matrix, expected_result):
        """Test that block encode is differentiable when using tensorflow."""
        import tensorflow as tf

        input_matrix = tf.Variable(input_matrix)

        dev = qp.device("default.qubit", wires=wires)

        @qp.qnode(dev)
        def circuit(input_matrix):
            qp.BlockEncode(input_matrix, wires=wires)
            return qp.expval(qp.PauliZ(wires=0))

        with tf.GradientTape() as tape:
            result = circuit(input_matrix)

        computed_grad = tape.gradient(result, input_matrix)
        assert np.allclose(computed_grad, expected_result)

    @pytest.mark.parametrize(
        ("input_matrix", "wires"),
        [
            (1, 0),
            (0.3, 0),
            (0.1, range(2)),
            (
                [[0.1, 0.2], [0.3, 0.4]],
                range(2),
            ),
            ([[0.1, 0.2, 0.3], [0.3, 0.4, 0.2], [0.1, 0.2, 0.3]], range(3)),
        ],
    )
    def test_adjoint(self, input_matrix, wires):
        """Test that the adjoint of a BlockEncode operation is correctly computed."""
        mat = qp.matrix(qp.BlockEncode(input_matrix, wires))
        adj = qp.matrix(qp.adjoint(qp.BlockEncode(input_matrix, wires)))
        other_adj = qp.matrix(qp.BlockEncode(input_matrix, wires).adjoint())
        assert np.allclose(np.eye(len(mat)), mat @ adj)
        assert np.allclose(np.eye(len(mat)), mat @ other_adj)

    def test_label(self):
        """Test the label method for BlockEncode op"""
        op = qp.BlockEncode(0.5, wires=[0, 1])
        assert op.label() == "BlockEncode"

    @pytest.mark.parametrize(
        ("input_matrix", "wires", "output_value"),
        [
            (1, [0], 1),
            ([[0.1, 0.2], [0.3, 0.4]], range(2), -0.8),
            (
                0.1,
                range(2),
                1,
            ),
        ],
    )
    def test_blockencode_integration(self, input_matrix, wires, output_value):
        """Test that the BlockEncode gate applied to a circuit produces the correct final state."""
        dev = qp.device("default.qubit", wires=wires)

        @qp.qnode(dev)
        def circuit(input_matrix):
            qp.BlockEncode(input_matrix, wires=wires)
            return qp.expval(qp.PauliZ(wires=0))

        assert circuit(input_matrix) == output_value

    @pytest.mark.parametrize(
        "matrix_data",
        [
            {
                "data": [0.1, 0.2, 0.3] * 4,
                "indices": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
                "indptr": [0, 3, 6, 9, 12],
                "shape": (4, 8),
            },
            {
                # A smaller example
                "data": [1.0, 2.0, 3.0],
                "indices": [0, 1, 2],
                "indptr": [0, 3],
                "shape": (1, 3),
            },
            {
                # A small 2x2 square example
                "data": [1.0, 2.0],
                "indices": [0, 1],
                "indptr": [0, 2, 2],
                "shape": (2, 2),
            },
            {
                # Another 3x3 square example
                "data": [0.5, 1.1, 1.2, 0.3],
                "indices": [0, 2, 1, 2],
                "indptr": [0, 2, 3, 4],
                "shape": (3, 3),
            },
        ],
    )
    @pytest.mark.parametrize("format", ["coo", "csr", "csc", "bsr"])
    def test_sparse_matrix(self, matrix_data, format):
        """Test that the BlockEncode works well with various sparse matrices."""
        data = matrix_data["data"]
        indices = matrix_data["indices"]
        indptr = matrix_data["indptr"]
        shape = matrix_data["shape"]

        num_wires = 5
        sparse_matrix = csr_matrix((data, indices, indptr), shape=shape).asformat(format)
        op = qp.BlockEncode(sparse_matrix, wires=range(num_wires))

        # Test the operator is unitary
        mat = op.sparse_matrix()
        assert np.allclose(np.eye(mat.shape[0]), (mat @ mat.T.conj()).toarray())
        mat_dense = qp.matrix(qp.BlockEncode(sparse_matrix.toarray(), wires=range(num_wires)))
        assert qp.math.allclose(mat, mat_dense)


class TestInterfaceMatricesLabel:
    """Test different interface matrices with qubit."""

    def check_interface(self, mat):
        """Interface independent helper method."""

        op = qp.QubitUnitary(mat, wires=0)

        cache = {"matrices": []}
        assert op.label(cache=cache) == "U\n(M0)"
        assert qp.math.allclose(cache["matrices"][0], mat)

        cache = {"matrices": [0, mat, 0]}
        assert op.label(cache=cache) == "U\n(M1)"
        assert len(cache["matrices"]) == 3

    @pytest.mark.torch
    def test_labelling_torch_tensor(self):
        """Test matrix cache labelling with torch interface."""

        import torch

        mat = torch.tensor([[1, 0], [0, -1]])
        self.check_interface(mat)

    @pytest.mark.tf
    def test_labelling_tf_variable(self):
        """Test matrix cache labelling with tf interface."""

        import tensorflow as tf

        mat = tf.Variable([[1, 0], [0, -1]])

        self.check_interface(mat)

    @pytest.mark.jax
    def test_labelling_jax_variable(self):
        """Test matrix cache labelling with jax interface."""

        import jax.numpy as jnp

        mat = jnp.array([[1, 0], [0, -1]])

        self.check_interface(mat)


control_data = [
    (qp.QubitUnitary(X, wires=0), Wires([])),
    (qp.DiagonalQubitUnitary([1, 1], wires=1), Wires([])),
    (qp.ControlledQubitUnitary(X, wires=[0, 1]), Wires([0])),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test ``control_wires`` attribute for matrix operations."""
    assert op.control_wires == control_wires

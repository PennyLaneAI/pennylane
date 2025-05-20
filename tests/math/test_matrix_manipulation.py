# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for matrix expand functions."""
# pylint: disable=too-few-public-methods,too-many-public-methods
from functools import reduce

import numpy as np
import pytest
from gate_data import CNOT, II, SWAP, I, Toffoli
from scipy.sparse import csr_matrix
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.math import expand_matrix, expand_vector, sqrt_matrix, sqrt_matrix_sparse
from pennylane.math.quantum import _denman_beavers_iterations

# Define a list of dtypes to test
dtypes = ["complex64", "complex128"]

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]

Toffoli_broadcasted = np.tensordot([0.1, -4.2j], Toffoli, axes=0)
CNOT_broadcasted = np.tensordot([1.4], CNOT, axes=0)
I_broadcasted = I[pnp.newaxis]


class TestExpandMatrix:
    """Tests for the expand_matrix helper function."""

    base_matrix_1 = np.arange(1, 5).reshape((2, 2))
    base_matrix_1_broadcasted = np.arange(1, 13).reshape((3, 2, 2))
    base_matrix_2 = np.arange(1, 17).reshape((4, 4))
    base_matrix_2_broadcasted = np.arange(1, 49).reshape((3, 4, 4))

    def test_no_expansion(self):
        """Tests the case where the original matrix is not changed"""
        res = qml.math.expand_matrix(self.base_matrix_2, wires=[0, 2], wire_order=[0, 2])
        assert np.allclose(self.base_matrix_2, res)

    def test_no_wire_order_returns_base_matrix(self):
        """Test the case where the wire_order is None it returns the original matrix"""
        res = qml.math.expand_matrix(self.base_matrix_2, wires=[0, 2])
        assert np.allclose(self.base_matrix_2, res)

    def test_no_expansion_broadcasted(self):
        """Tests the case where the broadcasted original matrix is not changed"""
        res = qml.math.expand_matrix(
            self.base_matrix_2_broadcasted, wires=[0, 2], wire_order=[0, 2]
        )
        assert np.allclose(self.base_matrix_2_broadcasted, res)

    def test_permutation(self):
        """Tests the case where the original matrix is permuted"""
        res = qml.math.expand_matrix(self.base_matrix_2, wires=[0, 2], wire_order=[2, 0])

        expected = np.array([[1, 3, 2, 4], [9, 11, 10, 12], [5, 7, 6, 8], [13, 15, 14, 16]])
        assert np.allclose(expected, res)

    def test_permutation_broadcasted(self):
        """Tests the case where the broadcasted original matrix is permuted"""
        res = qml.math.expand_matrix(
            self.base_matrix_2_broadcasted, wires=[0, 2], wire_order=[2, 0]
        )

        perm = [0, 2, 1, 3]
        expected = self.base_matrix_2_broadcasted[:, perm][:, :, perm]
        assert np.allclose(expected, res)

    def test_expansion(self):
        """Tests the case where the original matrix is expanded"""
        res = qml.math.expand_matrix(self.base_matrix_1, wires=[2], wire_order=[0, 2])
        expected = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]])
        assert np.allclose(expected, res)

        res = qml.math.expand_matrix(self.base_matrix_1, wires=[2], wire_order=[2, 0])
        expected = np.array([[1, 0, 2, 0], [0, 1, 0, 2], [3, 0, 4, 0], [0, 3, 0, 4]])
        assert np.allclose(expected, res)

    def test_expansion_broadcasted(self):
        """Tests the case where the broadcasted original matrix is expanded"""
        res = qml.math.expand_matrix(self.base_matrix_1_broadcasted, wires=[2], wire_order=[0, 2])
        expected = np.array(
            [
                [
                    [1, 2, 0, 0],
                    [3, 4, 0, 0],
                    [0, 0, 1, 2],
                    [0, 0, 3, 4],
                ],
                [
                    [5, 6, 0, 0],
                    [7, 8, 0, 0],
                    [0, 0, 5, 6],
                    [0, 0, 7, 8],
                ],
                [
                    [9, 10, 0, 0],
                    [11, 12, 0, 0],
                    [0, 0, 9, 10],
                    [0, 0, 11, 12],
                ],
            ]
        )
        assert np.allclose(expected, res)

        res = qml.math.expand_matrix(self.base_matrix_1_broadcasted, wires=[2], wire_order=[2, 0])
        expected = np.array(
            [
                [
                    [1, 0, 2, 0],
                    [0, 1, 0, 2],
                    [3, 0, 4, 0],
                    [0, 3, 0, 4],
                ],
                [
                    [5, 0, 6, 0],
                    [0, 5, 0, 6],
                    [7, 0, 8, 0],
                    [0, 7, 0, 8],
                ],
                [
                    [9, 0, 10, 0],
                    [0, 9, 0, 10],
                    [11, 0, 12, 0],
                    [0, 11, 0, 12],
                ],
            ]
        )
        assert np.allclose(expected, res)

    @staticmethod
    def func_for_autodiff(mat):
        """Expand a single-qubit matrix to two qubits where the
        matrix acts on the latter qubit."""
        return qml.math.expand_matrix(mat, wires=[2], wire_order=[0, 2])

    # the entries should be mapped by func_for_autodiff via
    # source -> destinations
    # (0, 0) -> (0, 0), (2, 2)
    # (0, 1) -> (0, 1), (2, 3)
    # (1, 0) -> (1, 0), (3, 2)
    # (1, 1) -> (1, 1), (3, 3)
    # so that the expected Jacobian is 0 everywhere except for the entries
    # (dest, source) from the above list, where it is 1.
    expected_autodiff_nobatch = np.zeros((4, 4, 2, 2), dtype=float)
    indices = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 1, 0, 1),
        (2, 3, 0, 1),
        (1, 0, 1, 0),
        (3, 2, 1, 0),
        (1, 1, 1, 1),
        (3, 3, 1, 1),
    ]
    for ind in indices:
        expected_autodiff_nobatch[ind] = 1.0

    # When using broadcasting, the expected Jacobian
    # of func_for_autodiff is diagonal in the dimensions 0 and 3
    expected_autodiff_broadcasted = np.zeros((3, 4, 4, 3, 2, 2), dtype=float)
    for ind in indices:
        expected_autodiff_broadcasted[:, ind[0], ind[1], :, ind[2], ind[3]] = np.eye(3)

    expected_autodiff = [expected_autodiff_nobatch, expected_autodiff_broadcasted]

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "i, base_matrix",
        [
            (0, [[0.2, 1.1], [-1.3, 1.9]]),
            (1, [[[0.2, 0.5], [1.2, 1.1]], [[-0.3, -0.2], [-1.3, 1.9]], [[0.2, 0.1], [0.2, 0.7]]]),
        ],
    )
    def test_autograd(self, i, base_matrix, tol):
        """Tests differentiation in autograd by computing the Jacobian of
        the expanded matrix with respect to the canonical matrix."""

        base_matrix = pnp.array(base_matrix, requires_grad=True)
        jac_fn = qml.jacobian(self.func_for_autodiff)
        jac = jac_fn(base_matrix)

        assert np.allclose(jac, self.expected_autodiff[i], atol=tol)

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "i, base_matrix",
        [
            (0, [[0.2, 1.1], [-1.3, 1.9]]),
            (1, [[[0.2, 0.5], [1.2, 1.1]], [[-0.3, -0.2], [-1.3, 1.9]], [[0.2, 0.1], [0.2, 0.7]]]),
        ],
    )
    def test_torch(self, i, base_matrix, tol):
        """Tests differentiation in torch by computing the Jacobian of
        the expanded matrix with respect to the canonical matrix."""
        import torch

        base_matrix = torch.tensor(base_matrix, requires_grad=True)
        jac = torch.autograd.functional.jacobian(self.func_for_autodiff, base_matrix)

        assert np.allclose(jac, self.expected_autodiff[i], atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "i, base_matrix",
        [
            (0, [[0.2, 1.1], [-1.3, 1.9]]),
            (1, [[[0.2, 0.5], [1.2, 1.1]], [[-0.3, -0.2], [-1.3, 1.9]], [[0.2, 0.1], [0.2, 0.7]]]),
        ],
    )
    def test_jax(self, i, base_matrix, tol):
        """Tests differentiation in jax by computing the Jacobian of
        the expanded matrix with respect to the canonical matrix."""
        import jax

        base_matrix = jax.numpy.array(base_matrix)
        jac_fn = jax.jacobian(self.func_for_autodiff)
        jac = jac_fn(base_matrix)

        assert np.allclose(jac, self.expected_autodiff[i], atol=tol)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "i, base_matrix",
        [
            (0, [[0.2, 1.1], [-1.3, 1.9]]),
            (1, [[[0.2, 0.5], [1.2, 1.1]], [[-0.3, -0.2], [-1.3, 1.9]], [[0.2, 0.1], [0.2, 0.7]]]),
        ],
    )
    def test_tf(self, i, base_matrix, tol):
        """Tests differentiation in TensorFlow by computing the Jacobian of
        the expanded matrix with respect to the canonical matrix."""
        import tensorflow as tf

        base_matrix = tf.Variable(base_matrix)
        with tf.GradientTape() as tape:
            res = self.func_for_autodiff(base_matrix)

        jac = tape.jacobian(res, base_matrix)
        assert np.allclose(jac, self.expected_autodiff[i], atol=tol)

    def test_expand_one(self, tol):
        """Test that a 1 qubit gate correctly expands to 3 qubits."""
        U = np.array(
            [
                [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
                [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
            ]
        )
        # test applied to wire 0
        res = qml.math.expand_matrix(U, [0], [0, 4, 9])
        expected = np.kron(np.kron(U, I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 4
        res = qml.math.expand_matrix(U, [4], [0, 4, 9])
        expected = np.kron(np.kron(I, U), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 9
        res = qml.math.expand_matrix(U, [9], [0, 4, 9])
        expected = np.kron(np.kron(I, I), U)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_one_broadcasted(self, tol):
        """Test that a broadcasted 1 qubit gate correctly expands to 3 qubits."""
        U = np.array(
            [
                [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
                [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
            ]
        )
        # outer product with batch vector
        U = np.tensordot([0.14, -0.23, 1.3j], U, axes=0)
        # test applied to wire 0
        res = qml.math.expand_matrix(U, [0], [0, 4, 9])
        expected = np.kron(np.kron(U, I_broadcasted), I_broadcasted)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 4
        res = qml.math.expand_matrix(U, [4], [0, 4, 9])
        expected = np.kron(np.kron(I_broadcasted, U), I_broadcasted)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 9
        res = qml.math.expand_matrix(U, [9], [0, 4, 9])
        expected = np.kron(np.kron(I_broadcasted, I_broadcasted), U)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_two_consecutive_wires(self, tol):
        """Test that a 2 qubit gate on consecutive wires correctly
        expands to 4 qubits."""
        U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)

        # test applied to wire 0+1
        res = qml.math.expand_matrix(U2, [0, 1], [0, 1, 2, 3])
        expected = np.kron(np.kron(U2, I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 1+2
        res = qml.math.expand_matrix(U2, [1, 2], [0, 1, 2, 3])
        expected = np.kron(np.kron(I, U2), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 2+3
        res = qml.math.expand_matrix(U2, [2, 3], [0, 1, 2, 3])
        expected = np.kron(np.kron(I, I), U2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_two_consecutive_wires_broadcasted(self, tol):
        """Test that a broadcasted 2 qubit gate on consecutive wires correctly
        expands to 4 qubits."""
        U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
        U2 = np.tensordot([2.31, 1.53, 0.7 - 1.9j], U2, axes=0)

        # test applied to wire 0+1
        res = qml.math.expand_matrix(U2, [0, 1], [0, 1, 2, 3])
        expected = np.kron(np.kron(U2, I_broadcasted), I_broadcasted)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 1+2
        res = qml.math.expand_matrix(U2, [1, 2], [0, 1, 2, 3])
        expected = np.kron(np.kron(I_broadcasted, U2), I_broadcasted)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 2+3
        res = qml.math.expand_matrix(U2, [2, 3], [0, 1, 2, 3])
        expected = np.kron(np.kron(I_broadcasted, I_broadcasted), U2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_two_reversed_wires(self, tol):
        """Test that a 2 qubit gate on reversed consecutive wires correctly
        expands to 4 qubits."""
        # CNOT with target on wire 1
        res = qml.math.expand_matrix(CNOT, [1, 0], [0, 1, 2, 3])
        rows = np.array([0, 2, 1, 3])
        expected = np.kron(np.kron(CNOT[:, rows][rows], I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_two_reversed_wires_broadcasted(self, tol):
        """Test that a broadcasted 2 qubit gate on reversed consecutive wires correctly
        expands to 4 qubits."""
        # CNOT with target on wire 1 and a batch dimension of size 1
        res = qml.math.expand_matrix(CNOT_broadcasted, [1, 0], [0, 1, 2, 3])
        rows = [0, 2, 1, 3]
        expected = np.kron(
            np.kron(CNOT_broadcasted[:, :, rows][:, rows], I_broadcasted), I_broadcasted
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_consecutive_wires(self, tol):
        """Test that a 3 qubit gate on consecutive
        wires correctly expands to 4 qubits."""
        # test applied to wire 0,1,2
        res = qml.math.expand_matrix(Toffoli, [0, 1, 2], [0, 1, 2, 3])
        expected = np.kron(Toffoli, I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 1,2,3
        res = qml.math.expand_matrix(Toffoli, [1, 2, 3], [0, 1, 2, 3])
        expected = np.kron(I, Toffoli)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_consecutive_wires_broadcasted(self, tol):
        """Test that a broadcasted 3 qubit gate on consecutive
        wires correctly expands to 4 qubits."""
        # test applied to wire 0,1,2
        res = qml.math.expand_matrix(Toffoli_broadcasted, [0, 1, 2], [0, 1, 2, 3])
        expected = np.kron(Toffoli_broadcasted, I_broadcasted)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 1,2,3
        res = qml.math.expand_matrix(Toffoli_broadcasted, [1, 2, 3], [0, 1, 2, 3])
        expected = np.kron(I_broadcasted, Toffoli_broadcasted)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_nonconsecutive_ascending_wires(self, tol):
        """Test that a 3 qubit gate on non-consecutive but ascending
        wires correctly expands to 4 qubits."""
        # test applied to wire 0,2,3
        res = qml.math.expand_matrix(Toffoli, [0, 2, 3], [0, 1, 2, 3])
        expected = np.kron(SWAP, II) @ np.kron(I, Toffoli) @ np.kron(SWAP, II)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 0,1,3
        res = qml.math.expand_matrix(Toffoli, [0, 1, 3], [0, 1, 2, 3])
        expected = np.kron(II, SWAP) @ np.kron(Toffoli, I) @ np.kron(II, SWAP)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_nonconsecutive_ascending_wires_broadcasted(self, tol):
        """Test that a broadcasted 3 qubit gate on non-consecutive but ascending
        wires correctly expands to 4 qubits."""
        # test applied to wire 0,2,3
        res = qml.math.expand_matrix(Toffoli_broadcasted[:1], [0, 2, 3], [0, 1, 2, 3])
        expected = np.tensordot(
            np.tensordot(
                np.kron(SWAP, II),
                np.kron(I_broadcasted, Toffoli_broadcasted[:1]),
                axes=[[1], [1]],
            ),
            np.kron(SWAP, II),
            axes=[[2], [0]],
        )
        expected = np.moveaxis(expected, 0, -2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 0,1,3
        res = qml.math.expand_matrix(Toffoli_broadcasted, [0, 1, 3], [0, 1, 2, 3])
        expected = np.tensordot(
            np.tensordot(
                np.kron(II, SWAP),
                np.kron(Toffoli_broadcasted, I_broadcasted),
                axes=[[1], [1]],
            ),
            np.kron(II, SWAP),
            axes=[[2], [0]],
        )
        expected = np.moveaxis(expected, 0, -2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_nonconsecutive_nonascending_wires(self, tol):
        """Test that a 3 qubit gate on non-consecutive non-ascending
        wires correctly expands to 4 qubits"""
        # test applied to wire 3, 1, 2
        res = qml.math.expand_matrix(Toffoli, [3, 1, 2], [0, 1, 2, 3])
        # change the control qubit on the Toffoli gate
        rows = [0, 4, 1, 5, 2, 6, 3, 7]
        Toffoli_perm = Toffoli[:, rows][rows]
        expected = np.kron(I, Toffoli_perm)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 3, 0, 2
        res = qml.math.expand_matrix(Toffoli, [3, 0, 2], [0, 1, 2, 3])
        # change the control qubit on the Toffoli gate
        expected = np.kron(SWAP, II) @ np.kron(I, Toffoli_perm) @ np.kron(SWAP, II)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_nonconsecutive_nonascending_wires_broadcasted(self, tol):
        """Test that a broadcasted 3 qubit gate on non-consecutive non-ascending
        wires correctly expands to 4 qubits"""
        # test applied to wire 3, 1, 2
        res = qml.math.expand_matrix(Toffoli_broadcasted, [3, 1, 2], [0, 1, 2, 3])
        # change the control qubit on the Toffoli gate
        rows = [0, 4, 1, 5, 2, 6, 3, 7]
        Toffoli_broadcasted_perm = Toffoli_broadcasted[:, :, rows][:, rows]
        expected = np.kron(I_broadcasted, Toffoli_broadcasted_perm)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 3, 0, 2
        res = qml.math.expand_matrix(Toffoli_broadcasted, [3, 0, 2], [0, 1, 2, 3])
        # change the control qubit on the Toffoli gate
        expected = np.tensordot(
            np.tensordot(
                np.kron(SWAP, II),
                np.kron(I_broadcasted, Toffoli_broadcasted_perm),
                axes=[[1], [1]],
            ),
            np.kron(SWAP, II),
            axes=[[2], [0]],
        )
        expected = np.moveaxis(expected, 0, -2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_matrix_usage_in_operator_class(self, tol):
        """Tests that the method is used correctly by defining a dummy operator and
        checking the permutation/expansion."""

        perm = [0, 2, 1, 3]
        permuted_matrix = self.base_matrix_2[perm][:, perm]

        expanded_matrix = np.array(
            [
                [1, 2, 0, 0, 3, 4, 0, 0],
                [5, 6, 0, 0, 7, 8, 0, 0],
                [0, 0, 1, 2, 0, 0, 3, 4],
                [0, 0, 5, 6, 0, 0, 7, 8],
                [9, 10, 0, 0, 11, 12, 0, 0],
                [13, 14, 0, 0, 15, 16, 0, 0],
                [0, 0, 9, 10, 0, 0, 11, 12],
                [0, 0, 13, 14, 0, 0, 15, 16],
            ]
        )

        class DummyOp(qml.operation.Operator):
            """Dummy operator for testing the expand_matrix method."""

            num_wires = 2

            @staticmethod
            def compute_matrix():
                return self.base_matrix_2

        op = DummyOp(wires=[0, 2])
        assert np.allclose(op.matrix(), self.base_matrix_2, atol=tol)
        assert np.allclose(op.matrix(wire_order=[2, 0]), permuted_matrix, atol=tol)
        assert np.allclose(op.matrix(wire_order=[0, 1, 2]), expanded_matrix, atol=tol)

    def test_expand_matrix_usage_in_operator_class_broadcasted(self, tol):
        """Tests that the method is used correctly with a broadcasted matrix by defining
        a dummy operator and checking the permutation/expansion."""

        perm = [0, 2, 1, 3]
        permuted_matrix = self.base_matrix_2_broadcasted[:, perm][:, :, perm]

        expanded_matrix = np.tensordot(
            np.tensordot(
                np.kron(SWAP, I),
                np.kron(I_broadcasted, self.base_matrix_2_broadcasted),
                axes=[[1], [1]],
            ),
            np.kron(SWAP, I),
            axes=[[2], [0]],
        )
        expanded_matrix = np.moveaxis(expanded_matrix, 0, -2)

        class DummyOp(qml.operation.Operator):
            """Dummy operator for testing the expand_matrix method."""

            num_wires = 2

            # pylint: disable=arguments-differ
            @staticmethod
            def compute_matrix():
                """Compute the matrix of the DummyOp."""
                return self.base_matrix_2_broadcasted

        op = DummyOp(wires=[0, 2])
        assert np.allclose(op.matrix(), self.base_matrix_2_broadcasted, atol=tol)
        assert np.allclose(op.matrix(wire_order=[2, 0]), permuted_matrix, atol=tol)
        assert np.allclose(op.matrix(wire_order=[0, 1, 2]), expanded_matrix, atol=tol)


class TestExpandMatrixQutrit:

    def test_add_wire_at_end(self):
        """Test that expand_matrix works on qutrit matrices when an additional wire is added at the end."""

        mat = np.reshape(np.arange(9), (3, 3))

        new_mat = expand_matrix(mat, 0, (0, 1))
        assert qml.math.allclose(new_mat, np.kron(mat, np.eye(3)))

    def test_add_wire_at_start(self):
        """Test that expand_matrix works on qutrit matrices when an additional wire is added at the start."""

        mat = np.reshape(np.arange(9), (3, 3))
        new_mat = expand_matrix(mat, 0, (1, 0))
        assert qml.math.allclose(new_mat, np.kron(np.eye(3), mat))

    def test_wire_permutation(self):
        """Test that wires can be permuted."""
        m1 = np.reshape(np.arange(81), (9, 9))
        m2 = expand_matrix(m1, (0, 1), (1, 0))

        # states across row are 00, 01, 02, 10, 11, 12, 20, 21, 22
        # extract out right qubit state with mod
        m1_wire_zero = m1 % 3
        m2_wire_zero = m2 % 3

        # extract out left qubit state  with floor then mod
        m1_wire_one = np.floor(m1 / 3) % 3
        m2_wire_one = np.floor(m2 / 3) % 3

        assert qml.math.allclose(m1_wire_zero, m2_wire_one)
        assert qml.math.allclose(m1_wire_one, m2_wire_zero)

        # check columns also switched
        # now matrix numbers indicate row number
        m1p = np.floor(m1 / 9)
        m2p = np.floor(m2 / 9)

        # states across column are 00, 01, 02, 10, 11, 12, 20, 21, 22
        # extract out right qubit state with mod
        m1_wire_zerop = m1p % 3
        m2_wire_zerop = m2p % 3

        # extract out left qubit state  with floor then mod
        m1_wire_onep = np.floor(m1p / 3) % 3
        m2_wire_onep = np.floor(m2p / 3) % 3

        assert qml.math.allclose(m1_wire_zerop, m2_wire_onep)
        assert qml.math.allclose(m1_wire_onep, m2_wire_zerop)

    def test_adding_wire_in_middle(self):
        """Test that expand_matrix can add an identity wire in the middle of a two qutrit matrix."""

        m1 = np.reshape(np.arange(9), (3, 3))
        m2 = np.reshape(np.arange(9, 18), (3, 3))
        m3 = np.kron(m1, m2)

        m3_added_wire = expand_matrix(m3, (0, 1), (1, 2, 0))
        m3_kron = np.kron(np.kron(m2, np.eye(3)), m1)

        assert qml.math.allclose(m3_added_wire, m3_kron)


class TestExpandMatrixSparse:
    """Tests for the _sparse_expand_matrix function."""

    base_matrix_1 = csr_matrix(np.arange(1, 5).reshape((2, 2)))
    base_matrix_2 = csr_matrix(np.arange(1, 17).reshape((4, 4)))

    def test_wires_pl_wires(self):
        """Tests the case wires is wires.Wires object"""
        mat = csr_matrix([[0, 1], [1, 0]])
        res = qml.math.expand_matrix(mat, wires=qml.wires.Wires([0]), wire_order=[0, 1])
        res.sort_indices()
        expected = csr_matrix(np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_wires_tuple(self):
        """Tests the case wires is a tuple"""
        mat = csr_matrix([[0, 1], [1, 0]])
        res = qml.math.expand_matrix(mat, wires=(0,), wire_order=[0, 1])
        res.sort_indices()
        expected = csr_matrix(np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_no_expansion(self):
        """Tests the case where the original matrix is not changed"""
        res = qml.math.expand_matrix(self.base_matrix_2, wires=[0, 2], wire_order=[0, 2])
        assert isinstance(res, type(self.base_matrix_2))
        assert all(res.data == self.base_matrix_2.data)
        assert all(res.indices == self.base_matrix_2.indices)

    def test_permutation(self):
        """Tests the case where the original matrix is permuted"""
        res = qml.math.expand_matrix(self.base_matrix_2, wires=[0, 2], wire_order=[2, 0])
        res.sort_indices()
        expected = csr_matrix(
            np.array([[1, 3, 2, 4], [9, 11, 10, 12], [5, 7, 6, 8], [13, 15, 14, 16]])
        )
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_expansion(self):
        """Tests the case where the original matrix is expanded"""
        res = qml.math.expand_matrix(self.base_matrix_1, wires=[2], wire_order=[0, 2])
        res.sort_indices()
        expected = csr_matrix(np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]]))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

        res = qml.math.expand_matrix(self.base_matrix_1, wires=[2], wire_order=[2, 0])
        res.sort_indices()
        expected = csr_matrix(np.array([[1, 0, 2, 0], [0, 1, 0, 2], [3, 0, 4, 0], [0, 3, 0, 4]]))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_expand_one(self):
        """Test that a 1 qubit gate correctly expands to 3 qubits."""
        U = np.array(
            [
                [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
                [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
            ]
        )
        U_sparse = csr_matrix(U)

        # test applied to wire 0
        res = qml.math.expand_matrix(U_sparse, [0], [0, 4, 9])
        res.sort_indices()
        expected = csr_matrix(np.kron(np.kron(U, I), I))

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

        # test applied to wire 4
        res = qml.math.expand_matrix(U_sparse, [4], [0, 4, 9])
        res.sort_indices()
        expected = csr_matrix(np.kron(np.kron(I, U), I))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

        # test applied to wire 9
        res = qml.math.expand_matrix(U_sparse, [9], [0, 4, 9])
        expected = csr_matrix(np.kron(np.kron(I, I), U))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_expand_two_consecutive_wires(self):
        """Test that a 2 qubit gate on consecutive wires correctly
        expands to 4 qubits."""
        U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
        U2_sparse = csr_matrix(U2)

        # test applied to wire 0+1
        res = qml.math.expand_matrix(U2_sparse, [0, 1], [0, 1, 2, 3])
        res.sort_indices()
        expected = csr_matrix(np.kron(np.kron(U2, I), I))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

        # test applied to wire 1+2
        res = qml.math.expand_matrix(U2_sparse, [1, 2], [0, 1, 2, 3])
        res.sort_indices()
        expected = csr_matrix(np.kron(np.kron(I, U2), I))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

        # test applied to wire 2+3
        res = qml.math.expand_matrix(U2_sparse, [2, 3], [0, 1, 2, 3])
        res.sort_indices()
        expected = csr_matrix(np.kron(np.kron(I, I), U2))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_expand_two_reversed_wires(self):
        """Test that a 2 qubit gate on reversed consecutive wires correctly
        expands to 4 qubits."""
        # CNOT with target on wire 1
        res = qml.math.expand_matrix(csr_matrix(CNOT), [1, 0], [0, 1, 2, 3])
        res.sort_indices()
        rows = np.array([0, 2, 1, 3])
        expected = csr_matrix(np.kron(np.kron(CNOT[:, rows][rows], I), I))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_expand_three_consecutive_wires(self):
        """Test that a 3 qubit gate on consecutive
        wires correctly expands to 4 qubits."""
        # test applied to wire 0,1,2
        res = qml.math.expand_matrix(csr_matrix(Toffoli), [0, 1, 2], [0, 1, 2, 3])
        res.sort_indices()
        expected = csr_matrix(np.kron(Toffoli, I))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

        # test applied to wire 1,2,3
        res = qml.math.expand_matrix(csr_matrix(Toffoli), [1, 2, 3], [0, 1, 2, 3])
        res.sort_indices()
        expected = csr_matrix(np.kron(I, Toffoli))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_expand_three_nonconsecutive_ascending_wires(self):
        """Test that a 3 qubit gate on non-consecutive but ascending
        wires correctly expands to 4 qubits."""
        # test applied to wire 0,2,3
        res = qml.math.expand_matrix(csr_matrix(Toffoli), [0, 2, 3], [0, 1, 2, 3])
        res.sort_indices()
        expected = csr_matrix(np.kron(SWAP, II) @ np.kron(I, Toffoli) @ np.kron(SWAP, II))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

        # test applied to wire 0,1,3
        res = qml.math.expand_matrix(csr_matrix(Toffoli), [0, 1, 3], [0, 1, 2, 3])
        res.sort_indices()
        expected = csr_matrix(np.kron(II, SWAP) @ np.kron(Toffoli, I) @ np.kron(II, SWAP))
        expected.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_expand_three_nonconsecutive_nonascending_wires(self):
        """Test that a 3 qubit gate on non-consecutive non-ascending
        wires correctly expands to 4 qubits"""
        # test applied to wire 3, 1, 2
        res = qml.math.expand_matrix(csr_matrix(Toffoli), [3, 1, 2], [0, 1, 2, 3])
        # change the control qubit on the Toffoli gate
        rows = [0, 4, 1, 5, 2, 6, 3, 7]
        Toffoli_perm = Toffoli[:, rows][rows]
        expected = csr_matrix(np.kron(I, Toffoli_perm))
        expected.sort_indices()
        res.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

        # test applied to wire 3, 0, 2
        res = qml.math.expand_matrix(csr_matrix(Toffoli), [3, 0, 2], [0, 1, 2, 3])
        # change the control qubit on the Toffoli gate
        expected = csr_matrix(np.kron(SWAP, II) @ np.kron(I, Toffoli_perm) @ np.kron(SWAP, II))
        expected.sort_indices()
        res.sort_indices()

        assert isinstance(res, type(expected))
        assert all(res.data == expected.data)
        assert all(res.indices == expected.indices)

    def test_sparse_swap_mat(self):
        """Test the swap matrix generated is as expected."""
        # pylint: disable=protected-access
        n = 4
        for i in range(n):
            for j in range(n):
                if i != j:
                    expected_mat = qml.SWAP(wires=[i, j]).matrix()
                    expected_mat = qml.math.expand_matrix(expected_mat, [i, j], wire_order=range(n))
                    computed_mat = qml.math.matrix_manipulation._sparse_swap_mat(i, j, n).toarray()
                    assert np.allclose(expected_mat, computed_mat)

    def test_sparse_swap_mat_same_index(self):
        """Test that if the indices are the same then the identity is returned."""
        # pylint: disable=protected-access
        computed_mat = qml.math.matrix_manipulation._sparse_swap_mat(2, 2, 3).toarray()
        expected_mat = np.eye(8)
        assert np.allclose(expected_mat, computed_mat)


class TestReduceMatrices:
    """Tests for the reduce_matrices function."""

    op_list = [
        qml.PauliX(0),
        qml.RX(1, 0),
        qml.CNOT([3, 4]),
        qml.PauliZ(0),
        qml.RX(2, 7),
        qml.Toffoli([4, 1, 7]),
    ]

    def test_sum_matrices(self):
        """Test the reduce_matrices function with the add method."""
        mats_and_wires_gen = ((op.matrix(), op.wires) for op in self.op_list)
        reduced_mat, final_wires = qml.math.reduce_matrices(mats_and_wires_gen, qml.math.add)

        expected_wires = reduce(lambda x, y: x + y, [op.wires for op in self.op_list])
        expected_matrix = reduce(
            qml.math.add, (op.matrix(wire_order=expected_wires) for op in self.op_list)
        )

        assert final_wires == expected_wires
        assert qml.math.allclose(reduced_mat, expected_matrix)
        assert reduced_mat.shape == (2**5, 2**5)

    def test_prod_matrices(self):
        """Test the reduce_matrices function with the dot method."""
        mats_and_wires_gen = ((op.matrix(), op.wires) for op in self.op_list)
        reduced_mat, final_wires = qml.math.reduce_matrices(mats_and_wires_gen, qml.math.dot)

        expected_wires = reduce(lambda x, y: x + y, [op.wires for op in self.op_list])
        expected_matrix = reduce(
            qml.math.dot, (op.matrix(wire_order=expected_wires) for op in self.op_list)
        )

        assert final_wires == expected_wires
        assert qml.math.allclose(reduced_mat, expected_matrix)
        assert reduced_mat.shape == (2**5, 2**5)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestPartialTrace:
    """Unit tests for the partial_trace function."""

    @pytest.mark.parametrize("c_dtype", dtypes)
    def test_single_density_matrix(self, ml_framework, c_dtype):
        """Test partial trace on a single density matrix."""
        # Define a 2-qubit density matrix
        rho = qml.math.asarray(
            np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]), like=ml_framework
        )

        # Expected result after tracing out the second qubit
        expected = qml.math.asarray(np.array([[[1, 0], [0, 0]]], dtype=c_dtype), like=ml_framework)

        # Perform the partial trace
        result = qml.math.quantum.partial_trace(rho, [0], c_dtype=c_dtype)
        assert qml.math.allclose(result, expected)

    @pytest.mark.parametrize("c_dtype", dtypes)
    def test_batched_density_matrices(self, ml_framework, c_dtype):
        """Test partial trace on a batch of density matrices."""
        # Define a batch of 2-qubit density matrices
        rho = qml.math.asarray(
            np.array(
                [
                    [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ]
            ),
            like=ml_framework,
        )

        # rho = qml.math.asarrays(rho)
        # Expected result after tracing out the first qubit for each matrix
        expected = qml.math.asarray(
            np.array([[[1, 0], [0, 0]], [[1, 0], [0, 0]]], dtype=c_dtype), like=ml_framework
        )

        # Perform the partial trace
        result = qml.math.quantum.partial_trace(rho, [1], c_dtype=c_dtype)
        assert qml.math.allclose(result, expected)

    @pytest.mark.parametrize("c_dtype", dtypes)
    def test_partial_trace_over_no_wires(self, ml_framework, c_dtype):
        """Test that tracing over no wires returns the original matrix."""
        # Define a 2-qubit density matrix
        rho = qml.math.asarray(
            np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=c_dtype),
            like=ml_framework,
        )

        # Perform the partial trace over no wires
        result = qml.math.quantum.partial_trace(rho, [], c_dtype=c_dtype)
        assert qml.math.allclose(result, rho)

    @pytest.mark.parametrize("c_dtype", dtypes)
    def test_partial_trace_over_all_wires(self, ml_framework, c_dtype):
        """Test that tracing over all wires returns the trace of the matrix."""
        # Define a 2-qubit density matrix
        rho = qml.math.asarray(
            np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]), like=ml_framework
        )
        # Expected result after tracing out all qubits
        expected = qml.math.asarray(np.array([1], dtype=c_dtype), like=ml_framework)

        # Perform the partial trace over all wires
        result = qml.math.quantum.partial_trace(rho, [0, 1], c_dtype=c_dtype)
        assert qml.math.allclose(result, expected)

    @pytest.mark.parametrize("c_dtype", dtypes)
    def test_invalid_wire_selection(self, ml_framework, c_dtype):
        """Test that an error is raised for an invalid wire selection."""

        # Define a 2-qubit density matrix
        rho = qml.math.asarray(
            np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]), like=ml_framework
        )

        # Attempt to trace over an invalid wire
        with pytest.raises(Exception) as e:
            import tensorflow as tf  # pylint: disable=Import outside toplevel (tensorflow) (import-outside-toplevel)

            qml.math.quantum.partial_trace(rho, [2], c_dtype=c_dtype)
            assert e.type in (
                ValueError,
                IndexError,
                tf.python.framework.errors_impl.InvalidArgumentError,
            )

    @pytest.mark.parametrize("c_dtype", dtypes)
    def test_partial_trace_single_matrix(self, ml_framework, c_dtype):
        """Test that partial_trace works on a single matrix."""
        # Define a 2-qubit density matrix
        rho = qml.math.asarray(
            np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), like=ml_framework
        )

        result = qml.math.quantum.partial_trace(rho, [0], c_dtype=c_dtype)
        expected = qml.math.asarray(np.array([[1, 0], [0, 0]], dtype=c_dtype), like=ml_framework)

        assert qml.math.allclose(result, expected)


class TestExpandVector:
    """Tests vector expansion to more wires"""

    VECTOR1 = np.array([1, -1])
    ONES = np.array([1, 1])

    @pytest.mark.parametrize(
        "original_wires,expanded_wires,expected",
        [
            ([0], 3, np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([1], 3, np.kron(np.kron(ONES, VECTOR1), ONES)),
            ([2], 3, np.kron(np.kron(ONES, ONES), VECTOR1)),
            ([0], [0, 4, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([4], [0, 4, 7], np.kron(np.kron(ONES, VECTOR1), ONES)),
            ([7], [0, 4, 7], np.kron(np.kron(ONES, ONES), VECTOR1)),
            ([0], [0, 4, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([4], [4, 0, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([7], [7, 4, 0], np.kron(np.kron(VECTOR1, ONES), ONES)),
        ],
    )
    def test_expand_vector_single_wire(self, original_wires, expanded_wires, expected, tol):
        """Test that expand_vector works with a single-wire vector."""

        res = expand_vector(TestExpandVector.VECTOR1, original_wires, expanded_wires)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    VECTOR2 = np.array([1, 2, 3, 4])
    ONES = np.array([1, 1])

    @pytest.mark.parametrize(
        "original_wires,expanded_wires,expected",
        [
            ([0, 1], 3, np.kron(VECTOR2, ONES)),
            ([1, 2], 3, np.kron(ONES, VECTOR2)),
            ([0, 2], 3, np.array([1, 2, 1, 2, 3, 4, 3, 4])),
            ([0, 5], [0, 5, 9], np.kron(VECTOR2, ONES)),
            ([5, 9], [0, 5, 9], np.kron(ONES, VECTOR2)),
            ([0, 9], [0, 5, 9], np.array([1, 2, 1, 2, 3, 4, 3, 4])),
            ([9, 0], [0, 5, 9], np.array([1, 3, 1, 3, 2, 4, 2, 4])),
            ([0, 1], [0, 1], VECTOR2),
        ],
    )
    def test_expand_vector_two_wires(self, original_wires, expanded_wires, expected, tol):
        """Test that expand_vector works with a single-wire vector."""

        res = expand_vector(TestExpandVector.VECTOR2, original_wires, expanded_wires)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_vector_invalid_wires(self):
        """Test exception raised if unphysical subsystems provided."""
        with pytest.raises(
            ValueError,
            match="Invalid target subsystems provided in 'original_wires' argument",
        ):
            expand_vector(TestExpandVector.VECTOR2, [-1, 5], 4)

    def test_expand_vector_invalid_vector(self):
        """Test exception raised if incorrect sized vector provided."""
        with pytest.raises(ValueError, match="Vector parameter must be of length"):
            expand_vector(TestExpandVector.VECTOR1, [0, 1], 4)


class TestSqrtMatrix:
    """Tests for the sqrt_matrix function."""

    # NOTE: make sure the matrix is positive definite
    dm_list = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 2]]),
        np.array([[4, 2], [2, 3]]),
    ]
    shape_list = range(2, 10)
    # NOTE: sqrt_matrix is frequently used by BlockEncode
    #       here below are some test matrices that are used
    #       in the BlockEncode tests
    matrices = [
        # 2x2 matrices
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        # Non-square matrices
        np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.2]]),
        # 3x3 matrix
        np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.2], [0.1, 0.2, 0.3]]),
        # Identity-like matrices
        np.array([[1, 0], [0, 1]]),
        np.identity(3),
        # Matrix with complex entries
        0.2 * np.array([[0.3, 0.9539392j], [0.9539392j, -0.3]]),
    ]
    # negative matrices: matrices that have negative eigenvalues
    matrices_negative = [
        # 2x2 matrix with negative eigenvalue
        np.array([[1, 2], [2, -3]]),
        # 3x3 matrix with mixed positive/negative eigenvalues
        np.array([[2, -1, 0], [-1, -2, 1], [0, 1, 3]]),
        # 4x4 matrix with negative eigenvalues
        np.array([[1, 2, 0, 1], [2, -2, 1, 0], [0, 1, -3, 2], [1, 0, 2, -1]]),
    ]
    # Known problematic matrices with informative error message.
    # NOTE: keep the entries here common. No fine-tuned corner cases.
    illmats_info_pairs = [
        (
            np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            ValueError,
            "Factor is exactly singular",
        )
    ]

    @pytest.mark.parametrize("dm", dm_list)
    def test_sqrt_matrix_sparse_dm(self, dm, tol):
        """Test the sqrt_matrix function."""

        A = qml.math.asarray(dm)
        A_sparse = csr_matrix(A)

        assert np.allclose(sqrt_matrix(A), sqrt_matrix_sparse(A_sparse).toarray(), atol=tol, rtol=0)

    @pytest.mark.parametrize("shape", shape_list)
    def test_sqrt_matrix_sparse_random(self, shape, tol):
        """Test the sqrt_matrix function."""

        # From unitary group
        dm = unitary_group.rvs(shape)
        dm = dm @ dm.T.conj()
        A = qml.math.asarray(dm)
        A_sparse = csr_matrix(A)

        assert np.allclose(sqrt_matrix(A), sqrt_matrix_sparse(A_sparse).toarray(), atol=tol, rtol=0)

    @pytest.mark.parametrize("matrix", matrices)
    def test_sqrt_matrix_inputs(self, matrix, tol):
        """Test sqrt_matrix function with various input matrices."""

        # type M^† M
        A = matrix.T.conj() @ matrix
        A = np.eye(A.shape[0]) - A
        A_sparse = csr_matrix(A)

        result_sparse = sqrt_matrix_sparse(A_sparse)
        result = result_sparse.toarray()
        result_2 = result @ result

        assert np.allclose(result_2, A, atol=tol, rtol=0)

        # type M M^†
        A = matrix @ matrix.T.conj()
        A = np.eye(A.shape[0]) - A
        A_sparse = csr_matrix(A)

        result_sparse = sqrt_matrix_sparse(A_sparse)
        result = result_sparse.toarray()
        result_2 = result @ result

        assert np.allclose(result_2, A, atol=tol, rtol=0)

    @pytest.mark.parametrize("m", matrices_negative)
    def test_sqrt_matrix_sparse_input_negative(self, m):
        """Test that sqrt_matrix_sparse raises ValueError for matrices with negative eigenvalues."""
        m = csr_matrix(m)
        with pytest.raises(ValueError):
            sqrt_matrix_sparse(m)

    @pytest.mark.parametrize("m, e, info", illmats_info_pairs)
    def test_sqrt_matrix_sparse_input_ill_conditioned(self, m, e, info):
        """Test that appropriate errors are raised for ill-conditioned matrices"""
        m = csr_matrix(m)
        with pytest.raises(e, match=info):
            sqrt_matrix_sparse(m)

    def test_sqrt_matrix_sparse_input_valid(self):
        """Test that if dense input errors raised"""
        A = np.array([[1, 0], [0, 1]])
        with pytest.raises(TypeError, match="only supports scipy.sparse matrices"):
            sqrt_matrix_sparse(A)


CONVERGENCE_ERROR = "Convergence threshold not reached"


def _reverse_det_hermitian(mat):
    """Helper function to reverse determinant sign of Hermitian matrix"""
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals[0] *= -1
    return eigvecs @ np.diag(eigvals) @ eigvecs.T.conj()


def _enforce_positivity(mat):
    """Helper function to ensure matrix is positive definite"""
    det_mat = np.linalg.det(mat)
    return mat if det_mat > 0 else _reverse_det_hermitian(mat)


class TestDenmanBeaversIterations:
    """Tests for the Denman-Beavers iteration method for matrix square root"""

    def test_singular_matrix(self):
        """Test that singular matrix raises appropriate error"""
        n = 4
        mat = csr_matrix(np.diag([0.0] + [1.0] * (n - 1)))

        with pytest.raises(ValueError, match="Factor is exactly singular"):
            _denman_beavers_iterations(mat)

    def test_overflow_matrix(self):
        """Test that matrix with very large values raises convergence warning"""
        n = 4
        mat = csr_matrix(np.eye(n) * 1e150)

        with pytest.raises(ValueError, match=CONVERGENCE_ERROR):
            _denman_beavers_iterations(mat)

    def test_invalid_value_matrix(self):
        """Test that matrix leading to invalid values raises appropriate error"""
        n = 4
        mat = csr_matrix(np.diag([-1e-10] + [1.0] * (n - 1)))

        with pytest.raises(ValueError, match="nan or inf"):
            _denman_beavers_iterations(mat)

    def test_non_convergent_matrix(self):
        """Test that non-convergent matrix raises appropriate error"""
        mat = csr_matrix([[1, 1e-6], [1e-6, 1e-12 + 1e-25]])
        with pytest.raises(ValueError, match=CONVERGENCE_ERROR):
            _denman_beavers_iterations(mat)

    def test_unstable_matrix(self):
        """Test that numerically unstable matrix raises convergence warning"""
        n = 4
        mat = csr_matrix(np.diag([1e-200, 1e200] + [1.0] * (n - 2)))

        with pytest.raises(ValueError, match=CONVERGENCE_ERROR):
            _denman_beavers_iterations(mat)

    @pytest.mark.parametrize("size", [2, 3, 4, 5])
    def test_valid_positive_definite(self, size, seed):
        """Test that valid real, positive definite matrices work correctly"""
        # Create a positive definite matrix
        rng = np.random.default_rng(seed)
        A = rng.random((size, size))
        mat = np.eye(size) - 0.1 * (A @ A.T)
        mat = _enforce_positivity(mat)
        mat = csr_matrix(mat)

        result = _denman_beavers_iterations(mat)
        # Check that result is a valid square root
        result_dense = result.toarray()
        original_dense = mat.toarray()
        qml.math.allclose(result_dense @ result_dense, original_dense, atol=1e-7, rtol=1e-7)

    def test_hermitian_matrix(self):
        """Test that Hermitian matrices work correctly on Hermitians of positive det.
        Emulate random users' random input; the iteration should pass the simple branch determined by determinant, or it needs extra examine.
        """
        n = 4
        A = np.random.random((n, n)) + 1j * np.random.random((n, n))
        mat = np.eye(n) - 0.2 * (A @ A.T.conj())

        det_mat = np.real(np.linalg.det(mat))
        good_mat = mat if det_mat > 0 else _reverse_det_hermitian(mat)

        result = _denman_beavers_iterations(csr_matrix(good_mat))
        result_dense = result.toarray()
        qml.math.allclose(result_dense @ result_dense, good_mat, atol=1e-7, rtol=1e-7)

        bad_mat = mat if det_mat < 0 else _reverse_det_hermitian(mat)
        with pytest.raises(ValueError, match="Invalid values encountered"):
            _denman_beavers_iterations(csr_matrix(bad_mat))


def _random_unitary(n):
    """Generates a random n x n unitary matrix."""
    Z = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    Q, _ = np.linalg.qr(Z)
    return Q


@pytest.mark.parametrize(
    "size, converter", [(2, qml.math.convert_to_su2), (4, qml.math.convert_to_su4)]
)
class TestConvertToSUMatrices:
    """Tests for the convert_to_SU2 and convert_to_SU4 functions."""

    @staticmethod
    def _assert_correct(original_matrix, su, phase):
        """Verifies that the converted matrix is correct."""
        assert qml.math.allclose(su * qml.math.exp(1j * phase), original_matrix)
        assert qml.math.allclose(qml.math.conj(su).T @ su, np.eye(su.shape[0]))
        assert qml.math.allclose(qml.math.linalg.det(su), 1)

    def test_simple_matrix(self, size, converter):
        """Tests the conversion of a simple matrix to SU(2)"""

        matrix = _random_unitary(size)
        su, phase = converter(matrix, return_global_phase=True)
        self._assert_correct(matrix, su, phase)

    def test_batched(self, size, converter):
        """Tests that the batched operation is correct."""

        matrices = [_random_unitary(size) for _ in range(3)]
        matrices = np.stack(matrices)
        sus, phases = converter(matrices, return_global_phase=True)
        assert qml.math.shape(sus) == (3, size, size)
        assert qml.math.shape(phases) == (3,)
        for matrix, su2, phase in zip(matrices, sus, phases):
            self._assert_correct(matrix, su2, phase)

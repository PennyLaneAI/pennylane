# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the eigvals transform
"""
# pylint: disable=too-few-public-methods
from functools import reduce

import pytest
import scipy
from gate_data import CNOT, H, I, S, X, Y, Z

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms import TransformError

one_qubit_no_parameter = [
    qml.PauliX,
    qml.PauliY,
    qml.PauliZ,
    qml.Hadamard,
    qml.S,
    qml.T,
    qml.SX,
]

one_qubit_one_parameter = [qml.RX, qml.RY, qml.RZ, qml.PhaseShift]


def test_invalid_argument():
    """Assert error raised when input is neither a tape, QNode, nor quantum function"""
    with pytest.raises(
        TransformError,
        match="Input is not an Operator, tape, QNode, or quantum function",
    ):
        _ = qml.eigvals(None)


class TestSingleOperation:
    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_instantiated(self, op_class):
        """Verify that the eigenvalues of non-parametric one-qubit gates are correct
        when provided as an instantiated operation"""
        op = op_class(wires=0)
        res = qml.eigvals(op)
        expected = op.eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_gates_qfunc(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as a qfunc"""
        res = qml.eigvals(op_class)(wires=0)
        expected = op_class(wires=0).eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_gates_qnode(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as a QNode"""
        dev = qml.device("default.qubit", wires=1)
        qnode = qml.QNode(lambda: op_class(wires=0) and qml.probs(wires=0), dev)
        res = qml.eigvals(qnode)()
        expected = op_class(wires=0).eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_instantiated(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as an instantiated operation"""
        op = op_class(0.54, wires=0)
        res = qml.eigvals(op)
        expected = op.eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_qfunc(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as a qfunc"""
        res = qml.eigvals(op_class)(0.54, wires=0)
        expected = op_class(0.54, wires=0).eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_qnode(self, op_class):
        """Verify that the eigenvalues of non-parametric one qubit gates is correct
        when provided as a QNode"""
        dev = qml.device("default.qubit", wires=1)
        qnode = qml.QNode(lambda x: op_class(x, wires=0) and qml.probs(wires=0), dev)
        res = qml.eigvals(qnode)(0.54)
        expected = op_class(0.54, wires=0).eigvals()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_adjoint(self, op_class):
        """Test that the adjoint is correctly taken into account"""
        rounding_precision = 6
        res = qml.eigvals(qml.adjoint(op_class))(0.54, wires=0)
        expected = op_class(-0.54, wires=0).eigvals()
        assert set(np.around(res, rounding_precision)) == set(
            np.around(expected, rounding_precision)
        )

    def test_ctrl(self):
        """Test that the ctrl is correctly taken into account"""
        res = qml.eigvals(qml.ctrl(qml.PauliX, 0))(wires=1)
        expected = np.linalg.eigvals(qml.matrix(qml.CNOT(wires=[0, 1])))
        assert np.allclose(np.sort(res), np.sort(expected))

    def test_tensor_product(self):
        """Test a tensor product"""
        res = qml.eigvals(qml.prod(qml.PauliX(0), qml.Identity(1), qml.PauliZ(1), lazy=False))
        expected = [1.0, -1.0, -1.0, 1.0]
        assert np.allclose(res, expected)

    def test_hamiltonian(self):
        """Test that the matrix of a Hamiltonian is correctly returned"""
        ham = qml.PauliZ(0) @ qml.PauliY(1) - 0.5 * qml.PauliX(1)

        res = qml.eigvals(ham)

        expected = np.linalg.eigvalsh(reduce(np.kron, [Z, Y]) - 0.5 * reduce(np.kron, [I, X]))
        assert np.allclose(res, expected)

    @pytest.mark.xfail(
        reason="This test will fail because Hamiltonians are not queued to tapes yet!"
    )
    def test_hamiltonian_qfunc(self):
        """Test that the matrix of a Hamiltonian is correctly returned"""

        def ansatz(x):
            return qml.PauliZ(0) @ qml.PauliY(1) - x * qml.PauliX(1)

        x = 0.5

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(ansatz)(x)

        expected = np.linalg.eigvalsh(reduce(np.kron, [Z, Y]) - 0.5 * reduce(np.kron, [I, X]))
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        ("row", "col", "dat"),
        [
            (
                # coordinates and values of a sparse Hamiltonian computed for H2
                # with geometry: np.array([[0.0, 0.0, 0.3674625962], [0.0, 0.0, -0.3674625962]])
                np.array([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15]),
                np.array([0, 1, 2, 3, 12, 4, 5, 6, 9, 7, 8, 6, 9, 10, 11, 3, 12, 13, 14, 15]),
                np.array(
                    [
                        0.72004228 + 0.0j,
                        0.2481941 + 0.0j,
                        0.2481941 + 0.0j,
                        0.47493346 + 0.0j,
                        0.18092703 + 0.0j,
                        -0.5363422 + 0.0j,
                        -0.52452264 + 0.0j,
                        -0.34359561 + 0.0j,
                        -0.18092703 + 0.0j,
                        0.36681148 + 0.0j,
                        -0.5363422 + 0.0j,
                        -0.18092703 + 0.0j,
                        -0.34359561 + 0.0j,
                        -0.52452264 + 0.0j,
                        0.36681148 + 0.0j,
                        0.18092703 + 0.0j,
                        -1.11700225 + 0.0j,
                        -0.44058792 + 0.0j,
                        -0.44058792 + 0.0j,
                        0.93441394 + 0.0j,
                    ]
                ),
            ),
        ],
    )
    def test_sparse_hamiltonian(self, row, col, dat):
        """Test that the eigenvalues of a sparse Hamiltonian are correctly returned"""
        # N x N matrix with N = 16
        h_mat = scipy.sparse.csr_matrix((dat, (row, col)), shape=(16, 16))
        h_sparse = qml.SparseHamiltonian(h_mat, wires=range(4))

        dense_mat = h_mat.todense()
        dense_eigvals = np.sort(np.linalg.eigvals(dense_mat))

        # k = 1  (< N-1) scipy.sparse.linalg is used:
        val_groundstate = qml.eigvals(h_sparse, k=1)
        assert np.allclose(val_groundstate, dense_eigvals[0])

        # k = 14  (< N-1) scipy.sparse.linalg is used:
        val_n_sparse = np.sort(qml.eigvals(h_sparse, k=14))
        assert np.allclose(val_n_sparse, dense_eigvals[0:-2])

        # k = 16 (> N-1) qml.math.linalg is used:
        val_all = np.sort(qml.eigvals(h_sparse, k=16))
        assert np.allclose(val_all, dense_eigvals)


class TestMultipleOperations:
    def test_multiple_operations_tape_no_overlaps(self):
        """Check the eigenvalues for a tape containing multiple gates
        assuming no overlap of wires"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(wires="a")
            qml.S(wires="b")
            qml.Hadamard(wires="c")

        tape = qml.tape.QuantumScript.from_queue(q)
        res = qml.eigvals(tape)
        expected = np.linalg.eigvals(np.kron(X, np.kron(S, H)))

        assert np.allclose(np.sort(res.real), np.sort(expected.real))
        assert np.allclose(np.sort(res.imag), np.sort(expected.imag))

    def test_multiple_operations_tape(self):
        """Check the eigenvalues for a tape containing multiple gates"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(wires="a")
            qml.S(wires="b")
            qml.Hadamard(wires="c")
            qml.CNOT(wires=["b", "c"])

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(tape)

        expected = np.linalg.eigvals(np.kron(I, CNOT) @ np.kron(X, np.kron(S, H)))
        assert np.allclose(res, expected)

    def test_multiple_operations_qfunc(self):
        """Check the eigenvalues for a qfunc containing multiple gates"""

        def testcircuit():
            qml.PauliX(wires="a")
            qml.S(wires="b")
            qml.Hadamard(wires="c")
            qml.CNOT(wires=["b", "c"])

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(testcircuit)()

        expected = np.linalg.eigvals(np.kron(I, CNOT) @ np.kron(X, np.kron(S, H)))
        assert np.allclose(res, expected)

    def test_multiple_operations_qnode(self):
        """Check the eigenvalues for a QNode containing multiple gates"""
        dev = qml.device("default.qubit", wires=["a", "b", "c"])

        @qml.qnode(dev)
        def testcircuit():
            qml.PauliX(wires="a")
            qml.adjoint(qml.S)(wires="b")
            qml.Hadamard(wires="c")
            qml.CNOT(wires=["b", "c"])
            return qml.expval(qml.PauliZ("a"))

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(testcircuit)()

        expected = np.linalg.eigvals(np.kron(I, CNOT) @ np.kron(X, np.kron(np.linalg.inv(S), H)))
        assert np.allclose(res, expected)


class TestCompositeOperations:
    """Composite Operations use math.linalg.eig instead of math.linalg.eigh since the operators
    may not be hermitian."""

    def test_sum_eigvals(self):
        """Test that a sum op returns the correct eigvals."""
        sum_op = qml.sum(qml.s_prod(1j, qml.PauliZ(wires=0)), qml.Identity(wires=0))
        sum_eigvals = qml.eigvals(sum_op)

        mat_rep = np.array([[1 + 1j, 0], [0, 1 - 1j]])
        mat_eigvals = np.linalg.eig(mat_rep)[0]

        assert np.allclose(mat_eigvals, sum_eigvals)

    def test_prod_eigvals(self):
        """Test that a prod op returns the correct eigvals."""

        prod_op = qml.prod(qml.s_prod(1j, qml.PauliZ(wires=0)), qml.Identity(wires=1))
        prod_eigvals = qml.eigvals(prod_op)

        mat_rep = np.array([[1j, 0, 0, 0], [0, 1j, 0, 0], [0, 0, -1j, 0], [0, 0, 0, -1j]])
        mat_eigvals = np.linalg.eig(mat_rep)[0]

        assert np.allclose(mat_eigvals, prod_eigvals)

    def test_composite_eigvals(self):
        """Test that an arithmetic op with non-hermitian base ops produces the correct eigen-values."""
        op = qml.prod(qml.PauliX(0), qml.adjoint(qml.PauliY(0)))
        op_adj = qml.adjoint(op)
        imag_op = -0.5j * (op - op_adj)
        op_eigvals = qml.eigvals(imag_op)

        mat_rep = np.array([[1.0, 0.0], [0.0, -1.0]])
        mat_eigvals = np.linalg.eig(mat_rep)[0]

        assert np.allclose(mat_eigvals, op_eigvals)


class TestTemplates:
    """These tests are useful as they test operators that might not have
    matrix forms defined, requiring decomposition."""

    def test_instantiated(self):
        """Test an instantiated template"""
        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        op = qml.StronglyEntanglingLayers(weights, wires=[0, 1])

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(op)

        with qml.queuing.AnnotatedQueue() as q:
            op.decomposition()

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            expected = qml.eigvals(tape)

        assert np.allclose(res, expected)

    def test_qfunc(self):
        """Test a template used within a qfunc"""

        def circuit(weights, x):
            qml.StronglyEntanglingLayers(weights, wires=[0, 1])
            qml.RX(x, wires=0)

        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        x = 0.54

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(circuit)(weights, x)

        op = qml.StronglyEntanglingLayers(weights, wires=[0, 1])

        with qml.queuing.AnnotatedQueue() as q:
            op.decomposition()
            qml.RX(x, wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            expected = qml.eigvals(tape)

        assert np.allclose(res, expected)

    def test_nested_instantiated(self):
        """Test an operation that must be decomposed twice"""

        class CustomOp(qml.operation.Operation):
            num_params = 1
            num_wires = 2

            @staticmethod
            def compute_decomposition(weights, wires):
                return [qml.StronglyEntanglingLayers(weights, wires=wires)]

        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        op = CustomOp(weights, wires=[0, 1])

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(op)

        op = qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        with qml.queuing.AnnotatedQueue() as q:
            op.decomposition()

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            expected = qml.eigvals(tape)

        assert np.allclose(res, expected)

    def test_nested_qfunc(self):
        """Test an operation that must be decomposed twice"""

        class CustomOp(qml.operation.Operation):
            num_params = 1
            num_wires = 2

            @staticmethod
            def compute_decomposition(weights, wires):
                return [qml.StronglyEntanglingLayers(weights, wires=wires)]

        def circuit(weights, x):
            CustomOp(weights, wires=[0, 1])
            qml.RX(x, wires=0)

        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        x = 0.54

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(circuit)(weights, x)

        op = qml.StronglyEntanglingLayers(weights, wires=[0, 1])

        with qml.queuing.AnnotatedQueue() as q:
            op.decomposition()
            qml.RX(x, wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            expected = qml.eigvals(tape)

        assert np.allclose(res, expected)


class TestDifferentiation:
    """Differentiation tests"""

    @pytest.mark.jax
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_jax(self, v):
        """Test that differentiation works correctly when using JAX"""

        import jax

        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])

        def loss(theta):
            U = qml.eigvals(circuit)(theta)
            return qml.math.sum(qml.math.real(U))

        x = jax.numpy.array(v)

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            l = loss(x)
            dl = jax.grad(loss)(x)

        assert isinstance(l, jax.numpy.ndarray)
        assert np.allclose(l, 2 * np.cos(v / 2))
        assert np.allclose(dl, -np.sin(v / 2))

    @pytest.mark.torch
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_torch(self, v):
        """Test that differentiation works correctly when using Torch"""

        import torch

        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])

        def loss(theta):
            U = qml.eigvals(circuit)(theta)
            return qml.math.sum(qml.math.real(U))

        x = torch.tensor(v, requires_grad=True)

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            l = loss(x)
            l.backward()

        dl = x.grad

        assert isinstance(l, torch.Tensor)
        assert np.allclose(l.detach(), 2 * np.cos(v / 2))
        assert np.allclose(dl.detach(), -np.sin(v / 2))

    @pytest.mark.tf
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_tensorflow(self, v):
        """Test that differentiation works correctly when using TF"""
        import tensorflow as tf

        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])

        def loss(theta):
            U = qml.eigvals(circuit)(theta)
            return qml.math.sum(qml.math.real(U))

        x = tf.Variable(v)
        with tf.GradientTape() as tape:
            with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
                l = loss(x)
        dl = tape.gradient(l, x)

        assert isinstance(l, tf.Tensor)
        assert np.allclose(l, 2 * np.cos(v / 2))
        assert np.allclose(dl, -np.sin(v / 2))

    @pytest.mark.autograd
    @pytest.mark.xfail(reason="np.linalg.eigvals not differentiable using Autograd")
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_autograd(self, v):
        """Test that differentiation works correctly when using Autograd"""

        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])

        def loss(theta):
            U = qml.eigvals(circuit)(theta)
            return qml.math.sum(qml.math.real(U))

        x = np.array(v, requires_grad=True)

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            l = loss(x)
            dl = qml.grad(loss)(x)

        assert isinstance(l, qml.numpy.tensor)
        assert np.allclose(l, 2 * np.cos(v / 2))
        assert np.allclose(dl, -np.sin(v / 2))

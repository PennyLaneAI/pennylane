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
from functools import reduce

import pytest
import scipy

import pennylane as qml
from gate_data import CNOT, H, I
from gate_data import Roty as RY
from gate_data import S, X, Y, Z
from pennylane import numpy as np
from pennylane.transforms.op_transforms import OperationTransformError

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
        assert set(np.around(res, rounding_precision)) == set(np.around(res, rounding_precision))

    def test_ctrl(self):
        """Test that the ctrl is correctly taken into account"""
        res = qml.eigvals(qml.ctrl(qml.PauliX, 0))(wires=1)
        expected = np.linalg.eigvals(qml.matrix(qml.CNOT(wires=[0, 1])))
        assert np.allclose(np.sort(res), np.sort(expected))

    def test_tensor_product(self):
        """Test a tensor product"""
        res = qml.eigvals(qml.PauliX(0) @ qml.Identity(1) @ qml.PauliZ(1))
        expected = reduce(np.kron, [[1, -1], [1, 1], [1, -1]])
        assert np.allclose(res, expected)

    def test_hamiltonian(self):
        """Test that the matrix of a Hamiltonian is correctly returned"""
        H = qml.PauliZ(0) @ qml.PauliY(1) - 0.5 * qml.PauliX(1)

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(H)

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
        ("row", "col", "dat", "val_ref"),
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
                # eigenvalues of the same matrix computed with np.linalg.eigh(H_dense)
                np.array(
                    [
                        -1.13730605,
                        -0.5363422,
                        -0.5363422,
                        -0.52452264,
                        -0.52452264,
                        -0.52452264,
                        -0.44058792,
                        -0.44058792,
                        -0.16266858,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.2481941,
                        0.2481941,
                        0.36681148,
                        0.36681148,
                        0.49523726,
                        0.72004228,
                        0.93441394,
                    ]
                ),
            ),
        ],
    )
    def test_sparse_hamiltonian(self, row, col, dat, val_ref):
        """Test that the eigenvalues of a sparse Hamiltonian are correctly returned"""
        # N x N matrix with N = 20
        h_sparse = scipy.sparse.csr_matrix((dat, (row, col)), shape=(len(row), len(col)))
        h_sparse = qml.SparseHamiltonian(h_sparse, wires=all)

        # k = 1  (< N-1) scipy.sparse.linalg is used:
        val_groundstate = qml.eigvals(h_sparse, k=1)
        # k = 18  (< N-1) scipy.sparse.linalg is used:
        val_n_sparse = qml.eigvals(h_sparse, k=len(val_ref) - 2)
        # k = 20 (> N-1) qml.math.linalg is used:
        val_all = qml.eigvals(h_sparse, k=len(val_ref))
        # k = 19 (= N-1) qml.math.linalg is used:
        val_n_dense = qml.eigvals(h_sparse, k=len(val_ref) - 1)

        assert np.allclose(val_groundstate, val_ref[0])
        assert np.allclose(np.sort(val_n_sparse), val_ref[0:18])
        assert np.allclose(np.sort(val_all), val_ref)
        assert np.allclose(np.sort(val_n_dense), val_ref)


class TestMultipleOperations:
    def test_multiple_operations_tape_no_overlaps(self):
        """Check the eigenvalues for a tape containing multiple gates
        assuming no overlap of wires"""

        with qml.tape.QuantumTape() as tape:
            qml.PauliX(wires="a")
            qml.S(wires="b")
            qml.Hadamard(wires="c")

        res = qml.eigvals(tape)
        expected = np.linalg.eigvals(np.kron(X, np.kron(S, H)))

        assert np.allclose(np.sort(res.real), np.sort(expected.real))
        assert np.allclose(np.sort(res.imag), np.sort(expected.imag))

    def test_multiple_operations_tape(self):
        """Check the eigenvalues for a tape containing multiple gates"""

        with qml.tape.QuantumTape() as tape:
            qml.PauliX(wires="a")
            qml.S(wires="b")
            qml.Hadamard(wires="c")
            qml.CNOT(wires=["b", "c"])

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


class TestTemplates:
    """These tests are useful as they test operators that might not have
    matrix forms defined, requiring decomposition."""

    def test_instantiated(self):
        """Test an instantiated template"""
        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        op = qml.StronglyEntanglingLayers(weights, wires=[0, 1])

        with pytest.warns(UserWarning, match="the eigenvalues will be computed numerically"):
            res = qml.eigvals(op)

        with qml.tape.QuantumTape() as tape:
            op.decomposition()

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

        with qml.tape.QuantumTape() as tape:
            op.decomposition()
            qml.RX(x, wires=0)

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
        with qml.tape.QuantumTape() as tape:
            op.decomposition()

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

        with qml.tape.QuantumTape() as tape:
            op.decomposition()
            qml.RX(x, wires=0)

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

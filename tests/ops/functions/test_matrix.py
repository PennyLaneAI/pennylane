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
Unit tests for the get_unitary_matrix transform
"""
from functools import reduce

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.op_transforms import OperationTransformError

from gate_data import I, X, Y, Z, H, S, CNOT, Roty as RY

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
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as an instantiated operation"""
        op = op_class(wires=0)
        res = qml.matrix(op)
        expected = op.matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_gates_qfunc(self, op_class):
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as a qfunc"""
        res = qml.matrix(op_class)(wires=0)
        expected = op_class(wires=0).matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_non_parametric_gates_qnode(self, op_class):
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as a QNode"""
        dev = qml.device("default.qubit", wires=1)
        qnode = qml.QNode(lambda: op_class(wires=0) and qml.probs(wires=0), dev)
        res = qml.matrix(qnode)()
        expected = op_class(wires=0).matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_no_parameter)
    def test_matrix_expansion(self, op_class):
        """Verify that matrices are correctly expanded when a wire order is provided"""
        res = qml.matrix(op_class, wire_order=[1, 0, 2])(wires=0)
        expected = np.kron(np.eye(2), np.kron(op_class(wires=0).matrix(), np.eye(2)))
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_instantiated(self, op_class):
        """Verify that the matrix of non-parametric one qubit gates is correct
        when provided as an instantiated operation"""
        op = op_class(0.54, wires=0)
        res = qml.matrix(op)
        expected = op.matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_qfunc(self, op_class):
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as a qfunc"""
        res = qml.matrix(op_class)(0.54, wires=0)
        expected = op_class(0.54, wires=0).matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_parametric_gates_qnode(self, op_class):
        """Verify that the matrices of non-parametric one qubit gates is correct
        when provided as a QNode"""
        dev = qml.device("default.qubit", wires=1)
        qnode = qml.QNode(lambda x: op_class(x, wires=0) and qml.probs(wires=0), dev)
        res = qml.matrix(qnode)(0.54)
        expected = op_class(0.54, wires=0).matrix()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("op_class", one_qubit_one_parameter)
    def test_adjoint(self, op_class):
        """Test that the adjoint is correctly taken into account"""
        res = qml.matrix(qml.adjoint(op_class))(0.54, wires=0)
        expected = op_class(-0.54, wires=0).matrix()
        assert np.allclose(res, expected)

    def test_ctrl(self):
        """Test that the ctrl is correctly taken into account"""
        res = qml.matrix(qml.ctrl(qml.PauliX, 0))(wires=1)
        expected = CNOT
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("target_wire", [0, 2, 3, 4])
    def test_CNOT_permutations(self, target_wire):
        """Test CNOT: 2-qubit gate with different target wires, some non-adjacent."""
        res = qml.matrix(qml.CNOT, wire_order=[0, 1, 2, 3, 4])(wires=[1, target_wire])

        # compute the expected matrix
        perm = np.swapaxes(
            np.swapaxes(np.arange(2**5).reshape([2] * 5), 0, 1), 0, target_wire
        ).flatten()
        expected = reduce(np.kron, [CNOT, I, I, I])[:, perm][perm]
        assert np.allclose(res, expected)

    def test_hamiltonian(self):
        """Test that the matrix of a Hamiltonian is correctly returned"""
        H = qml.PauliZ(0) @ qml.PauliY(1) - 0.5 * qml.PauliX(1)
        mat = qml.matrix(H, wire_order=[1, 0, 2])
        expected = reduce(np.kron, [Y, Z, I]) - 0.5 * reduce(np.kron, [X, I, I])

    @pytest.mark.xfail(
        reason="This test will fail because Hamiltonians are not queued to tapes yet!"
    )
    def test_hamiltonian_qfunc(self):
        """Test that the matrix of a Hamiltonian is correctly returned"""

        def ansatz(x):
            return qml.PauliZ(0) @ qml.PauliY(1) - x * qml.PauliX(1)

        x = 0.5
        mat = qml.matrix(ansatz, wire_order=[1, 0, 2])(x)
        expected = reduce(np.kron, [Y, Z, I]) - x * reduce(np.kron, [X, I, I])


class TestMultipleOperations:
    def test_multiple_operations_tape(self):
        """Check the total matrix for a tape containing multiple gates"""
        wire_order = ["a", "b", "c"]

        with qml.tape.QuantumTape() as tape:
            qml.PauliX(wires="a")
            qml.S(wires="b")
            qml.Hadamard(wires="c")
            qml.CNOT(wires=["b", "c"])

        matrix = qml.matrix(tape, wire_order)
        expected_matrix = np.kron(I, CNOT) @ np.kron(X, np.kron(S, H))
        assert np.allclose(matrix, expected_matrix)

    def test_multiple_operations_qfunc(self):
        """Check the total matrix for a qfunc containing multiple gates"""
        wire_order = ["a", "b", "c"]

        def testcircuit():
            qml.PauliX(wires="a")
            qml.S(wires="b")
            qml.Hadamard(wires="c")
            qml.CNOT(wires=["b", "c"])

        matrix = qml.matrix(testcircuit, wire_order)()
        expected_matrix = np.kron(I, CNOT) @ np.kron(X, np.kron(S, H))
        assert np.allclose(matrix, expected_matrix)

    def test_multiple_operations_qnode(self):
        """Check the total matrix for a QNode containing multiple gates"""
        dev = qml.device("default.qubit", wires=["a", "b", "c"])

        @qml.qnode(dev)
        def testcircuit():
            qml.PauliX(wires="a")
            qml.adjoint(qml.S)(wires="b")
            qml.Hadamard(wires="c")
            qml.CNOT(wires=["b", "c"])
            return qml.expval(qml.PauliZ("a"))

        matrix = qml.matrix(testcircuit)()
        expected_matrix = np.kron(I, CNOT) @ np.kron(X, np.kron(np.linalg.inv(S), H))
        assert np.allclose(matrix, expected_matrix)


class TestWithParameterBroadcasting:
    def test_multiple_operations_tape_single_broadcasted_op(self):
        """Check the total matrix for a tape containing multiple gates
        and a single broadcasted gate."""
        wire_order = ["a", "b", "c"]

        angles = np.array([0.0, np.pi, 0.0])
        with qml.tape.QuantumTape() as tape:
            qml.S(wires="b")
            qml.RX(angles, wires="a")
            qml.Hadamard(wires="c")
            qml.CNOT(wires=["b", "c"])

        matrix = qml.matrix(tape, wire_order)
        expected_matrix = [
            np.kron(I, CNOT) @ np.kron(I, np.kron(S, H)),
            -1j * np.kron(I, CNOT) @ np.kron(X, np.kron(S, H)),
            np.kron(I, CNOT) @ np.kron(I, np.kron(S, H)),
        ]
        assert np.allclose(matrix, expected_matrix)

    def test_multiple_operations_tape_multi_broadcasted_op(self):
        """Check the total matrix for a tape containing multiple gates
        and a multiple broadcasted gate."""
        wire_order = ["a", "b", "c"]

        angles1 = np.array([0.0, np.pi, 0.0, np.pi])
        angles2 = np.array([0.0, 0.0, np.pi, np.pi])
        with qml.tape.QuantumTape() as tape:
            qml.S(wires="b")
            qml.RX(angles1, wires="a")
            qml.Hadamard(wires="c")
            qml.CNOT(wires=["b", "c"])
            qml.RX(angles2, wires="c")

        matrix = qml.matrix(tape, wire_order)
        expected_matrix = [
            np.kron(I, np.kron(I, I)) @ np.kron(I, CNOT) @ np.kron(I, np.kron(S, H)),
            -1j * np.kron(I, np.kron(I, I)) @ np.kron(I, CNOT) @ np.kron(X, np.kron(S, H)),
            -1j * np.kron(I, np.kron(I, X)) @ np.kron(I, CNOT) @ np.kron(I, np.kron(S, H)),
            -np.kron(I, np.kron(I, X)) @ np.kron(I, CNOT) @ np.kron(X, np.kron(S, H)),
        ]
        assert np.allclose(matrix, expected_matrix)


class TestCustomWireOrdering:
    def test_tensor_wire_oder(self):
        """Test wire order of a tensor product"""
        H = qml.PauliZ(0) @ qml.PauliX(1)
        res = qml.matrix(H, wire_order=[0, 2, 1])
        expected = np.kron(Z, np.kron(I, X))
        assert np.allclose(res, expected)

    def test_tape_wireorder(self):
        """Test changing the wire order when using a tape"""

        with qml.tape.QuantumTape() as tape:
            qml.PauliX(wires=0)
            qml.PauliY(wires=1)
            qml.PauliZ(wires=2)

        matrix = qml.matrix(tape)
        expected_matrix = np.kron(X, np.kron(Y, Z))
        assert np.allclose(matrix, expected_matrix)

        matrix = qml.matrix(tape, wire_order=[1, 0, 2])
        expected_matrix = np.kron(Y, np.kron(X, Z))
        assert np.allclose(matrix, expected_matrix)

    def test_qfunc_wireorder(self):
        """Test changing the wire order when using a qfunc"""

        def testcircuit():
            qml.PauliX(wires=0)
            qml.PauliY(wires=1)
            qml.PauliZ(wires=2)

        matrix = qml.matrix(testcircuit)()
        expected_matrix = np.kron(X, np.kron(Y, Z))
        assert np.allclose(matrix, expected_matrix)

        matrix = qml.matrix(testcircuit, wire_order=[1, 0, 2])()
        expected_matrix = np.kron(Y, np.kron(X, Z))
        assert np.allclose(matrix, expected_matrix)

    def test_qnode_wireorder(self):
        """Test changing the wire order when using a QNode"""
        dev = qml.device("default.qubit", wires=[1, 0, 2, 3])

        @qml.matrix()
        @qml.qnode(dev)
        def testcircuit(x):
            qml.PauliX(wires=0)
            qml.RY(x, wires=1)
            qml.PauliZ(wires=2)
            return qml.expval(qml.PauliZ(0))

        x = 0.5

        # default wire ordering will come from the device
        expected_matrix = np.kron(RY(x), np.kron(X, np.kron(Z, I)))
        assert np.allclose(testcircuit(x), expected_matrix)

        @qml.matrix(wire_order=[1, 0, 2])
        @qml.qnode(dev)
        def testcircuit(x):
            qml.PauliX(wires=0)
            qml.RY(x, wires=1)
            qml.PauliZ(wires=2)
            return qml.expval(qml.PauliZ(0))

        expected_matrix = np.kron(RY(x), np.kron(X, Z))
        assert np.allclose(testcircuit(x), expected_matrix)


class TestTemplates:
    """These tests are useful as they test operators that might not have
    matrix forms defined, requiring decomposition."""

    def test_instantiated(self):
        """Test an instantiated template"""
        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        op = qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        res = qml.matrix(op)

        with qml.tape.QuantumTape() as tape:
            op.decomposition()

        expected = qml.matrix(tape)
        np.allclose(res, expected)

    def test_qfunc(self):
        """Test a template used within a qfunc"""

        def circuit(weights, x):
            qml.StronglyEntanglingLayers(weights, wires=[0, 1])
            qml.RX(x, wires=0)

        weights = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        x = 0.54
        res = qml.matrix(circuit)(weights, x)

        op = qml.StronglyEntanglingLayers(weights, wires=[0, 1])

        with qml.tape.QuantumTape() as tape:
            op.decomposition()
            qml.RX(x, wires=0)

        expected = qml.matrix(tape)
        np.allclose(res, expected)

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
        res = qml.matrix(op)

        op = qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        with qml.tape.QuantumTape() as tape:
            op.decomposition()

        expected = qml.matrix(tape)
        np.allclose(res, expected)

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
        res = qml.matrix(circuit)(weights, x)

        op = qml.StronglyEntanglingLayers(weights, wires=[0, 1])

        with qml.tape.QuantumTape() as tape:
            op.decomposition()
            qml.RX(x, wires=0)

        expected = qml.matrix(tape)
        np.allclose(res, expected)


class TestValidation:
    def test_invalid_argument(self):
        """Assert error raised when input is neither a tape, QNode, nor quantum function"""
        with pytest.raises(
            OperationTransformError,
            match="Input is not an Operator, tape, QNode, or quantum function",
        ):
            qml.matrix(None)(0.5)

    def test_wrong_function(self):
        """Assert error raised when input function is not a quantum function"""

        def testfunction(x):
            return x

        with pytest.raises(OperationTransformError, match="function contains no quantum operation"):
            qml.matrix(testfunction)(0)

    def test_inconsistent_wires(self):
        """Assert error raised when wire labels in wire_order and circuit are inconsistent"""

        def circuit():
            qml.PauliX(wires=1)
            qml.PauliZ(wires=0)

        wires = [0, "b"]

        with pytest.raises(
            OperationTransformError,
            match=r"Wires in circuit \[1, 0\] are inconsistent with those in wire_order \[0, 'b'\]",
        ):
            matrix = qml.matrix(circuit, wire_order=wires)()


class TestInterfaces:
    @pytest.mark.tf
    def test_tf(self):
        """Test with tensorflow interface"""
        import tensorflow as tf

        @qml.matrix
        def circuit(beta, theta):
            qml.RZ(beta, wires=0)
            qml.RZ(theta[0], wires=1)
            qml.CRY(theta[1], wires=[1, 2])

        beta = 0.1
        # input tensorflow parameters
        theta = tf.Variable([0.2, 0.3])
        matrix = circuit(beta, theta)

        # expected matrix
        theta_np = theta.numpy()
        matrix1 = np.kron(
            qml.RZ(beta, wires=0).matrix(),
            np.kron(qml.RZ(theta_np[0], wires=1).matrix(), I),
        )
        matrix2 = np.kron(I, qml.CRY(theta_np[1], wires=[1, 2]).matrix())
        expected_matrix = matrix2 @ matrix1

        assert np.allclose(matrix, expected_matrix)

    @pytest.mark.torch
    def test_torch(self):
        """Test with torch interface"""

        import torch

        dev = qml.device("default.qubit", wires=3)

        @qml.matrix
        @qml.qnode(dev, interface="torch")
        def circuit(theta):
            qml.RZ(theta[0], wires=0)
            qml.RZ(theta[1], wires=1)
            qml.CRY(theta[2], wires=[1, 2])
            return qml.expval(qml.PauliZ(1))

        # input torch parameters
        theta = torch.tensor([0.1, 0.2, 0.3])
        matrix = circuit(theta)

        # expected matrix
        matrix1 = np.kron(
            qml.RZ(theta[0], wires=0).matrix(),
            np.kron(qml.RZ(theta[1], wires=1).matrix(), I),
        )
        matrix2 = np.kron(I, qml.CRY(theta[2], wires=[1, 2]).matrix())
        expected_matrix = matrix2 @ matrix1

        assert np.allclose(matrix, expected_matrix)

    @pytest.mark.autograd
    def test_autograd(self):
        """Test with autograd interface"""

        @qml.matrix
        def circuit(theta):
            qml.RZ(theta[0], wires=0)
            qml.RZ(theta[1], wires=1)
            qml.CRY(theta[2], wires=[1, 2])

        # set input parameters
        theta = np.array([0.1, 0.2, 0.3], requires_grad=True)
        matrix = circuit(theta)

        # expected matrix
        matrix1 = np.kron(
            qml.RZ(theta[0], wires=0).matrix(),
            np.kron(qml.RZ(theta[1], wires=1).matrix(), I),
        )
        matrix2 = np.kron(I, qml.CRY(theta[2], wires=[1, 2]).matrix())
        expected_matrix = matrix2 @ matrix1

        assert np.allclose(matrix, expected_matrix)

    @pytest.mark.jax
    def test_get_unitary_matrix_interface_jax(self):
        """Test with JAX interface"""

        from jax import numpy as jnp
        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        @qml.matrix
        def circuit(theta):
            qml.RZ(theta[0], wires=0)
            qml.RZ(theta[1], wires=1)
            qml.CRY(theta[2], wires=[1, 2])

        # input jax parameters
        theta = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64)

        matrix = circuit(theta)

        # expected matrix
        matrix1 = np.kron(
            qml.RZ(theta[0], wires=0).matrix(),
            np.kron(qml.RZ(theta[1], wires=1).matrix(), I),
        )
        matrix2 = np.kron(I, qml.CRY(theta[2], wires=[1, 2]).matrix())
        expected_matrix = matrix2 @ matrix1

        assert np.allclose(matrix, expected_matrix)


class TestDifferentiation:
    @pytest.mark.jax
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_jax(self, v):

        import jax

        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])

        def loss(theta):
            U = qml.matrix(circuit)(theta)
            return qml.math.real(qml.math.trace(U))

        x = jax.numpy.array(v)

        l = loss(x)
        dl = jax.grad(loss)(x)
        matrix = qml.matrix(circuit)(x)

        assert isinstance(matrix, jax.numpy.ndarray)
        assert np.allclose(l, 2 * np.cos(v / 2))
        assert np.allclose(dl, -np.sin(v / 2))

    @pytest.mark.torch
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_torch(self, v):

        import torch

        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])

        def loss(theta):
            U = qml.matrix(circuit)(theta)
            return qml.math.real(qml.math.trace(U))

        x = torch.tensor(v, requires_grad=True)
        l = loss(x)
        l.backward()
        dl = x.grad
        matrix = qml.matrix(circuit)(x)

        assert isinstance(matrix, torch.Tensor)
        assert np.allclose(l.detach(), 2 * np.cos(v / 2))
        assert np.allclose(dl.detach(), -np.sin(v / 2))

    @pytest.mark.tf
    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_tensorflow(self, v):

        import tensorflow as tf

        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])

        def loss(theta):
            U = qml.matrix(circuit)(theta)
            return qml.math.real(qml.math.trace(U))

        x = tf.Variable(v)
        with tf.GradientTape() as tape:
            l = loss(x)
        dl = tape.gradient(l, x)
        matrix = qml.matrix(circuit)(x)

        assert isinstance(matrix, tf.Tensor)
        assert np.allclose(l, 2 * np.cos(v / 2))
        assert np.allclose(dl, -np.sin(v / 2))

    @pytest.mark.parametrize("v", np.linspace(0.2, 1.6, 8))
    def test_get_unitary_matrix_autograd_differentiable(self, v):
        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])

        def loss(theta):
            U = qml.matrix(circuit)(theta)
            return qml.math.real(qml.math.trace(U))

        x = np.array(v, requires_grad=True)
        l = loss(x)
        dl = qml.grad(loss)(x)
        matrix = qml.matrix(circuit)(x)

        assert isinstance(matrix, qml.numpy.tensor)
        assert np.allclose(l, 2 * np.cos(v / 2))
        assert np.allclose(dl, -np.sin(v / 2))

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
"""Tests for the gradients.gradient_transform module."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients.gradient_transform import gradient_transform, gradient_expand


class TestGradientExpand:
    """Tests for the gradient expand function"""

    def test_no_expansion(self, mocker):
        """Test that a circuit with differentiable
        operations is not expanded"""
        x = np.array(0.2, requires_grad=True)
        y = np.array(0.1, requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        spy = mocker.spy(tape, "expand")
        new_tape = gradient_expand(tape)

        assert new_tape is tape
        spy.assert_not_called()

    def test_trainable_nondiff_expansion(self, mocker):
        """Test that a circuit with non-differentiable
        trainable operations is expanded"""
        x = np.array(0.2, requires_grad=True)
        y = np.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qml.PhaseShift):
            grad_method = None

        with qml.tape.QuantumTape() as tape:
            NonDiffPhaseShift(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        spy = mocker.spy(tape, "expand")
        new_tape = gradient_expand(tape)

        assert new_tape is not tape
        spy.assert_called()

        new_tape.operations[0].name == "RZ"
        new_tape.operations[0].grad_method == "A"
        new_tape.operations[1].name == "RY"
        new_tape.operations[2].name == "CNOT"

    def test_nontrainable_nondiff(self, mocker):
        """Test that a circuit with non-differentiable
        non-trainable operations is not expanded"""
        x = np.array(0.2, requires_grad=False)
        y = np.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qml.PhaseShift):
            grad_method = None

        with qml.tape.QuantumTape() as tape:
            NonDiffPhaseShift(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)
        assert tape.trainable_params == {1}

        spy = mocker.spy(tape, "expand")
        new_tape = gradient_expand(tape)

        assert new_tape is tape
        spy.assert_not_called()

    def test_trainable_numeric(self, mocker):
        """Test that a circuit with numeric differentiable
        trainable operations is *not* expanded"""
        x = np.array(0.2, requires_grad=True)
        y = np.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qml.PhaseShift):
            grad_method = "F"

        with qml.tape.QuantumTape() as tape:
            NonDiffPhaseShift(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        spy = mocker.spy(tape, "expand")
        new_tape = gradient_expand(tape)

        assert new_tape is tape
        spy.assert_not_called()


class TestGradientTransformIntegration:
    """Test integration of the gradient transform decorator"""

    def test_acting_on_qnodes(self, tol):
        """Test that a gradient transform acts on QNodes
        correctly"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=[0])
            qml.RY(weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

        grad_fn = qml.gradients.param_shift(circuit)

        w = np.array([0.543, -0.654], requires_grad=True)
        res = grad_fn(w)

        x, y = w
        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_decorator(self, tol):
        """Test that a gradient transform decorating a QNode
        acts correctly"""
        dev = qml.device("default.qubit", wires=2)

        @qml.gradients.param_shift
        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=[0])
            qml.RY(weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

        w = np.array([0.543, -0.654], requires_grad=True)
        res = circuit(w)

        x, y = w
        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_passing_arguments(self, mocker, tol):
        """Test that a gradient transform correctly
        passes arguments"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=[0])
            qml.RY(weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

        grad_fn = qml.gradients.param_shift(circuit, shift=np.pi / 4)

        w = np.array([0.543, -0.654], requires_grad=True)
        res = grad_fn(w)

        x, y = w
        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        assert spy.call_args[0][2] == np.pi / 4

    def test_expansion(self, mocker, tol):
        """Test that a gradient transform correctly
        expands gates with no gradient recipe"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        class NonDiffRXGate(qml.PhaseShift):
            grad_method = "F"

            @staticmethod
            def decomposition(x, wires):
                return [qml.RX(x, wires=wires)]

        @qml.qnode(dev)
        def circuit(weights):
            NonDiffRXGate(weights[0], wires=[0])
            qml.RY(weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

        grad_fn = qml.gradients.param_shift(circuit)

        w = np.array([0.543, -0.654], requires_grad=True)
        res = grad_fn(w)

        x, y = w
        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert spy.call_args[0][0].operations[0].name == "RX"

    def test_permuted_arguments(self, tol):
        """Test that a gradient transform acts on QNodes
        correctly when the QNode arguments are permuted"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[1], wires=[0])
            qml.RY(weights[0], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

        w = np.array([-0.654, 0.543], requires_grad=True)
        res = qml.gradients.param_shift(circuit)(w)

        expected = qml.jacobian(circuit)(w)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_classical_processing_arguments(self, mocker, tol):
        """Test that a gradient transform acts on QNodes
        correctly when the QNode arguments are classically processed"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.transforms, "classical_jacobian")

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0] ** 2, wires=[0])
            qml.RY(weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        w = np.array([0.543, -0.654], requires_grad=True)
        res = qml.gradients.param_shift(circuit)(w)

        classical_jac = spy.spy_return(w)
        assert isinstance(classical_jac, np.ndarray)
        assert np.allclose(classical_jac, np.array([[2 * w[0], 0], [0, 1]]))

        x, y = w
        expected = [-2 * x * np.sin(x ** 2), 0]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_classical_processing_multiple_arguments(self, mocker, tol):
        """Test that a gradient transform acts on QNodes
        correctly when multiple QNode arguments are classically processed"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.transforms, "classical_jacobian")

        @qml.qnode(dev)
        def circuit(data, weights):
            qml.RY(np.cos(data), wires=0)
            qml.RX(weights[0] ** 2, wires=[0])
            qml.RY(weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # set d as non-differentiable
        d = np.array(0.56, requires_grad=False)
        w = np.array([0.543, -0.654], requires_grad=True)
        x, y = w

        res = qml.gradients.param_shift(circuit)(d, w)
        classical_jac = spy.spy_return(d, w)
        assert np.allclose(classical_jac, np.array([[2 * w[0], 0], [0, 1]]).T)

        expected = np.array([-2 * x * np.cos(np.cos(d)) * np.sin(x ** 2), 0])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # set d as differentiable
        d = np.array(0.56, requires_grad=True)
        w = np.array([0.543, -0.654], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(d, w)
        classical_jac = spy.spy_return(d, w)
        assert isinstance(classical_jac, tuple)
        assert np.allclose(classical_jac[0], [-np.sin(d), 0, 0])
        assert np.allclose(classical_jac[1], np.array([[0, 2 * w[0], 0], [0, 0, 1]]).T)

        expected_dd = np.cos(x ** 2) * np.sin(d) * np.sin(np.cos(d))
        expected_dw = np.array([-2 * x * np.cos(np.cos(d)) * np.sin(x ** 2), 0])
        assert np.allclose(res[0], expected_dd, atol=tol, rtol=0)
        assert np.allclose(res[1], expected_dw, atol=tol, rtol=0)

    def test_advanced_classical_processing_arguments(self, tol):
        """Test that a gradient transform acts on QNodes
        correctly when the QNode arguments are classically processed,
        and the input weights and the output weights have weird shape."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0, 0] ** 2, wires=[0])
            qml.RY(weights[0, 1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        w = np.array([[0.543, -0.654], [0.0, 0.0]], requires_grad=True)
        res = qml.gradients.param_shift(circuit)(w)
        assert res.shape == (4, 2, 2)

        expected = qml.jacobian(circuit)(w)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # when executed with hybrid=False, only the quantum jacobian is returned
        res = qml.gradients.param_shift(circuit, hybrid=False)(w)
        assert res.shape == (4, 2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=[0])
            qml.RY(weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        w = np.array([0.543 ** 2, -0.654], requires_grad=True)
        expected = qml.jacobian(circuit)(w)
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestInterfaceIntegration:
    """Test that the gradient transforms are differentiable
    using each interface"""

    def test_autograd(self, tol):
        """Test that a gradient transform remains differentiable
        with autograd"""
        dev = qml.device("default.qubit", wires=2)

        @qml.gradients.param_shift
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x ** 2, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliX(1))

        x = np.array(-0.654, requires_grad=True)

        res = circuit(x)
        expected = -4 * x * np.cos(x ** 2) * np.sin(x ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.grad(circuit)(x)
        expected = -2 * (4 * x ** 2 * np.cos(2 * x ** 2) + np.sin(2 * x ** 2))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_tf(self, tol):
        """Test that a gradient transform remains differentiable
        with TF"""
        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit", wires=2)

        @qml.gradients.param_shift
        @qml.qnode(dev, interface="tf", diff_method="parameter-shift")
        def circuit(x):
            qml.RY(x ** 2, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliX(1))

        x_ = -0.654
        x = tf.Variable(x_, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(x)

        expected = -4 * x_ * np.cos(x_ ** 2) * np.sin(x_ ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.gradient(res, x)
        expected = -2 * (4 * x_ ** 2 * np.cos(2 * x_ ** 2) + np.sin(2 * x_ ** 2))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_torch(self, tol):
        """Test that a gradient transform remains differentiable
        with PyTorch"""
        torch = pytest.importorskip("torch")
        dev = qml.device("default.qubit", wires=2)

        @qml.gradients.param_shift
        @qml.qnode(dev, interface="torch")
        def circuit(x):
            qml.RY(x, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliX(1))

        x_ = -0.654
        x = torch.tensor(x_, dtype=torch.float64, requires_grad=True)
        res = circuit(x)[0]

        expected = -2 * np.cos(x_) * np.sin(x_)
        assert np.allclose(res.detach(), expected, atol=tol, rtol=0)

        res.backward()
        expected = -2 * np.cos(2 * x_)
        assert np.allclose(x.grad.detach(), expected, atol=tol, rtol=0)

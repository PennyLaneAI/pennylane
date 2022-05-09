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
from pennylane.gradients.gradient_transform import (
    gradient_transform,
    gradient_analysis,
    choose_grad_methods,
    grad_method_validation,
)


class TestGradAnalysis:
    """Tests for parameter gradient methods"""

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is correctly marked"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        gradient_analysis(tape)

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "A"
        assert tape._par_info[2]["grad_method"] == "A"

    def test_analysis_caching(self, mocker):
        """Test that the gradient analysis is only executed once per tape
        if grad_fn is set an unchanged."""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        spy = mocker.spy(qml.operation, "has_grad_method")
        gradient_analysis(tape, grad_fn=5)
        spy.assert_called()

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "A"
        assert tape._par_info[2]["grad_method"] == "A"

        spy = mocker.spy(qml.operation, "has_grad_method")
        gradient_analysis(tape, grad_fn=5)
        spy.assert_not_called()

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliY(0))

        gradient_analysis(tape)

        assert tape._par_info[0]["grad_method"] == "A"
        assert tape._par_info[1]["grad_method"] == "0"

    def test_independent_no_graph_mode(self):
        """In non-graph mode, it is impossible to determine
        if a parameter is independent or not"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliY(0))

        gradient_analysis(tape, use_graph=False)

        assert tape._par_info[0]["grad_method"] == "A"
        assert tape._par_info[1]["grad_method"] == "A"

    def test_finite_diff(self, monkeypatch):
        """If an op has grad_method=F, this should be respected"""
        monkeypatch.setattr(qml.RX, "grad_method", "F")

        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        gradient_analysis(tape)

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "F"
        assert tape._par_info[2]["grad_method"] == "A"


class TestGradMethodValidation:
    """Test the helper function grad_method_validation."""

    @pytest.mark.parametrize("method", ["analytic", "best"])
    def test_with_nondiff_parameters(self, method):
        """Test that trainable parameters without grad_method
        are detected correctly, raising an exception."""
        with qml.tape.QuantumTape() as tape:
            qml.RX(np.array(0.1, requires_grad=True), wires=0)
            qml.RX(np.array(0.1, requires_grad=True), wires=0)
            qml.expval(qml.PauliZ(0))
        tape._par_info[0]["grad_method"] = "A"
        tape._par_info[1]["grad_method"] = None
        with pytest.raises(ValueError, match="Cannot differentiate with respect"):
            grad_method_validation(method, tape)

    def test_with_numdiff_parameters_and_analytic(self):
        """Test that trainable parameters with numerical grad_method ``"F"``
        together with ``method="analytic"`` raises an exception."""
        with qml.tape.QuantumTape() as tape:
            qml.RX(np.array(0.1, requires_grad=True), wires=0)
            qml.RX(np.array(0.1, requires_grad=True), wires=0)
            qml.expval(qml.PauliZ(0))
        tape._par_info[0]["grad_method"] = "A"
        tape._par_info[1]["grad_method"] = "F"
        with pytest.raises(ValueError, match="The analytic gradient method cannot be used"):
            grad_method_validation("analytic", tape)


class TestChooseGradMethods:
    """Test the helper function choose_grad_methods"""

    all_diff_methods = [
        ["A"] * 2,
        [None, "A", "F", "A"],
        ["F"],
        ["0", "A"],
    ]

    @pytest.mark.parametrize("diff_methods", all_diff_methods)
    def test_without_argnum(self, diff_methods):
        """Test that the method returns all diff_methods when
        used with ``argnum=None``."""
        chosen = choose_grad_methods(diff_methods, None)
        assert chosen == dict(enumerate(diff_methods))

    @pytest.mark.parametrize(
        "diff_methods, argnum, expected",
        zip(all_diff_methods, [1, 2, 0, 0], [{1: "A"}, {2: "F"}, {0: "F"}, {0: "0"}]),
    )
    def test_with_integer_argnum(self, diff_methods, argnum, expected):
        """Test that the method returns the correct single diff_method when
        used with an integer ``argnum``."""
        chosen = choose_grad_methods(diff_methods, argnum)
        assert chosen == expected

    @pytest.mark.parametrize("diff_methods", all_diff_methods[:2] + [[]])
    def test_warning_with_empty_argnum(self, diff_methods):
        """Test that the method raises a warning when an empty iterable
        is passed as ``argnum``."""
        with pytest.warns(UserWarning, match="No trainable parameters were specified"):
            chosen = choose_grad_methods(diff_methods, [])
        assert chosen == {}


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

        shifts = [(np.pi / 4,), (np.pi / 3,)]
        grad_fn = qml.gradients.param_shift(circuit, shifts=shifts)

        w = np.array([0.543, -0.654], requires_grad=True)
        res = grad_fn(w)

        x, y = w
        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        assert spy.call_args[0][2] == shifts

    def test_expansion(self, mocker, tol):
        """Test that a gradient transform correctly
        expands gates with no gradient recipe"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        class NonDiffRXGate(qml.PhaseShift):
            grad_method = None

            @staticmethod
            def compute_decomposition(x, wires):
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

    def test_multiple_tensor_arguments(self, tol):
        """Test that a gradient transform acts on QNodes
        correctly when multiple tensor QNode arguments are present"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x[0, 0], wires=0)
            qml.RY(y[0, 0], wires=0)
            qml.RZ(x[1, 0], wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array([[0.1, 0.3], [0.2, -0.1]], requires_grad=True)
        y = np.array([[0.2, 0.2], [0.3, 0.5]], requires_grad=True)

        expected = qml.jacobian(circuit)(x, y)
        res = qml.gradients.param_shift(circuit, hybrid=True)(x, y)
        assert isinstance(res, tuple) and len(res) == 2
        assert all(np.allclose(_r, _e, atol=tol, rtol=0) for _r, _e in zip(res, expected))

    def test_multiple_tensor_arguments_old_version(self, tol):
        """Test that a gradient transform acts on QNodes
        correctly when multiple tensor QNode arguments are present"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x[0, 0], wires=0)
            qml.RY(y[0, 0], wires=0)
            qml.RZ(x[1, 0], wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array([[0.1], [0.2]], requires_grad=True)
        y = np.array([[0.2], [0.3]], requires_grad=True)

        expected = qml.jacobian(circuit)(x, y)
        res = qml.gradients.param_shift(circuit)(x, y)
        assert isinstance(res, tuple) and len(res) == 2
        assert all(np.allclose(_r, _e, atol=tol, rtol=0) for _r, _e in zip(res, expected))

    def test_high_dimensional_single_parameter_arg(self, tol):
        """Test that a gradient transform acts on QNodes correctly
        when a single high-dimensional tensor QNode arguments is used"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0, 0] / 2, wires=0)
            qml.RX(x[0, 0] / 2, wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array([[0.1, 0.1], [0.1, 0.1]], requires_grad=True)

        expected = qml.jacobian(circuit)(x)
        res = qml.gradients.param_shift(circuit)(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_high_dimensional_single_parameter_arg_and_single_gate(self, tol):
        """Test that a gradient transform acts on QNodes correctly
        when a single high-dimensional tensor QNode arguments is used"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0, 0], wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array([[0.1]], requires_grad=True)

        expected = qml.jacobian(circuit)(x)
        res = qml.gradients.param_shift(circuit)(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_gate_arg(self, tol):
        """Test that a gradient transform acts on QNodes correctly
        when a single QNode argument and gate are present"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qml.jacobian(circuit)(x)
        cjac = qml.transforms.classical_jacobian(circuit)(x)
        res = qml.gradients.param_shift(circuit)(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_first_non_trainable_argument(self, tol):
        """Test that a gradient transform acts on QNodes
        correctly when the first argument is non-trainable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(y[0], wires=0)
            qml.RY(y[0], wires=0)
            qml.RZ(y[1], wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1], requires_grad=False)
        y = np.array([0.2, 0.3], requires_grad=True)

        expected = qml.jacobian(circuit)(x, y)
        res = qml.gradients.param_shift(circuit)(x, y)

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
        expected = [-2 * x * np.sin(x**2), 0]
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

        expected = np.array([-2 * x * np.cos(np.cos(d)) * np.sin(x**2), 0])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # set d as differentiable
        d = np.array(0.56, requires_grad=True)
        w = np.array([0.543, -0.654], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(d, w)
        classical_jac = spy.spy_return(d, w)
        assert isinstance(classical_jac, tuple)
        assert np.allclose(classical_jac[0], [-np.sin(d), 0, 0])
        assert np.allclose(classical_jac[1], np.array([[0, 2 * w[0], 0], [0, 0, 1]]).T)

        expected_dd = np.cos(x**2) * np.sin(d) * np.sin(np.cos(d))
        expected_dw = np.array([-2 * x * np.cos(np.cos(d)) * np.sin(x**2), 0])
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

        w = np.array([0.543**2, -0.654], requires_grad=True)
        expected = qml.jacobian(circuit)(w)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("strategy", ["gradient", "device"])
    def test_template_integration(self, strategy, tol):
        """Test that the gradient transform acts on QNodes
        correctly when the QNode contains a template"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, expansion_strategy=strategy)
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.probs(wires=[0, 1])

        weights = np.ones([2, 3, 3], dtype=np.float64, requires_grad=True)
        res = qml.gradients.param_shift(circuit)(weights)
        assert res.shape == (4, 2, 3, 3)

        expected = qml.jacobian(circuit)(weights)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_setting_shots(self):
        """Test that setting the number of shots works correctly for
        a gradient transform"""

        dev = qml.device("default.qubit", wires=1, shots=1000)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.543, requires_grad=True)

        # the gradient function can be called with different shot values
        grad_fn = qml.gradients.param_shift(circuit)
        assert grad_fn(x).shape == ()
        assert grad_fn(x, shots=[(1, 1000)]).shape == (1000,)

        # the original QNode is unaffected
        assert circuit(x).shape == tuple()
        assert circuit(x, shots=1000).shape == tuple()

    def test_shots_error(self):
        """Raise an exception if shots is used within the QNode"""
        dev = qml.device("default.qubit", wires=1, shots=1000)

        def circuit(x, shots):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(UserWarning, match="Detected 'shots' as an argument to the given"):
            qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match="Detected 'shots' as an argument of the quantum"):
            qml.gradients.param_shift(qnode)(0.2, shots=100)


class TestInterfaceIntegration:
    """Test that the gradient transforms are differentiable
    using each interface"""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Test that a gradient transform remains differentiable
        with autograd"""
        dev = qml.device("default.qubit", wires=2)

        @qml.gradients.param_shift
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x**2, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliX(1))

        x = np.array(-0.654, requires_grad=True)

        res = circuit(x)
        expected = -4 * x * np.cos(x**2) * np.sin(x**2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = qml.grad(circuit)(x)
        expected = -2 * (4 * x**2 * np.cos(2 * x**2) + np.sin(2 * x**2))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Test that a gradient transform remains differentiable
        with TF"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.gradients.param_shift
        @qml.qnode(dev, interface="tf", diff_method="parameter-shift")
        def circuit(x):
            qml.RY(x**2, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliX(1))

        x_ = -0.654
        x = tf.Variable(x_, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(x)

        expected = -4 * x_ * np.cos(x_**2) * np.sin(x_**2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.gradient(res, x)
        expected = -2 * (4 * x_**2 * np.cos(2 * x_**2) + np.sin(2 * x_**2))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Test that a gradient transform remains differentiable
        with PyTorch"""
        import torch

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

    @pytest.mark.jax
    def test_jax(self, tol):
        """Test that a gradient transform remains differentiable
        with JAX"""
        import jax

        jnp = jax.numpy
        dev = qml.device("default.qubit", wires=2)

        @qml.gradients.param_shift
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.RY(x**2, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliX(1))

        x = jnp.array(-0.654)

        res = circuit(x)
        expected = -4 * x * np.cos(x**2) * np.sin(x**2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.grad(circuit)(x)
        expected = -2 * (4 * x**2 * np.cos(2 * x**2) + np.sin(2 * x**2))
        assert np.allclose(res, expected, atol=tol, rtol=0)

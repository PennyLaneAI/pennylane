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
import inspect

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients.gradient_transform import (
    SUPPORTED_GRADIENT_KWARGS,
    _find_gradient_methods,
    _validate_gradient_methods,
    choose_trainable_param_indices,
)
from pennylane.transforms.core import TransformDispatcher


def test_supported_gradient_kwargs():
    """Test that all keyword arguments of gradient transforms are
    registered as supported gradient kwargs, and no others."""
    # Collect all gradient transforms

    # Non-diff_methods to skip
    methods_to_skip = ("metric_tensor", "classical_fisher", "quantum_fisher")

    grad_transforms = []
    for attr in dir(qml.gradients):
        if attr in methods_to_skip:
            continue
        obj = getattr(qml.gradients, attr)
        if isinstance(obj, TransformDispatcher):
            grad_transforms.append(obj)

    # Collect arguments of all gradient transforms
    grad_kwargs = set()
    for tr in grad_transforms:
        grad_kwargs |= set(inspect.signature(tr).parameters)

    # Remove arguments that are not keyword arguments
    grad_kwargs -= {"tape"}
    # Remove "dev", because we decided against supporting this kwarg, although
    # it is an argument to param_shift_cv, to avoid confusion.
    grad_kwargs -= {"dev"}

    # Check equality of required and supported gradient kwargs
    assert grad_kwargs == SUPPORTED_GRADIENT_KWARGS
    # Shots should never be used as a gradient kwarg
    assert "shots" not in grad_kwargs


def test_repr():
    """Test the repr method of gradient transforms."""
    assert repr(qml.gradients.param_shift) == "<transform: param_shift>"
    assert repr(qml.gradients.spsa_grad) == "<transform: spsa_grad>"
    assert repr(qml.gradients.finite_diff) == "<transform: finite_diff>"


class TestGradAnalysis:
    """Tests for parameter gradient methods"""

    # pylint: disable=protected-access

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is correctly marked"""

        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        trainable_params = choose_trainable_param_indices(tape, None)
        diff_methods = _find_gradient_methods(tape, trainable_params)

        assert diff_methods[0] is None
        assert diff_methods[1] == "A"
        assert diff_methods[2] == "A"

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliY(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        trainable_params = choose_trainable_param_indices(tape, None)
        diff_methods = _find_gradient_methods(tape, trainable_params)

        assert diff_methods[0] == "A"
        assert diff_methods[1] == "0"

    def test_independent_no_graph_mode(self):
        """In non-graph mode, it is impossible to determine
        if a parameter is independent or not"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliY(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        trainable_params = choose_trainable_param_indices(tape, None)
        diff_methods = _find_gradient_methods(tape, trainable_params, use_graph=False)

        assert diff_methods[0] == "A"
        assert diff_methods[1] == "A"

    def test_finite_diff(self, monkeypatch):
        """If an op has grad_method=F, this should be respected"""

        monkeypatch.setattr(qml.RX, "grad_method", "F")

        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        trainable_params = choose_trainable_param_indices(tape, None)
        diff_methods = _find_gradient_methods(tape, trainable_params)

        assert diff_methods[0] is None
        assert diff_methods[1] == "F"
        assert diff_methods[2] == "A"


class TestGradMethodValidation:
    """Test the helper function _grad_method_validation."""

    # pylint: disable=protected-access

    @pytest.mark.parametrize("method", ["analytic", "best"])
    def test_with_nondiff_parameters(self, method):
        """Test that trainable parameters without grad_method
        are detected correctly, raising an exception."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(np.array(0.1, requires_grad=True), wires=0)
            qml.RX(np.array(0.1, requires_grad=True), wires=0)
            qml.expval(qml.PauliZ(0))
        tape = qml.tape.QuantumScript.from_queue(q)
        diff_methods = {0: "A", 1: None}
        with pytest.raises(ValueError, match="Cannot differentiate with respect"):
            _validate_gradient_methods(tape, method, diff_methods)

    def test_with_numdiff_parameters_and_analytic(self):
        """Test that trainable parameters with numerical grad_method ``"F"``
        together with ``method="analytic"`` raises an exception."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(np.array(0.1, requires_grad=True), wires=0)
            qml.RX(np.array(0.1, requires_grad=True), wires=0)
            qml.expval(qml.PauliZ(0))
        tape = qml.tape.QuantumScript.from_queue(q)
        diff_methods = {
            0: "A",
            1: "F",
        }
        with pytest.raises(ValueError, match="The analytic gradient method cannot be used"):
            _validate_gradient_methods(tape, "analytic", diff_methods)


class TestChooseParams:
    """Test the helper function choose_trainable_param_indices."""

    def test_without_argnum(self):
        """Test that the method returns all params when used with ``argnum=None``."""
        tape = qml.tape.QuantumScript(
            [qml.RX(1.2, wires=0), qml.RY(2.3, wires=0), qml.RZ(3.4, wires=0)],
            [qml.expval(qml.PauliZ(0))],
            trainable_params=[1, 2],
        )
        chosen = choose_trainable_param_indices(tape, None)
        assert chosen == [0, 1]

    def test_with_integer_argnum(self):
        """Test that the method returns the correct single param when
        used with an integer ``argnum``."""
        tape = qml.tape.QuantumScript(
            [qml.RX(1.2, wires=0), qml.RY(2.3, wires=0), qml.RZ(3.4, wires=0)],
            [qml.expval(qml.PauliZ(0))],
            trainable_params=[1, 2],
        )
        chosen = choose_trainable_param_indices(tape, argnum=1)
        assert chosen == [1]

    def test_warning_with_empty_argnum(self):
        """Test that the method raises a warning when an empty iterable
        is passed as ``argnum``."""
        tape = qml.tape.QuantumScript(
            [qml.RX(1.2, wires=0), qml.RY(2.3, wires=0), qml.RZ(3.4, wires=0)],
            [qml.expval(qml.PauliZ(0))],
            trainable_params=[1, 2],
        )
        with pytest.warns(UserWarning, match="No trainable parameters were specified"):
            chosen = choose_trainable_param_indices(tape, [])
        assert chosen == []


class TestGradientTransformIntegration:
    """Test integration of the gradient transform decorator"""

    @pytest.mark.parametrize("shots, atol", [(None, 1e-6), (1000, 1e-1), ([1000, 500], 3e-1)])
    @pytest.mark.parametrize("slicing", [False, True])
    @pytest.mark.parametrize("prefactor", [1.0, 2.0])
    def test_acting_on_qnodes_single_param(self, shots, slicing, prefactor, atol):
        """Test that a gradient transform acts on QNodes with a single parameter correctly"""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(weights):
            if slicing:
                qml.RX(prefactor * weights[0], wires=[0])
            else:
                qml.RX(prefactor * weights, wires=[0])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

        grad_fn = qml.gradients.param_shift(circuit)

        w = np.array([0.543] if slicing else 0.543, requires_grad=True)
        res = grad_fn(w)
        assert circuit.interface == "auto"

        # Need to multiply 0 with w to get the right output shape for non-scalar w
        expected = (-prefactor * np.sin(prefactor * w), w * 0)

        if isinstance(shots, list):
            assert all(np.allclose(r, expected, atol=atol, rtol=0) for r in res)
        else:
            assert np.allclose(res, expected, atol=atol, rtol=0)

    @pytest.mark.parametrize("shots, atol", [(None, 1e-6), (1000, 2e-1), ([1000, 1500], 2e-1)])
    @pytest.mark.parametrize("prefactor", [1.0, 2.0])
    def test_acting_on_qnodes_multi_param(self, shots, prefactor, atol, seed):
        """Test that a gradient transform acts on QNodes with multiple parameters correctly"""

        dev = qml.device("default.qubit", wires=2, seed=seed)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=[0])
            qml.RY(prefactor * weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        grad_fn = qml.gradients.param_shift(circuit)

        w = np.array([0.543, -0.654], requires_grad=True)
        res = grad_fn(w)
        assert circuit.interface == "auto"
        x, y = w
        y *= prefactor
        expected = np.array(
            [
                [-np.sin(x), 0],
                [
                    2 * np.cos(x) * np.sin(x) * np.cos(y) ** 2,
                    2 * prefactor * np.cos(y) * np.sin(y) * np.cos(x) ** 2,
                ],
            ]
        )
        if isinstance(shots, list):
            assert all(np.allclose(r, expected, atol=atol) for r in res)
        else:
            assert np.allclose(res, expected, atol=atol)

    @pytest.mark.xfail(reason="Gradient transforms are not compatible with shots and mixed shapes")
    @pytest.mark.parametrize("shots, atol", [(None, 1e-6), (1000, 1e-1), ([1000, 100], 2e-1)])
    def test_acting_on_qnodes_multi_param_multi_arg(self, shots, atol):
        """Test that a gradient transform acts on QNodes with multiple parameters
        in both the tape and the QNode correctly"""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(weight0, weight1):
            qml.RX(weight0, wires=[0])
            qml.RY(weight1[0], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

        grad_fn = qml.gradients.param_shift(circuit)

        w = [np.array(0.543, requires_grad=True), np.array([-0.654], requires_grad=True)]
        res = grad_fn(*w)
        assert circuit.interface == "auto"
        x, (y,) = w
        expected = (np.array([-np.sin(x), 0]), np.array([[0], [-2 * np.cos(y) * np.sin(y)]]))
        if isinstance(shots, list):
            assert isinstance(res, tuple) and len(res) == len(shots)
            for _res in res:
                assert isinstance(_res, tuple) and len(_res) == 2
                assert all(np.allclose(r, e, atol=atol, rtol=0) for r, e in zip(_res, expected))
        else:
            assert isinstance(res, tuple) and len(res) == 2
            assert all(np.allclose(r, e, atol=atol, rtol=0) for r, e in zip(res, expected))

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

        # pylint: disable=too-few-public-methods
        class NonDiffRXGate(qml.RX):
            """A non-differentiable gate that decomposes into RX."""

            grad_method = None

            @staticmethod
            def compute_decomposition(x, wires):
                """Decompose into a qml.RX gate."""
                return [qml.RX(x, wires=wires)]

        @qml.qnode(dev)
        def circuit(weights):
            """A quantum circuit using the above non-differentiable RX gate."""
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

        def cost(w):
            return qml.math.stack(circuit(w))

        w = np.array([-0.654, 0.543], requires_grad=True)
        res = qml.gradients.param_shift(circuit)(w)

        expected = qml.jacobian(cost)(w)
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
        # pylint:disable=unexpected-keyword-arg
        res = qml.gradients.param_shift(circuit)(x, y)
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
        for _r, _e in zip(res, expected):
            assert qml.math.allclose(_r, _e, atol=tol, rtol=0)

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

        @qml.qnode(dev, interface="autograd")
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qml.jacobian(circuit)(x)
        res = qml.gradients.param_shift(circuit)(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_first_non_trainable_argument(self, tol):
        """Test that a gradient transform acts on QNodes
        correctly when the first argument is non-trainable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            """A quantum circuit that does not use its first parameter."""
            # pylint: disable=unused-argument
            qml.RX(y[0], wires=0)
            qml.RY(y[0], wires=0)
            qml.RZ(y[1], wires=0)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1], requires_grad=False)
        y = np.array([0.2, 0.3], requires_grad=True)

        expected = qml.jacobian(circuit)(x, y)
        res = qml.gradients.param_shift(circuit)(x, y)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_classical_processing_arguments(self, tol):
        """Test that a gradient transform acts on QNodes
        correctly when the QNode arguments are classically processed"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0] ** 2, wires=[0])
            qml.RY(weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        w = np.array([0.543, -0.654], requires_grad=True)
        res = qml.gradients.param_shift(circuit)(w)

        x, _ = w
        expected = [-2 * x * np.sin(x**2), 0]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_classical_processing_multiple_arguments(self, tol):
        """Test that a gradient transform acts on QNodes
        correctly when multiple QNode arguments are classically processed"""
        dev = qml.device("default.qubit", wires=2)

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
        x, _ = w

        res = qml.gradients.param_shift(circuit)(d, w)

        expected = np.array([-2 * x * np.cos(np.cos(d)) * np.sin(x**2), 0])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # set d as differentiable
        d = np.array(0.56, requires_grad=True)
        w = np.array([0.543, -0.654], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(d, w)

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

        @qml.qnode(dev)
        def circuit1(weights):
            qml.RX(weights[0], wires=[0])
            qml.RY(weights[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        w = np.array([0.543**2, -0.654], requires_grad=True)
        expected = qml.jacobian(circuit1)(w)

        assert np.allclose(res[0][0], expected[0], atol=10e-2, rtol=0)
        assert np.allclose(res[1][0], expected[1], atol=10e-2, rtol=0)

    def test_template_integration(self, tol):
        """Test that the gradient transform acts on QNodes
        correctly when the QNode contains a template"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.probs(wires=[0, 1])

        weights = np.ones([2, 3, 3], dtype=np.float64, requires_grad=True)
        res = qml.gradients.param_shift(circuit)(weights)
        assert res.shape == (4, 2, 3, 3)

        expected = qml.jacobian(circuit)(weights)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    # pylint: disable=unexpected-keyword-arg
    def test_setting_shots(self):
        """Test that setting the number of shots works correctly for
        a gradient transform"""

        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(shots=1000)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.543, requires_grad=True)

        # the gradient function can be called with different shot values
        grad_fn = qml.gradients.param_shift(circuit)
        assert grad_fn(x).shape == ()

        assert len(qml.set_shots(shots=[(1, 1000)])(grad_fn)(x)) == 1000

        # the original QNode is unaffected
        assert circuit(x).shape == tuple()
        assert qml.set_shots(shots=1000)(circuit)(x).shape == tuple()

    @pytest.mark.parametrize(
        "interface",
        (
            pytest.param("autograd", marks=pytest.mark.autograd),
            pytest.param("jax", marks=pytest.mark.jax),
            pytest.param("torch", marks=pytest.mark.torch),
            pytest.param("tensorflow", marks=pytest.mark.tf),
        ),
    )
    def test_use_with_batch_transform(self, interface):
        """Test that a gradient transform can be chained with a batch transform."""

        dev = qml.device("default.qubit")

        # @qml.transforms.split_non_commuting
        @qml.qnode(dev)
        def c(x):
            qml.RX(x**2, 0)
            return qml.expval(qml.Z(0)), qml.expval(qml.Y(0)), qml.expval(qml.X(0))

        x = qml.math.asarray(0.5, like=interface, requires_grad=True)

        if interface == "tensorflow":
            import tensorflow as tf

            with tf.GradientTape():  # need to make x trainable
                grad_z, grad_y, grad_x = qml.gradients.param_shift(c)(x)
        else:
            grad_z, grad_y, grad_x = qml.gradients.param_shift(c)(x)

        expected_z = -2 * x * qml.math.sin(x**2)
        expected_y = -2 * x * qml.math.cos(x**2)
        assert qml.math.allclose(expected_z, grad_z)
        assert qml.math.allclose(grad_y, expected_y)
        assert qml.math.allclose(grad_x, 0)


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
        res = circuit(x)

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

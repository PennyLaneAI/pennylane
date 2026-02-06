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
"""Tests for the gradients.parameter_shift_cv module."""
# pylint: disable=protected-access, no-self-use, not-callable, no-value-for-parameter

from unittest import mock

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.gradients import param_shift_cv
from pennylane.gradients.gradient_transform import choose_trainable_param_indices
from pennylane.gradients.parameter_shift_cv import (
    _find_gradient_methods_cv,
    _grad_method_cv,
    _transform_observable,
)

hbar = 2


class TestGradAnalysis:
    """Tests for parameter gradient methods"""

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is
        correctly marked"""

        with qp.queuing.AnnotatedQueue() as q:
            qp.FockState(1, wires=0)
            qp.Displacement(0.543, 0, wires=[1])
            qp.Beamsplitter(0, 0, wires=[0, 1])
            qp.expval(qp.QuadX(wires=[0]))

        tape = qp.tape.QuantumScript.from_queue(q)
        assert _grad_method_cv(tape, 0) is None
        assert _grad_method_cv(tape, 1) == "A"
        assert _grad_method_cv(tape, 2) == "A"
        assert _grad_method_cv(tape, 3) == "A"
        assert _grad_method_cv(tape, 4) == "A"

        trainable_params = choose_trainable_param_indices(tape, None)
        diff_methods = _find_gradient_methods_cv(tape, trainable_params)

        assert diff_methods[0] is None
        assert diff_methods[1] == "A"
        assert diff_methods[2] == "A"
        assert diff_methods[3] == "A"
        assert diff_methods[4] == "A"

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(0.543, wires=[0])
            qp.Rotation(-0.654, wires=[1])
            qp.expval(qp.QuadP(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        assert _grad_method_cv(tape, 0) == "A"
        assert _grad_method_cv(tape, 1) == "0"

        trainable_params = choose_trainable_param_indices(tape, None)
        diff_methods = _find_gradient_methods_cv(tape, trainable_params)

        assert diff_methods[0] == "A"
        assert diff_methods[1] == "0"

    def test_finite_diff(self, monkeypatch):
        """If an op has grad_method=F, this should be respected
        by the qp.tape.QuantumScript"""
        monkeypatch.setattr(qp.Rotation, "grad_method", "F")

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(0.543, wires=[0])
            qp.Squeezing(0.543, 0, wires=[0])
            qp.expval(qp.QuadP(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        assert _grad_method_cv(tape, 0) == "F"
        assert _grad_method_cv(tape, 1) == "A"
        assert _grad_method_cv(tape, 2) == "A"

    def test_non_gaussian_operation(self):
        """Test that a non-Gaussian operation succeeding
        a differentiable Gaussian operation results in
        numeric differentiation."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(1.0, wires=[0])
            qp.Rotation(1.0, wires=[1])
            # Non-Gaussian
            qp.Kerr(1.0, wires=[1])
            qp.expval(qp.QuadP(0))
            qp.expval(qp.QuadX(1))

        tape = qp.tape.QuantumScript.from_queue(q)
        # First rotation gate has no succeeding non-Gaussian operation
        assert _grad_method_cv(tape, 0) == "A"
        # Second rotation gate does no succeeding non-Gaussian operation
        assert _grad_method_cv(tape, 1) == "F"
        # Kerr gate does not support the parameter-shift rule
        assert _grad_method_cv(tape, 2) == "F"

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(1.0, wires=[0])
            qp.Rotation(1.0, wires=[1])
            # entangle the modes
            qp.Beamsplitter(1.0, 0.0, wires=[0, 1])
            # Non-Gaussian
            qp.Kerr(1.0, wires=[1])
            qp.expval(qp.QuadP(0))
            qp.expval(qp.QuadX(1))

        tape = qp.tape.QuantumScript.from_queue(q)
        # After entangling the modes, the Kerr gate now succeeds
        # both initial rotations
        assert _grad_method_cv(tape, 0) == "F"
        assert _grad_method_cv(tape, 1) == "F"
        assert _grad_method_cv(tape, 2) == "F"

    def test_probability(self):
        """Probability is the expectation value of a
        higher order observable, and thus only supports numerical
        differentiation"""
        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(0.543, wires=[0])
            qp.Squeezing(0.543, 0, wires=[0])
            qp.probs(wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert _grad_method_cv(tape, 0) == "F"
        assert _grad_method_cv(tape, 1) == "F"
        assert _grad_method_cv(tape, 2) == "F"

    def test_variance(self):
        """If the variance of the observable is first order, then
        parameter-shift is supported. If the observable is second order,
        however, only finite-differences is supported."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(1.0, wires=[0])
            qp.var(qp.QuadP(0))  # first order

        tape = qp.tape.QuantumScript.from_queue(q)
        assert _grad_method_cv(tape, 0) == "A"

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(1.0, wires=[0])
            qp.var(qp.NumberOperator(0))  # second order

        tape = qp.tape.QuantumScript.from_queue(q)
        assert _grad_method_cv(tape, 0) == "F"

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(1.0, wires=[0])
            qp.Rotation(1.0, wires=[1])
            qp.Beamsplitter(0.5, 0.0, wires=[0, 1])
            qp.var(qp.NumberOperator(0))  # fourth order
            qp.expval(qp.NumberOperator(1))

        tape = qp.tape.QuantumScript.from_queue(q)
        assert _grad_method_cv(tape, 0) == "F"
        assert _grad_method_cv(tape, 1) == "F"
        assert _grad_method_cv(tape, 2) == "F"
        assert _grad_method_cv(tape, 3) == "F"

    def test_second_order_expectation(self):
        """Test that the expectation of a second-order observable forces
        the gradient method to use the second-order parameter-shift rule"""

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(1.0, wires=[0])
            qp.expval(qp.NumberOperator(0))  # second order

        tape = qp.tape.QuantumScript.from_queue(q)
        assert _grad_method_cv(tape, 0) == "A2"

    def test_unknown_op_grad_method(self, monkeypatch):
        """Test that an exception is raised if an operator has a
        grad method defined that the CV parameter-shift tape
        doesn't recognize"""
        monkeypatch.setattr(qp.Rotation, "grad_method", "B")

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(1.0, wires=0)
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.raises(ValueError, match="unknown gradient method"):
            _grad_method_cv(tape, 0)


class TestTransformObservable:
    """Tests for the _transform_observable method"""

    def test_incorrect_heisenberg_size(self, monkeypatch):
        """The number of dimensions of a CV observable Heisenberg representation does
        not match the ev_order attribute."""
        monkeypatch.setattr(qp.QuadP, "ev_order", 2)

        with pytest.raises(ValueError, match="Mismatch between the polynomial order"):
            _transform_observable(qp.QuadP(0), np.identity(3), device_wires=[0])

    def test_higher_order_observable(self, monkeypatch):
        """An exception should be raised if the observable is higher than 2nd order."""
        monkeypatch.setattr(qp.QuadP, "ev_order", 3)

        with pytest.raises(NotImplementedError, match="order > 2 not implemented"):
            _transform_observable(qp.QuadP(0), np.identity(3), device_wires=[0])

    def test_first_order_transform(self, tol):
        """Test that a first order observable is transformed correctly"""
        # create a symmetric transformation
        Z = np.arange(3**2).reshape(3, 3)
        Z = Z.T + Z

        obs = qp.QuadX(0)
        res = _transform_observable(obs, Z, device_wires=[0])

        # The Heisenberg representation of the X
        # operator is simply... X
        expected = np.array([0, 1, 0]) @ Z

        assert isinstance(res, qp.PolyXP)
        assert res.wires.labels == (0,)
        assert np.allclose(res.data[0], expected, atol=tol, rtol=0)

    def test_second_order_transform(self, tol):
        """Test that a second order observable is transformed correctly"""
        # create a symmetric transformation
        Z = np.arange(3**2).reshape(3, 3)
        Z = Z.T + Z

        obs = qp.NumberOperator(0)
        res = _transform_observable(obs, Z, device_wires=[0])

        # The Heisenberg representation of the number operator
        # is (X^2 + P^2) / (2*hbar) - 1/2
        A = np.array([[-0.5, 0, 0], [0, 0.25, 0], [0, 0, 0.25]])
        expected = A @ Z + Z @ A

        assert isinstance(res, qp.PolyXP)
        assert res.wires.labels == (0,)
        assert np.allclose(res.data[0], expected, atol=tol, rtol=0)

    def test_device_wire_expansion(self, tol):
        """Test that the transformation works correctly
        for the case where the transformation applies to more wires
        than the observable."""

        # create a 3-mode symmetric transformation
        wires = qp.wires.Wires([0, "a", 2])
        ndim = 1 + 2 * len(wires)

        Z = np.arange(ndim**2).reshape(ndim, ndim)
        Z = Z.T + Z

        obs = qp.NumberOperator(0)
        res = _transform_observable(obs, Z, device_wires=wires)

        # The Heisenberg representation of the number operator
        # is (X^2 + P^2) / (2*hbar) - 1/2. We use the ordering
        # I, X0, Xa, X2, P0, Pa, P2.
        A = np.diag([-0.5, 0.25, 0.25, 0, 0, 0, 0])
        expected = A @ Z + Z @ A

        assert isinstance(res, qp.PolyXP)
        assert res.wires == wires
        assert np.allclose(res.data[0], expected, atol=tol, rtol=0)


class TestParameterShiftLogic:
    """Test for the dispatching logic of the parameter shift method"""

    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qp.device("default.gaussian", wires=2)

        @qp.qnode(device=dev, interface="autograd")
        def circuit(weights):
            qp.Displacement(weights[0], 0.0, wires=[0])
            qp.Rotation(weights[1], wires=[0])
            return qp.expval(qp.QuadX(0))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qp.gradients.param_shift_cv(circuit, dev)(weights)

    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qp.device("default.gaussian", wires=2)

        @qp.qnode(dev, interface="torch")
        def circuit(weights):
            qp.Displacement(weights[0], 0.0, wires=[0])
            qp.Rotation(weights[1], wires=[0])
            return qp.expval(qp.QuadX(0))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qp.gradients.param_shift_cv(circuit, dev)(weights)

    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qp.device("default.gaussian", wires=2)

        @qp.qnode(dev, interface="tf")
        def circuit(weights):
            qp.Displacement(weights[0], 0.0, wires=[0])
            qp.Rotation(weights[1], wires=[0])
            return qp.expval(qp.QuadX(0))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qp.gradients.param_shift_cv(circuit, dev)(weights)

    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qp.device("default.gaussian", wires=2)

        @qp.qnode(device=dev, interface="jax")
        def circuit(weights):
            qp.Displacement(weights[0], 0.0, wires=[0])
            qp.Rotation(weights[1], wires=[0])
            return qp.expval(qp.QuadX(0))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qp.gradients.param_shift_cv(circuit, dev)(weights)

    def test_no_trainable_params_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qp.device("default.gaussian", wires=2)

        weights = [0.1, 0.2]
        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(weights[0], 0.0, wires=[0])
            qp.Rotation(weights[1], wires=[0])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        # TODO: remove once #2155 is resolved
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qp.gradients.param_shift_cv(tape, dev)
        res = post_processing(qp.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert res.shape == (0,)

    def test_all_zero_diff_methods(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qp.device("default.gaussian", wires=2)

        @qp.qnode(dev)
        def circuit(params):
            qp.Rotation(params[0], wires=0)
            return qp.expval(qp.QuadX(1))

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qp.gradients.param_shift_cv(circuit, dev)(params)
        assert np.allclose(result, np.zeros(3), atol=0, rtol=0)

    def test_state_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a state"""
        with qp.queuing.AnnotatedQueue() as q:
            qp.state()

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.raises(ValueError, match=r"return the state is not supported"):
            qp.gradients.param_shift_cv(tape, None)

    def test_force_order2(self, mocker):
        """Test that if the force_order2 keyword argument is provided,
        the second order parameter shift rule is forced"""
        spy = mocker.spy(qp.gradients.parameter_shift_cv, "second_order_param_shift")

        dev = qp.device("default.gaussian", wires=1)

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(1.0, 0.0, wires=[0])
            qp.Rotation(2.0, wires=[0])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1, 2}

        qp.gradients.param_shift_cv(tape, dev, force_order2=False)
        spy.assert_not_called()

        qp.gradients.param_shift_cv(tape, dev, force_order2=True)
        spy.assert_called()

    def test_force_order2_dim_2(self, mocker, monkeypatch):
        """Test that if the force_order2 keyword argument is provided, the
        second order parameter shift rule is forced for an observable with
        2-dimensional parameters"""
        spy = mocker.spy(qp.gradients.parameter_shift_cv, "second_order_param_shift")

        def _mock_transform_observable(obs, Z, device_wires):  # pylint: disable=unused-argument
            """A mock version of the _transform_observable internal function
            such that an operator ``transformed_obs`` of two-dimensions is
            returned. This function is created such that when definining ``A =
            transformed_obs.parameters[0]`` the condition ``len(A.nonzero()[0])
            == 1 and A.ndim == 2 and A[0, 0] != 0`` is ``True``."""
            iden = qp.Identity(0)
            iden.data = (np.array([[1, 0], [0, 0]]),)
            return iden

        monkeypatch.setattr(
            qp.gradients.parameter_shift_cv,
            "_transform_observable",
            _mock_transform_observable,
        )

        dev = qp.device("default.gaussian", wires=1)

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(1.0, 0.0, wires=[0])
            qp.Rotation(2.0, wires=[0])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1, 2}

        qp.gradients.param_shift_cv(tape, dev, force_order2=False)
        spy.assert_not_called()

        qp.gradients.param_shift_cv(tape, dev, force_order2=True)
        spy.assert_called()

    def test_no_poly_xp_support(self, mocker, monkeypatch):
        """Test that if a device does not support PolyXP
        and the second-order parameter-shift rule is required,
        we fallback to finite differences."""
        spy_second_order = mocker.spy(qp.gradients.parameter_shift_cv, "second_order_param_shift")

        dev = qp.device("default.gaussian", wires=1)

        monkeypatch.delitem(dev._observable_map, "PolyXP")

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(1.0, wires=[0])
            qp.expval(qp.NumberOperator(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0}

        with pytest.warns(UserWarning, match="does not support the PolyXP observable"):
            qp.gradients.param_shift_cv(tape, dev)

        spy_second_order.assert_not_called()

    def test_no_poly_xp_support_variance(self, mocker, monkeypatch):
        """Test that if a device does not support PolyXP
        and the variance parameter-shift rule is required,
        we fallback to finite differences."""
        spy = mocker.spy(qp.gradients.parameter_shift_cv, "var_param_shift")
        dev = qp.device("default.gaussian", wires=1)

        monkeypatch.delitem(dev._observable_map, "PolyXP")

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rotation(1.0, wires=[0])
            qp.var(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0}

        with pytest.warns(UserWarning, match="does not support the PolyXP observable"):
            qp.gradients.param_shift_cv(tape, dev)

        spy.assert_not_called()

    def test_independent_parameters_analytic(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qp.device("default.gaussian", wires=2)

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(1.0, 0.0, wires=[1])
            qp.Displacement(1.0, 0.0, wires=[0])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 2}
        tapes, fn = qp.gradients.param_shift_cv(tape, dev)

        # We should only be executing the device to differentiate 1 parameter
        # (first order, so 2 executions)
        assert len(tapes) == 2

        res = fn(dev.batch_execute(tapes))
        assert np.allclose(res, [0, 2])

        tape.trainable_params = {0, 2}
        tapes, fn = qp.gradients.param_shift_cv(tape, dev, force_order2=True)

        # We should only be executing the device to differentiate 1 parameter
        # (second order, so 0 executions)
        assert len(tapes) == 0

        res = fn(dev.batch_execute(tapes))
        assert np.allclose(res, [0, 2])

    def test_all_independent(self):
        """Test the case where expectation values are independent of all parameters."""
        dev = qp.device("default.gaussian", wires=2)

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(1, 0, wires=[1])
            qp.Displacement(1, 0, wires=[1])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 2}
        tapes, fn = qp.gradients.param_shift_cv(tape, dev)
        assert len(tapes) == 0

        grad = fn(dev.batch_execute(tapes))
        assert np.allclose(grad, [0, 0])


class TestExpectationQuantumGradients:
    """Tests for the quantum gradients of various gates
    with expectation value output"""

    @pytest.mark.parametrize(
        "gradient_recipes",
        [None, ([[1 / np.sqrt(2), 1, np.pi / 4], [-1 / np.sqrt(2), 1, -np.pi / 4]],)],
    )
    def test_rotation_gradient(self, gradient_recipes, mocker, tol):
        """Test the gradient of the rotation gate"""
        dev = qp.device("default.gaussian", wires=2, hbar=hbar)

        alpha = 0.5643
        theta = 0.23354

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(alpha, 0.0, wires=[0])
            qp.Rotation(theta, wires=[0])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {2}

        spy2 = mocker.spy(qp.gradients.parameter_shift_cv, "second_order_param_shift")

        tapes, fn = param_shift_cv(tape, dev, gradient_recipes=gradient_recipes)
        grad_A = fn(dev.batch_execute(tapes))
        spy2.assert_not_called()

        tapes, fn = param_shift_cv(tape, dev, gradient_recipes=gradient_recipes, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))
        spy2.assert_called()

        expected = -hbar * alpha * np.sin(theta)
        assert np.allclose(grad_A, expected, atol=tol, rtol=0)
        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    def test_beamsplitter_gradient(self, mocker, tol):
        """Test the gradient of the beamsplitter gate"""
        dev = qp.device("default.gaussian", wires=2, hbar=hbar)

        alpha = 0.5643
        theta = 0.23354

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(alpha, 0.0, wires=[0])
            qp.Beamsplitter(theta, 0.0, wires=[0, 1])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {2}

        spy2 = mocker.spy(qp.gradients.parameter_shift_cv, "second_order_param_shift")

        tapes, fn = param_shift_cv(tape, dev)
        grad_A = fn(dev.batch_execute(tapes))
        spy2.assert_not_called()

        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))
        spy2.assert_called()

        expected = -hbar * alpha * np.sin(theta)
        assert np.allclose(grad_A, expected, atol=tol, rtol=0)
        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    def test_displacement_gradient(self, mocker, tol):
        """Test the gradient of the displacement gate"""
        dev = qp.device("default.gaussian", wires=2, hbar=hbar)

        r = 0.5643
        phi = 0.23354

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(r, phi, wires=[0])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}

        spy2 = mocker.spy(qp.gradients.parameter_shift_cv, "second_order_param_shift")

        tapes, fn = param_shift_cv(tape, dev)
        grad_A = fn(dev.batch_execute(tapes))
        spy2.assert_not_called()

        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))
        spy2.assert_called()

        expected = [hbar * np.cos(phi), -hbar * r * np.sin(phi)]
        assert np.allclose(grad_A, expected, atol=tol, rtol=0)
        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    def test_squeezed_gradient(self, mocker, tol):
        """Test the gradient of the squeezed gate. We also
        ensure that the gradient is correct even when an operation
        with no Heisenberg representation is a descendent."""
        dev = qp.device("default.gaussian", wires=2, hbar=hbar)

        # pylint: disable=too-few-public-methods
        class Rotation(qp.operation.CVOperation):
            """Dummy operation that does not support
            heisenberg representation"""

            num_wires = 1
            par_domain = "R"
            grad_method = "A"

        alpha = 0.5643
        r = 0.23354

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(alpha, 0.0, wires=[0])
            qp.Squeezing(r, 0.0, wires=[0])

            # The following two gates have no effect
            # on the circuit gradient and expectation value
            qp.Beamsplitter(0.0, 0.0, wires=[0, 1])
            Rotation(0.543, wires=[1])

            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {2}

        spy2 = mocker.spy(qp.gradients.parameter_shift_cv, "second_order_param_shift")

        tapes, fn = param_shift_cv(tape, dev)
        grad_A = fn(dev.batch_execute(tapes))
        spy2.assert_not_called()

        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))
        spy2.assert_called()

        expected = -np.exp(-r) * hbar * alpha
        assert np.allclose(grad_A, expected, atol=tol, rtol=0)
        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    def test_squeezed_number_state_gradient(self, mocker, tol):
        """Test the numerical gradient of the squeeze gate with
        with number state expectation is correct"""
        dev = qp.device("default.gaussian", wires=2, hbar=hbar)

        r = 0.23354

        with qp.queuing.AnnotatedQueue() as q:
            qp.Squeezing(r, 0.0, wires=[0])
            # the fock state projector is a 'non-Gaussian' observable
            qp.expval(qp.FockStateProjector(np.array([2, 0]), wires=[0, 1]))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0}

        spy = mocker.spy(qp.gradients.parameter_shift_cv, "second_order_param_shift")
        spy2 = mocker.spy(qp.gradients.parameter_shift_cv, "_find_gradient_methods_cv")

        tapes, fn = param_shift_cv(tape, dev)
        grad = fn(dev.batch_execute(tapes))

        spy.assert_not_called()
        assert spy2.spy_return == {0: "F"}

        # (d/dr) |<2|S(r)>|^2 = 0.5 tanh(r)^3 (2 csch(r)^2 - 1) sech(r)
        expected = 0.5 * np.tanh(r) ** 3 * (2 / (np.sinh(r) ** 2) - 1) / np.cosh(r)
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_multiple_squeezing_gradient(self, mocker, tol):
        """Test that the gradient of a circuit with two squeeze
        gates is correct."""
        dev = qp.device("default.gaussian", wires=2, hbar=hbar)

        r0, phi0, r1, phi1 = [0.4, -0.3, -0.7, 0.2]

        with qp.queuing.AnnotatedQueue() as q:
            qp.Squeezing(r0, phi0, wires=[0])
            qp.Squeezing(r1, phi1, wires=[0])
            qp.expval(qp.NumberOperator(0))  # second order

        tape = qp.tape.QuantumScript.from_queue(q)
        spy2 = mocker.spy(qp.gradients.parameter_shift_cv, "second_order_param_shift")
        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))
        spy2.assert_called()

        # check against the known analytic formula
        expected = np.zeros([4])
        expected[0] = np.cosh(2 * r1) * np.sinh(2 * r0) + np.cos(phi0 - phi1) * np.cosh(
            2 * r0
        ) * np.sinh(2 * r1)
        expected[1] = -0.5 * np.sin(phi0 - phi1) * np.sinh(2 * r0) * np.sinh(2 * r1)
        expected[2] = np.cos(phi0 - phi1) * np.cosh(2 * r1) * np.sinh(2 * r0) + np.cosh(
            2 * r0
        ) * np.sinh(2 * r1)
        expected[3] = 0.5 * np.sin(phi0 - phi1) * np.sinh(2 * r0) * np.sinh(2 * r1)

        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    def test_multiple_second_order_observables(self):
        """Test that the gradient of a circuit with multiple
        second order observables is correct"""

        dev = qp.device("default.gaussian", wires=2, hbar=hbar)
        r = [0.4, -0.7, 0.1, 0.2]
        p = [0.1, 0.2, 0.3, 0.4]

        with qp.queuing.AnnotatedQueue() as q:
            qp.Squeezing(r[0], p[0], wires=[0])
            qp.Squeezing(r[1], p[1], wires=[0])
            qp.Squeezing(r[2], p[2], wires=[1])
            qp.Squeezing(r[3], p[3], wires=[1])
            qp.expval(qp.NumberOperator(0))  # second order
            qp.expval(qp.NumberOperator(1))  # second order

        tape = qp.tape.QuantumScript.from_queue(q)

        with pytest.raises(
            ValueError, match="Computing the gradient of CV circuits that return more"
        ):
            param_shift_cv(tape, dev)

    @pytest.mark.parametrize("obs", [qp.QuadP, qp.Identity])
    @pytest.mark.parametrize(
        "op", [qp.Displacement(0.1, 0.2, wires=0), qp.TwoModeSqueezing(0.1, 0.2, wires=[0, 1])]
    )
    def test_gradients_gaussian_circuit(self, mocker, op, obs, tol):
        """Tests that the gradients of circuits of gaussian gates match between the
        finite difference and analytic methods."""
        tol = 1e-2

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(0.5, 0, wires=0)
            qp.apply(op)
            qp.Beamsplitter(1.3, -2.3, wires=[0, 1])
            qp.Displacement(-0.5, 0.1, wires=0)
            qp.Squeezing(0.5, -1.5, wires=0)
            qp.Rotation(-1.1, wires=0)
            qp.expval(obs(wires=0))

        tape = qp.tape.QuantumScript.from_queue(q)
        dev = qp.device("default.gaussian", wires=2)

        tape.trainable_params = set(range(2, 2 + op.num_params))

        spy = mocker.spy(qp.gradients.parameter_shift_cv, "_find_gradient_methods_cv")

        tapes, fn = qp.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))

        for method in spy.spy_return.values():
            assert method == "A"

        assert np.allclose(grad_A2, grad_F, atol=tol, rtol=0)

        if obs.ev_order == 1:
            tapes, fn = param_shift_cv(tape, dev)
            grad_A = fn(dev.batch_execute(tapes))
            assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize("t", [0, 1])
    def test_interferometer_unitary(self, mocker, t, tol):
        """An integration test for CV gates that support analytic differentiation
        if succeeding the gate to be differentiated, but cannot be differentiated
        themselves (for example, they may be Gaussian but accept no parameters,
        or may accept a numerical array parameter.).

        This ensures that, assuming their _heisenberg_rep is defined, the quantum
        gradient analytic method can still be used, and returns the correct result.

        Currently, the only such operation is qp.InterferometerUnitary. In the future,
        we may consider adding a qp.GaussianTransfom operator.
        """

        if t == 1:
            pytest.xfail(
                "There is a bug in the second order CV parameter-shift rule; "
                "phase arguments return the incorrect derivative."
            )

            # Note: this bug currently affects PL core as well:
            #
            # dev = qp.device("default.gaussian", wires=2)
            #
            # U = np.array([[ 0.51310276+0.81702166j,  0.13649626+0.22487759j],
            #         [ 0.26300233+0.00556194j, -0.96414101-0.03508489j]])
            #
            # @qp.qnode(dev)
            # def circuit(r, phi):
            #     qp.Displacement(r, phi, wires=0)
            #     qp.InterferometerUnitary(U, wires=[0, 1])
            #     return qp.expval(qp.QuadX(0))
            #
            # r = 0.543
            # phi = 0.
            #
            # >>> print(circuit.jacobian([r, phi], options={"force_order2":False}))
            # [[ 1.02620552 0.14823494]]
            # >>> print(circuit.jacobian([r, phi], options={"force_order2":True}))
            # [[ 1.02620552 -0.88728552]]

        U = np.array(
            [
                [0.51310276 + 0.81702166j, 0.13649626 + 0.22487759j],
                [0.26300233 + 0.00556194j, -0.96414101 - 0.03508489j],
            ],
            requires_grad=False,
        )

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(0.543, 0, wires=0)
            qp.InterferometerUnitary(U, wires=[0, 1])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {t}

        dev = qp.device("default.gaussian", wires=2)

        spy = mocker.spy(qp.gradients.parameter_shift_cv, "_find_gradient_methods_cv")

        tapes, fn = qp.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev)
        grad_A = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))

        assert spy.spy_return[0] == "A"

        # the different methods agree
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)
        assert np.allclose(grad_A2, grad_F, atol=tol, rtol=0)


class TestVarianceQuantumGradients:
    """Tests for the quantum gradients of various gates
    with variance measurements"""

    def test_first_order_observable(self, tol):
        """Test variance of a first order CV observable"""
        dev = qp.device("default.gaussian", wires=1)

        r = 0.543
        phi = -0.654

        with qp.queuing.AnnotatedQueue() as q:
            qp.Squeezing(r, 0, wires=0)
            qp.Rotation(phi, wires=0)
            qp.var(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 2}

        res = qp.execute([tape], dev, None)
        expected = np.exp(2 * r) * np.sin(phi) ** 2 + np.exp(-2 * r) * np.cos(phi) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qp.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev)
        grad_A = fn(dev.batch_execute(tapes))

        expected = np.array(
            [
                [
                    2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                    2 * np.sinh(2 * r) * np.sin(2 * phi),
                ]
            ]
        )
        assert np.allclose(grad_A, expected, atol=tol, rtol=0)
        assert np.allclose(grad_F, expected, atol=tol, rtol=0)

    def test_second_order_cv(self, tol):
        """Test variance of a second order CV expectation value"""
        dev = qp.device("default.gaussian", wires=1)

        n = 0.12
        a = 0.765

        with qp.queuing.AnnotatedQueue() as q:
            qp.ThermalState(n, wires=0)
            qp.Displacement(a, 0, wires=0)
            qp.var(qp.NumberOperator(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}

        res = qp.execute([tape], dev, None)
        expected = n**2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qp.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        expected = np.array([[2 * a**2 + 2 * n + 1, 2 * a * (2 * n + 1)]])
        assert np.allclose(grad_F, expected, atol=tol, rtol=0)

    def test_expval_and_variance(self):
        """Test that the gradient works for a combination of CV expectation
        values and variances"""
        dev = qp.device("default.gaussian", wires=3)

        a, b = [0.54, -0.423]

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(0.5, 0, wires=0)
            qp.Squeezing(a, 0, wires=0)
            qp.Squeezing(b, 0, wires=1)
            qp.Beamsplitter(0.6, -0.3, wires=[0, 1])
            qp.Squeezing(-0.3, 0, wires=2)
            qp.Beamsplitter(1.4, 0.5, wires=[1, 2])
            qp.var(qp.QuadX(0))
            qp.expval(qp.QuadX(1))
            qp.var(qp.QuadX(2))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {2, 4}

        with pytest.raises(
            ValueError, match="Computing the gradient of CV circuits that return more"
        ):
            param_shift_cv(tape, dev)

    def test_error_analytic_second_order(self):
        """Test exception raised if attempting to use a second
        order observable to compute the variance derivative analytically"""
        dev = qp.device("default.gaussian", wires=1)

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(1.0, 0, wires=0)
            qp.var(qp.NumberOperator(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0}

        with pytest.raises(ValueError, match=r"cannot be used with the parameter\(s\) \[0\]"):
            param_shift_cv(tape, dev, fallback_fn=None)

    @mock.patch("pennylane.gradients.parameter_shift_cv._grad_method_cv", return_value="A")
    def test_error_unsupported_grad_recipe(self, _):
        """Test exception raised if attempting to use the second order rule for
        computing the gradient analytically of an expectation value that
        contains an operation with more than two terms in the gradient recipe"""

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.operation.CVOperation):
            """Dummy op"""

            num_wires = 1
            par_domain = "R"
            grad_method = "A"
            grad_recipe = ([[1, 1, 1], [1, 1, 1], [1, 1, 1]],)

        dev = qp.device("default.gaussian", wires=1)

        dev.operations.add(DummyOp)

        with qp.queuing.AnnotatedQueue() as q:
            DummyOp(1, wires=[0])
            qp.expval(qp.QuadX(0))

        tape = qp.tape.QuantumScript.from_queue(q)

        with pytest.raises(
            NotImplementedError, match=r"analytic gradient for order-2 operators is unsupported"
        ):
            param_shift_cv(tape, dev, force_order2=True)

    @pytest.mark.parametrize("obs", [qp.QuadX, qp.NumberOperator])
    @pytest.mark.parametrize(
        "op", [qp.Squeezing(0.1, 0.2, wires=0), qp.Beamsplitter(0.1, 0.2, wires=[0, 1])]
    )
    def test_gradients_gaussian_circuit(self, mocker, op, obs, tol):
        """Tests that the gradients of circuits of selected gaussian gates match between the
        finite difference and analytic methods."""
        tol = 1e-2

        with qp.queuing.AnnotatedQueue() as q:
            qp.Displacement(0.5, 0, wires=0)
            qp.apply(op)
            qp.Beamsplitter(1.3, -2.3, wires=[0, 1])
            qp.Displacement(-0.5, 0.1, wires=0)
            qp.Squeezing(0.5, -1.5, wires=0)
            qp.Rotation(-1.1, wires=0)
            qp.var(obs(wires=0))

        tape = qp.tape.QuantumScript.from_queue(q)
        dev = qp.device("default.gaussian", wires=2)

        tape.trainable_params = set(range(2, 2 + op.num_params))

        spy = mocker.spy(qp.gradients.parameter_shift_cv, "_find_gradient_methods_cv")

        # jacobians must match
        tapes, fn = qp.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev)
        grad_A = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))

        assert np.allclose(grad_A2, grad_F, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

        if obs != qp.NumberOperator:
            assert all(m == "A" for m in spy.spy_return.values())

    def test_squeezed_mean_photon_variance(self, tol):
        """Test gradient of the photon variance of a displaced thermal state"""
        dev = qp.device("default.gaussian", wires=1)

        r = 0.12
        phi = 0.105

        with qp.queuing.AnnotatedQueue() as q:
            qp.Squeezing(r, 0, wires=0)
            qp.Rotation(phi, wires=0)
            qp.var(qp.QuadX(wires=[0]))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 2}
        tapes, fn = param_shift_cv(tape, dev)
        grad = fn(dev.batch_execute(tapes))
        expected = np.array(
            [
                2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                2 * np.sinh(2 * r) * np.sin(2 * phi),
            ]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_displaced_thermal_mean_photon_variance(self, tol):
        """Test gradient of the photon variance of a displaced thermal state"""
        dev = qp.device("default.gaussian", wires=1)

        n = 0.12
        a = 0.105

        with qp.queuing.AnnotatedQueue() as q:
            qp.ThermalState(n, wires=0)
            qp.Displacement(a, 0, wires=0)
            qp.var(qp.TensorN(wires=[0]))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}
        tapes, fn = param_shift_cv(tape, dev)
        grad = fn(dev.batch_execute(tapes))
        expected = np.array([2 * a**2 + 2 * n + 1, 2 * a * (2 * n + 1)])
        assert np.allclose(grad, expected, atol=tol, rtol=0)


class TestParamShiftInterfaces:
    """Test that the transform is differentiable"""

    @pytest.mark.autograd
    def test_autograd_gradient(self, tol):
        """Tests that the output of the parameter-shift CV transform
        can be differentiated using autograd."""
        dev = qp.device("default.gaussian", wires=1)

        r = 0.12
        phi = 0.105

        @qp.qnode(device=dev, interface="autograd", max_diff=2)
        def cost_fn(x):
            qp.Squeezing(x[0], 0, wires=0)
            qp.Rotation(x[1], wires=0)
            return qp.var(qp.QuadX(wires=[0]))

        params = np.array([r, phi], requires_grad=True)

        grad = qp.jacobian(cost_fn)(params)
        expected = np.array(
            [
                2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                2 * np.sinh(2 * r) * np.sin(2 * phi),
            ]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests that the output of the parameter-shift CV transform
        can be executed using TF"""
        import tensorflow as tf

        dev = qp.device("default.gaussian", wires=1)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape() as t:
            with qp.queuing.AnnotatedQueue() as q:
                qp.Squeezing(params[0], 0, wires=0)
                qp.Rotation(params[1], wires=0)
                qp.var(qp.QuadX(wires=[0]))

            tape = qp.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 2}
            tapes, fn = param_shift_cv(tape, dev)
            jac = fn(
                qp.execute(
                    tapes, dev, param_shift_cv, gradient_kwargs={"dev": dev}, interface="tf"
                )
            )
            res = jac[1]

        r, phi = 1.0 * params

        expected = np.array(
            [
                2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                2 * np.sinh(2 * r) * np.sin(2 * phi),
            ]
        )
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        grad = t.jacobian(res, params)
        expected = np.array(
            [4 * np.cosh(2 * r) * np.sin(2 * phi), 4 * np.cos(2 * phi) * np.sinh(2 * r)]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests that the output of the parameter-shift CV transform
        can be executed using Torch."""
        import torch

        dev = qp.device("default.gaussian", wires=1)
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.Squeezing(params[0], 0, wires=0)
            qp.Rotation(params[1], wires=0)
            qp.var(qp.QuadX(wires=[0]))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 2}
        tapes, fn = qp.gradients.param_shift_cv(tape, dev)
        jac = fn(
            qp.execute(tapes, dev, param_shift_cv, gradient_kwargs={"dev": dev}, interface="torch")
        )

        r, phi = params.detach().numpy()

        expected = np.array(
            [
                2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                2 * np.sinh(2 * r) * np.sin(2 * phi),
            ]
        )
        assert np.allclose(jac[0].detach().numpy(), expected[0], atol=tol, rtol=0)
        assert np.allclose(jac[1].detach().numpy(), expected[1], atol=tol, rtol=0)

        cost = jac[1]
        cost.backward()
        hess = params.grad
        expected = np.array(
            [4 * np.cosh(2 * r) * np.sin(2 * phi), 4 * np.cos(2 * phi) * np.sinh(2 * r)]
        )

        assert np.allclose(hess.detach().numpy(), expected, atol=0.1, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests that the output of the parameter-shift CV transform
        can be differentiated using JAX, yielding second derivatives."""
        import jax
        from jax import numpy as jnp

        dev = qp.device("default.gaussian", wires=1)
        params = jnp.array([0.543, -0.654])

        def cost_fn(x):
            with qp.queuing.AnnotatedQueue() as q:
                qp.Squeezing(x[0], 0, wires=0)
                qp.Rotation(x[1], wires=0)
                qp.var(qp.QuadX(wires=[0]))

            tape = qp.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 2}
            tapes, fn = qp.gradients.param_shift_cv(tape, dev)
            jac = fn(
                qp.execute(
                    tapes, dev, param_shift_cv, gradient_kwargs={"dev": dev}, interface="jax"
                )
            )
            return jac

        r, phi = params
        res = cost_fn(params)
        expected = np.array(
            [
                2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                2 * np.sinh(2 * r) * np.sin(2 * phi),
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.jacobian(cost_fn)(params)

        expected = np.array(
            [4 * np.cosh(2 * r) * np.sin(2 * phi), 4 * np.cos(2 * phi) * np.sinh(2 * r)]
        )

        assert np.allclose(res[1], expected, atol=tol, rtol=0)

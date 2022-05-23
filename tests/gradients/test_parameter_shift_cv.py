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
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift_cv
from pennylane.gradients.parameter_shift_cv import (
    _grad_method,
    _gradient_analysis_cv,
    _transform_observable,
)


hbar = 2


class TestGradAnalysis:
    """Tests for parameter gradient methods"""

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is
        correctly marked"""

        with qml.tape.QuantumTape() as tape:
            qml.FockState(1, wires=0)
            qml.Displacement(0.543, 0, wires=[1])
            qml.Beamsplitter(0, 0, wires=[0, 1])
            qml.expval(qml.X(wires=[0]))

        assert _grad_method(tape, 0) is None
        assert _grad_method(tape, 1) == "A"
        assert _grad_method(tape, 2) == "A"
        assert _grad_method(tape, 3) == "A"
        assert _grad_method(tape, 4) == "A"

        _gradient_analysis_cv(tape)

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "A"
        assert tape._par_info[2]["grad_method"] == "A"
        assert tape._par_info[3]["grad_method"] == "A"
        assert tape._par_info[4]["grad_method"] == "A"

        _gradient_analysis_cv(tape)

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(0.543, wires=[0])
            qml.Rotation(-0.654, wires=[1])
            qml.expval(qml.P(0))

        assert _grad_method(tape, 0) == "A"
        assert _grad_method(tape, 1) == "0"

        _gradient_analysis_cv(tape)

        assert tape._par_info[0]["grad_method"] == "A"
        assert tape._par_info[1]["grad_method"] == "0"

    def test_finite_diff(self, monkeypatch):
        """If an op has grad_method=F, this should be respected
        by the qml.tape.QuantumTape"""
        monkeypatch.setattr(qml.Rotation, "grad_method", "F")

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(0.543, wires=[0])
            qml.Squeezing(0.543, 0, wires=[0])
            qml.expval(qml.P(0))

        assert _grad_method(tape, 0) == "F"
        assert _grad_method(tape, 1) == "A"
        assert _grad_method(tape, 2) == "A"

    def test_non_gaussian_operation(self):
        """Test that a non-Gaussian operation succeeding
        a differentiable Gaussian operation results in
        numeric differentiation."""

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.Rotation(1.0, wires=[1])
            # Non-Gaussian
            qml.Kerr(1.0, wires=[1])
            qml.expval(qml.P(0))
            qml.expval(qml.X(1))

        # First rotation gate has no succeeding non-Gaussian operation
        assert _grad_method(tape, 0) == "A"
        # Second rotation gate does no succeeding non-Gaussian operation
        assert _grad_method(tape, 1) == "F"
        # Kerr gate does not support the parameter-shift rule
        assert _grad_method(tape, 2) == "F"

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.Rotation(1.0, wires=[1])
            # entangle the modes
            qml.Beamsplitter(1.0, 0.0, wires=[0, 1])
            # Non-Gaussian
            qml.Kerr(1.0, wires=[1])
            qml.expval(qml.P(0))
            qml.expval(qml.X(1))

        # After entangling the modes, the Kerr gate now succeeds
        # both initial rotations
        assert _grad_method(tape, 0) == "F"
        assert _grad_method(tape, 1) == "F"
        assert _grad_method(tape, 2) == "F"

    def test_probability(self):
        """Probability is the expectation value of a
        higher order observable, and thus only supports numerical
        differentiation"""
        with qml.tape.QuantumTape() as tape:
            qml.Rotation(0.543, wires=[0])
            qml.Squeezing(0.543, 0, wires=[0])
            qml.probs(wires=0)

        assert _grad_method(tape, 0) == "F"
        assert _grad_method(tape, 1) == "F"
        assert _grad_method(tape, 2) == "F"

    def test_variance(self):
        """If the variance of the observable is first order, then
        parameter-shift is supported. If the observable is second order,
        however, only finite-differences is supported."""

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.var(qml.P(0))  # first order

        assert _grad_method(tape, 0) == "A"

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.var(qml.NumberOperator(0))  # second order

        assert _grad_method(tape, 0) == "F"

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.Rotation(1.0, wires=[1])
            qml.Beamsplitter(0.5, 0.0, wires=[0, 1])
            qml.var(qml.NumberOperator(0))  # fourth order
            qml.expval(qml.NumberOperator(1))

        assert _grad_method(tape, 0) == "F"
        assert _grad_method(tape, 1) == "F"
        assert _grad_method(tape, 2) == "F"
        assert _grad_method(tape, 3) == "F"

    def test_second_order_expectation(self):
        """Test that the expectation of a second-order observable forces
        the gradient method to use the second-order parameter-shift rule"""

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.expval(qml.NumberOperator(0))  # second order

        assert _grad_method(tape, 0) == "A2"

    def test_unknown_op_grad_method(self, monkeypatch):
        """Test that an exception is raised if an operator has a
        grad method defined that the CV parameter-shift tape
        doesn't recognize"""
        monkeypatch.setattr(qml.Rotation, "grad_method", "B")

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(1.0, wires=0)
            qml.expval(qml.X(0))

        with pytest.raises(ValueError, match="unknown gradient method"):
            _grad_method(tape, 0)


class TestTransformObservable:
    """Tests for the _transform_observable method"""

    def test_incorrect_heisenberg_size(self, monkeypatch):
        """The number of dimensions of a CV observable Heisenberg representation does
        not match the ev_order attribute."""
        monkeypatch.setattr(qml.P, "ev_order", 2)

        with pytest.raises(ValueError, match="Mismatch between the polynomial order"):
            _transform_observable(qml.P(0), np.identity(3), device_wires=[0])

    def test_higher_order_observable(self, monkeypatch):
        """An exception should be raised if the observable is higher than 2nd order."""
        monkeypatch.setattr(qml.P, "ev_order", 3)

        with pytest.raises(NotImplementedError, match="order > 2 not implemented"):
            _transform_observable(qml.P(0), np.identity(3), device_wires=[0])

    def test_first_order_transform(self, tol):
        """Test that a first order observable is transformed correctly"""
        # create a symmetric transformation
        Z = np.arange(3**2).reshape(3, 3)
        Z = Z.T + Z

        obs = qml.X(0)
        res = _transform_observable(obs, Z, device_wires=[0])

        # The Heisenberg representation of the X
        # operator is simply... X
        expected = np.array([0, 1, 0]) @ Z

        assert isinstance(res, qml.PolyXP)
        assert res.wires.labels == (0,)
        assert np.allclose(res.data[0], expected, atol=tol, rtol=0)

    def test_second_order_transform(self, tol):
        """Test that a second order observable is transformed correctly"""
        # create a symmetric transformation
        Z = np.arange(3**2).reshape(3, 3)
        Z = Z.T + Z

        obs = qml.NumberOperator(0)
        res = _transform_observable(obs, Z, device_wires=[0])

        # The Heisenberg representation of the number operator
        # is (X^2 + P^2) / (2*hbar) - 1/2
        A = np.array([[-0.5, 0, 0], [0, 0.25, 0], [0, 0, 0.25]])
        expected = A @ Z + Z @ A

        assert isinstance(res, qml.PolyXP)
        assert res.wires.labels == (0,)
        assert np.allclose(res.data[0], expected, atol=tol, rtol=0)

    def test_device_wire_expansion(self, tol):
        """Test that the transformation works correctly
        for the case where the transformation applies to more wires
        than the observable."""

        # create a 3-mode symmetric transformation
        wires = qml.wires.Wires([0, "a", 2])
        ndim = 1 + 2 * len(wires)

        Z = np.arange(ndim**2).reshape(ndim, ndim)
        Z = Z.T + Z

        obs = qml.NumberOperator(0)
        res = _transform_observable(obs, Z, device_wires=wires)

        # The Heisenberg representation of the number operator
        # is (X^2 + P^2) / (2*hbar) - 1/2. We use the ordering
        # I, X0, Xa, X2, P0, Pa, P2.
        A = np.diag([-0.5, 0.25, 0.25, 0, 0, 0, 0])
        expected = A @ Z + Z @ A

        assert isinstance(res, qml.PolyXP)
        assert res.wires == wires
        assert np.allclose(res.data[0], expected, atol=tol, rtol=0)


class TestParameterShiftLogic:
    """Test for the dispatching logic of the parameter shift method"""

    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            qml.Displacement(weights[0], 0.0, wires=[0])
            qml.Rotation(weights[1], wires=[0])
            return qml.expval(qml.X(0))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
            res = qml.gradients.param_shift_cv(circuit)(weights)

        assert res == ()

    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(weights):
            qml.Displacement(weights[0], 0.0, wires=[0])
            qml.Rotation(weights[1], wires=[0])
            return qml.expval(qml.X(0))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
            res = qml.gradients.param_shift_cv(circuit)(weights)

        assert res == ()

    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit(weights):
            qml.Displacement(weights[0], 0.0, wires=[0])
            qml.Rotation(weights[1], wires=[0])
            return qml.expval(qml.X(0))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
            res = qml.gradients.param_shift_cv(circuit)(weights)

        assert res == ()

    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(weights):
            qml.Displacement(weights[0], 0.0, wires=[0])
            qml.Rotation(weights[1], wires=[0])
            return qml.expval(qml.X(0))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
            res = qml.gradients.param_shift_cv(circuit)(weights)

        assert res == ()

    def test_no_trainable_params_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.gaussian", wires=2)

        weights = [0.1, 0.2]
        with qml.tape.QuantumTape() as tape:
            qml.Displacement(weights[0], 0.0, wires=[0])
            qml.Rotation(weights[1], wires=[0])
            qml.expval(qml.X(0))

        # TODO: remove once #2155 is resolved
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.param_shift_cv(tape, dev)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert res.shape == (1, 0)

    def test_all_zero_diff_methods(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rotation(params[0], wires=0)
            return qml.expval(qml.X(1))

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qml.gradients.param_shift_cv(circuit, dev)(params)
        assert np.allclose(result, np.zeros((2, 3)), atol=0, rtol=0)

        tapes, _ = qml.gradients.param_shift_cv(circuit.tape, dev)
        assert tapes == []

    def test_state_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a state"""
        with qml.tape.QuantumTape() as tape:
            qml.state()

        with pytest.raises(ValueError, match=r"return the state is not supported"):
            qml.gradients.param_shift_cv(tape, None)

    def test_force_order2(self, mocker):
        """Test that if the force_order2 keyword argument is provided,
        the second order parameter shift rule is forced"""
        spy = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")

        dev = qml.device("default.gaussian", wires=1)

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(1.0, 0.0, wires=[0])
            qml.Rotation(2.0, wires=[0])
            qml.expval(qml.X(0))

        tape.trainable_params = {0, 1, 2}

        qml.gradients.param_shift_cv(tape, dev, force_order2=False)
        spy.assert_not_called()

        qml.gradients.param_shift_cv(tape, dev, force_order2=True)
        spy.assert_called()

    def test_force_order2_dim_2(self, mocker, monkeypatch):
        """Test that if the force_order2 keyword argument is provided, the
        second order parameter shift rule is forced for an observable with
        2-dimensional parameters"""
        spy = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")

        def _mock_transform_observable(obs, Z, device_wires):
            """A mock version of the _transform_observable internal function
            such that an operator ``transformed_obs`` of two-dimensions is
            returned. This function is created such that when definining ``A =
            transformed_obs.parameters[0]`` the condition ``len(A.nonzero()[0])
            == 1 and A.ndim == 2 and A[0, 0] != 0`` is ``True``."""
            iden = qml.Identity(0)
            iden.data = [np.array([[1, 0], [0, 0]])]
            return iden

        monkeypatch.setattr(
            qml.gradients.parameter_shift_cv,
            "_transform_observable",
            _mock_transform_observable,
        )

        dev = qml.device("default.gaussian", wires=1)

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(1.0, 0.0, wires=[0])
            qml.Rotation(2.0, wires=[0])
            qml.expval(qml.X(0))

        tape.trainable_params = {0, 1, 2}

        qml.gradients.param_shift_cv(tape, dev, force_order2=False)
        spy.assert_not_called()

        qml.gradients.param_shift_cv(tape, dev, force_order2=True)
        spy.assert_called()

    def test_no_poly_xp_support(self, mocker, monkeypatch, caplog):
        """Test that if a device does not support PolyXP
        and the second-order parameter-shift rule is required,
        we fallback to finite differences."""
        spy_second_order = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")

        dev = qml.device("default.gaussian", wires=1)

        monkeypatch.delitem(dev._observable_map, "PolyXP")

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.expval(qml.NumberOperator(0))

        tape.trainable_params = {0}

        with pytest.warns(UserWarning, match="does not support the PolyXP observable"):
            qml.gradients.param_shift_cv(tape, dev)

        spy_second_order.assert_not_called()

    def test_no_poly_xp_support_variance(self, mocker, monkeypatch, caplog):
        """Test that if a device does not support PolyXP
        and the variance parameter-shift rule is required,
        we fallback to finite differences."""
        spy = mocker.spy(qml.gradients.parameter_shift_cv, "var_param_shift")
        dev = qml.device("default.gaussian", wires=1)

        monkeypatch.delitem(dev._observable_map, "PolyXP")

        with qml.tape.QuantumTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.var(qml.X(0))

        tape.trainable_params = {0}

        with pytest.warns(UserWarning, match="does not support the PolyXP observable"):
            qml.gradients.param_shift_cv(tape, dev)

        spy.assert_not_called()

    def test_independent_parameters_analytic(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.gaussian", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(1.0, 0.0, wires=[1])
            qml.Displacement(1.0, 0.0, wires=[0])
            qml.expval(qml.X(0))

        tape.trainable_params = {0, 2}
        tapes, fn = qml.gradients.param_shift_cv(tape, dev)

        # We should only be executing the device to differentiate 1 parameter
        # (first order, so 2 executions)
        assert len(tapes) == 2

        res = fn(dev.batch_execute(tapes))
        assert np.allclose(res, [0, 2])

        tape.trainable_params = {0, 2}
        tapes, fn = qml.gradients.param_shift_cv(tape, dev, force_order2=True)

        # We should only be executing the device to differentiate 1 parameter
        # (second order, so 0 executions)
        assert len(tapes) == 0

        res = fn(dev.batch_execute(tapes))
        assert np.allclose(res, [0, 2])

    def test_all_independent(self):
        """Test the case where expectation values are independent of all parameters."""
        dev = qml.device("default.gaussian", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(1, 0, wires=[1])
            qml.Displacement(1, 0, wires=[1])
            qml.expval(qml.X(0))

        tape.trainable_params = {0, 2}
        tapes, fn = qml.gradients.param_shift_cv(tape, dev)
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
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        alpha = 0.5643
        theta = 0.23354

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(alpha, 0.0, wires=[0])
            qml.Rotation(theta, wires=[0])
            qml.expval(qml.X(0))

        tape.trainable_params = {2}

        spy2 = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")

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
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        alpha = 0.5643
        theta = 0.23354

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(alpha, 0.0, wires=[0])
            qml.Beamsplitter(theta, 0.0, wires=[0, 1])
            qml.expval(qml.X(0))

        tape.trainable_params = {2}

        spy2 = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")

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
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        r = 0.5643
        phi = 0.23354

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(r, phi, wires=[0])
            qml.expval(qml.X(0))

        tape.trainable_params = {0, 1}

        spy2 = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")

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
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        class Rotation(qml.operation.CVOperation):
            """Dummy operation that does not support
            heisenberg representation"""

            num_wires = 1
            par_domain = "R"
            grad_method = "A"

        alpha = 0.5643
        r = 0.23354

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(alpha, 0.0, wires=[0])
            qml.Squeezing(r, 0.0, wires=[0])

            # The following two gates have no effect
            # on the circuit gradient and expectation value
            qml.Beamsplitter(0.0, 0.0, wires=[0, 1])
            Rotation(0.543, wires=[1])

            qml.expval(qml.X(0))

        tape.trainable_params = {2}

        spy2 = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")

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
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        r = 0.23354

        with qml.tape.QuantumTape() as tape:
            qml.Squeezing(r, 0.0, wires=[0])
            # the fock state projector is a 'non-Gaussian' observable
            qml.expval(qml.FockStateProjector(np.array([2, 0]), wires=[0, 1]))

        tape.trainable_params = {0}

        spy = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")

        tapes, fn = param_shift_cv(tape, dev)
        grad = fn(dev.batch_execute(tapes))
        assert tape._par_info[0]["grad_method"] == "F"

        spy.assert_not_called()

        # (d/dr) |<2|S(r)>|^2 = 0.5 tanh(r)^3 (2 csch(r)^2 - 1) sech(r)
        expected = 0.5 * np.tanh(r) ** 3 * (2 / (np.sinh(r) ** 2) - 1) / np.cosh(r)
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_multiple_squeezing_gradient(self, mocker, tol):
        """Test that the gradient of a circuit with two squeeze
        gates is correct."""
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        r0, phi0, r1, phi1 = [0.4, -0.3, -0.7, 0.2]

        with qml.tape.QuantumTape() as tape:
            qml.Squeezing(r0, phi0, wires=[0])
            qml.Squeezing(r1, phi1, wires=[0])
            qml.expval(qml.NumberOperator(0))  # second order

        spy2 = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")
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

    def test_multiple_second_order_observables(self, mocker, tol):
        """Test that the gradient of a circuit with multiple
        second order observables is correct"""

        dev = qml.device("default.gaussian", wires=2, hbar=hbar)
        r = [0.4, -0.7, 0.1, 0.2]
        p = [0.1, 0.2, 0.3, 0.4]

        with qml.tape.QuantumTape() as tape:
            qml.Squeezing(r[0], p[0], wires=[0])
            qml.Squeezing(r[1], p[1], wires=[0])
            qml.Squeezing(r[2], p[2], wires=[1])
            qml.Squeezing(r[3], p[3], wires=[1])
            qml.expval(qml.NumberOperator(0))  # second order
            qml.expval(qml.NumberOperator(1))  # second order

        spy2 = mocker.spy(qml.gradients.parameter_shift_cv, "second_order_param_shift")
        tapes, fn = param_shift_cv(tape, dev)
        grad_A2 = fn(dev.batch_execute(tapes))
        spy2.assert_called()

        # check against the known analytic formula

        def expected_grad(r, p):
            return np.array(
                [
                    np.cosh(2 * r[1]) * np.sinh(2 * r[0])
                    + np.cos(p[0] - p[1]) * np.cosh(2 * r[0]) * np.sinh(2 * r[1]),
                    -0.5 * np.sin(p[0] - p[1]) * np.sinh(2 * r[0]) * np.sinh(2 * r[1]),
                    np.cos(p[0] - p[1]) * np.cosh(2 * r[1]) * np.sinh(2 * r[0])
                    + np.cosh(2 * r[0]) * np.sinh(2 * r[1]),
                    0.5 * np.sin(p[0] - p[1]) * np.sinh(2 * r[0]) * np.sinh(2 * r[1]),
                ]
            )

        expected = np.zeros([2, 8])
        expected[0, :4] = expected_grad(r[:2], p[:2])
        expected[1, 4:] = expected_grad(r[2:], p[2:])

        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("obs", [qml.P, qml.Identity])
    @pytest.mark.parametrize(
        "op", [qml.Displacement(0.1, 0.2, wires=0), qml.TwoModeSqueezing(0.1, 0.2, wires=[0, 1])]
    )
    def test_gradients_gaussian_circuit(self, op, obs, tol):
        """Tests that the gradients of circuits of gaussian gates match between the
        finite difference and analytic methods."""
        tol = 1e-2

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(0.5, 0, wires=0)
            qml.apply(op)
            qml.Beamsplitter(1.3, -2.3, wires=[0, 1])
            qml.Displacement(-0.5, 0.1, wires=0)
            qml.Squeezing(0.5, -1.5, wires=0)
            qml.Rotation(-1.1, wires=0)
            qml.expval(obs(wires=0))

        dev = qml.device("default.gaussian", wires=2)
        res = qml.execute([tape], dev, None)

        tape.trainable_params = set(range(2, 2 + op.num_params))

        tapes, fn = qml.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))

        # check that every parameter is analytic
        for i in range(op.num_params):
            assert tape._par_info[2 + i]["grad_method"][0] == "A"

        assert np.allclose(grad_A2, grad_F, atol=tol, rtol=0)

        if obs.ev_order == 1:
            tapes, fn = param_shift_cv(tape, dev)
            grad_A = fn(dev.batch_execute(tapes))
            assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize("t", [0, 1])
    def test_interferometer_unitary(self, t, tol):
        """An integration test for CV gates that support analytic differentiation
        if succeeding the gate to be differentiated, but cannot be differentiated
        themselves (for example, they may be Gaussian but accept no parameters,
        or may accept a numerical array parameter.).

        This ensures that, assuming their _heisenberg_rep is defined, the quantum
        gradient analytic method can still be used, and returns the correct result.

        Currently, the only such operation is qml.InterferometerUnitary. In the future,
        we may consider adding a qml.GaussianTransfom operator.
        """

        if t == 1:
            pytest.xfail(
                "There is a bug in the second order CV parameter-shift rule; "
                "phase arguments return the incorrect derivative."
            )

            # Note: this bug currently affects PL core as well:
            #
            # dev = qml.device("default.gaussian", wires=2)
            #
            # U = np.array([[ 0.51310276+0.81702166j,  0.13649626+0.22487759j],
            #         [ 0.26300233+0.00556194j, -0.96414101-0.03508489j]])
            #
            # @qml.qnode(dev)
            # def circuit(r, phi):
            #     qml.Displacement(r, phi, wires=0)
            #     qml.InterferometerUnitary(U, wires=[0, 1])
            #     return qml.expval(qml.X(0))
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

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(0.543, 0, wires=0)
            qml.InterferometerUnitary(U, wires=[0, 1])
            qml.expval(qml.X(0))

        tape.trainable_params = {t}

        dev = qml.device("default.gaussian", wires=2)

        tapes, fn = qml.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev)
        grad_A = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))

        assert tape._par_info[0]["grad_method"] == "A"
        assert tape._par_info[1]["grad_method"] == "A"

        # the different methods agree
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)
        assert np.allclose(grad_A2, grad_F, atol=tol, rtol=0)


class TestVarianceQuantumGradients:
    """Tests for the quantum gradients of various gates
    with variance measurements"""

    def test_first_order_observable(self, tol):
        """Test variance of a first order CV observable"""
        dev = qml.device("default.gaussian", wires=1)

        r = 0.543
        phi = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            qml.var(qml.X(0))

        tape.trainable_params = {0, 2}

        res = qml.execute([tape], dev, None)
        expected = np.exp(2 * r) * np.sin(phi) ** 2 + np.exp(-2 * r) * np.cos(phi) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.finite_diff(tape)
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
        dev = qml.device("default.gaussian", wires=1)

        n = 0.12
        a = 0.765

        with qml.tape.QuantumTape() as tape:
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            qml.var(qml.NumberOperator(0))

        tape.trainable_params = {0, 1}

        res = qml.execute([tape], dev, None)
        expected = n**2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        expected = np.array([[2 * a**2 + 2 * n + 1, 2 * a * (2 * n + 1)]])
        assert np.allclose(grad_F, expected, atol=tol, rtol=0)

    def test_expval_and_variance(self, tol):
        """Test that the gradient works for a combination of CV expectation
        values and variances"""
        dev = qml.device("default.gaussian", wires=3)

        a, b = [0.54, -0.423]

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(0.5, 0, wires=0)
            qml.Squeezing(a, 0, wires=0)
            qml.Squeezing(b, 0, wires=1)
            qml.Beamsplitter(0.6, -0.3, wires=[0, 1])
            qml.Squeezing(-0.3, 0, wires=2)
            qml.Beamsplitter(1.4, 0.5, wires=[1, 2])
            qml.var(qml.X(0))
            qml.expval(qml.X(1))
            qml.var(qml.X(2))

        tape.trainable_params = {2, 4}

        # jacobians must match
        tapes, fn = qml.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev)
        grad_A = fn(dev.batch_execute(tapes))

        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    def test_error_analytic_second_order(self):
        """Test exception raised if attempting to use a second
        order observable to compute the variance derivative analytically"""
        dev = qml.device("default.gaussian", wires=1)

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(1.0, 0, wires=0)
            qml.var(qml.NumberOperator(0))

        tape.trainable_params = {0}

        with pytest.raises(ValueError, match=r"cannot be used with the parameter\(s\) \{0\}"):
            param_shift_cv(tape, dev, fallback_fn=None)

    def test_error_unsupported_grad_recipe(self, monkeypatch):
        """Test exception raised if attempting to use the second order rule for
        computing the gradient analytically of an expectation value that
        contains an operation with more than two terms in the gradient recipe"""

        class DummyOp(qml.operation.CVOperation):
            num_wires = 1
            par_domain = "R"
            grad_method = "A"
            grad_recipe = ([[1, 1, 1], [1, 1, 1], [1, 1, 1]],)

        dev = qml.device("default.gaussian", wires=1)

        dev.operations.add(DummyOp)

        with qml.tape.QuantumTape() as tape:
            DummyOp(1, wires=[0])
            qml.expval(qml.X(0))

        tape._gradient_fn = param_shift_cv
        tape._par_info[0]["grad_method"] = "A"
        tape.trainable_params = {0}

        with pytest.raises(
            NotImplementedError, match=r"analytic gradient for order-2 operators is unsupported"
        ):
            param_shift_cv(tape, dev, force_order2=True)

    @pytest.mark.parametrize("obs", [qml.X, qml.NumberOperator])
    @pytest.mark.parametrize(
        "op", [qml.Squeezing(0.1, 0.2, wires=0), qml.Beamsplitter(0.1, 0.2, wires=[0, 1])]
    )
    def test_gradients_gaussian_circuit(self, op, obs, tol):
        """Tests that the gradients of circuits of selected gaussian gates match between the
        finite difference and analytic methods."""
        tol = 1e-2

        with qml.tape.QuantumTape() as tape:
            qml.Displacement(0.5, 0, wires=0)
            qml.apply(op)
            qml.Beamsplitter(1.3, -2.3, wires=[0, 1])
            qml.Displacement(-0.5, 0.1, wires=0)
            qml.Squeezing(0.5, -1.5, wires=0)
            qml.Rotation(-1.1, wires=0)
            qml.var(obs(wires=0))

        dev = qml.device("default.gaussian", wires=2)
        res = qml.execute([tape], dev, None)

        tape.trainable_params = set(range(2, 2 + op.num_params))

        # jacobians must match
        tapes, fn = qml.gradients.finite_diff(tape)
        grad_F = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev)
        grad_A = fn(dev.batch_execute(tapes))

        tapes, fn = param_shift_cv(tape, dev, force_order2=True)
        grad_A2 = fn(dev.batch_execute(tapes))

        assert np.allclose(grad_A2, grad_F, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

        # check that every parameter is analytic
        if obs != qml.NumberOperator:
            for i in range(op.num_params):
                assert tape._par_info[2 + i]["grad_method"][0] == "A"

    def test_squeezed_mean_photon_variance(self, tol):
        """Test gradient of the photon variance of a displaced thermal state"""
        dev = qml.device("default.gaussian", wires=1)

        r = 0.12
        phi = 0.105

        with qml.tape.QuantumTape() as tape:
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            qml.var(qml.X(wires=[0]))

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
        dev = qml.device("default.gaussian", wires=1)

        n = 0.12
        a = 0.105

        with qml.tape.QuantumTape() as tape:
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            qml.var(qml.TensorN(wires=[0]))

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
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.gaussian", wires=1)

        r = 0.12
        phi = 0.105

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                qml.Squeezing(x[0], 0, wires=0)
                qml.Rotation(x[1], wires=0)
                qml.var(qml.X(wires=[0]))

            tapes, fn = param_shift_cv(tape, dev)
            jac = fn(qml.execute(tapes, dev, param_shift_cv, gradient_kwargs={"dev": dev}))
            return jac[0, 2]

        params = np.array([r, phi], requires_grad=True)
        grad = qml.jacobian(cost_fn)(params)
        expected = np.array(
            [4 * np.cosh(2 * r) * np.sin(2 * phi), 4 * np.cos(2 * phi) * np.sinh(2 * r)]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests that the output of the parameter-shift CV transform
        can be executed using TF"""
        import tensorflow as tf

        dev = qml.device("default.gaussian", wires=1)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.Squeezing(params[0], 0, wires=0)
                qml.Rotation(params[1], wires=0)
                qml.var(qml.X(wires=[0]))

            tape.trainable_params = {0, 2}
            tapes, fn = param_shift_cv(tape, dev)
            jac = fn(
                qml.execute(
                    tapes, dev, param_shift_cv, gradient_kwargs={"dev": dev}, interface="tf"
                )
            )
            res = jac[0, 1]

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

        dev = qml.device("default.gaussian", wires=1)
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.Squeezing(params[0], 0, wires=0)
            qml.Rotation(params[1], wires=0)
            qml.var(qml.X(wires=[0]))

        tape.trainable_params = {0, 2}
        tapes, fn = qml.gradients.param_shift_cv(tape, dev)
        jac = fn(
            qml.execute(tapes, dev, param_shift_cv, gradient_kwargs={"dev": dev}, interface="torch")
        )

        r, phi = params.detach().numpy()

        expected = np.array(
            [
                2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                2 * np.sinh(2 * r) * np.sin(2 * phi),
            ]
        )
        assert np.allclose(jac.detach().numpy(), expected, atol=tol, rtol=0)

        cost = jac[0, 1]
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
        from jax.config import config

        config.update("jax_enable_x64", True)

        dev = qml.device("default.gaussian", wires=2)
        params = jnp.array([0.543, -0.654])

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                qml.Squeezing(params[0], 0, wires=0)
                qml.Rotation(params[1], wires=0)
                qml.var(qml.X(wires=[0]))

            tape.trainable_params = {0, 2}
            tapes, fn = qml.gradients.param_shift_cv(tape, dev)
            jac = fn(
                qml.execute(
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

        pytest.xfail("The CV Operation methods have not been updated to support autodiff")

        res = jax.jacobian(cost_fn)(params)
        expected = np.array(
            [
                [
                    4 * np.exp(-2 * r) * (np.cos(phi) ** 2 + np.exp(4 * r) * np.sin(phi) ** 2),
                    4 * np.cosh(2 * r) * np.sin(2 * phi),
                ],
                [4 * np.cosh(2 * r) * np.sin(2 * phi), 4 * np.cos(2 * phi) * np.sinh(2 * r)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the CV parameter-shift CVParamShiftTape"""
import pytest
import numpy as np

import pennylane as qml
from pennylane.beta.tapes import CVParamShiftTape, qnode
from pennylane.beta.queuing import expval, var, sample, probs, MeasurementProcess


class TestGradMethod:
    """Tests for parameter gradient methods"""

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is
        correctly marked"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with CVParamShiftTape() as tape:
            qml.FockState(1, wires=0)
            qml.Displacement(0.543, 0, wires=[1])
            qml.Beamsplitter(0, 0, wires=[0, 1])
            expval(qml.X(wires=[0]))

        assert tape._grad_method(0) is None
        assert tape._grad_method(1) == "A"
        assert tape._grad_method(2) == "A"
        assert tape._grad_method(3) == "A"
        assert tape._grad_method(4) == "A"

        tape._update_gradient_info()

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "A"
        assert tape._par_info[2]["grad_method"] == "A"
        assert tape._par_info[3]["grad_method"] == "A"
        assert tape._par_info[4]["grad_method"] == "A"

    def test_no_graph_exception(self):
        """Test that an exception is raised for analytically differentiable
        operations if use_graph=False"""
        with CVParamShiftTape() as tape:
            qml.Rotation(0.543, wires=[0])
            expval(qml.P(0))

        with pytest.raises(ValueError, match="must always use the graph"):
            tape._grad_method(0, use_graph=False)

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with CVParamShiftTape() as tape:
            qml.Rotation(0.543, wires=[0])
            qml.Rotation(-0.654, wires=[1])
            expval(qml.P(0))

        assert tape._grad_method(0) == "A"
        assert tape._grad_method(1) == "0"

        tape._update_gradient_info()

        assert tape._par_info[0]["grad_method"] == "A"
        assert tape._par_info[1]["grad_method"] == "0"

    def test_finite_diff(self, monkeypatch):
        """If an op has grad_method=F, this should be respected
        by the CVParamShiftTape"""
        monkeypatch.setattr(qml.Rotation, "grad_method", "F")

        with CVParamShiftTape() as tape:
            qml.Rotation(0.543, wires=[0])
            qml.Squeezing(0.543, 0, wires=[0])
            expval(qml.P(0))

        assert tape._grad_method(0) == "F"
        assert tape._grad_method(1) == "A"
        assert tape._grad_method(2) == "A"

    def test_non_gaussian_operation(self):
        """Test that a non-Gaussian operation succeeding
        a differentiable Gaussian operation results in
        numeric differentiation."""

        with CVParamShiftTape() as tape:
            qml.Rotation(1., wires=[0])
            qml.Rotation(1., wires=[1])
            # Non-Gaussian
            qml.Kerr(1., wires=[1])
            expval(qml.P(0))
            expval(qml.X(1))

        # First rotation gate has no succeeding non-Gaussian operation
        assert tape._grad_method(0) == "A"
        # Second rotation gate does no succeeding non-Gaussian operation
        assert tape._grad_method(1) == "F"
        # Kerr gate does not support the parameter-shift rule
        assert tape._grad_method(2) == "F"

        with CVParamShiftTape() as tape:
            qml.Rotation(1., wires=[0])
            qml.Rotation(1., wires=[1])
            # entangle the modes
            qml.Beamsplitter(1., 0., wires=[0, 1])
            # Non-Gaussian
            qml.Kerr(1., wires=[1])
            expval(qml.P(0))
            expval(qml.X(1))

        # After entangling the modes, the Kerr gate now succeeds
        # both initial rotations
        assert tape._grad_method(0) == "F"
        assert tape._grad_method(1) == "F"
        assert tape._grad_method(2) == "F"

    def test_probability(self):
        """Probability is the expectation value of a
        higher order observable, and thus only supports numerical
        differentiation"""
        with CVParamShiftTape() as tape:
            qml.Rotation(0.543, wires=[0])
            qml.Squeezing(0.543, 0, wires=[0])
            probs(wires=0)

        assert tape._grad_method(0) == "F"
        assert tape._grad_method(1) == "F"
        assert tape._grad_method(2) == "F"

    def test_variance(self):
        """If the variance of the observable is first order, then
        parameter-shift is supported. If the observable is second order,
        however, only finite-differences is supported."""

        with CVParamShiftTape() as tape:
            qml.Rotation(1., wires=[0])
            var(qml.P(0))  # first order

        assert tape._grad_method(0) == "A"

        with CVParamShiftTape() as tape:
            qml.Rotation(1., wires=[0])
            var(qml.NumberOperator(0))  # second order

        assert tape._grad_method(0) == "F"

        with CVParamShiftTape() as tape:
            qml.Rotation(1., wires=[0])
            qml.Rotation(1., wires=[1])
            qml.Beamsplitter(0.5, 0., wires=[0, 1])
            var(qml.NumberOperator(0))  # second order
            expval(qml.NumberOperator(0))

        assert tape._grad_method(0) == "F"
        assert tape._grad_method(1) == "F"

    def test_second_order_expectation(self):
        """Test that the expectation of a second-order observable forces
        the gradient method to use the second-order parameter-shift rule"""

        with CVParamShiftTape() as tape:
            qml.Rotation(1., wires=[0])
            expval(qml.NumberOperator(0))  # second order

        assert tape._grad_method(0) == "A2"

    def test_unknown_op_grad_method(self, monkeypatch):
        """Test that an exception is raised if an operator has a
        grad method defined that the CV parameter-shift tape
        doesn't recognize"""
        monkeypatch.setattr(qml.Rotation, "grad_method", "B")

        with CVParamShiftTape() as tape:
            qml.Rotation(1., wires=0)
            expval(qml.X(0))

        with pytest.raises(ValueError, match="unknown gradient method"):
            tape._grad_method(0)


class TestTransformObservable:
    """Tests for the _transform_observable method"""

    def test_incorrect_heisenberg_size(self, monkeypatch):
        """The number of dimensions of a CV observable Heisenberg representation does
        not match the ev_order attribute."""
        monkeypatch.setattr(qml.P, "ev_order", 2)

        with pytest.raises(ValueError, match="Mismatch between the polynomial order"):
            CVParamShiftTape._transform_observable(qml.P(0), np.identity(3), device_wires=[0])

    def test_higher_order_observable(self, monkeypatch):
        """An exception should be raised if the observable is higher than 2nd order."""
        monkeypatch.setattr(qml.P, "ev_order", 3)

        with pytest.raises(NotImplementedError, match="order > 2 not implemented"):
            CVParamShiftTape._transform_observable(qml.P(0), np.identity(3), device_wires=[0])

    def test_first_order_transform(self, tol):
        """Test that a first order observable is transformed correctly"""
        # create a symmetric transformation
        Z = np.arange(3 ** 2).reshape(3, 3)
        Z = Z.T + Z

        obs = qml.X(0)
        res = CVParamShiftTape._transform_observable(obs, Z, device_wires=[0])

        # The Heisenberg representation of the X
        # operator is simply... X
        expected = np.array([0, 1, 0]) @ Z

        assert isinstance(res, qml.PolyXP)
        assert res.wires.labels == (0,)
        assert np.allclose(res.data[0], expected, atol=tol, rtol=0)

    def test_second_order_transform(self, tol):
        """Test that a second order observable is transformed correctly"""
        # create a symmetric transformation
        Z = np.arange(3 ** 2).reshape(3, 3)
        Z = Z.T + Z

        obs = qml.NumberOperator(0)
        res = CVParamShiftTape._transform_observable(obs, Z, device_wires=[0])

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

        Z = np.arange(ndim ** 2).reshape(ndim, ndim)
        Z = Z.T + Z

        obs = qml.NumberOperator(0)
        res = CVParamShiftTape._transform_observable(obs, Z, device_wires=wires)

        # The Heisenberg representation of the number operator
        # is (X^2 + P^2) / (2*hbar) - 1/2. We use the ordering
        # I, X0, Xa, X2, P0, Pa, P2.
        A = np.diag([-0.5, 0.25, 0.25, 0, 0, 0, 0])
        expected = A @ Z + Z @ A

        assert isinstance(res, qml.PolyXP)
        assert res.wires == wires
        assert np.allclose(res.data[0], expected, atol=tol, rtol=0)


class TestCVParameterShiftRule:
    """Tests for the first-order parameter shift implementation"""

    def test_rotation_gradient(self, mocker, tol):
        """Test the gradient of the rotation gate"""
        dev = qml.device("default.gaussian", wires=1)

        alpha = 0.5643
        theta = 0.23354

        with CVParamShiftTape() as tape:
            qml.Displacement(alpha, 0., wires=[0])
            qml.Rotation(theta, wires=[0])
            expval(qml.X(0))

        tape._update_gradient_info()
        tape.trainable_params = {2}

        spy = mocker.spy(CVParamShiftTape, "parameter_shift_second_order")

        grad_A = tape.jacobian(dev, method="analytic")
        spy.assert_not_called()

        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)
        spy.assert_called()

        expected = - 2 * alpha * np.sin(theta)
        assert np.allclose(grad_A, expected, atol=tol, rtol=0)
        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    cv_ops = [getattr(qml, name) for name in qml.ops._cv__ops__]
    analytic_cv_ops = [cls for cls in cv_ops if cls.supports_parameter_shift]

    @pytest.mark.parametrize('obs', [qml.X, qml.P, qml.NumberOperator, qml.Identity])
    @pytest.mark.parametrize('op', analytic_cv_ops)
    def test_gradients_gaussian_circuit(self, op, obs, mocker, tol):
        """Tests that the gradients of circuits of gaussian gates match between the
        finite difference and analytic methods."""
        tol = 1e-2

        args = np.linspace(0.2, 0.5, op.num_params)

        with CVParamShiftTape() as tape:
            qml.Displacement(0.5, 0, wires=0)
            op(*args, wires=range(op.num_wires))
            qml.Beamsplitter(1.3, -2.3, wires=[0, 1])
            qml.Displacement(-0.5, 0.1, wires=0)
            qml.Squeezing(0.5, -1.5, wires=0)
            qml.Rotation(-1.1, wires=0)
            expval(obs(wires=0))

        dev = qml.device("default.gaussian", wires=2)
        res = tape.execute(dev)

        tape._update_gradient_info()
        tape.trainable_params = set(range(2, 2 + op.num_params))

        # check that every parameter is analytic
        for i in range(op.num_params):
            assert tape._par_info[2 + i]["grad_method"][0] == "A"

        spy = mocker.spy(CVParamShiftTape, "parameter_shift_first_order")
        grad_F  = tape.jacobian(dev, method="numeric")
        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)

        spy.assert_not_called()
        assert np.allclose(grad_A2, grad_F, atol=tol, rtol=0)

        if obs.ev_order == 1:
            grad_A = tape.jacobian(dev, method="analytic")
            spy.assert_called()
            assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

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
from pennylane.tape import CVParamShiftTape


hbar = 2


class TestGradMethod:
    """Tests for parameter gradient methods"""

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is
        correctly marked"""

        with CVParamShiftTape() as tape:
            qml.FockState(1, wires=0)
            qml.Displacement(0.543, 0, wires=[1])
            qml.Beamsplitter(0, 0, wires=[0, 1])
            qml.expval(qml.X(wires=[0]))

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
            qml.expval(qml.P(0))

        with pytest.raises(ValueError, match="must always use the graph"):
            tape._grad_method(0, use_graph=False)

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with CVParamShiftTape() as tape:
            qml.Rotation(0.543, wires=[0])
            qml.Rotation(-0.654, wires=[1])
            qml.expval(qml.P(0))

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
            qml.expval(qml.P(0))

        assert tape._grad_method(0) == "F"
        assert tape._grad_method(1) == "A"
        assert tape._grad_method(2) == "A"

    def test_non_gaussian_operation(self):
        """Test that a non-Gaussian operation succeeding
        a differentiable Gaussian operation results in
        numeric differentiation."""

        with CVParamShiftTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.Rotation(1.0, wires=[1])
            # Non-Gaussian
            qml.Kerr(1.0, wires=[1])
            qml.expval(qml.P(0))
            qml.expval(qml.X(1))

        # First rotation gate has no succeeding non-Gaussian operation
        assert tape._grad_method(0) == "A"
        # Second rotation gate does no succeeding non-Gaussian operation
        assert tape._grad_method(1) == "F"
        # Kerr gate does not support the parameter-shift rule
        assert tape._grad_method(2) == "F"

        with CVParamShiftTape() as tape:
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
            qml.probs(wires=0)

        assert tape._grad_method(0) == "F"
        assert tape._grad_method(1) == "F"
        assert tape._grad_method(2) == "F"

    def test_variance(self):
        """If the variance of the observable is first order, then
        parameter-shift is supported. If the observable is second order,
        however, only finite-differences is supported."""

        with CVParamShiftTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.var(qml.P(0))  # first order

        assert tape._grad_method(0) == "A"

        with CVParamShiftTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.var(qml.NumberOperator(0))  # second order

        assert tape._grad_method(0) == "F"

        with CVParamShiftTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.Rotation(1.0, wires=[1])
            qml.Beamsplitter(0.5, 0.0, wires=[0, 1])
            qml.var(qml.NumberOperator(0))  # second order
            qml.expval(qml.NumberOperator(1))

        assert tape._grad_method(0) == "F"
        assert tape._grad_method(1) == "F"
        assert tape._grad_method(2) == "F"
        assert tape._grad_method(3) == "F"

    def test_second_order_expectation(self):
        """Test that the expectation of a second-order observable forces
        the gradient method to use the second-order parameter-shift rule"""

        with CVParamShiftTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.expval(qml.NumberOperator(0))  # second order

        assert tape._grad_method(0) == "A2"

    def test_unknown_op_grad_method(self, monkeypatch):
        """Test that an exception is raised if an operator has a
        grad method defined that the CV parameter-shift tape
        doesn't recognize"""
        monkeypatch.setattr(qml.Rotation, "grad_method", "B")

        with CVParamShiftTape() as tape:
            qml.Rotation(1.0, wires=0)
            qml.expval(qml.X(0))

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
        Z = np.arange(3**2).reshape(3, 3)
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
        Z = np.arange(3**2).reshape(3, 3)
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

        Z = np.arange(ndim**2).reshape(ndim, ndim)
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


class TestParameterShiftLogic:
    """Test for the dispatching logic of the parameter shift method"""

    def test_force_order2(self, mocker):
        """Test that if the force_order2 keyword argument is provided,
        the second order parameter shift rule is forced"""
        dev = qml.device("default.gaussian", wires=1)

        with CVParamShiftTape() as tape:
            qml.Displacement(1.0, 0.0, wires=[0])
            qml.Rotation(2.0, wires=[0])
            qml.expval(qml.X(0))

        tape.trainable_params = {0, 1, 2}

        spy1 = mocker.spy(tape, "parameter_shift_first_order")
        spy2 = mocker.spy(tape, "parameter_shift_second_order")

        tape.jacobian(dev, method="analytic", force_order2=False)
        spy1.assert_called()
        spy2.assert_not_called()

        tape.jacobian(dev, method="analytic", force_order2=True)
        spy2.assert_called()

    def test_no_poly_xp_support(self, mocker, monkeypatch, caplog):
        """Test that if a device does not support PolyXP
        and the second-order parameter-shift rule is required,
        we fallback to finite differences."""
        dev = qml.device("default.gaussian", wires=1)

        monkeypatch.delitem(dev._observable_map, "PolyXP")

        with CVParamShiftTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.expval(qml.NumberOperator(0))

        tape.trainable_params = {0}
        assert tape.analytic_pd == tape.parameter_shift

        spy_analytic = mocker.spy(tape, "analytic_pd")
        spy_first_order_shift = mocker.spy(tape, "parameter_shift_first_order")
        spy_second_order_shift = mocker.spy(tape, "parameter_shift_second_order")
        spy_transform = mocker.spy(qml.operation.CVOperation, "heisenberg_tr")
        spy_numeric = mocker.spy(tape, "numeric_pd")

        with pytest.warns(UserWarning, match="does not support the PolyXP observable"):
            tape.jacobian(dev, method="analytic")

        spy_analytic.assert_called()
        spy_first_order_shift.assert_not_called()
        spy_second_order_shift.assert_not_called()
        spy_transform.assert_not_called()
        spy_numeric.assert_called()

    def test_no_poly_xp_support_variance(self, mocker, monkeypatch, caplog):
        """Test that if a device does not support PolyXP
        and the variance parameter-shift rule is required,
        we fallback to finite differences."""
        dev = qml.device("default.gaussian", wires=1)

        monkeypatch.delitem(dev._observable_map, "PolyXP")

        with CVParamShiftTape() as tape:
            qml.Rotation(1.0, wires=[0])
            qml.var(qml.X(0))

        tape.trainable_params = {0}
        assert tape.analytic_pd == tape.parameter_shift_var

        spy1 = mocker.spy(tape, "parameter_shift_first_order")
        spy2 = mocker.spy(tape, "parameter_shift_second_order")
        spy_numeric = mocker.spy(tape, "numeric_pd")

        with pytest.warns(UserWarning, match="does not support the PolyXP observable"):
            tape.jacobian(dev, method="analytic")

        spy1.assert_not_called()
        spy2.assert_not_called()
        spy_numeric.assert_called()


class TestExpectationQuantumGradients:
    """Tests for the quantum gradients of various gates

    with expectation value output"""

    def test_rotation_gradient(self, mocker, tol):
        """Test the gradient of the rotation gate"""
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        alpha = 0.5643
        theta = 0.23354

        with CVParamShiftTape() as tape:
            qml.Displacement(alpha, 0.0, wires=[0])
            qml.Rotation(theta, wires=[0])
            qml.expval(qml.X(0))

        tape._update_gradient_info()
        tape.trainable_params = {2}

        spy1 = mocker.spy(CVParamShiftTape, "parameter_shift_first_order")
        spy2 = mocker.spy(CVParamShiftTape, "parameter_shift_second_order")

        grad_A = tape.jacobian(dev, method="analytic")
        spy1.assert_called()
        spy2.assert_not_called()

        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)
        spy2.assert_called()

        expected = -hbar * alpha * np.sin(theta)
        assert np.allclose(grad_A, expected, atol=tol, rtol=0)
        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    def test_beamsplitter_gradient(self, mocker, tol):
        """Test the gradient of the beamsplitter gate"""
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        alpha = 0.5643
        theta = 0.23354

        with CVParamShiftTape() as tape:
            qml.Displacement(alpha, 0.0, wires=[0])
            qml.Beamsplitter(theta, 0.0, wires=[0, 1])
            qml.expval(qml.X(0))

        tape._update_gradient_info()
        tape.trainable_params = {2}

        spy1 = mocker.spy(CVParamShiftTape, "parameter_shift_first_order")
        spy2 = mocker.spy(CVParamShiftTape, "parameter_shift_second_order")

        grad_A = tape.jacobian(dev, method="analytic")
        spy1.assert_called()
        spy2.assert_not_called()

        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)
        spy2.assert_called()

        expected = -hbar * alpha * np.sin(theta)
        assert np.allclose(grad_A, expected, atol=tol, rtol=0)
        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    def test_displacement_gradient(self, mocker, tol):
        """Test the gradient of the displacement gate"""
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        r = 0.5643
        phi = 0.23354

        with CVParamShiftTape() as tape:
            qml.Displacement(r, phi, wires=[0])
            qml.expval(qml.X(0))

        tape._update_gradient_info()
        tape.trainable_params = {0, 1}

        spy1 = mocker.spy(CVParamShiftTape, "parameter_shift_first_order")
        spy2 = mocker.spy(CVParamShiftTape, "parameter_shift_second_order")

        grad_A = tape.jacobian(dev, method="analytic")
        spy1.assert_called()
        spy2.assert_not_called()

        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)
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
            num_params = 1
            grad_method = "A"

        alpha = 0.5643
        r = 0.23354

        with CVParamShiftTape() as tape:
            qml.Displacement(alpha, 0.0, wires=[0])
            qml.Squeezing(r, 0.0, wires=[0])

            # The following two gates have no effect
            # on the circuit gradient and expectation value
            qml.Beamsplitter(0.0, 0.0, wires=[0, 1])
            Rotation(0.543, wires=[1])

            qml.expval(qml.X(0))

        tape._update_gradient_info()
        tape.trainable_params = {2}

        spy1 = mocker.spy(CVParamShiftTape, "parameter_shift_first_order")
        spy2 = mocker.spy(CVParamShiftTape, "parameter_shift_second_order")

        grad_A = tape.jacobian(dev, method="analytic")
        spy1.assert_called()
        spy2.assert_not_called()

        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)
        spy2.assert_called()

        expected = -np.exp(-r) * hbar * alpha
        assert np.allclose(grad_A, expected, atol=tol, rtol=0)
        assert np.allclose(grad_A2, expected, atol=tol, rtol=0)

    def test_squeezed_number_state_gradient(self, mocker, tol):
        """Test the numerical gradient of the squeeze gate with
        with number state expectation is correct"""
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        r = 0.23354

        with CVParamShiftTape() as tape:
            qml.Squeezing(r, 0.0, wires=[0])
            # the fock state projector is a 'non-Gaussian' observable
            qml.expval(qml.FockStateProjector(np.array([2, 0]), wires=[0, 1]))

        tape._update_gradient_info()
        tape.trainable_params = {0}
        assert tape._par_info[0]["grad_method"] == "F"

        spy = mocker.spy(CVParamShiftTape, "parameter_shift")
        grad = tape.jacobian(dev)
        spy.assert_not_called()

        # (d/dr) |<2|S(r)>|^2 = 0.5 tanh(r)^3 (2 csch(r)^2 - 1) sech(r)
        expected = 0.5 * np.tanh(r) ** 3 * (2 / (np.sinh(r) ** 2) - 1) / np.cosh(r)
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_multiple_squeezing_gradient(self, mocker, tol):
        """Test that the gradient of a circuit with two squeeze
        gates is correct."""
        dev = qml.device("default.gaussian", wires=2, hbar=hbar)

        r0, phi0, r1, phi1 = [0.4, -0.3, -0.7, 0.2]

        with CVParamShiftTape() as tape:
            qml.Squeezing(r0, phi0, wires=[0])
            qml.Squeezing(r1, phi1, wires=[0])
            qml.expval(qml.NumberOperator(0))  # second order

        tape._update_gradient_info()

        spy2 = mocker.spy(CVParamShiftTape, "parameter_shift_second_order")
        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)
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

        with CVParamShiftTape() as tape:
            qml.Squeezing(r[0], p[0], wires=[0])
            qml.Squeezing(r[1], p[1], wires=[0])
            qml.Squeezing(r[2], p[2], wires=[1])
            qml.Squeezing(r[3], p[3], wires=[1])
            qml.expval(qml.NumberOperator(0))  # second order
            qml.expval(qml.NumberOperator(1))  # second order

        tape._update_gradient_info()

        spy2 = mocker.spy(CVParamShiftTape, "parameter_shift_second_order")
        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)
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

    @pytest.mark.parametrize("obs", [qml.X, qml.Identity])
    @pytest.mark.parametrize(
        "op", [qml.Displacement(0.1, 0.2, wires=0), qml.TwoModeSqueezing(0.1, 0.2, wires=[0, 1])]
    )
    def test_gradients_gaussian_circuit(self, op, obs, mocker, tol):
        """Tests that the gradients of circuits of gaussian gates match between the
        finite difference and analytic methods."""
        tol = 1e-2

        with CVParamShiftTape() as tape:
            qml.Displacement(0.5, 0, wires=0)
            qml.apply(op)
            qml.Beamsplitter(1.3, -2.3, wires=[0, 1])
            qml.Displacement(-0.5, 0.1, wires=0)
            qml.Squeezing(0.5, -1.5, wires=0)
            qml.Rotation(-1.1, wires=0)
            qml.expval(obs(wires=0))

        dev = qml.device("default.gaussian", wires=2)
        res = tape.execute(dev)

        tape._update_gradient_info()
        tape.trainable_params = set(range(2, 2 + op.num_params))

        # check that every parameter is analytic
        for i in range(op.num_params):
            assert tape._par_info[2 + i]["grad_method"][0] == "A"

        spy = mocker.spy(CVParamShiftTape, "parameter_shift_first_order")
        grad_F = tape.jacobian(dev, method="numeric")
        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)

        spy.assert_not_called()
        assert np.allclose(grad_A2, grad_F, atol=tol, rtol=0)

        if obs.ev_order == 1:
            grad_A = tape.jacobian(dev, method="analytic")
            spy.assert_called()
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
            ]
        )

        with CVParamShiftTape() as tape:
            qml.Displacement(0.543, 0, wires=0)
            qml.InterferometerUnitary(U, wires=[0, 1])
            qml.expval(qml.X(0))

        tape._update_gradient_info()
        tape.trainable_params = {t}
        assert tape._par_info[0]["grad_method"] == "A"
        assert tape._par_info[1]["grad_method"] == "A"

        dev = qml.device("default.gaussian", wires=2)
        grad_F = tape.jacobian(dev, method="numeric")
        grad_A = tape.jacobian(dev, method="analytic")
        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)

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

        with CVParamShiftTape() as tape:
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            qml.var(qml.X(0))

        tape.trainable_params = {0, 2}

        res = tape.execute(dev)
        expected = np.exp(2 * r) * np.sin(phi) ** 2 + np.exp(-2 * r) * np.cos(phi) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        grad_F = tape.jacobian(dev, method="numeric")
        grad_A = tape.jacobian(dev, method="analytic")
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

        with CVParamShiftTape() as tape:
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            qml.var(qml.NumberOperator(0))

        tape.trainable_params = {0, 1}

        res = tape.execute(dev)
        expected = n**2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        grad_F = tape.jacobian(dev, method="numeric")
        expected = np.array([[2 * a**2 + 2 * n + 1, 2 * a * (2 * n + 1)]])
        assert np.allclose(grad_F, expected, atol=tol, rtol=0)

    def test_expval_and_variance(self, tol):
        """Test that the gradient works for a combination of CV expectation
        values and variances"""
        dev = qml.device("default.gaussian", wires=3)

        a, b = [0.54, -0.423]

        with CVParamShiftTape() as tape:
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
        grad_F = tape.jacobian(dev, method="numeric")
        grad_A = tape.jacobian(dev, method="analytic")
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    def test_error_analytic_second_order(self):
        """Test exception raised if attempting to use a second
        order observable to compute the variance derivative analytically"""
        dev = qml.device("default.gaussian", wires=1)

        with CVParamShiftTape() as tape:
            qml.Displacement(1.0, 0, wires=0)
            qml.var(qml.NumberOperator(0))

        tape.trainable_params = {0}

        with pytest.raises(ValueError, match=r"cannot be used with the argument\(s\) \{0\}"):
            tape.jacobian(dev, method="analytic")

    def test_error_unsupported_grad_recipe(self, monkeypatch):
        """Test exception raised if attempting to use the second order rule for
        computing the gradient analytically of an expectation value that
        contains an operation with more than two terms in the gradient recipe"""

        class DummyOp(qml.operation.CVOperation):
            num_wires = 1
            num_params = 1
            grad_method = "A"
            grad_recipe = ([[1, 1, 1], [1, 1, 1], [1, 1, 1]],)

        dev = qml.device("default.gaussian", wires=1)

        dev.operations.add(DummyOp)

        with CVParamShiftTape() as tape:
            DummyOp(1, wires=[0])
            qml.expval(qml.X(0))

        with monkeypatch.context() as m:
            tape._par_info[0]["grad_method"] = "A"
            tape.trainable_params = {0}

            with pytest.raises(
                NotImplementedError, match=r"analytic gradient for order-2 operators is unsupported"
            ):
                tape.jacobian(dev, method="analytic", force_order2=True)

    @pytest.mark.parametrize("obs", [qml.X, qml.P, qml.Identity])
    @pytest.mark.parametrize(
        "op", [qml.Squeezing(0.1, 0.2, wires=0), qml.Beamsplitter(0.1, 0.2, wires=[0, 1])]
    )
    def test_gradients_gaussian_circuit(self, op, obs, mocker, tol):
        """Tests that the gradients of circuits of gaussian gates match between the
        finite difference and analytic methods."""
        tol = 1e-2

        args = np.linspace(0.2, 0.5, op.num_params)

        with CVParamShiftTape() as tape:
            qml.Displacement(0.5, 0, wires=0)
            qml.apply(op)
            qml.Beamsplitter(1.3, -2.3, wires=[0, 1])
            qml.Displacement(-0.5, 0.1, wires=0)
            qml.Squeezing(0.5, -1.5, wires=0)
            qml.Rotation(-1.1, wires=0)
            qml.var(obs(wires=0))

        dev = qml.device("default.gaussian", wires=2)
        res = tape.execute(dev)

        tape._update_gradient_info()
        tape.trainable_params = set(range(2, 2 + op.num_params))

        # check that every parameter is analytic
        for i in range(op.num_params):
            assert tape._par_info[2 + i]["grad_method"][0] == "A"

        spy = mocker.spy(CVParamShiftTape, "parameter_shift_first_order")
        grad_F = tape.jacobian(dev, method="numeric")
        grad_A = tape.jacobian(dev, method="analytic")
        grad_A2 = tape.jacobian(dev, method="analytic", force_order2=True)

        assert np.allclose(grad_A2, grad_F, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    def test_squeezed_mean_photon_variance(self, tol):
        """Test gradient of the photon variance of a displaced thermal state"""
        dev = qml.device("default.gaussian", wires=1)

        r = 0.12
        phi = 0.105

        with CVParamShiftTape() as tape:
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            qml.var(qml.X(wires=[0]))

        tape.trainable_params = {0, 2}
        grad = tape.jacobian(dev, method="analytic")
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

        with CVParamShiftTape() as tape:
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            qml.var(qml.TensorN(wires=[0]))

        tape.trainable_params = {0, 1}
        grad = tape.jacobian(dev)
        expected = np.array([2 * a**2 + 2 * n + 1, 2 * a * (2 * n + 1)])
        assert np.allclose(grad, expected, atol=tol, rtol=0)

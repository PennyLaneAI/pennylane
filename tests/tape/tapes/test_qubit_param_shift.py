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
"""Unit tests for the qubit parameter-shift QubitParamShiftTape"""
import pytest
import numpy as np

import pennylane as qml
from pennylane.tape import QubitParamShiftTape, qnode
from pennylane.tape.measure import MeasurementProcess


class TestGradMethod:
    """Tests for parameter gradient methods"""

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is
        correctly marked"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with QubitParamShiftTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        assert tape._grad_method(0) is None
        assert tape._grad_method(1) == "A"
        assert tape._grad_method(2) == "A"

        tape._update_gradient_info()

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "A"
        assert tape._par_info[2]["grad_method"] == "A"

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with QubitParamShiftTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliY(0))

        assert tape._grad_method(0) == "A"
        assert tape._grad_method(1) == "0"

        tape._update_gradient_info()

        assert tape._par_info[0]["grad_method"] == "A"
        assert tape._par_info[1]["grad_method"] == "0"

        # in non-graph mode, it is impossible to determine
        # if a parameter is independent or not
        tape._graph = None
        assert tape._grad_method(1, use_graph=False) == "A"

    def test_finite_diff(self, monkeypatch):
        """If an op has grad_method=F, this should be respected
        by the QubitParamShiftTape"""
        monkeypatch.setattr(qml.RX, "grad_method", "F")

        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with QubitParamShiftTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        assert tape._grad_method(0) is None
        assert tape._grad_method(1) == "F"
        assert tape._grad_method(2) == "A"


class TestParameterShiftRule:
    """Tests for the parameter shift implementation"""

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift])
    def test_pauli_rotation_gradient(self, mocker, G, theta, shift, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        spy = mocker.spy(QubitParamShiftTape, "parameter_shift")
        dev = qml.device("default.qubit", wires=1)

        with QubitParamShiftTape() as tape:
            qml.QubitStateVector(np.array([1., -1.]) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        autograd_val = tape.jacobian(dev, shift=shift, method="analytic")
        manualgrad_val = (
            tape.execute(dev, params=[theta + np.pi / 2])
            - tape.execute(dev, params=[theta - np.pi / 2])
        ) / 2
        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)

        assert spy.call_args[1]["shift"] == shift

        # compare to finite differences
        numeric_val = tape.jacobian(dev, shift=shift, method="numeric")
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    def test_Rot_gradient(self, mocker, theta, shift, tol):
        """Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."""
        spy = mocker.spy(QubitParamShiftTape, "parameter_shift")
        dev = qml.device("default.qubit", wires=1)
        params = np.array([theta, theta ** 3, np.sqrt(2) * theta])

        with QubitParamShiftTape() as tape:
            qml.QubitStateVector(np.array([1., -1.]) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        autograd_val = tape.jacobian(dev, shift=shift, method="analytic")
        manualgrad_val = np.zeros_like(autograd_val)

        for idx in list(np.ndindex(*params.shape)):
            s = np.zeros_like(params)
            s[idx] += np.pi / 2

            forward = tape.execute(dev, params=params + s)
            backward = tape.execute(dev, params=params - s)

            manualgrad_val[0, idx] = (forward - backward) / 2

        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)
        assert spy.call_args[1]["shift"] == shift

        # compare to finite differences
        numeric_val = tape.jacobian(dev, shift=shift, method="numeric")
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, tol):
        """Test gradient of controlled rotation gates"""
        dev = qml.device("default.qubit", wires=2)
        b = 0.123

        with QubitParamShiftTape() as tape:
            qml.QubitStateVector(np.array([1., -1.]) / np.sqrt(2), wires=0)
            G(b, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {1}

        res = tape.execute(dev)
        assert np.allclose(res, -np.cos(b / 2), atol=tol, rtol=0)

        grad = tape.jacobian(dev, method="analytic")
        expected = np.sin(b / 2) / 2
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        # compare to finite differences
        numeric_val = tape.jacobian(dev, method="numeric")
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, mocker, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parameterized
        gate is correct."""
        spy = mocker.spy(QubitParamShiftTape, "parameter_shift")
        dev = qml.device("default.qubit", wires=2)
        a, b, c = np.array([theta, theta ** 3, np.sqrt(2) * theta])

        with QubitParamShiftTape() as tape:
            qml.QubitStateVector(np.array([1., -1.]) / np.sqrt(2), wires=0)
            qml.CRot(a, b, c, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {1, 2, 3}

        res = tape.execute(dev)
        expected = -np.cos(b / 2) * np.cos(0.5 * (a + c))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = tape.jacobian(dev, method="analytic")
        expected = np.array([[
            0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
            0.5 * np.sin(b / 2) * np.cos(0.5 * (a + c)),
            0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
        ]])
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        # compare to finite differences
        numeric_val = tape.jacobian(dev, method="numeric")
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    def test_gradients_agree_finite_differences(self, mocker, tol):
        """Tests that the parameter-shift rule agrees with the first and second
        order finite differences"""
        params = np.array([0.1, -1.6, np.pi / 5])

        with QubitParamShiftTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(params[2], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 2, 3}
        dev = qml.device("default.qubit", wires=2)

        spy_numeric = mocker.spy(tape, "numeric_pd")
        spy_analytic = mocker.spy(tape, "analytic_pd")

        grad_F1 = tape.jacobian(dev, method="numeric", order=1)
        grad_F2 = tape.jacobian(dev, method="numeric", order=2)

        spy_numeric.assert_called()
        spy_analytic.assert_not_called()

        grad_A = tape.jacobian(dev, method="analytic")
        spy_analytic.assert_called()

        # gradients computed with different methods must agree
        assert np.allclose(grad_A, grad_F1, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F2, atol=tol, rtol=0)

    def test_variance_gradients_agree_finite_differences(self, mocker, tol):
        """Tests that the variance parameter-shift rule agrees with the first and second
        order finite differences"""
        params = np.array([0.1, -1.6, np.pi / 5])

        with QubitParamShiftTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(params[2], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

        tape.trainable_params = {0, 2, 3}
        dev = qml.device("default.qubit", wires=2)

        spy_numeric = mocker.spy(tape, "numeric_pd")
        spy_analytic = mocker.spy(tape, "analytic_pd")

        grad_F1 = tape.jacobian(dev, method="numeric", order=1)
        grad_F2 = tape.jacobian(dev, method="numeric", order=2)

        spy_numeric.assert_called()
        spy_analytic.assert_not_called()

        grad_A = tape.jacobian(dev, method="analytic")
        spy_analytic.assert_called()

        # gradients computed with different methods must agree
        assert np.allclose(grad_A, grad_F1, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F2, atol=tol, rtol=0)

class TestJacobianIntegration:
    """Tests for general Jacobian integration"""

    def test_single_expectation_value(self, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QubitParamShiftTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        res = tape.jacobian(dev, method="analytic")
        assert res.shape == (1, 2)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QubitParamShiftTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        res = tape.jacobian(dev, method="analytic")
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QubitParamShiftTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        res = tape.jacobian(dev, method="analytic")
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QubitParamShiftTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        res = tape.jacobian(dev, method="analytic")
        assert res.shape == (5, 2)

        expected = (
            np.array(
                [
                    [-2 * np.sin(x), 0],
                    [
                        -(np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        -(np.sin(x) * np.sin(y / 2) ** 2),
                        (np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.sin(x) * np.sin(y / 2) ** 2),
                        (np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                ]
            )
            / 2
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_involutory_variance(self, mocker, tol):
        """Tests qubit observable that are involutory"""
        dev = qml.device("default.qubit", wires=1)
        a = 0.54

        spy_analytic_var = mocker.spy(QubitParamShiftTape, "parameter_shift_var")
        spy_numeric = mocker.spy(QubitParamShiftTape, "numeric_pd")
        spy_execute = mocker.spy(dev, "execute")


        with QubitParamShiftTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.PauliZ(0))

        res = tape.execute(dev)
        expected = 1 - np.cos(a) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy_execute.call_args_list = []

        # circuit jacobians
        gradA = tape.jacobian(dev, method="analytic")
        spy_analytic_var.assert_called()
        spy_numeric.assert_not_called()
        assert len(spy_execute.call_args_list) == 1 + 2 * 1

        spy_execute.call_args_list = []

        gradF = tape.jacobian(dev, method="numeric")
        spy_numeric.assert_called()
        assert len(spy_execute.call_args_list) == 2

        expected = 2 * np.sin(a) * np.cos(a)

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_non_involutory_variance(self, mocker, tol):
        """Tests a qubit Hermitian observable that is not involutory"""
        dev = qml.device("default.qubit", wires=1)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        spy_analytic_var = mocker.spy(QubitParamShiftTape, "parameter_shift_var")
        spy_numeric = mocker.spy(QubitParamShiftTape, "numeric_pd")
        spy_execute = mocker.spy(dev, "execute")

        with QubitParamShiftTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.Hermitian(A, 0))

        tape.trainable_params = {0}

        res = tape.execute(dev)
        expected = (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy_execute.call_args_list = []

        # circuit jacobians
        gradA = tape.jacobian(dev, method="analytic")
        spy_analytic_var.assert_called()
        spy_numeric.assert_not_called()
        assert len(spy_execute.call_args_list) == 1 + 4 * 1

        spy_execute.call_args_list = []

        gradF = tape.jacobian(dev, method="numeric")
        spy_numeric.assert_called()
        assert len(spy_execute.call_args_list) == 2

        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_involutory_and_noninvolutory_variance(self, mocker, tol):
        """Tests a qubit Hermitian observable that is not involutory alongside
        a involutory observable."""
        dev = qml.device("default.qubit", wires=2)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        spy_analytic_var = mocker.spy(QubitParamShiftTape, "parameter_shift_var")
        spy_numeric = mocker.spy(QubitParamShiftTape, "numeric_pd")
        spy_execute = mocker.spy(dev, "execute")

        with QubitParamShiftTape() as tape:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        tape.trainable_params = {0, 1}

        res = tape.execute(dev)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy_execute.call_args_list = []

        # circuit jacobians
        gradA = tape.jacobian(dev, method="analytic")
        spy_analytic_var.assert_called()
        spy_numeric.assert_not_called()
        assert len(spy_execute.call_args_list) == 1 + 2 * 4

        spy_execute.call_args_list = []

        gradF = tape.jacobian(dev, method="numeric")
        spy_numeric.assert_called()
        assert len(spy_execute.call_args_list) == 1 + 2

        expected = [2 * np.sin(a) * np.cos(a), -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]
        assert np.diag(gradA) == pytest.approx(expected, abs=tol)
        assert np.diag(gradF) == pytest.approx(expected, abs=tol)

    def test_expval_and_variance(self, tol):
        """Test that the qnode works for a combination of expectation
        values and variances"""
        dev = qml.device("default.qubit", wires=3)

        a = 0.54
        b = -0.423
        c = 0.123

        with QubitParamShiftTape() as tape:
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))
            qml.var(qml.PauliZ(2))

        res = tape.execute(dev)
        expected = np.array(
            [
                np.sin(a) ** 2,
                np.cos(a) * np.cos(b),
                0.25 * (3 - 2 * np.cos(b) ** 2 * np.cos(2 * c) - np.cos(2 * b)),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # # circuit jacobians
        gradA = tape.jacobian(dev, method="analytic")
        gradF = tape.jacobian(dev, method="numeric")
        expected = np.array(
            [
                [2 * np.cos(a) * np.sin(a), -np.cos(b) * np.sin(a), 0],
                [
                    0,
                    -np.cos(a) * np.sin(b),
                    0.5 * (2 * np.cos(b) * np.cos(2 * c) * np.sin(b) + np.sin(2 * b)),
                ],
                [0, 0, np.cos(b) ** 2 * np.sin(2 * c)],
            ]
        ).T
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the gradients.parameter_shift module using the new return types."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift
from pennylane.gradients.parameter_shift import _get_operation_recipe
from pennylane.devices import DefaultQubit
from pennylane.operation import Observable, AnyWires

# TODO: port more tests
# Expval

# Note: class TestGetOperationRecipe removed
# Note: class TestParamShift removed


def grad_fn(tape, dev, fn=qml.gradients.param_shift, **kwargs):
    """Utility function to automate execution and processing of gradient tapes"""
    tapes, fn = fn(tape, **kwargs)
    return fn(dev.batch_execute_new(tapes))


class TestParameterShiftRule:
    """Tests for the parameter shift implementation"""

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift])
    def test_pauli_rotation_gradient(self, mocker, G, theta, shift, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        qml.enable_return()
        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        dev = qml.device("default.qubit", wires=1)

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)])
        assert len(tapes) == 2

        autograd_val = fn(dev.batch_execute_new(tapes))

        tape_fwd, tape_bwd = tape.copy(copy_operations=True), tape.copy(copy_operations=True)
        tape_fwd.set_parameters([theta + np.pi / 2])
        tape_bwd.set_parameters([theta - np.pi / 2])

        manualgrad_val = np.subtract(*dev.batch_execute_new([tape_fwd, tape_bwd])) / 2
        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)
        assert isinstance(autograd_val, tuple)

        num_params = len(tape.trainable_params)
        assert len(autograd_val) == num_params

        assert spy.call_args[1]["shifts"] == (shift,)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute_new(tapes))
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)
        qml.disable_return()

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    def test_Rot_gradient(self, mocker, theta, shift, tol):
        """Tests that the automatic gradient of an arbitrary Euler-angle-parameterized gate is correct."""
        qml.enable_return()
        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        dev = qml.device("default.qubit", wires=1)
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)] * 3)
        num_params = len(tape.trainable_params)
        assert len(tapes) == 2 * num_params

        autograd_val = fn(dev.batch_execute_new(tapes))
        assert isinstance(autograd_val, tuple)
        assert len(autograd_val) == num_params
        manualgrad_val = np.zeros((1, num_params))

        manualgrad_val = []
        for idx in list(np.ndindex(*params.shape)):
            s = np.zeros_like(params)
            s[idx] += np.pi / 2

            tape.set_parameters(params + s)
            forward = dev.execute_new(tape)

            tape.set_parameters(params - s)
            backward = dev.execute_new(tape)

            component = (forward - backward) / 2
            manualgrad_val.append(component)

        assert len(autograd_val) == len(manualgrad_val)

        for a_val, m_val in zip(autograd_val, manualgrad_val):
            assert np.allclose(a_val, m_val, atol=tol, rtol=0)
            assert spy.call_args[1]["shifts"] == (shift,)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = np.squeeze(fn(dev.batch_execute_new(tapes)))
        for a_val, n_val in zip(autograd_val, numeric_val):
            assert np.allclose(a_val, n_val, atol=tol, rtol=0)

        qml.disable_return()

    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, tol):
        """Test gradient of controlled rotation gates"""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=2)
        b = 0.123

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(b, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {1}

        res = dev.execute_new(tape)
        assert np.allclose(res, -np.cos(b / 2), atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape)
        grad = fn(dev.batch_execute_new(tapes))
        expected = np.sin(b / 2) / 2
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute_new(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)
        qml.disable_return()

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parameterized
        gate is correct."""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=2)
        a, b, c = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.CRot(a, b, c, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {1, 2, 3}

        res = dev.execute_new(tape)
        expected = -np.cos(b / 2) * np.cos(0.5 * (a + c))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4 * len(tape.trainable_params)

        grad = fn(dev.batch_execute_new(tapes))
        expected = np.array(
            [
                0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                0.5 * np.sin(b / 2) * np.cos(0.5 * (a + c)),
                0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
            ]
        )
        assert isinstance(grad, tuple)
        assert len(grad) == 3
        for idx, g in enumerate(grad):
            assert np.allclose(g, expected[idx], atol=tol, rtol=0)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = np.squeeze(fn(dev.batch_execute_new(tapes)))
        for idx, g in enumerate(grad):
            assert np.allclose(g, numeric_val[idx], atol=tol, rtol=0)
        qml.disable_return()

    def test_gradients_agree_finite_differences(self, tol):
        """Tests that the parameter-shift rule agrees with the first and second
        order finite differences"""
        qml.enable_return()
        params = np.array([0.1, -1.6, np.pi / 5])

        with qml.tape.QuantumTape() as tape:
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

        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1)
        grad_F2 = grad_fn(
            tape, dev, fn=qml.gradients.finite_diff, approx_order=2, strategy="center"
        )
        grad_A = grad_fn(tape, dev)

        # gradients computed with different methods must agree
        assert np.allclose(grad_A, grad_F1, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F2, atol=tol, rtol=0)
        qml.disable_return()

    # TODO: remove xfail when var/finite diff works
    @pytest.mark.xfail
    def test_variance_gradients_agree_finite_differences(self, tol):
        """Tests that the variance parameter-shift rule agrees with the first and second
        order finite differences"""
        qml.enable_return()
        params = np.array([0.1, -1.6, np.pi / 5])

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(params[2], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        tape.trainable_params = {0, 2, 3}
        dev = qml.device("default.qubit", wires=2)

        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1)
        grad_F2 = grad_fn(
            tape, dev, fn=qml.gradients.finite_diff, approx_order=2, strategy="center"
        )
        grad_A = grad_fn(tape, dev)

        # gradients computed with different methods must agree
        assert np.allclose(grad_A, grad_F1, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F2, atol=tol, rtol=0)
        qml.disable_return()

    # TODO: remove xfail when var/finite diff works
    @pytest.mark.autograd
    @pytest.mark.xfail
    def test_fallback(self, mocker, tol):
        """Test that fallback gradient functions are correctly used"""
        qml.enable_return()
        spy = mocker.spy(qml.gradients, "finite_diff")
        dev = qml.device("default.qubit.autograd", wires=2)
        x = 0.543
        y = -0.654

        params = np.array([x, y], requires_grad=True)

        class RY(qml.RY):
            grad_method = "F"

        def cost_fn(params):
            with qml.tape.QuantumTape() as tape:
                qml.RX(params[0], wires=[0])
                RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.var(qml.PauliX(1))

            tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff)
            assert len(tapes) == 5

            # check that the fallback method was called for the specified argnums
            spy.assert_called()
            assert spy.call_args[1]["argnum"] == {1}

            return fn(dev.batch_execute_new(tapes))

        res = cost_fn(params)
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # double check the derivative
        jac = qml.jacobian(cost_fn)(params)
        assert np.allclose(jac[0, 0, 0], -np.cos(x), atol=tol, rtol=0)
        assert np.allclose(jac[1, 1, 1], -2 * np.cos(2 * y), atol=tol, rtol=0)
        qml.disable_return()

    @pytest.mark.autograd
    @pytest.mark.xfail
    def test_all_fallback(self, mocker, tol):
        """Test that *only* the fallback logic is called if no parameters
        support the parameter-shift rule"""
        qml.enable_return()
        spy_fd = mocker.spy(qml.gradients, "finite_diff")
        spy_ps = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        dev = qml.device("default.qubit.autograd", wires=2)
        x = 0.543
        y = -0.654

        params = np.array([x, y], requires_grad=True)

        class RY(qml.RY):
            grad_method = "F"

        class RX(qml.RX):
            grad_method = "F"

        with qml.tape.QuantumTape() as tape:
            RX(x, wires=[0])
            RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff)
        assert len(tapes) == 1 + 2

        # check that the fallback method was called for all argnums
        spy_fd.assert_called()
        spy_ps.assert_not_called()

        res = fn(dev.batch_execute_new(tapes))
        assert res.shape == (1, 2)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value(self, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

        res = fn(dev.batch_execute_new(tapes))
        assert len(res) == 2
        assert not isinstance(res[0], tuple)
        assert not isinstance(res[1], tuple)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)
        qml.disable_return()

    def test_multiple_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

        res = fn(dev.batch_execute_new(tapes))
        assert len(res) == 2
        assert len(res[0]) == 2
        assert len(res[1]) == 2

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)
        qml.disable_return()

    def test_var_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 5

        res = fn(dev.batch_execute_new(tapes))
        assert len(res) == 2
        assert len(res[0]) == 2
        assert len(res[1]) == 2

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])

        for a, e in zip(res, expected):
            assert np.allclose(np.squeeze(np.stack(a)), e, atol=tol, rtol=0)
        qml.disable_return()

    def test_prob_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        qml.enable_return()

        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

        res = fn(dev.batch_execute_new(tapes))
        assert len(res) == 2

        for r in res:
            assert len(r) == 2

        expval_expected = [-2 * np.sin(x) / 2, 0]
        probs_expected = (
            np.array(
                [
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

        # Expvals
        assert np.allclose(res[0][0], expval_expected[0])
        assert np.allclose(res[1][0], expval_expected[1])

        # Probs
        assert np.allclose(res[0][1], probs_expected[:, 0])
        assert np.allclose(res[1][1], probs_expected[:, 1])
        qml.disable_return()

    def test_involutory_variance(self, tol):
        """Tests qubit observables that are involutory"""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=1)
        a = 0.54

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.PauliZ(0))

        res = dev.execute_new(tape)
        expected = 1 - np.cos(a) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute_new(tapes))
        assert len(tapes) == 1 + 2 * 1

        # TODO: check when finite diff ready:
        # tapes, fn = qml.gradients.finite_diff(tape)
        # gradF = fn(dev.batch_execute_new(tapes))
        # assert len(tapes) == 2

        expected = 2 * np.sin(a) * np.cos(a)

        # assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)
        qml.disable_return()

    def test_non_involutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory"""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=1)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.Hermitian(A, 0))

        tape.trainable_params = {0}

        res = dev.execute_new(tape)
        expected = (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute_new(tapes))
        assert len(tapes) == 1 + 4 * 1

        # TODO: check when finite diff ready:
        # tapes, fn = qml.gradients.finite_diff(tape)
        # gradF = fn(dev.batch_execute_new(tapes))
        # assert len(tapes) == 2

        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        assert gradA == pytest.approx(expected, abs=tol)
        # assert gradF == pytest.approx(expected, abs=tol)
        qml.disable_return()

    def test_involutory_and_noninvolutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable."""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=2)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        tape.trainable_params = {0, 1}

        res = dev.execute_new(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute_new(tapes))
        assert len(tapes) == 1 + 2 * 4

        # TODO: check when finite diff ready:
        # tapes, fn = qml.gradients.finite_diff(tape)
        # gradF = fn(dev.batch_execute_new(tapes))
        # assert len(tapes) == 1 + 2

        expected = [2 * np.sin(a) * np.cos(a), -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]
        assert np.diag(gradA) == pytest.approx(expected, abs=tol)
        # TODO: check when finite diff ready:
        # assert np.diag(gradF) == pytest.approx(expected, abs=tol)
        qml.disable_return()

    def test_expval_and_variance(self, tol):
        """Test that the qnode works for a combination of expectation
        values and variances"""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=3)

        a = 0.54
        b = -0.423
        c = 0.123

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))
            qml.var(qml.PauliZ(2))

        res = dev.execute_new(tape)
        expected = np.array(
            [
                np.sin(a) ** 2,
                np.cos(a) * np.cos(b),
                0.25 * (3 - 2 * np.cos(b) ** 2 * np.cos(2 * c) - np.cos(2 * b)),
            ]
        )
        assert isinstance(res, tuple)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute_new(tapes))

        # TODO: check when finite diff ready:
        # tapes, fn = qml.gradients.finite_diff(tape)
        # gradF = fn(dev.batch_execute_new(tapes))

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
        assert isinstance(gradA, tuple)
        for a, e in zip(gradA, expected):
            assert np.allclose(np.squeeze(np.stack(a)), e, atol=tol, rtol=0)
        # assert gradF == pytest.approx(expected, abs=tol)
        qml.disable_return()

    def test_projector_variance(self, tol):
        """Test that the variance of a projector is correctly returned"""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=2)
        P = np.array([1])
        x, y = 0.765, -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.Projector(P, wires=0) @ qml.PauliX(1))

        tape.trainable_params = {0, 1}

        res = dev.execute_new(tape)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute_new(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute_new(tapes))

        expected = np.array(
            [
                [
                    0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                    -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
                ]
            ]
        )
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)
        qml.disable_return()

    def cost1(x):
        qml.Rot(*x, wires=0)
        return qml.expval(qml.PauliZ(0))

    def cost2(x):
        qml.Rot(*x, wires=0)
        return [qml.expval(qml.PauliZ(0))]

    def cost3(x):
        qml.Rot(*x, wires=0)
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    def cost4(x):
        qml.Rot(*x, wires=0)
        return qml.probs([0, 1])

    def cost5(x):
        qml.Rot(*x, wires=0)
        return [qml.probs([0, 1])]

    def cost6(x):
        qml.Rot(*x, wires=0)
        return [qml.probs([0, 1]), qml.probs([2, 3])]

    costs_and_expected_expval = [
        (cost1, [3], False),
        (cost2, [3], True),
        (cost3, [3, 2], True),
    ]

    @pytest.mark.parametrize("cost, expected_shape, list_output", costs_and_expected_expval)
    def test_output_shape_matches_qnode_expval(self, cost, expected_shape, list_output):
        """Test that the transform output shape matches that of the QNode."""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res = qml.gradients.param_shift(circuit)(x)
        assert isinstance(res, tuple)
        assert len(res) == expected_shape[0]

        if len(expected_shape) > 1:
            for r in res:
                assert isinstance(r, tuple)
                assert len(r) == expected_shape[1]

        qml.disable_return()

    costs_and_expected_probs = [
        (cost4, [3, 4], False),
        (cost5, [3, 4], True),
        # The output shape of transforms for 2D qnode outputs (cost6) is currently
        # transposed, e.g. (4, 1, 3) instead of (1, 4, 3).
        # TODO: fix qnode/expected once #2296 is resolved
        (cost6, [3, 2, 4], True),
    ]

    @pytest.mark.parametrize("cost, expected_shape, list_output", costs_and_expected_probs)
    def test_output_shape_matches_qnode_probs(self, cost, expected_shape, list_output):
        """Test that the transform output shape matches that of the QNode."""
        qml.enable_return()
        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res = qml.gradients.param_shift(circuit)(x)
        assert isinstance(res, tuple)
        assert len(res) == expected_shape[0]

        if len(expected_shape) > 2:
            for r in res:
                assert isinstance(r, tuple)
                assert len(r) == expected_shape[1]

                for idx in range(len(r)):
                    assert isinstance(r[idx], qml.numpy.ndarray)
                    assert len(r[idx]) == expected_shape[2]

        elif len(expected_shape) > 1:
            for r in res:
                assert isinstance(r, qml.numpy.ndarray)
                assert len(r) == expected_shape[1]

        qml.disable_return()

    def test_special_observable_qnode_differentiation(self):
        """Test differentiation of a QNode on a device supporting a
        special observable that returns an object rather than a number."""
        qml.enable_return()

        class SpecialObject:
            """SpecialObject

            A special object that conveniently encapsulates the return value of
            a special observable supported by a special device and which supports
            multiplication with scalars and addition.
            """

            def __init__(self, val):
                self.val = val

            def __mul__(self, other):
                return SpecialObject(self.val * other)

            def __add__(self, other):
                newval = self.val + other.val if isinstance(other, self.__class__) else other
                return SpecialObject(newval)

        class SpecialObservable(Observable):
            """SpecialObservable"""

            num_wires = AnyWires

            def diagonalizing_gates(self):
                """Diagonalizing gates"""
                return []

        class DeviceSupporingSpecialObservable(DefaultQubit):
            name = "Device supporting SpecialObservable"
            short_name = "default.qubit.specialobservable"
            observables = DefaultQubit.observables.union({"SpecialObservable"})

            @staticmethod
            def _asarray(arr, dtype=None):
                return arr

            def init(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.R_DTYPE = SpecialObservable

            def expval(self, observable, **kwargs):
                if self.analytic and isinstance(observable, SpecialObservable):
                    val = super().expval(qml.PauliZ(wires=0), **kwargs)
                    return np.array(SpecialObject(val))

                return super().expval(observable, **kwargs)

        dev = DeviceSupporingSpecialObservable(wires=1, shots=None)

        @qml.qnode(dev, diff_method="parameter-shift")
        def qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(SpecialObservable(wires=0))

        @qml.qnode(dev, diff_method="parameter-shift")
        def reference_qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        par = np.array(0.2, requires_grad=True)
        assert np.isclose(qnode(par).item().val, reference_qnode(par))
        assert np.isclose(qml.jacobian(qnode)(par).item().val, qml.jacobian(reference_qnode)(par))
        qml.disable_return()


class TestParamShiftGradients:
    """Test that the transform is differentiable"""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests that the output of the parameter-shift transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.param_shift(tape)
            jac = fn(dev.batch_execute_new(tapes))
            return jac

        res = qml.jacobian(cost_fn)(params)
        x, y = params
        expected = np.array(
            [
                [2 * np.cos(2 * x) * np.sin(y) ** 2, np.sin(2 * x) * np.sin(2 * y)],
                [np.sin(2 * x) * np.sin(2 * y), -2 * np.cos(x) ** 2 * np.cos(2 * y)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)
        qml.disable_return()


# class TestHamiltonianExpvalGradients:
#     """Test that tapes ending with expval(H) can be
#     differentiated"""
#
#     def test_not_expval_error(self):
#         """Test that if the variance of the Hamiltonian is requested,
#         an error is raised"""
#         dev = qml.device("default.qubit", wires=2)
#
#         obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
#         coeffs = np.array([0.1, 0.2, 0.3])
#         H = qml.Hamiltonian(coeffs, obs)
#
#         weights = np.array([0.4, 0.5])
#
#         with qml.tape.QuantumTape() as tape:
#             qml.RX(weights[0], wires=0)
#             qml.RY(weights[1], wires=1)
#             qml.CNOT(wires=[0, 1])
#             qml.var(H)
#
#         tape.trainable_params = {2, 3, 4}
#
#         with pytest.raises(ValueError, match="for expectations, not var"):
#             qml.gradients.param_shift(tape)
#
#     def test_no_trainable_coeffs(self, mocker, tol):
#         """Test no trainable Hamiltonian coefficients"""
#         dev = qml.device("default.qubit", wires=2)
#         spy = mocker.spy(qml.gradients, "hamiltonian_grad")
#
#         obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
#         coeffs = np.array([0.1, 0.2, 0.3])
#         H = qml.Hamiltonian(coeffs, obs)
#
#         weights = np.array([0.4, 0.5])
#
#         with qml.tape.QuantumTape() as tape:
#             qml.RX(weights[0], wires=0)
#             qml.RY(weights[1], wires=1)
#             qml.CNOT(wires=[0, 1])
#             qml.expval(H)
#
#         a, b, c = coeffs
#         x, y = weights
#         tape.trainable_params = {0, 1}
#
#         res = dev.batch_execute_new([tape])
#         expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
#         assert np.allclose(res, expected, atol=tol, rtol=0)
#
#         tapes, fn = qml.gradients.param_shift(tape)
#         # two shifts per rotation gate, one circuit per trainable H term
#         assert len(tapes) == 2 * 2
#         spy.assert_not_called()
#
#         res = fn(dev.batch_execute_new(tapes))
#         assert res.shape == (1, 2)
#
#         expected = [
#             -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
#             b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
#         ]
#         assert np.allclose(res, expected, atol=tol, rtol=0)
#
#     def test_trainable_coeffs(self, mocker, tol):
#         """Test trainable Hamiltonian coefficients"""
#         dev = qml.device("default.qubit", wires=2)
#         spy = mocker.spy(qml.gradients, "hamiltonian_grad")
#
#         obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
#         coeffs = np.array([0.1, 0.2, 0.3])
#         H = qml.Hamiltonian(coeffs, obs)
#
#         weights = np.array([0.4, 0.5])
#
#         with qml.tape.QuantumTape() as tape:
#             qml.RX(weights[0], wires=0)
#             qml.RY(weights[1], wires=1)
#             qml.CNOT(wires=[0, 1])
#             qml.expval(H)
#
#         a, b, c = coeffs
#         x, y = weights
#         tape.trainable_params = {0, 1, 2, 4}
#
#         res = dev.batch_execute_new([tape])
#         expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
#         assert np.allclose(res, expected, atol=tol, rtol=0)
#
#         tapes, fn = qml.gradients.param_shift(tape)
#         # two shifts per rotation gate, one circuit per trainable H term
#         assert len(tapes) == 2 * 2 + 2
#         spy.assert_called()
#
#         res = fn(dev.batch_execute_new(tapes))
#         assert res.shape == (1, 4)
#
#         expected = [
#             -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
#             b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
#             np.cos(x),
#             -(np.sin(x) * np.sin(y)),
#         ]
#         assert np.allclose(res, expected, atol=tol, rtol=0)
#
#     def test_multiple_hamiltonians(self, mocker, tol):
#         """Test multiple trainable Hamiltonian coefficients"""
#         dev = qml.device("default.qubit", wires=2)
#         spy = mocker.spy(qml.gradients, "hamiltonian_grad")
#
#         obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
#         coeffs = np.array([0.1, 0.2, 0.3])
#         a, b, c = coeffs
#         H1 = qml.Hamiltonian(coeffs, obs)
#
#         obs = [qml.PauliZ(0)]
#         coeffs = np.array([0.7])
#         d = coeffs[0]
#         H2 = qml.Hamiltonian(coeffs, obs)
#
#         weights = np.array([0.4, 0.5])
#         x, y = weights
#
#         with qml.tape.QuantumTape() as tape:
#             qml.RX(weights[0], wires=0)
#             qml.RY(weights[1], wires=1)
#             qml.CNOT(wires=[0, 1])
#             qml.expval(H1)
#             qml.expval(H2)
#
#         tape.trainable_params = {0, 1, 2, 4, 5}
#
#         res = dev.batch_execute_new([tape])
#         expected = [-c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y)), d * np.cos(x)]
#         assert np.allclose(res, expected, atol=tol, rtol=0)
#
#         tapes, fn = qml.gradients.param_shift(tape)
#         # two shifts per rotation gate, one circuit per trainable H term
#         assert len(tapes) == 2 * 2 + 3
#         spy.assert_called()
#
#         res = fn(dev.batch_execute_new(tapes))
#         assert res.shape == (2, 5)
#
#         expected = [
#             [
#                 -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
#                 b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
#                 np.cos(x),
#                 -(np.sin(x) * np.sin(y)),
#                 0,
#             ],
#             [-d * np.sin(x), 0, 0, 0, np.cos(x)],
#         ]
#
#         assert np.allclose(res, expected, atol=tol, rtol=0)
#
#     @staticmethod
#     def cost_fn(weights, coeffs1, coeffs2, dev=None):
#         """Cost function for gradient tests"""
#         obs1 = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
#         H1 = qml.Hamiltonian(coeffs1, obs1)
#
#         obs2 = [qml.PauliZ(0)]
#         H2 = qml.Hamiltonian(coeffs2, obs2)
#
#         with qml.tape.QuantumTape() as tape:
#             qml.RX(weights[0], wires=0)
#             qml.RY(weights[1], wires=1)
#             qml.CNOT(wires=[0, 1])
#             qml.expval(H1)
#             qml.expval(H2)
#
#         tape.trainable_params = {0, 1, 2, 3, 4, 5}
#         tapes, fn = qml.gradients.param_shift(tape)
#         jac = fn(dev.batch_execute_new(tapes))
#         return jac
#
#     @staticmethod
#     def cost_fn_expected(weights, coeffs1, coeffs2):
#         """Analytic jacobian of cost_fn above"""
#         a, b, c = coeffs1
#         d = coeffs2[0]
#         x, y = weights
#         return [
#             [
#                 -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
#                 b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
#                 np.cos(x),
#                 np.cos(x) * np.sin(y),
#                 -(np.sin(x) * np.sin(y)),
#                 0,
#             ],
#             [-d * np.sin(x), 0, 0, 0, 0, np.cos(x)],
#         ]
#
#     @pytest.mark.autograd
#     def test_autograd(self, tol):
#         """Test gradient of multiple trainable Hamiltonian coefficients
#         using autograd"""
#         coeffs1 = np.array([0.1, 0.2, 0.3], requires_grad=True)
#         coeffs2 = np.array([0.7], requires_grad=True)
#         weights = np.array([0.4, 0.5], requires_grad=True)
#         dev = qml.device("default.qubit.autograd", wires=2)
#
#         res = self.cost_fn(weights, coeffs1, coeffs2, dev=dev)
#         expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
#         assert np.allclose(res, expected, atol=tol, rtol=0)
#
#         # second derivative wrt to Hamiltonian coefficients should be zero
#         res = qml.jacobian(self.cost_fn)(weights, coeffs1, coeffs2, dev=dev)
#         assert np.allclose(res[1][:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
#         assert np.allclose(res[2][:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)
#
#     @pytest.mark.tf
#     @pytest.mark.slow
#     def test_tf(self, tol):
#         """Test gradient of multiple trainable Hamiltonian coefficients
#         using tf"""
#         import tensorflow as tf
#
#         coeffs1 = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)
#         coeffs2 = tf.Variable([0.7], dtype=tf.float64)
#         weights = tf.Variable([0.4, 0.5], dtype=tf.float64)
#
#         dev = qml.device("default.qubit.tf", wires=2)
#
#         with tf.GradientTape() as t:
#             jac = self.cost_fn(weights, coeffs1, coeffs2, dev=dev)
#
#         expected = self.cost_fn_expected(weights.numpy(), coeffs1.numpy(), coeffs2.numpy())
#         assert np.allclose(jac, expected, atol=tol, rtol=0)
#
#         # second derivative wrt to Hamiltonian coefficients should be zero
#         hess = t.jacobian(jac, [coeffs1, coeffs2])
#         assert np.allclose(hess[0][:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
#         assert np.allclose(hess[1][:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)
#
#     @pytest.mark.torch
#     def test_torch(self, tol):
#         """Test gradient of multiple trainable Hamiltonian coefficients
#         using torch"""
#         import torch
#
#         coeffs1 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
#         coeffs2 = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
#         weights = torch.tensor([0.4, 0.5], dtype=torch.float64, requires_grad=True)
#
#         dev = qml.device("default.qubit.torch", wires=2)
#
#         res = self.cost_fn(weights, coeffs1, coeffs2, dev=dev)
#         expected = self.cost_fn_expected(
#             weights.detach().numpy(), coeffs1.detach().numpy(), coeffs2.detach().numpy()
#         )
#         assert np.allclose(res.detach(), expected, atol=tol, rtol=0)
#
#         # second derivative wrt to Hamiltonian coefficients should be zero
#         hess = torch.autograd.functional.jacobian(
#             lambda *args: self.cost_fn(*args, dev=dev), (weights, coeffs1, coeffs2)
#         )
#         assert np.allclose(hess[1][:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
#         assert np.allclose(hess[2][:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)
#
#     @pytest.mark.jax
#     @pytest.mark.slow
#     def test_jax(self, tol):
#         """Test gradient of multiple trainable Hamiltonian coefficients
#         using JAX"""
#         import jax
#
#         jnp = jax.numpy
#
#         coeffs1 = jnp.array([0.1, 0.2, 0.3])
#         coeffs2 = jnp.array([0.7])
#         weights = jnp.array([0.4, 0.5])
#         dev = qml.device("default.qubit.jax", wires=2)
#
#         res = self.cost_fn(weights, coeffs1, coeffs2, dev=dev)
#         expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
#         assert np.allclose(res, expected, atol=tol, rtol=0)
#
#         # second derivative wrt to Hamiltonian coefficients should be zero
#         res = jax.jacobian(self.cost_fn, argnums=1)(weights, coeffs1, coeffs2, dev=dev)
#         assert np.allclose(res[:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
#
#         res = jax.jacobian(self.cost_fn, argnums=1)(weights, coeffs1, coeffs2, dev=dev)
#         assert np.allclose(res[:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)

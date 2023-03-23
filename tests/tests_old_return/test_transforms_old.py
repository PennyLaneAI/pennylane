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
Tests for transforms.
"""
import pytest
import numpy as onp
from pennylane import numpy as np

import pennylane as qml
from pennylane.gradients import finite_diff, param_shift, hadamard_grad, stoch_pulse_grad, spsa_grad
from pennylane.qinfo import classical_fisher, quantum_fisher
from pennylane.transforms import (
    mitigate_with_zne,
    richardson_extrapolate,
    fold_global,
)
from pennylane.measurements.classical_shadow import ShadowExpvalMP


@pytest.mark.jax
def test_stoch_pulse_grad_raises():
    """Test that stoch_pulse_grad raises a NotImplementedError."""
    tape = qml.tape.QuantumScript()
    with pytest.raises(NotImplementedError, match="The stochastic pulse parameter-shift"):
        stoch_pulse_grad(tape)


def test_stoch_pulse_grad_raises_without_jax_installed():
    """Test that an error is raised if a stoch_pulse_grad is called without jax installed"""
    try:
        import jax  # pylint: disable=unused-import

        pytest.skip()
    except ImportError:
        tape = qml.tape.QuantumScript([], [])
        with pytest.raises(ImportError, match="Module jax is required"):
            stoch_pulse_grad(tape)


def test_hadamard_grad_raises():
    """Test that hadamard_grad function raises a NotImplementedError."""
    tape = qml.tape.QuantumScript()
    with pytest.raises(NotImplementedError, match="The Hadamard gradient"):
        hadamard_grad(tape)


@pytest.mark.parametrize("approx_order", [2, 4])
@pytest.mark.parametrize("strategy", ["forward", "backward", "center"])
class TestFiniteDiffIntegration:
    """Tests for the finite difference gradient transform"""

    def test_ragged_output(self, approx_order, strategy):
        """Test that the Jacobian is correctly returned for a tape
        with ragged output"""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1, 2])

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = finite_diff(tape, approx_order=approx_order, strategy=strategy)
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (6, 3)

    def test_single_expectation_value(self, approx_order, strategy, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = finite_diff(tape, approx_order=approx_order, strategy=strategy)
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestParameterShiftRule:
    """Tests for the parameter shift implementation"""

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift])
    def test_pauli_rotation_gradient(self, mocker, G, theta, shift, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        dev = qml.device("default.qubit", wires=1)

        with qml.queuing.AnnotatedQueue() as q:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)])
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == None

        autograd_val = fn(dev.batch_execute(tapes))

        tape_fwd, tape_bwd = tape.copy(copy_operations=True), tape.copy(copy_operations=True)
        tape_fwd.set_parameters([theta + np.pi / 2])
        tape_bwd.set_parameters([theta - np.pi / 2])

        manualgrad_val = np.subtract(*dev.batch_execute([tape_fwd, tape_bwd])) / 2
        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)

        assert spy.call_args[1]["shifts"] == (shift,)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_fallback(self, mocker, tol):
        """Test that fallback gradient functions are correctly used"""
        spy = mocker.spy(qml.gradients, "finite_diff")
        dev = qml.device("default.qubit.autograd", wires=2)
        x = 0.543
        y = -0.654

        params = np.array([x, y], requires_grad=True)

        class RY(qml.RY):
            grad_method = "F"

        def cost_fn(params):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                RY(params[1], wires=[1])  # Use finite differences for this op
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.var(qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff)
            assert len(tapes) == 5

            # check that the fallback method was called for the specified argnums
            spy.assert_called()
            assert spy.call_args[1]["argnum"] == {1}

            return fn(dev.batch_execute(tapes))

        res = cost_fn(params)
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # double check the derivative
        jac = qml.jacobian(cost_fn)(params)
        assert np.allclose(jac[0, 0, 0], -np.cos(x), atol=tol, rtol=0)
        assert np.allclose(jac[1, 1, 1], -2 * np.cos(2 * y), atol=tol, rtol=0)

    def test_single_expectation_value(self, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4
        assert [t.batch_size for t in tapes] == [None] * 4

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4
        assert [t.batch_size for t in tapes] == [None] * 4

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 5  # One unshifted, four shifted tapes
        assert [t.batch_size for t in tapes] == [None] * 5

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        dev.execute(tape)

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4
        assert [t.batch_size for t in tapes] == [None] * 4

        res = fn(dev.batch_execute(tapes))
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

    def test_involutory_variance(self, tol):
        """Tests qubit observables that are involutory"""
        dev = qml.device("default.qubit", wires=1)
        a = 0.54

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.var(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)
        expected = 1 - np.cos(a) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 2 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 2

        expected = 2 * np.sin(a) * np.cos(a)

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_non_involutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory"""
        dev = qml.device("default.qubit", wires=1)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.var(qml.Hermitian(A, 0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0}

        res = dev.execute(tape)
        expected = (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 4 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 2

        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_involutory_and_noninvolutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable."""
        dev = qml.device("default.qubit", wires=2)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}

        res = dev.execute(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 2 * 4

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 2

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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))
            qml.var(qml.PauliZ(2))

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)
        expected = np.array(
            [
                np.sin(a) ** 2,
                np.cos(a) * np.cos(b),
                0.25 * (3 - 2 * np.cos(b) ** 2 * np.cos(2 * c) - np.cos(2 * b)),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))

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

    def test_recycling_unshifted_tape_result(self):
        """Test that an unshifted term in the used gradient recipe is reused
        for the chain rule computation within the variance parameter shift rule."""
        dev = qml.device("default.qubit", wires=2)
        gradient_recipes = ([[-1e-5, 1, 0], [1e-5, 1, 0], [-1e5, 1, -5e-6], [1e5, 1, 5e-6]], None)
        x = [0.543, -0.654]

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[0])
            qml.var(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)
        # 2 operations x 2 shifted positions + 1 unshifted term overall
        assert len(tapes) == 2 * 2 + 1

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[0])
            qml.var(qml.Projector([1], wires=0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = [0, 1]
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)
        for tape in tapes:
            print(tape.measurements)
        # 2 operations x 2 shifted positions + 1 unshifted term overall    <-- <H>
        # + 2 operations x 2 shifted positions + 1 unshifted term          <-- <H^2>
        assert len(tapes) == (2 * 2 + 1) + (2 * 2 + 1)

    def test_projector_variance(self, tol):
        """Test that the variance of a projector is correctly returned"""
        dev = qml.device("default.qubit", wires=2)
        P = np.array([1])
        x, y = 0.765, -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.Projector(P, wires=0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = [0, 1]

        res = dev.execute(tape)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))

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


class TestParameterShiftHessian:
    """Test the general functionality of the param_shift_hessian method
    on the default interface (autograd)"""

    def test_single_two_term_gate(self):
        """Test that the correct hessian is calculated for a QNode with single RX operator
        and single expectation value output (0d -> 0d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        expected = qml.jacobian(qml.grad(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert circuit.interface == "auto"
        assert np.allclose(expected, hessian)

    def test_fixed_params(self):
        """Test that the correct hessian is calculated for a QNode with single RX operator
        and single expectation value output (0d -> 0d) where some fixed parameters gate are added"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RZ(0.1, wires=0)
            qml.RZ(-0.1, wires=0)
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        expected = qml.jacobian(qml.grad(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_gate_without_impact(self):
        """Test that the correct hessian is calculated for a QNode with an operator
        that does not have any impact on the QNode output."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.PauliZ(0))

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qml.jacobian(qml.grad(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    @pytest.mark.filterwarnings("ignore:Output seems independent of input.")
    def test_no_gate_with_impact(self):
        """Test that the correct hessian is calculated for a QNode without any
        operators that have an impact on the QNode output."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=2)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.PauliZ(0))

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qml.jacobian(qml.grad(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_single_multi_term_gate(self):
        """Test that the correct hessian is calculated for a QNode with single operation
        with more than two terms in the shift rule, parameter frequencies defined,
        and single expectation value output (0d -> 0d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.Hadamard(wires=1)
            qml.CRX(x, wires=[1, 0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        expected = qml.jacobian(qml.grad(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_single_gate_custom_recipe(self):
        """Test that the correct hessian is calculated for a QNode with single operation
        with more than two terms in the shift rule, parameter frequencies defined,
        and single expectation value output (0d -> 0d)"""

        dev = qml.device("default.qubit", wires=2)

        c, s = qml.gradients.generate_shift_rule((0.5, 1)).T
        recipe = list(zip(c, np.ones_like(c), s))

        class DummyOp(qml.CRX):
            grad_recipe = (recipe,)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.Hadamard(wires=1)
            DummyOp(x, wires=[1, 0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        expected = qml.jacobian(qml.grad(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_single_two_term_gate_vector_output(self):
        """Test that the correct hessian is calculated for a QNode with single RY operator
        and probabilies as output (0d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = np.array(0.1, requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates(self):
        """Test that the correct hessian is calculated for a QNode with two rotation operators
        and one expectation value output (1d -> 0d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x[2], wires=1)
            return qml.expval(qml.PauliZ(1))

        x = np.array([0.1, 0.2, -0.8], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates_vector_output(self):
        """Test that the correct hessian is calculated for a QNode with two rotation operators
        and probabilities output (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x[2], wires=1)
            return qml.probs(wires=1)

        x = np.array([0.1, 0.2, -0.8], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_quantum_hessian_shape_vector_input_vector_output(self):
        """Test that the purely "quantum" hessian has the correct shape (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(x[2], wires=1)
            qml.Rot(x[0], x[1], x[2], wires=1)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)
        shape = (4, 6, 6)  # (num_output_vals, num_gate_args, num_gate_args)

        hessian = qml.gradients.param_shift_hessian(circuit, hybrid=False)(x)

        assert qml.math.shape(hessian) == shape

    def test_multiple_two_term_gates_reusing_parameters(self):
        """Test that the correct hessian is calculated when reusing parameters (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(x[2], wires=1)
            qml.Rot(x[0], x[1], x[2], wires=1)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates_classical_processing(self):
        """Test that the correct hessian is calculated when manipulating parameters (1d -> 1d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0] + x[1] + x[2], wires=0)
            qml.RY(x[1] - x[0] + 3 * x[2], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(x[2] / x[0] - x[1], wires=1)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates_matrix_output(self):
        """Test that the correct hessian is calculated for higher dimensional QNode outputs
        (1d -> 2d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        x = np.ones([2], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates_matrix_input(self):
        """Test that the correct hessian is calculated for higher dimensional cl. jacobians
        (2d -> 2d)"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0, 0], wires=0)
            qml.RY(x[0, 1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[0, 2], wires=0)
            qml.RY(x[0, 0], wires=0)
            return qml.probs(wires=0), qml.probs(wires=1)

        x = np.ones([1, 3], requires_grad=True)

        expected = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_qnode_arguments_scalar(self):
        """Test that the correct Hessian is calculated with multiple QNode arguments (0D->1D)"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.SingleExcitation(z, wires=[1, 0])
            qml.RY(y, wires=0)
            qml.RX(x, wires=0)
            return qml.probs(wires=[0, 1])

        wrapper = lambda X: circuit(*X)

        x = np.array(0.1, requires_grad=True)
        y = np.array(0.5, requires_grad=True)
        z = np.array(0.3, requires_grad=True)
        X = qml.math.stack([x, y, z])

        expected = qml.jacobian(qml.jacobian(wrapper))(X)
        expected = tuple(expected[:, i, i] for i in range(3))
        circuit.interface = "autograd"
        hessian = qml.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(expected, hessian)

    def test_multiple_qnode_arguments_vector(self):
        """Test that the correct Hessian is calculated with multiple QNode arguments (1D->1D)"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x, y, z):
            qml.RX(x[0], wires=1)
            qml.RY(y[0], wires=0)
            qml.CRZ(z[0] + z[1], wires=[1, 0])
            qml.RY(y[1], wires=1)
            qml.RX(x[1], wires=0)
            return qml.probs(wires=[0, 1])

        wrapper = lambda X: circuit(*X)

        x = np.array([0.1, 0.3], requires_grad=True)
        y = np.array([0.5, 0.7], requires_grad=True)
        z = np.array([0.3, 0.2], requires_grad=True)
        X = qml.math.stack([x, y, z])

        expected = qml.jacobian(qml.jacobian(wrapper))(X)
        expected = tuple(expected[:, i, :, i] for i in range(3))

        circuit.interface = "autograd"
        hessian = qml.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(expected, hessian)

    def test_multiple_qnode_arguments_matrix(self):
        """Test that the correct Hessian is calculated with multiple QNode arguments (2D->1D)"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x, y, z):
            qml.RX(x[0, 0], wires=0)
            qml.RY(y[0, 0], wires=1)
            qml.CRZ(z[0, 0] + z[1, 1], wires=[1, 0])
            qml.RY(y[1, 0], wires=0)
            qml.RX(x[1, 0], wires=1)
            return qml.probs(wires=[0, 1])

        wrapper = lambda X: circuit(*X)

        x = np.array([[0.1, 0.3], [0.2, 0.4]], requires_grad=True)
        y = np.array([[0.5, 0.7], [0.2, 0.4]], requires_grad=True)
        z = np.array([[0.3, 0.2], [0.2, 0.4]], requires_grad=True)
        X = qml.math.stack([x, y, z])

        expected = qml.jacobian(qml.jacobian(wrapper))(X)
        expected = tuple(expected[:, i, :, :, i] for i in range(3))

        circuit.interface = "autograd"
        hessian = qml.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(expected, hessian)

    def test_multiple_qnode_arguments_mixed(self):
        """Test that the correct Hessian is calculated with multiple mixed-shape QNode arguments"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, max_diff=2, diff_method="parameter-shift")
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(z[0] + z[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(y[1, 0], wires=0)
            qml.CRY(y[0, 1], wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        x = np.array(0.1, requires_grad=True)
        y = np.array([[0.5, 0.6], [0.2, 0.1]], requires_grad=True)
        z = np.array([0.3, 0.4], requires_grad=True)

        expected = tuple(
            qml.jacobian(qml.jacobian(circuit, argnum=i), argnum=i)(x, y, z) for i in range(3)
        )
        hessian = qml.gradients.param_shift_hessian(circuit)(x, y, z)

        assert all(np.allclose(expected[i], hessian[i]) for i in range(3))


def coordinate_sampler(indices, num_params, idx, seed=None):
    """Return a single canonical basis vector, corresponding
    to the index ``indices[idx]``. This is a sequential coordinate sampler
    that allows to exactly reproduce derivatives, instead of using SPSA in the
    intended way."""
    # pylint: disable=unused-argument
    idx = idx % len(indices)
    direction = np.zeros(num_params)
    direction[indices[idx]] = 1.0
    return direction


@pytest.mark.parametrize("approx_order", [2, 4])
@pytest.mark.parametrize("strategy", ["forward", "backward", "center"])
@pytest.mark.parametrize("validate", [True, False])
class TestSpsaGradientIntegration:
    """Tests for the SPSA gradient transform"""

    def test_ragged_output(self, approx_order, strategy, validate):
        """Test that the Jacobian is correctly returned for a tape with ragged output"""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1, 2])

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=11,
            validate_params=validate,
        )
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (6, 3)

    def test_single_expectation_value(self, approx_order, strategy, tol, validate):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            h=1e-6,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=6,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res *= 2

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value_with_argnum_all(self, approx_order, strategy, tol, validate):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where all parameters are chosen to compute
        the jacobian"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # we choose both trainable parameters
        tapes, fn = spsa_grad(
            tape,
            argnum=[0, 1],
            approx_order=approx_order,
            strategy=strategy,
            num_directions=8,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res *= 2

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value_with_argnum_one(self, approx_order, strategy, tol, validate):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where only one parameter is chosen to
        estimate the jacobian.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # we choose only 1 trainable parameter - do not need to account for the multiplicative
        # error of using the coordinate_sampler
        tapes, fn = spsa_grad(
            tape,
            argnum=1,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=11,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        expected = np.array([[0, np.cos(y) * np.cos(x)]])
        res = res.flatten()
        expected = expected.flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, approx_order, strategy, tol, validate):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=12,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2, 2)

        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res *= 2

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_expectation_values(self, approx_order, strategy, tol, validate):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=12,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2, 2)

        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res *= 2

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, approx_order, strategy, tol, validate):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = spsa_grad(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            num_directions=10,
            sampler=coordinate_sampler,
            validate_params=validate,
        )
        res = fn(dev.batch_execute(tapes))

        assert res.shape == (5, 2)
        # The coordinate_sampler produces the right evaluation points, but the tape execution
        # results are averaged instead of added, so that we need to account for the prefactor
        # 1 / num_params here.
        res *= 2

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


fixed_pars = [-0.2, 0.2, 0.5, 0.3, 0.7]


def fubini_ansatz0(params, wires=None):
    qml.RX(params[0], wires=0)
    qml.RY(fixed_pars[0], wires=0)
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(params[1], wires=0)
    qml.CNOT(wires=[wires[0], wires[1]])


def fubini_ansatz1(params, wires=None):
    qml.RX(fixed_pars[1], wires=0)
    for wire in wires:
        qml.Rot(*params[0][wire], wires=wire)
    qml.CNOT(wires=[0, 1])
    qml.adjoint(qml.RY(fixed_pars[1], wires=0))
    qml.CNOT(wires=[1, 2])
    for wire in wires:
        qml.Rot(*params[1][wire], wires=wire)
    qml.CNOT(wires=[1, 2])
    qml.RX(fixed_pars[2], wires=1)


def fubini_ansatz2(params, wires=None):
    params0 = params[0]
    params1 = params[1]
    qml.RX(fixed_pars[1], wires=0)
    qml.Rot(*fixed_pars[2:5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params0, wires=0)
    qml.RY(params0, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.adjoint(qml.RX(params1, wires=0))
    qml.RX(params1, wires=1)


def fubini_ansatz3(params, wires=None):
    params0 = params[0]
    params1 = params[1]
    params2 = params[2]
    qml.RX(fixed_pars[1], wires=0)
    qml.RX(fixed_pars[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RX(params0, wires=0)
    qml.RX(params0, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    qml.RY(params1, wires=0)
    qml.RY(params1, wires=1)
    qml.RY(params1, wires=2)
    qml.RZ(params2, wires=0)
    qml.RZ(params2, wires=1)
    qml.RZ(params2, wires=2)


def fubini_ansatz4(params00, params_rest, wires=None):
    params01 = params_rest[0]
    params10 = params_rest[1]
    params11 = params_rest[2]
    qml.RY(fixed_pars[3], wires=0)
    qml.RY(fixed_pars[2], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RY(fixed_pars[4], wires=0)
    qml.RX(params00, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(params01, wires=1)
    qml.RZ(params10, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params11, wires=1)


def fubini_ansatz5(params, wires=None):
    fubini_ansatz4(params[0], [params[0], params[1], params[1]], wires=wires)


def fubini_ansatz6(params, wires=None):
    fubini_ansatz4(params[0], [params[0], params[1], -params[1]], wires=wires)


def fubini_ansatz7(params0, params1, wires=None):
    fubini_ansatz4(params0[0], [params0[1], params1[0], params1[1]], wires=wires)


def fubini_ansatz8(x, wires=None):
    qml.RX(fixed_pars[0], wires=0)
    qml.RX(x, wires=0)


def fubini_ansatz9(params, wires=None):
    params0 = params[0]
    params1 = params[1]
    qml.RX(fixed_pars[1], wires=[0])
    qml.RY(fixed_pars[3], wires=[0])
    qml.RZ(fixed_pars[2], wires=[0])
    qml.RX(fixed_pars[2], wires=[1])
    qml.RY(fixed_pars[2], wires=[1])
    qml.RZ(fixed_pars[4], wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.RX(fixed_pars[0], wires=[0])
    qml.RY(fixed_pars[1], wires=[0])
    qml.RZ(fixed_pars[3], wires=[0])
    qml.RX(fixed_pars[1], wires=[1])
    qml.RY(fixed_pars[2], wires=[1])
    qml.RZ(fixed_pars[0], wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.RX(params0, wires=[0])
    qml.RX(params0, wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.RY(fixed_pars[4], wires=[1])
    qml.RY(params1, wires=[0])
    qml.RY(params1, wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.RX(fixed_pars[2], wires=[1])


def fubini_ansatz10(weights, wires=None):
    qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])


B = np.array(
    [
        [
            [0.73, 0.49, 0.04],
            [0.29, 0.45, 0.59],
            [0.64, 0.06, 0.26],
        ],
        [
            [0.93, 0.14, 0.46],
            [0.31, 0.83, 0.79],
            [0.25, 0.40, 0.16],
        ],
    ],
    requires_grad=True,
)
fubini_ansatze_tape = [fubini_ansatz0, fubini_ansatz1, fubini_ansatz8]
fubini_params_tape = [
    (np.array([0.3434, -0.7245345], requires_grad=True),),
    (B,),
    (np.array(-0.1735, requires_grad=True),),
]

fubini_ansatze = [
    fubini_ansatz0,
    fubini_ansatz1,
    fubini_ansatz2,
    fubini_ansatz3,
    fubini_ansatz4,
    fubini_ansatz5,
    fubini_ansatz6,
    fubini_ansatz7,
    fubini_ansatz8,
    fubini_ansatz9,
    fubini_ansatz10,
]

fubini_params = [
    (np.array([0.3434, -0.7245345], requires_grad=True),),
    (B,),
    (np.array([-0.1111, -0.2222], requires_grad=True),),
    (np.array([-0.1111, -0.2222, 0.4554], requires_grad=True),),
    (
        np.array(-0.1735, requires_grad=True),
        np.array([-0.1735, -0.2846, -0.2846], requires_grad=True),
    ),
    (np.array([-0.1735, -0.2846], requires_grad=True),),
    (np.array([-0.1735, -0.2846], requires_grad=True),),
    (
        np.array([-0.1735, -0.2846], requires_grad=True),
        np.array([0.9812, -0.1492], requires_grad=True),
    ),
    (np.array(-0.1735, requires_grad=True),),
    (np.array([-0.1111, 0.3333], requires_grad=True),),
    (np.array([[0.21, 9.29], [-0.2, 0.12], [0.3, -2.1]], requires_grad=True),),
]


def autodiff_metric_tensor(ansatz, num_wires):
    """Compute the metric tensor by full state vector
    differentiation via autograd."""
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev)
    def qnode(*params):
        ansatz(*params, wires=dev.wires)
        return qml.state()

    def mt(*params):
        state = qnode(*params)
        rqnode = lambda *params: np.real(qnode(*params))
        iqnode = lambda *params: np.imag(qnode(*params))
        rjac = qml.jacobian(rqnode)(*params)
        ijac = qml.jacobian(iqnode)(*params)

        if isinstance(rjac, tuple):
            out = []
            for rc, ic in zip(rjac, ijac):
                c = rc + 1j * ic
                psidpsi = np.tensordot(np.conj(state), c, axes=([0], [0]))
                out.append(
                    np.real(
                        np.tensordot(np.conj(c), c, axes=([0], [0]))
                        - np.tensordot(np.conj(psidpsi), psidpsi, axes=0)
                    )
                )
            return tuple(out)

        jac = rjac + 1j * ijac
        psidpsi = np.tensordot(np.conj(state), jac, axes=([0], [0]))
        return np.real(
            np.tensordot(np.conj(jac), jac, axes=([0], [0]))
            - np.tensordot(np.conj(psidpsi), psidpsi, axes=0)
        )

    return mt


class TestAdjointMetricTensorQNode:
    """Test the adjoint method for the metric tensor when calling it on
    a QNode.
    """

    num_wires = 3

    @pytest.mark.autograd
    @pytest.mark.parametrize("ansatz, params", list(zip(fubini_ansatze, fubini_params)))
    @pytest.mark.parametrize("interface", ["auto", "autograd"])
    def test_correct_output_qnode_autograd(self, ansatz, params, interface):
        """Test that the output is correct when using Autograd and
        calling the adjoint metric tensor on a QNode."""
        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        dev = qml.device("default.qubit", wires=self.num_wires)

        @qml.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qml.expval(qml.PauliZ(0))

        mt = qml.adjoint_metric_tensor(circuit)(*params)

        if isinstance(mt, tuple):
            assert all(qml.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qml.math.allclose(mt, expected)


@pytest.mark.autograd
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint", "parameter-shift"])
def test_batch_input_autograd(diff_method, tol):
    """Test derivatives when using autograd"""
    dev = qml.device("default.qubit", wires=2)

    @qml.batch_input(argnum=0)
    @qml.qnode(dev, diff_method=diff_method)
    def circuit(input, x):
        qml.RY(input, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    batch_size = 3

    def cost(input, x):
        return np.sum(circuit(input, x))

    input = np.linspace(0.1, 0.5, batch_size, requires_grad=False)
    x = np.array(0.1, requires_grad=True)

    res = qml.grad(cost)(input, x)
    expected = -np.sin(0.1) * sum(np.sin(input))
    assert np.allclose(res, expected, atol=tol, rtol=0)


def test_batch_params_simple_circuit(mocker):
    """Test that batching works for a simple circuit"""
    dev = qml.device("default.qubit", wires=3)

    @qml.batch_params
    @qml.qnode(dev, interface="autograd")
    def circuit(data, x, weights):
        qml.templates.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True)
        qml.RX(x, wires=0)
        qml.RY(0.2, wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 2])

    batch_size = 5
    data = np.random.random((batch_size, 8))
    x = np.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = np.ones((batch_size, 10, 3, 3), requires_grad=True)

    spy = mocker.spy(circuit.device, "batch_execute")
    res = circuit(data, x, weights)
    assert res.shape == (batch_size, 4)
    assert len(spy.call_args[0][0]) == batch_size


class TestFisherIntegration:
    """Integration test of classical and quantum fisher information matrices"""

    @pytest.mark.parametrize("n_wires", onp.arange(1, 5))
    @pytest.mark.parametrize("n_params", onp.arange(1, 5))
    def test_different_sizes(self, n_wires, n_params):
        """Testing that for any number of wires and parameters, the correct size and values are computed"""
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            for i in range(n_wires):
                qml.Hadamard(wires=i)

            for x in params:
                for j in range(n_wires):
                    qml.RX(x, wires=j)
                    qml.RY(x, wires=j)
                    qml.RZ(x, wires=j)

            return qml.probs(wires=range(n_wires))

        params = np.zeros(n_params, requires_grad=True)
        res = classical_fisher(circ)(params)
        assert circ.interface == "auto"
        assert np.allclose(res, n_wires * onp.ones((n_params, n_params)))

    def test_quantum_fisher_info(
        self,
    ):
        """Integration test of quantum fisher information matrix CFIM. This is just calling ``qml.metric_tensor`` or
        ``qml.adjoint_metric_tensor`` and multiplying by a factor of 4"""

        n_wires = 2

        dev = qml.device("default.qubit", wires=n_wires)
        dev_hard = qml.device("default.qubit", wires=n_wires + 1, shots=1000)

        def qfunc(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=0)
            qml.CNOT(wires=(0, 1))
            return qml.state()

        params = np.random.random(2)

        circ_hard = qml.QNode(qfunc, dev_hard)
        QFIM_hard = quantum_fisher(circ_hard)(params)
        QFIM1_hard = 4.0 * qml.metric_tensor(circ_hard)(params)

        circ = qml.QNode(qfunc, dev)
        QFIM = quantum_fisher(circ)(params)
        QFIM1 = 4.0 * qml.adjoint_metric_tensor(circ)(params)
        assert np.allclose(QFIM, QFIM1)
        assert np.allclose(QFIM_hard, QFIM1_hard, atol=1e-1)


class TestFullMetricTensor:
    num_wires = 3
    interfaces = ["auto", "autograd"]

    @pytest.mark.autograd
    @pytest.mark.parametrize("ansatz, params", zip(fubini_ansatze, fubini_params))
    @pytest.mark.parametrize("interface", interfaces)
    def test_correct_output_autograd(self, ansatz, params, interface):
        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        dev = qml.device("default.qubit.autograd", wires=self.num_wires + 1)

        @qml.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires[:-1])
            return qml.expval(qml.PauliZ(0))

        mt = qml.metric_tensor(circuit, approx=None)(*params)

        if interface == "auto":
            assert circuit.interface == "auto"

        if isinstance(mt, tuple):
            assert all(qml.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qml.math.allclose(mt, expected)


def qfunc(theta):
    qml.RY(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    return qml.expval(1 * qml.PauliZ(0) + 2 * qml.PauliZ(1))


n_wires = 2

# Describe noise
noise_gate = qml.PhaseDamping
noise_strength = 0.05

# Load devices
dev_ideal = qml.device("default.mixed", wires=n_wires)
dev_noisy = qml.transforms.insert(noise_gate, noise_strength)(dev_ideal)

out_ideal = np.sqrt(2) / 2 + np.sqrt(2)
grad_ideal_0 = [-np.sqrt(2) / 2, -np.sqrt(2)]


class TestDifferentiableZNE:
    """Testing differentiable ZNE"""

    @pytest.mark.autograd
    def test_diffability_autograd(self):
        """Testing that the mitigated qnode can be differentiated and returns the correct gradient in autograd"""
        qnode_noisy = qml.QNode(qfunc, dev_noisy)
        qnode_ideal = qml.QNode(qfunc, dev_ideal)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = mitigate_with_zne(scale_factors, fold_global, richardson_extrapolate)(
            qnode_noisy
        )

        theta = np.array([np.pi / 4, np.pi / 4], requires_grad=True)

        res = mitigated_qnode(theta)
        assert qml.math.allclose(res, out_ideal, atol=1e-2)

        grad = qml.grad(mitigated_qnode)(theta)
        grad_ideal = qml.grad(qnode_ideal)(theta)

        assert qml.math.allclose(grad_ideal, grad_ideal_0)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)


def hadamard_circuit(wires, shots=10000, interface="autograd"):
    """Hadamard circuit to put all qubits in equal superposition (locally)"""
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit():
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.classical_shadow(wires=range(wires))

    return circuit


@pytest.mark.autograd
class TestExpvalTransform:
    """Test that the expval transform is applied correctly"""

    def test_hadamard_transform(self):
        """
        Test that the transform is correct for a circuit that prepares
        the uniform superposition
        """
        obs = qml.PauliZ(0)
        circuit = hadamard_circuit(3, shots=100000)
        circuit = qml.shadows.shadow_expval(obs)(circuit)

        tape = circuit.construct((), {})[0][0]

        assert all(qml.equal(qml.Hadamard(i), tape.operations[i]) for i in range(3))
        assert len(tape.observables) == 1
        assert isinstance(tape.observables[0], ShadowExpvalMP)
        assert tape.observables[0].H == obs


def test_expval_non_commuting_observables():
    """Test expval with multiple non-commuting operators"""
    dev = qml.device("default.qubit", wires=6)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(1)
        qml.Hadamard(0)
        qml.PauliZ(0)
        qml.Hadamard(3)
        qml.Hadamard(5)
        qml.T(5)
        return [
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliX(1) @ qml.PauliX(4)),
            qml.expval(qml.PauliX(3)),
            qml.expval(qml.PauliY(5)),
        ]

    assert all(np.isclose(circuit(), np.array([0.0, -1.0, 0.0, 0.0, 1.0, 1 / np.sqrt(2)])))

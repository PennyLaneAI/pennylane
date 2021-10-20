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
Tests for the gradients.finite_difference module.
"""
import pytest

from pennylane import numpy as np

import pennylane as qml
from pennylane.gradients import finite_diff, finite_diff_coeffs, generate_shifted_tapes


class TestCoeffs:
    """Tests for the finite_diff_coeffs function"""

    def test_invalid_derivative_error(self):
        """Test that an error is raised if n<1 or not an integer"""
        with pytest.raises(ValueError, match="n must be a positive integer"):
            finite_diff_coeffs(0, 1, 1)

        with pytest.raises(ValueError, match="n must be a positive integer"):
            finite_diff_coeffs(1.3, 1, 1)

    def test_invalid_approx_order_error(self):
        """Test that an error is raised if order < 1 or not an integer"""
        with pytest.raises(ValueError, match="order must be a positive integer"):
            finite_diff_coeffs(1, 0, 1)

        with pytest.raises(ValueError, match="order must be a positive integer"):
            finite_diff_coeffs(1, 1.7, 1)

        with pytest.raises(ValueError, match="Centered finite-difference requires an even order"):
            finite_diff_coeffs(1, 1, "center")

    def test_invalid_strategy(self):
        """Test that an error is raised if the strategy is not recognized"""
        with pytest.raises(ValueError, match="Unknown strategy"):
            finite_diff_coeffs(1, 1, 1)

    def test_correct_forward_order1(self):
        """Test that the correct forward order 1 method is returned"""
        coeffs, shifts = finite_diff_coeffs(1, 1, "forward")
        assert np.allclose(coeffs, [-1, 1])
        assert np.allclose(shifts, [0, 1])

    def test_correct_forward_order2(self):
        """Test that the correct forward order 2 method is returned"""
        coeffs, shifts = finite_diff_coeffs(1, 2, "forward")
        assert np.allclose(coeffs, [-1.5, 2, -0.5])
        assert np.allclose(shifts, [0, 1, 2])

    def test_correct_center_order2(self):
        """Test that the correct centered order 2 method is returned"""
        coeffs, shifts = finite_diff_coeffs(1, 2, "center")
        assert np.allclose(coeffs, [-0.5, 0.5])
        assert np.allclose(shifts, [-1, 1])

    def test_correct_backward_order1(self):
        """Test that the correct backward order 1 method is returned"""
        coeffs, shifts = finite_diff_coeffs(1, 1, "backward")
        assert np.allclose(coeffs, [1, -1])
        assert np.allclose(shifts, [0, -1])

    def test_correct_second_derivative_forward_order1(self):
        """Test that the correct forward order 1 method is returned"""
        coeffs, shifts = finite_diff_coeffs(2, 1, "forward")
        assert np.allclose(coeffs, [1, -2, 1])
        assert np.allclose(shifts, [0, 1, 2])

    def test_correct_second_derivative_center_order4(self):
        """Test that the correct forward order 4 method is returned"""
        coeffs, shifts = finite_diff_coeffs(2, 4, "center")
        assert np.allclose(coeffs, [-2.5, 4 / 3, 4 / 3, -1 / 12, -1 / 12])
        assert np.allclose(shifts, [0, -1, 1, -2, 2])


class TestShiftedTapes:
    """Tests for the generate_shifted_tapes function"""

    def test_behaviour(self):
        """Test that the function behaves as expected"""

        with qml.tape.QuantumTape() as tape:
            qml.PauliZ(0)
            qml.RX(1.0, wires=0)
            qml.CNOT(wires=[0, 2])
            qml.Rot(2.0, 3.0, 4.0, wires=0)
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 2}
        shifts = [0.1, -0.2, 1.6]
        res = generate_shifted_tapes(tape, 1, shifts=shifts)

        assert len(res) == len(shifts)
        assert res[0].get_parameters(trainable_only=False) == [1.0, 2.0, 3.1, 4.0]
        assert res[1].get_parameters(trainable_only=False) == [1.0, 2.0, 2.8, 4.0]
        assert res[2].get_parameters(trainable_only=False) == [1.0, 2.0, 4.6, 4.0]

    def test_multipliers(self):
        """Test that the function behaves as expected when multipliers are used"""

        with qml.tape.JacobianTape() as tape:
            qml.PauliZ(0)
            qml.RX(1.0, wires=0)
            qml.CNOT(wires=[0, 2])
            qml.Rot(2.0, 3.0, 4.0, wires=0)
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 2}
        shifts = [0.3, 0.6]
        multipliers = [0.2, 0.5]
        res = generate_shifted_tapes(tape, 0, shifts=shifts, multipliers=multipliers)

        assert len(res) == 2
        assert res[0].get_parameters(trainable_only=False) == [0.2 * 1.0 + 0.3, 2.0, 3.0, 4.0]
        assert res[1].get_parameters(trainable_only=False) == [0.5 * 1.0 + 0.6, 2.0, 3.0, 4.0]


class TestFiniteDiff:
    """Tests for the finite difference gradient transform"""

    def test_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a non-differentiable argument"""
        psi = np.array([1, 0, 1, 0], requires_grad=False) / np.sqrt(2)

        with qml.tape.JacobianTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        # by default all parameters are assumed to be trainable
        with pytest.raises(
            ValueError, match=r"Cannot differentiate with respect to parameter\(s\) {0}"
        ):
            finite_diff(tape, _expand=False)

        # setting trainable parameters avoids this
        tape.trainable_params = {1, 2}
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = finite_diff(tape)

        # For now, we must squeeze the results of the device execution, since
        # qml.probs results in a nested result. Later, we will revisit device
        # execution to avoid this issue.
        res = fn(qml.math.squeeze(dev.batch_execute(tapes)))
        assert res.shape == (4, 2)

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.finite_difference, "generate_shifted_tapes")

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        tapes, fn = finite_diff(tape)
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        assert len(spy.call_args_list) == 1

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, 0)

    def test_no_trainable_parameters(self, mocker):
        """Test that if the tape has no trainable parameters, no
        subroutines are called and the returned Jacobian is empty"""
        spy = mocker.spy(qml.gradients.finite_difference, "generate_shifted_tapes")

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        tape.trainable_params = {}

        tapes, fn = finite_diff(tape)
        res = fn(dev.batch_execute(tapes))
        assert res.size == 0
        assert np.all(res == np.array([[]]))

        spy.assert_not_called()
        assert len(tapes) == 0

    def test_y0(self, mocker):
        """Test that if first order finite differences is used, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        tapes, fn = finite_diff(tape, approx_order=1)

        # one tape per parameter, plus one global call
        assert len(tapes) == tape.num_params + 1

    def test_y0_provided(self):
        """Test that if first order finite differences is used,
        and the original tape output is provided, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        f0 = dev.execute(tape)
        tapes, fn = finite_diff(tape, approx_order=1, f0=f0)

        # one tape per parameter, plus one global call
        assert len(tapes) == tape.num_params

    def test_independent_parameters(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.JacobianTape() as tape1:
            qml.RX(1, wires=[0])
            qml.RX(1, wires=[1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.JacobianTape() as tape2:
            qml.RX(1, wires=[0])
            qml.RX(1, wires=[1])
            qml.expval(qml.PauliZ(1))

        tapes, fn = finite_diff(tape1, approx_order=1)
        j1 = fn(dev.batch_execute(tapes))

        # We should only be executing the device to differentiate 1 parameter (2 executions)
        assert dev.num_executions == 2

        tapes, fn = finite_diff(tape2, approx_order=1)
        j2 = fn(dev.batch_execute(tapes))

        exp = -np.sin(1)

        assert np.allclose(j1, [exp, 0])
        assert np.allclose(j2, [0, exp])


@pytest.mark.parametrize("approx_order", [2, 4])
@pytest.mark.parametrize("strategy", ["forward", "backward", "center"])
class TestFiniteDiffIntegration:
    """Tests for the finite difference gradient transform"""

    def test_ragged_output(self, approx_order, strategy):
        """Test that the Jacobian is correctly returned for a tape
        with ragged output"""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with qml.tape.JacobianTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1, 2])

        tapes, fn = finite_diff(tape, approx_order=approx_order, strategy=strategy)
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (6, 3)

    def test_single_expectation_value(self, approx_order, strategy, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tapes, fn = finite_diff(tape, approx_order=approx_order, strategy=strategy)
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value_with_argnum_all(self, approx_order, strategy, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where all parameters are chosen to compute
        the jacobian"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        # we choose both trainable parameters
        tapes, fn = finite_diff(tape, argnum=[0, 1], approx_order=approx_order, strategy=strategy)
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value_with_argnum_one(self, approx_order, strategy, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where only one parameter is chosen to
        estimate the jacobian.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        # we choose only 1 trainable parameter
        tapes, fn = finite_diff(tape, argnum=1, approx_order=approx_order, strategy=strategy)
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        expected = np.array([[0, np.cos(y) * np.cos(x)]])
        res = res.flatten()
        expected = expected.flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, approx_order, strategy, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tapes, fn = finite_diff(tape, approx_order=approx_order, strategy=strategy)
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_expectation_values(self, approx_order, strategy, tol):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tapes, fn = finite_diff(tape, approx_order=approx_order, strategy=strategy)
        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, approx_order, strategy, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tapes, fn = finite_diff(tape, approx_order=approx_order, strategy=strategy)
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


@pytest.mark.parametrize("approx_order", [2])
@pytest.mark.parametrize("strategy", ["center"])
class TestFiniteDiffGradients:
    """Test that the transform is differentiable"""

    def test_autograd(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.tape.JacobianTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            jac = fn(dev.batch_execute(tapes))
            return jac

        res = qml.jacobian(cost_fn)(params)
        x, y = params
        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_autograd_ragged(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        of a ragged tape can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.tape.JacobianTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            jac = fn(dev.batch_execute(tapes))
            return jac[1, 0]

        x, y = params
        res = qml.grad(cost_fn)(params)
        expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.slow
    def test_tf(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using TF, yielding second derivatives."""
        tf = pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit.tf", wires=2)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape() as t:
            with qml.tape.JacobianTape() as tape:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            jac = fn(dev.batch_execute(tapes))

        x, y = 1.0 * params

        expected = np.array([-np.sin(x) * np.sin(y), np.cos(x) * np.cos(y)])
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        res = t.jacobian(jac, params)
        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_tf_ragged(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        of a ragged tape can be differentiated using TF, yielding second derivatives."""
        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit.tf", wires=2)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape() as t:
            with qml.tape.JacobianTape() as tape:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            jac = fn(dev.batch_execute(tapes))[1, 0]

        x, y = 1.0 * params
        res = t.gradient(jac, params)
        expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_torch(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using Torch, yielding second derivatives."""
        torch = pytest.importorskip("torch")
        from pennylane.interfaces.torch import TorchInterface

        dev = qml.device("default.qubit.torch", wires=2)
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)

        with TorchInterface.apply(qml.tape.QubitParamShiftTape()) as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
        jac = fn(dev.batch_execute(tapes))
        cost = torch.sum(jac)
        cost.backward()
        hess = params.grad

        x, y = params.detach().numpy()

        expected = np.array([-np.sin(x) * np.sin(y), np.cos(x) * np.cos(y)])
        assert np.allclose(jac.detach().numpy(), expected, atol=tol, rtol=0)

        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        assert np.allclose(hess.detach().numpy(), np.sum(expected, axis=0), atol=tol, rtol=0)

    def test_jax(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using JAX, yielding second derivatives."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp
        from pennylane.interfaces.jax import JAXInterface
        from jax.config import config

        config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit.jax", wires=2)
        params = jnp.array([0.543, -0.654])

        def cost_fn(x):
            with JAXInterface.apply(qml.tape.QubitParamShiftTape()) as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            jac = fn(dev.batch_execute(tapes))
            return jac

        res = jax.jacobian(cost_fn)(params)
        x, y = params
        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

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
from pennylane.devices import DefaultQubit
from pennylane.operation import Observable, AnyWires


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

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape:
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

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch", "tensorflow"])
    def test_no_trainable_params_qnode(self, interface):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        if interface != "autograd":
            pytest.importorskip(interface)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
            res = qml.gradients.finite_diff(circuit)(weights)

        assert res == ()

    def test_no_trainable_params_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        # TODO: remove once #2155 is resolved
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.finite_diff(tape)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert res.shape == (1, 0)

    def test_all_zero_diff_methods(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qml.gradients.finite_diff(circuit)(params)
        assert np.allclose(result, np.zeros((4, 3)), atol=0, rtol=0)

        tapes, _ = qml.gradients.finite_diff(circuit.tape)
        assert tapes == []

    def test_y0(self, mocker):
        """Test that if first order finite differences is used, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape1:
            qml.RX(1, wires=[0])
            qml.RX(1, wires=[1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
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

    def test_output_shape_matches_qnode(self):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=4)

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

        x = np.random.rand(3)
        circuits = [qml.QNode(cost, dev) for cost in (cost1, cost2, cost3, cost4, cost5, cost6)]

        transform = [qml.math.shape(qml.gradients.finite_diff(c)(x)) for c in circuits]
        # The output shape of transforms for 2D qnode outputs (cost5 & cost6) is currently
        # transposed, e.g. (4, 1, 3) instead of (1, 4, 3).
        # TODO: fix qnode/expected once #2296 is resolved
        qnode = [qml.math.shape(c(x)) + (3,) for c in circuits[:4]] + [(4, 1, 3), (4, 2, 3)]
        expected = [(3,), (1, 3), (2, 3), (4, 3), (4, 1, 3), (4, 2, 3)]

        assert all(t == q == e for t, q, e in zip(transform, qnode, expected))

    def test_special_observable_qnode_differentiation(self):
        """Test differentiation of a QNode on a device supporting a
        special observable that returns an object rather than a number."""

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
                new = self.val + other.val if isinstance(other, self.__class__) else other
                return SpecialObject(new)

        class SpecialObservable(Observable):
            """SpecialObservable"""

            num_wires = AnyWires

            def diagonalizing_gates(self):
                """Diagonalizing gates"""
                return []

        class DeviceSupportingSpecialObservable(DefaultQubit):
            name = "Device supporting SpecialObservable"
            short_name = "default.qubit.specialobservable"
            observables = DefaultQubit.observables.union({"SpecialObservable"})
            R_DTYPE = SpecialObservable

            def expval(self, observable, **kwargs):
                if self.analytic and isinstance(observable, SpecialObservable):
                    val = super().expval(qml.PauliZ(wires=0), **kwargs)
                    return SpecialObject(val)

                return super().expval(observable, **kwargs)

        dev = DeviceSupportingSpecialObservable(wires=1, shots=None)

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


@pytest.mark.parametrize("approx_order", [2, 4])
@pytest.mark.parametrize("strategy", ["forward", "backward", "center"])
class TestFiniteDiffIntegration:
    """Tests for the finite difference gradient transform"""

    def test_ragged_output(self, approx_order, strategy):
        """Test that the Jacobian is correctly returned for a tape
        with ragged output"""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape:
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

        with qml.tape.QuantumTape() as tape:
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
            with qml.tape.QuantumTape() as tape:
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
            with qml.tape.QuantumTape() as tape:
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
            with qml.tape.QuantumTape() as tape:
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
            with qml.tape.QuantumTape() as tape:
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

        dev = qml.device("default.qubit.torch", wires=2)
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)

        with qml.tape.QuantumTape() as tape:
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
        from jax.config import config

        config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit.jax", wires=2)
        params = jnp.array([0.543, -0.654])

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
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

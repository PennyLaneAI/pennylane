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
"""
Tests for the gradients.finite_difference module.
"""
# pylint: disable=use-implicit-booleaness-not-comparison,abstract-method
import numpy
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.gradients import finite_diff, finite_diff_coeffs


def test_float32_warning():
    """Test that a warning is raised if provided float32 parameters."""
    x = np.array(0.1, dtype=np.float32)
    tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
    with pytest.warns(UserWarning, match="Finite differences with float32 detected."):
        finite_diff(tape)


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


class TestFiniteDiff:
    """Tests for the finite difference gradient transform"""

    def test_finite_diff_non_commuting_observables(self):
        """Test that finite differences work even if the measurements do not commute with each other."""

        ops = (qml.RX(0.5, wires=0),)
        ms = (qml.expval(qml.X(0)), qml.expval(qml.Z(0)))
        tape = qml.tape.QuantumScript(ops, ms, trainable_params=[0])

        batch, _ = qml.gradients.finite_diff(tape)
        assert len(batch) == 2
        tape0 = qml.tape.QuantumScript((qml.RX(0.5, 0),), ms, trainable_params=[0])
        tape1 = qml.tape.QuantumScript((qml.RX(0.5 + 1e-7, 0),), ms, trainable_params=[0])
        qml.assert_equal(batch[0], tape0)
        qml.assert_equal(batch[1], tape1)

    def test_trainable_batched_tape_raises(self):
        """Test that an error is raised for a broadcasted/batched tape if the broadcasted
        parameter is differentiated."""
        tape = qml.tape.QuantumScript([qml.RX([0.4, 0.2], 0)], [qml.expval(qml.PauliZ(0))])
        _match = r"Computing the gradient of broadcasted tapes .* using the finite difference"
        with pytest.raises(NotImplementedError, match=_match):
            finite_diff(tape)

    def test_nontrainable_batched_tape(self):
        """Test that no error is raised for a broadcasted/batched tape if the broadcasted
        parameter is not differentiated, and that the results correspond to the stacked
        results of the single-tape derivatives."""
        dev = qml.device("default.qubit")
        x = [0.4, 0.2]
        tape = qml.tape.QuantumScript(
            [qml.RY(0.6, 0), qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))], trainable_params=[0]
        )
        batched_tapes, batched_fn = finite_diff(tape)
        batched_grad = batched_fn(dev.execute(batched_tapes))
        separate_tapes = [
            qml.tape.QuantumScript(
                [qml.RY(0.6, 0), qml.RX(_x, 0)], [qml.expval(qml.PauliZ(0))], trainable_params=[0]
            )
            for _x in x
        ]
        separate_tapes_and_fns = [finite_diff(t) for t in separate_tapes]
        separate_grad = [_fn(dev.execute(_tapes)) for _tapes, _fn in separate_tapes_and_fns]
        assert np.allclose(batched_grad, separate_grad)

    def test_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a non-differentiable argument"""
        psi = np.array([1, 0, 1, 0], requires_grad=False) / np.sqrt(2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        # by default all parameters are assumed to be trainable
        with pytest.raises(
            ValueError, match=r"Cannot differentiate with respect to parameter\(s\) {0}"
        ):
            finite_diff(tape)

        # setting trainable parameters avoids this
        tape.trainable_params = {1, 2}
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = finite_diff(tape)

        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)

        assert isinstance(res[0], numpy.ndarray)
        assert res[0].shape == (4,)

        assert isinstance(res[1], numpy.ndarray)
        assert res[1].shape == (4,)

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.finite_difference, "generate_shifted_tapes")

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = finite_diff(tape)
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], numpy.ndarray)
        assert isinstance(res[1], numpy.ndarray)

        assert len(spy.call_args_list) == 1

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, 0)

    def test_no_trainable_params_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # TODO: remove once #2155 is resolved
        tape.trainable_params = []
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.finite_diff(tape)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, numpy.ndarray)
        assert res.shape == (0,)

    def test_no_trainable_params_multiple_return_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters with multiple returns."""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = []
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.finite_diff(tape)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, tuple)

    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.finite_diff(circuit)(weights)

    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.finite_diff(circuit)(weights)

    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.finite_diff(circuit)(weights)

    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.finite_diff(circuit)(weights)

    @pytest.mark.parametrize("prefactor", [1.0, 2.0])
    def test_all_zero_diff_methods(self, prefactor):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*(prefactor * params), wires=0)
            return qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qml.gradients.finite_diff(circuit)(params)

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3)
        assert np.allclose(result, 0)

    def test_all_zero_diff_methods_multiple_returns(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.expval(qml.PauliZ(wires=2)), qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qml.gradients.finite_diff(circuit)(params)

        assert isinstance(result, tuple)

        assert len(result) == 2

        for r, exp_shape in zip(result, [(3,), (4, 3)]):
            assert isinstance(r, np.ndarray)
            assert r.shape == exp_shape
            assert np.allclose(r, 0)

    def test_y0(self):
        """Test that if first order finite differences is used, then
        the tape is executed only once using the current parameter
        values."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = finite_diff(tape, approx_order=1)

        # one tape per parameter, plus one global call
        assert len(tapes) == tape.num_params + 1

    def test_y0_provided(self):
        """Test that by providing y0 the number of tapes is equal the number of parameters."""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        f0 = dev.execute(tape)
        tapes, _ = finite_diff(tape, approx_order=1, f0=f0)

        assert len(tapes) == tape.num_params

    def test_independent_parameters(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.qubit")

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(1))

        tape2 = qml.tape.QuantumScript.from_queue(q2)
        tapes, fn = finite_diff(tape1, approx_order=1)
        with qml.Tracker(dev) as tracker:
            j1 = fn(dev.execute(tapes))

        # We should only be executing the device to differentiate 1 parameter (2 executions)
        assert tracker.totals["executions"] == 2

        tapes, fn = finite_diff(tape2, approx_order=1)
        j2 = fn(dev.execute(tapes))

        exp = -np.sin(1)

        assert np.allclose(j1, [exp, 0])
        assert np.allclose(j2, [0, exp])

    def test_output_shape_matches_qnode(self):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=4)

        def cost1(x):
            qml.Rot(x[0], 0.3 * x[1], x[2], wires=0)
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
            return qml.probs([0, 1]), qml.probs([2, 3])

        x = np.random.rand(3)
        circuits = [qml.QNode(cost, dev) for cost in (cost1, cost2, cost3, cost4, cost5, cost6)]

        transform = [qml.math.shape(qml.gradients.finite_diff(c)(x)) for c in circuits]

        expected_shapes = [
            (3,),
            (1, 3),
            (2, 3),
            (4, 3),
            (1, 4, 3),
            (2, 4, 3),
        ]
        assert all(t == q for t, q in zip(transform, expected_shapes))

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

            def __rmul__(self, other):
                return SpecialObject(other * self.val)

            def __matmul__(self, other):
                return SpecialObject(self.val @ other)

            def __rmatmul__(self, other):
                return SpecialObject(other @ self.val)

            def __add__(self, other):
                new = self.val + (other.val if isinstance(other, self.__class__) else other)
                return SpecialObject(new)

            def __radd__(self, other):
                return self + other

        # pylint: disable=too-few-public-methods
        class SpecialObservable(qml.operation.Operator):
            """SpecialObservable"""

            def diagonalizing_gates(self):
                """Diagonalizing gates"""
                return []

        # pylint: disable=too-few-public-methods
        class DeviceSupportingSpecialObservable(DefaultQubitLegacy):
            """A device that supports the above SpecialObservable as a return type."""

            name = "Device supporting SpecialObservable"
            short_name = "default.qubit.specialobservable"
            observables = DefaultQubitLegacy.observables.union({"SpecialObservable"})

            # pylint: disable=unused-argument
            @staticmethod
            def _asarray(arr, dtype=None):
                return np.asarray(arr)

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.R_DTYPE = SpecialObservable

            def expval(self, observable, **kwargs):
                """Compute the expectation value of an observable."""
                if self.analytic and isinstance(observable, SpecialObservable):
                    val = super().expval(qml.PauliZ(wires=0), **kwargs)
                    return SpecialObject(val)

                return super().expval(observable, **kwargs)

        dev = DeviceSupportingSpecialObservable(wires=1, shots=None)

        @qml.qnode(dev, diff_method="finite-diff")
        def qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(SpecialObservable(wires=0))

        @qml.qnode(dev, diff_method="finite-diff")
        def reference_qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        par = np.array(0.2, requires_grad=True)
        assert np.isclose(qnode(par).item().val, reference_qnode(par))
        assert np.isclose(qml.jacobian(qnode)(par).item().val, qml.jacobian(reference_qnode)(par))


@pytest.mark.parametrize("approx_order", [2, 4])
@pytest.mark.parametrize("strategy", ["forward", "backward", "center"])
@pytest.mark.parametrize("validate", [True, False])
class TestFiniteDiffIntegration:
    """Tests for the finite difference gradient transform"""

    def test_ragged_output(self, approx_order, strategy, validate):
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
        tapes, fn = finite_diff(
            tape, approx_order=approx_order, strategy=strategy, validate_params=validate
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)

        assert len(res) == 2

        assert len(res[0]) == 3
        assert res[0][0].shape == (2,)
        assert res[0][1].shape == (2,)
        assert res[0][2].shape == (2,)

        assert len(res[1]) == 3
        assert res[1][0].shape == (4,)
        assert res[1][1].shape == (4,)
        assert res[1][2].shape == (4,)

    def test_single_expectation_value(self, approx_order, strategy, validate, tol):
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
        tapes, fn = finite_diff(
            tape, approx_order=approx_order, strategy=strategy, validate_params=validate
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], numpy.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], numpy.ndarray)
        assert res[1].shape == ()

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value_with_argnum_all(self, approx_order, strategy, validate, tol):
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
        tapes, fn = finite_diff(
            tape,
            argnum=[0, 1],
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], numpy.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], numpy.ndarray)
        assert res[1].shape == ()

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value_with_argnum_one(self, approx_order, strategy, validate, tol):
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
        # we choose only 1 trainable parameter
        tapes, fn = finite_diff(
            tape, argnum=1, approx_order=approx_order, strategy=strategy, validate_params=validate
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], numpy.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], numpy.ndarray)
        assert res[1].shape == ()

        expected = [0, np.cos(y) * np.cos(x)]

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_value_with_argnum_one(
        self, approx_order, strategy, validate, tol
    ):
        """Tests correct output shape and evaluation for a tape
        with a multiple measurement, where only one parameter is chosen to
        be trainable.

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
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        # we choose only 1 trainable parameter
        tapes, fn = finite_diff(
            tape, argnum=1, approx_order=approx_order, strategy=strategy, validate_params=validate
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert isinstance(res[0], tuple)
        assert np.allclose(res[0][0], 0, atol=tol, rtol=0)
        assert isinstance(res[1], tuple)
        assert np.allclose(res[1][0], 0, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, approx_order, strategy, validate, tol):
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
        tapes, fn = finite_diff(
            tape, approx_order=approx_order, strategy=strategy, validate_params=validate
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2
        assert np.allclose(res[0], [-np.sin(x), 0], atol=tol, rtol=0)
        assert isinstance(res[0][0], numpy.ndarray)
        assert isinstance(res[0][1], numpy.ndarray)

        assert isinstance(res[1], tuple)
        assert len(res[1]) == 2
        assert np.allclose(res[1], [0, np.cos(y)], atol=tol, rtol=0)
        assert isinstance(res[1][0], numpy.ndarray)
        assert isinstance(res[1][1], numpy.ndarray)

    def test_var_expectation_values(self, approx_order, strategy, validate, tol):
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
        tapes, fn = finite_diff(
            tape, approx_order=approx_order, strategy=strategy, validate_params=validate
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2
        assert np.allclose(res[0], [-np.sin(x), 0], atol=tol, rtol=0)
        assert isinstance(res[0][0], numpy.ndarray)
        assert isinstance(res[0][1], numpy.ndarray)

        assert isinstance(res[1], tuple)
        assert len(res[1]) == 2
        assert np.allclose(res[1], [0, -2 * np.cos(y) * np.sin(y)], atol=tol, rtol=0)
        assert isinstance(res[1][0], numpy.ndarray)
        assert isinstance(res[1][1], numpy.ndarray)

    def test_prob_expectation_values(self, approx_order, strategy, validate, tol):
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
        tapes, fn = finite_diff(
            tape, approx_order=approx_order, strategy=strategy, validate_params=validate
        )
        res = fn(dev.execute(tapes))

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2
        assert np.allclose(res[0][0], -np.sin(x), atol=tol, rtol=0)
        assert isinstance(res[0][0], numpy.ndarray)
        assert np.allclose(res[0][1], 0, atol=tol, rtol=0)
        assert isinstance(res[0][1], numpy.ndarray)

        assert isinstance(res[1], tuple)
        assert len(res[1]) == 2
        assert np.allclose(
            res[1][0],
            [
                -(np.cos(y / 2) ** 2 * np.sin(x)) / 2,
                -(np.sin(x) * np.sin(y / 2) ** 2) / 2,
                (np.sin(x) * np.sin(y / 2) ** 2) / 2,
                (np.cos(y / 2) ** 2 * np.sin(x)) / 2,
            ],
            atol=tol,
            rtol=0,
        )
        assert isinstance(res[1][0], numpy.ndarray)
        assert np.allclose(
            res[1][1],
            [
                -(np.cos(x / 2) ** 2 * np.sin(y)) / 2,
                (np.cos(x / 2) ** 2 * np.sin(y)) / 2,
                (np.sin(x / 2) ** 2 * np.sin(y)) / 2,
                -(np.sin(x / 2) ** 2 * np.sin(y)) / 2,
            ],
            atol=tol,
            rtol=0,
        )
        assert isinstance(res[1][1], numpy.ndarray)


@pytest.mark.parametrize("approx_order", [2])
@pytest.mark.parametrize("strategy", ["center"])
class TestFiniteDiffGradients:
    """Test that the transform is differentiable"""

    @pytest.mark.autograd
    def test_autograd(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            return np.array(fn(dev.execute(tapes)))

        res = qml.jacobian(cost_fn)(params)
        x, y = params
        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd_ragged(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        of a ragged tape can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            jac = fn(dev.execute(tapes))
            return jac[1][0]

        x, y = params
        res = qml.jacobian(cost_fn)(params)[0]
        expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_tf(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using TF, yielding second derivatives."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape(persistent=True) as t:
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            jac_0, jac_1 = fn(dev.execute(tapes))

        x, y = 1.0 * params

        res_0 = t.jacobian(jac_0, params)
        res_1 = t.jacobian(jac_1, params)

        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        assert np.allclose([res_0, res_1], expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf_ragged(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        of a ragged tape can be differentiated using TF, yielding second derivatives."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape(persistent=True) as t:
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)

            jac_01 = fn(dev.execute(tapes))[1][0]

        x, y = 1.0 * params

        res_01 = t.jacobian(jac_01, params)

        expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])

        assert np.allclose(res_01[0], expected, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using Torch, yielding second derivatives."""
        import torch

        dev = qml.device("default.qubit", wires=2)
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)

        def cost_fn(params):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            jac = fn(dev.execute(tapes))
            return jac

        hess = torch.autograd.functional.jacobian(cost_fn, params)

        x, y = params.detach().numpy()

        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )

        assert np.allclose(hess[0].detach().numpy(), expected[0], atol=tol, rtol=0)
        assert np.allclose(hess[1].detach().numpy(), expected[1], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using JAX, yielding second derivatives."""
        import jax
        from jax import numpy as jnp

        dev = qml.device("default.qubit", wires=2)
        params = jnp.array([0.543, -0.654])

        def cost_fn(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(tape, n=1, approx_order=approx_order, strategy=strategy)
            return fn(dev.execute(tapes))

        res = jax.jacobian(cost_fn)(params)
        assert isinstance(res, tuple)
        x, y = params
        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_probs(self, approx_order, strategy, tol):  # pylint: disable=unused-argument
        """Tests that the output of the finite-difference transform is similar using or not diff method on the QNode."""
        import jax

        dev = qml.device("default.qubit", wires=2)
        x = jax.numpy.array(0.543)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=[0])
            qml.RY(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        @qml.qnode(dev, diff_method="finite-diff")
        def circuit_fd(x):
            qml.RX(x, wires=[0])
            qml.RY(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        transform_output = qml.gradients.finite_diff(circuit)(x)
        framework_output = jax.jacobian(circuit)(x)
        assert jax.numpy.allclose(
            jax.numpy.stack(transform_output), jax.numpy.stack(framework_output)
        )

        transform_output = qml.gradients.finite_diff(circuit_fd)(x)
        framework_output = jax.jacobian(circuit_fd)(x)
        assert jax.numpy.allclose(
            jax.numpy.stack(transform_output), jax.numpy.stack(framework_output)
        )


@pytest.mark.parametrize("argnums", [[0], [1], [0, 1]])
@pytest.mark.parametrize("interface", ["jax"])
@pytest.mark.parametrize("approx_order", [2, 4])
@pytest.mark.parametrize("strategy", ["forward", "backward", "center"])
@pytest.mark.jax
class TestJaxArgnums:
    """Class to test the integration of argnums (Jax) and the finite-diff transform."""

    expected_jacs = []
    interfaces = ["auto", "jax"]

    def test_single_expectation_value(self, argnums, interface, approx_order, strategy):
        """Test for single expectation value."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = qml.gradients.finite_diff(
            circuit, argnums=argnums, approx_order=approx_order, strategy=strategy, h=1e-5
        )(x, y)

        expected_0 = np.array([-np.sin(y) * np.sin(x[0]), 0])
        expected_1 = np.array(np.cos(y) * np.cos(x[0]))

        if argnums == [0]:
            assert np.allclose(res, expected_0)
        if argnums == [1]:
            assert np.allclose(res, expected_1)
        if argnums == [0, 1]:
            assert np.allclose(res[0], expected_0)
            assert np.allclose(res[1], expected_1)

    def test_multi_expectation_values(self, argnums, interface, approx_order, strategy):
        """Test for multiple expectation values."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = qml.gradients.finite_diff(
            circuit, argnums=argnums, approx_order=approx_order, strategy=strategy, h=1e-5
        )(x, y)

        expected_0 = np.array([[-np.sin(x[0]), 0.0], [0.0, 0.0]])
        expected_1 = np.array([0, np.cos(y)])

        if argnums == [0]:
            assert np.allclose(res[0], expected_0[0])
            assert np.allclose(res[1], expected_0[1])
        if argnums == [1]:
            assert np.allclose(res[0][0], expected_1[0])
            assert np.allclose(res[1][0], expected_1[1])
        if argnums == [0, 1]:
            assert np.allclose(res[0][0], expected_0[0])
            assert np.allclose(res[0][1], expected_0[1])
            assert np.allclose(res[1][0], expected_1[0])
            assert np.allclose(res[1][1], expected_1[1])

    def test_hessian(self, argnums, interface, approx_order, strategy):
        """Test for hessian."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.RY(y, wires=[1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, -0.654])
        y = jax.numpy.array(-0.123)

        res = jax.jacobian(
            qml.gradients.finite_diff(
                circuit, approx_order=approx_order, strategy=strategy, h=1e-5, argnums=argnums
            ),
            argnums=argnums,
        )(x, y)
        res_expected = jax.hessian(circuit, argnums=argnums)(x, y)

        tol = 10e-6

        if len(argnums) == 1:
            # jax.hessian produces an additional tuple axis, which we have to index away here
            assert np.allclose(res, res_expected[0], atol=tol)
        else:
            # The Hessian is a 2x2 nested tuple "matrix" for argnums=[0, 1]
            for r, r_e in zip(res, res_expected):
                for r_, r_e_ in zip(r, r_e):
                    assert np.allclose(r_, r_e_, atol=tol)

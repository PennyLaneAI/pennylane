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
Tests for the gradients.finite_difference module using shot vectors.
"""
import numpy
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.gradients import finite_diff
from pennylane.measurements import Shots

# pylint:disable = use-implicit-booleaness-not-comparison,abstract-method

h_val = 0.1

default_shot_vector = (1000, 2000, 3000)
many_shots_shot_vector = tuple([1000000] * 3)


class TestFiniteDiff:
    """Tests for the finite difference gradient transform"""

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

        tape = qml.tape.QuantumScript.from_queue(q, shots=default_shot_vector)
        # by default all parameters are assumed to be trainable
        with pytest.raises(
            ValueError, match=r"Cannot differentiate with respect to parameter\(s\) {0}"
        ):
            finite_diff(tape)

        # setting trainable parameters avoids this
        tape.trainable_params = {1, 2}
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = finite_diff(tape, h=h_val)

        all_res = fn(dev.execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        for res in all_res:
            assert isinstance(res, tuple)

            assert isinstance(res[0], numpy.ndarray)
            assert res[0].shape == (4,)

            assert isinstance(res[1], numpy.ndarray)
            assert res[1].shape == (4,)

    def test_independent_parameter_skipped(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.finite_difference, "generate_shifted_tapes")

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=default_shot_vector)
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = finite_diff(tape, h=h_val)
        all_res = fn(dev.execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        for res in all_res:
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

        tape = qml.tape.QuantumScript.from_queue(q, shots=default_shot_vector)
        # TODO: remove once #2155 is resolved
        tape.trainable_params = []
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.finite_diff(tape, h=h_val)
        all_res = post_processing(qml.execute(g_tapes, dev, None))
        assert len(all_res) == len(default_shot_vector)

        for res in all_res:
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

        tape = qml.tape.QuantumScript.from_queue(q, shots=default_shot_vector)
        tape.trainable_params = []
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.finite_diff(tape, h=h_val)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, tuple)

    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(default_shot_vector)
        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.finite_diff(circuit, h=h_val)(weights)

    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(default_shot_vector)
        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.finite_diff(circuit, h=h_val)(weights)

    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(default_shot_vector)
        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.finite_diff(circuit, h=h_val)(weights)

    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(default_shot_vector)
        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.finite_diff(circuit, h=h_val)(weights)

    def test_all_zero_diff_methods(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4)

        @qml.set_shots(default_shot_vector)
        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        all_result = qml.gradients.finite_diff(circuit, h=h_val)(params)

        assert isinstance(all_result, tuple)
        assert len(all_result) == len(default_shot_vector)

        for result in all_result:
            assert isinstance(result, np.ndarray)
            assert result.shape == (4, 3)
            assert np.allclose(result, 0)

    def test_all_zero_diff_methods_multiple_returns(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""

        dev = qml.device("default.qubit", wires=4)

        @qml.set_shots(many_shots_shot_vector)
        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.expval(qml.PauliZ(wires=2)), qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        all_result = qml.gradients.finite_diff(circuit, h=h_val)(params)
        assert isinstance(all_result, tuple)
        assert len(all_result) == len(many_shots_shot_vector)

        for res in all_result:
            assert isinstance(res, tuple)
            assert len(res) == 2
            for r, exp_shape in zip(res, [(3,), (4, 3)]):
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

        tape = qml.tape.QuantumScript.from_queue(q, shots=default_shot_vector)
        tapes, _ = finite_diff(tape, approx_order=1, h=h_val)

        # one tape per parameter, plus one global call
        assert len(tapes) == tape.num_params + 1

    def test_y0_provided(self):
        """Test that by providing y0 the number of tapes is equal the number of parameters."""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=default_shot_vector)
        f0 = dev.execute(tape)
        tapes, _ = finite_diff(tape, approx_order=1, f0=f0, h=h_val)

        assert len(tapes) == tape.num_params

    def test_independent_parameters(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1, shots=many_shots_shot_vector)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(1))

        tape2 = qml.tape.QuantumScript.from_queue(q2, shots=many_shots_shot_vector)
        tapes, fn = finite_diff(tape1, approx_order=1, h=h_val)
        with qml.Tracker(dev) as tracker:
            j1 = fn(dev.execute(tapes))

        # We should only be executing the device to differentiate 1 parameter (2 executions)
        assert tracker.totals["executions"] == 2

        tapes, fn = finite_diff(tape2, approx_order=1, h=h_val)
        j2 = fn(dev.execute(tapes))

        exp = -np.sin(1)

        assert isinstance(j1, tuple)
        assert len(j1) == len(many_shots_shot_vector)
        assert isinstance(j2, tuple)
        assert len(j2) == len(many_shots_shot_vector)

        for _j1, _j2 in zip(j1, j2):
            assert np.allclose(_j1, [exp, 0], atol=0.07)
            assert np.allclose(_j2, [0, exp], atol=0.07)

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
            qml.Rot(x[0], 0.3 * x[1], x[2], wires=0)
            return [qml.probs([0, 1])]

        def cost6(x):
            qml.Rot(*x, wires=0)
            return qml.probs([0, 1]), qml.probs([2, 3])

        x = np.random.rand(3)
        circuits = [
            qml.set_shots(qml.QNode(cost, dev), shots=(1, 1000, 1001))
            for cost in (cost1, cost2, cost3, cost4, cost5, cost6)
        ]

        transform = [qml.math.shape(qml.gradients.finite_diff(c, h=h_val)(x)) for c in circuits]
        expected = [(3, 3), (3, 1, 3), (3, 2, 3), (3, 4, 3), (3, 1, 4, 3), (3, 2, 4, 3)]
        assert all(t == q for t, q in zip(transform, expected))

    def test_output_shape_matches_qnode_two_args(self):
        """Test that the transform output shape matches that of a QNode with multiple args."""
        dev = qml.device("default.qubit", wires=4)

        def cost1(x, y, z):
            qml.Rot(x[0], 2 * y[1], -0.1 * z[0], wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost2(x, y, z):
            qml.Rot(x[0], 2 * y[1], -0.1 * z[0], wires=0)
            return qml.probs([0, 1]), qml.probs([2, 3])

        x, y, z = np.random.rand(3, 2)
        circuits = [
            qml.set_shots(qml.QNode(cost, dev), shots=(1, 1000, 1001)) for cost in (cost1, cost2)
        ]

        transform = [
            qml.math.shape(qml.gradients.finite_diff(c, h=h_val)(x, y, z)) for c in circuits
        ]
        expected = [
            (3, 3, 2),  # shot vector, params, param shape
            (3, 2, 3, 4, 2),  # shot vector, measurements, params, measurement shape, param shape
        ]
        assert all(t == q for t, q in zip(transform, expected))

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
                return np.array(arr)

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

        tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
        tapes, fn = finite_diff(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
        )
        all_res = fn(dev.execute(tapes))
        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        for res in all_res:
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

    def test_single_expectation_value(self, approx_order, strategy, validate, seed):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2, seed=seed)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
        tapes, fn = finite_diff(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=h_val,
        )
        all_res = fn(dev.execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], numpy.ndarray)
            assert res[0].shape == ()

            assert isinstance(res[1], numpy.ndarray)
            assert res[1].shape == ()

            assert np.allclose(res, expected, atol=0.15, rtol=0)

    def test_single_expectation_value_with_argnum_all(self, approx_order, strategy, validate, seed):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where all parameters are chosen to compute
        the jacobian"""
        dev = qml.device("default.qubit", wires=2, seed=seed)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
        # we choose both trainable parameters
        tapes, fn = finite_diff(
            tape,
            argnum=[0, 1],
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=h_val,
        )
        all_res = fn(dev.execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], numpy.ndarray)
            assert res[0].shape == ()

            assert isinstance(res[1], numpy.ndarray)
            assert res[1].shape == ()

            assert np.allclose(res, expected, atol=0.15, rtol=0)

    def test_single_expectation_value_with_argnum_one(self, approx_order, strategy, validate, seed):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where only one parameter is chosen to
        estimate the jacobian.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        dev = qml.device("default.qubit", wires=2, seed=seed)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
        # we choose only 1 trainable parameter
        tapes, fn = finite_diff(
            tape,
            argnum=1,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=h_val,
        )
        all_res = fn(dev.execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        expected = [0, np.cos(y) * np.cos(x)]

        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], numpy.ndarray)
            assert res[0].shape == ()

            assert isinstance(res[1], numpy.ndarray)
            assert res[1].shape == ()

            assert np.allclose(res, expected, atol=0.12, rtol=0)

    def test_probs_expval_with_argnum_one(self, approx_order, strategy, validate, seed):
        """Tests correct output shape and evaluation for a tape
        with a multiple measurement, where only one parameter is chosen to
        be trainable.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """

        dev = qml.device("default.qubit", wires=2, seed=seed)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
        # we choose only 1 trainable parameter
        tapes, fn = finite_diff(
            tape,
            argnum=1,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=h_val,
        )
        all_res = fn(dev.execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        cx, sx, cy, sy = np.cos(x / 2), np.sin(x / 2), np.cos(y / 2), np.sin(y / 2)
        # probability vector is [cx**2 * cy**2, cx**2 * sy**2, sx**2 * sy**2, sx**2 * cy**2]
        exp_dprob = np.array([-(cx**2), cx**2, sx**2, -(sx**2)]) * (sy * cy)
        exp_probs = [np.zeros(4), exp_dprob]
        # expval is np.sin(y) * np.cos(x)
        exp_expval = [0, np.cos(y) * np.cos(x)]

        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2  # two measurements
            assert np.allclose(res[0], exp_probs, atol=0.07)
            assert np.allclose(res[1], exp_expval, atol=0.2)

    def test_multiple_expectation_values(self, approx_order, strategy, validate, seed):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2, seed=seed)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
        tapes, fn = finite_diff(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=h_val,
        )
        all_res = fn(dev.execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], tuple)
            assert len(res[0]) == 2
            assert np.allclose(res[0], [-np.sin(x), 0], atol=0.15, rtol=0)
            assert isinstance(res[0][0], numpy.ndarray)
            assert isinstance(res[0][1], numpy.ndarray)

            assert isinstance(res[1], tuple)
            assert len(res[1]) == 2
            assert np.allclose(res[1], [0, np.cos(y)], atol=0.15, rtol=0)
            assert isinstance(res[1][0], numpy.ndarray)
            assert isinstance(res[1][1], numpy.ndarray)

    def test_var_expectation_values(self, approx_order, strategy, validate, seed):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        dev = qml.device("default.qubit", wires=2, seed=seed)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
        tapes, fn = finite_diff(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=h_val,
        )
        all_res = fn(dev.execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], tuple)
            assert len(res[0]) == 2
            assert np.allclose(res[0], [-np.sin(x), 0], atol=0.2, rtol=0)
            assert isinstance(res[0][0], numpy.ndarray)
            assert isinstance(res[0][1], numpy.ndarray)

            assert isinstance(res[1], tuple)
            assert len(res[1]) == 2
            assert np.allclose(res[1], [0, -2 * np.cos(y) * np.sin(y)], atol=0.2, rtol=0)
            assert isinstance(res[1][0], numpy.ndarray)
            assert isinstance(res[1][1], numpy.ndarray)

    def test_prob_expectation_values(self, approx_order, strategy, validate, seed):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2, seed=seed)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
        tapes, fn = finite_diff(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=h_val,
        )
        all_res = fn(dev.execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], tuple)
            assert len(res[0]) == 2
            assert np.allclose(res[0][0], -np.sin(x), atol=0.1, rtol=0)
            assert isinstance(res[0][0], numpy.ndarray)
            assert np.allclose(res[0][1], 0, atol=0.1, rtol=0)
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
                atol=0.07,
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
                atol=0.08,
                rtol=0,
            )
            assert isinstance(res[1][1], numpy.ndarray)


@pytest.mark.parametrize("approx_order", [2])
@pytest.mark.parametrize("strategy", ["center"])
@pytest.mark.xfail(reason="TODO: higher-order derivatives with finite shots")
class TestFiniteDiffGradients:
    """Test that the transform is differentiable"""

    @pytest.mark.autograd
    def test_autograd(self, approx_order, strategy):
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

            tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=h_val,
            )
            return np.array(fn(dev.execute(tapes)))

        all_res = qml.jacobian(cost_fn)(params)

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        for res in all_res:
            x, y = params
            expected = np.array(
                [
                    [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                    [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
                ]
            )

            assert np.allclose(res, expected, atol=0.3, rtol=0)

    @pytest.mark.autograd
    def test_autograd_ragged(self, approx_order, strategy):
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

            tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=h_val,
            )
            jac = fn(dev.execute(tapes))
            return jac[1][0]

        x, y = params
        all_res = qml.jacobian(cost_fn)(params)[0]

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        for res in all_res:
            expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])
            assert np.allclose(res, expected, atol=0.3, rtol=0)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_tf(self, approx_order, strategy):
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

            tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=h_val,
            )
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
        assert np.allclose([res_0, res_1], expected, atol=0.3, rtol=0)

    @pytest.mark.tf
    def test_tf_ragged(self, approx_order, strategy):
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

            tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=h_val,
            )

            jac_01 = fn(dev.execute(tapes))[1][0]

        x, y = 1.0 * params

        res_01 = t.jacobian(jac_01, params)

        expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])

        assert np.allclose(res_01[0], expected, atol=0.3, rtol=0)

    @pytest.mark.torch
    def test_torch(self, approx_order, strategy):
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

            tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=h_val,
            )
            return fn(dev.execute(tapes))

        hess = torch.autograd.functional.jacobian(cost_fn, params)

        x, y = params.detach().numpy()

        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )

        assert np.allclose(hess[0].detach().numpy(), expected[0], atol=0.3, rtol=0)
        assert np.allclose(hess[1].detach().numpy(), expected[1], atol=0.3, rtol=0)

    @pytest.mark.jax
    def test_jax(self, approx_order, strategy):
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

            tape = qml.tape.QuantumScript.from_queue(q, shots=many_shots_shot_vector)
            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=h_val,
            )
            return fn(dev.execute(tapes))

        all_res = jax.jacobian(cost_fn)(params)

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        x, y = params
        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        for res in all_res:
            assert isinstance(res, tuple)
            assert np.allclose(res, expected, atol=0.3, rtol=0)


pauliz = qml.PauliZ(wires=0)
proj = qml.Projector([1], wires=0)
A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
hermitian = qml.Hermitian(A, wires=0)

expval = qml.expval(pauliz)
probs = qml.probs(wires=[1, 0])
var_involutory = qml.var(proj)
var_non_involutory = qml.var(hermitian)

single_scalar_output_measurements = [
    expval,
    probs,
    var_involutory,
    var_non_involutory,
]

single_meas_with_shape = list(zip(single_scalar_output_measurements, [(), (4,), (), ()]))


@pytest.mark.parametrize(
    "shot_vec", [(100, 1, 10), (1, 1, 10), ((1, 2), 10), (10, (1, 2)), (1, 1, 1)]
)
class TestReturn:
    """Class to test the shape of Jacobian with different return types.

    The return types have the following major cases:

    1. 1 trainable param, 1 measurement
    2. 1 trainable param, >1 measurement
    3. >1 trainable param, 1 measurement
    4. >1 trainable param, >1 measurement
    """

    @pytest.mark.parametrize("meas, shape", single_meas_with_shape)
    @pytest.mark.parametrize("op_wires", [0, 2])
    def test_1_1(self, shot_vec, meas, shape, op_wires):
        """Test one param one measurement case"""
        dev = qml.device("default.qubit", wires=3)
        x = 0.543

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(
                x, wires=[op_wires]
            )  # Op acts either on wire 0 (non-zero grad) or wire 2 (zero grad)
            qml.apply(meas)  # Measurements act on wires 0 and 1

        grad_transform_shots = Shots(shot_vec)
        tape = qml.tape.QuantumScript.from_queue(q, shots=grad_transform_shots)
        # One trainable param
        tape.trainable_params = {0}

        tapes, fn = qml.gradients.finite_diff(tape)
        all_res = fn(dev.execute(tapes))

        assert len(all_res) == grad_transform_shots.num_copies
        assert isinstance(all_res, tuple)

        for res in all_res:
            assert isinstance(res, np.ndarray)
            assert res.shape == shape

    @pytest.mark.parametrize("op_wire", [0, 1])
    def test_1_N(self, shot_vec, op_wire):
        """Test single param multi-measurement case"""
        dev = qml.device("default.qubit", wires=6)
        x = 0.543

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(
                x, wires=[op_wire]
            )  # Op acts either on wire 0 (non-zero grad) or wire 1 (zero grad)

            # 4 measurements
            qml.expval(qml.PauliZ(wires=0))

            # Note: wire 1 is skipped as a measurement to allow for zero grad case to be tested
            qml.probs(wires=[3, 2])
            qml.var(qml.Projector([1], wires=4))
            qml.var(qml.Hermitian(A, wires=5))

        grad_transform_shots = Shots(shot_vec)
        tape = qml.tape.QuantumScript.from_queue(q, shots=grad_transform_shots)
        # Multiple trainable params
        tape.trainable_params = {0}

        tapes, fn = qml.gradients.finite_diff(tape)
        all_res = fn(dev.execute(tapes))

        assert len(all_res) == grad_transform_shots.num_copies
        assert isinstance(all_res, tuple)

        expected_shapes = [(), (4,), (), ()]
        for meas_res in all_res:
            for res, shape in zip(meas_res, expected_shapes):
                assert isinstance(res, np.ndarray)
                assert res.shape == shape

    @pytest.mark.parametrize("meas, shape", single_meas_with_shape)
    @pytest.mark.parametrize("op_wires", [0, 2])
    def test_N_1(self, shot_vec, meas, shape, op_wires):
        """Test multi-param single measurement case"""
        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = 0.213

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(
                x, wires=[op_wires]
            )  # Op acts either on wire 0 (non-zero grad) or wire 2 (zero grad)
            qml.RY(
                y, wires=[op_wires]
            )  # Op acts either on wire 0 (non-zero grad) or wire 2 (zero grad)
            qml.apply(meas)  # Measurements act on wires 0 and 1

        grad_transform_shots = Shots(shot_vec)
        tape = qml.tape.QuantumScript.from_queue(q, shots=grad_transform_shots)
        # Multiple trainable params
        tape.trainable_params = {0, 1}

        tapes, fn = qml.gradients.finite_diff(tape)
        all_res = fn(dev.execute(tapes))

        assert len(all_res) == grad_transform_shots.num_copies
        assert isinstance(all_res, tuple)

        for param_res in all_res:
            for res in param_res:
                assert isinstance(res, np.ndarray)
                assert res.shape == shape

    @pytest.mark.parametrize("op_wires", [(0, 1, 2, 3, 4), (5, 5, 5, 5, 5)])
    def test_N_N(self, shot_vec, op_wires):
        """Test multi-param multi-measurement case"""
        dev = qml.device("default.qubit", wires=6)
        params = np.random.random(6)

        with qml.queuing.AnnotatedQueue() as q:
            for idx, w in enumerate(op_wires):
                qml.RY(
                    params[idx], wires=[w]
                )  # Op acts either on wire 0-4 (non-zero grad) or wire 5 (zero grad)

            # The following asserts that there actually is a w defined, or raises an
            # error if op_wires was empty
            w = op_wires[-1]
            # Extra op - 5 measurements in total
            qml.RY(
                params[5], wires=[w]
            )  # Op acts either on wire 0-4 (non-zero grad) or wire 5 (zero grad)

            # 4 measurements
            qml.expval(qml.PauliZ(wires=0))
            qml.probs(wires=[2, 1])
            qml.var(qml.Projector([1], wires=3))
            qml.var(qml.Hermitian(A, wires=4))

        grad_transform_shots = Shots(shot_vec)
        tape = qml.tape.QuantumScript.from_queue(q, shots=grad_transform_shots)
        # Multiple trainable params
        tape.trainable_params = {0, 1, 2, 3, 4}

        tapes, fn = qml.gradients.finite_diff(tape)
        all_res = fn(dev.execute(tapes))

        assert len(all_res) == grad_transform_shots.num_copies
        assert isinstance(all_res, tuple)

        expected_shapes = [(), (4,), (), ()]
        for meas_res in all_res:
            assert len(meas_res) == 4
            for idx, param_res in enumerate(meas_res):
                assert len(param_res) == 5
                for res in param_res:
                    assert isinstance(res, np.ndarray)
                    assert res.shape == expected_shapes[idx]

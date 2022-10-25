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

from pennylane import numpy as np

import pennylane as qml
from pennylane.gradients import (
    finite_diff,
    finite_diff_coeffs,
    generate_shifted_tapes,
)
from pennylane.devices import DefaultQubit
from pennylane.operation import Observable, AnyWires


finite_diff_shot_vec_tol = 0.3

default_shot_vector = (1000, 2000, 3000)
many_shots_shot_vector = tuple([1000000] * 3)


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
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)
        tapes, fn = finite_diff(tape, h=10e-2, shots=default_shot_vector)

        all_res = fn(dev.batch_execute(tapes))

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

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)
        tapes, fn = finite_diff(tape, h=10e-2, shots=default_shot_vector)
        all_res = fn(dev.batch_execute(tapes))

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
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        weights = [0.1, 0.2]
        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        # TODO: remove once #2155 is resolved
        tape.trainable_params = []
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.finite_diff(
                tape, h=10e-2, shots=default_shot_vector
            )
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, numpy.ndarray)
        assert res.shape == (0,)

    def test_no_trainable_params_multiple_return_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters with multiple returns."""
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        weights = [0.1, 0.2]
        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            qml.probs(wires=[0, 1])

        tape.trainable_params = []
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.finite_diff(
                tape, h=10e-2, shots=default_shot_vector
            )
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, tuple)

    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
            res = qml.gradients.finite_diff(circuit)(weights)

        assert res == ()

    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        @qml.qnode(dev, interface="torch")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
            res = qml.gradients.finite_diff(circuit)(weights)

        assert res == ()

    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        @qml.qnode(dev, interface="tf")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
            res = qml.gradients.finite_diff(circuit)(weights)

        assert res == ()

    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        @qml.qnode(dev, interface="jax")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
            res = qml.gradients.finite_diff(circuit)(weights)

        assert res == ()

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

        assert isinstance(result, tuple)

        assert len(result) == 3

        assert isinstance(result[0], numpy.ndarray)
        assert result[0].shape == (4,)
        assert np.allclose(result[0], 0)

        assert isinstance(result[1], numpy.ndarray)
        assert result[1].shape == (4,)
        assert np.allclose(result[1], 0)

        assert isinstance(result[2], numpy.ndarray)
        assert result[2].shape == (4,)
        assert np.allclose(result[2], 0)

        tapes, _ = qml.gradients.finite_diff(circuit.tape)
        assert tapes == []

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

        # First elem
        assert len(result[0]) == 3

        assert isinstance(result[0][0], numpy.ndarray)
        assert result[0][0].shape == (1,)
        assert np.allclose(result[0][0], 0)

        assert isinstance(result[0][1], numpy.ndarray)
        assert result[0][1].shape == (1,)
        assert np.allclose(result[0][1], 0)

        assert isinstance(result[0][2], numpy.ndarray)
        assert result[0][2].shape == (1,)
        assert np.allclose(result[0][2], 0)

        # Second elem
        assert len(result[0]) == 3

        assert isinstance(result[1][0], numpy.ndarray)
        assert result[1][0].shape == (4,)
        assert np.allclose(result[1][0], 0)

        assert isinstance(result[1][1], numpy.ndarray)
        assert result[1][1].shape == (4,)
        assert np.allclose(result[1][1], 0)

        assert isinstance(result[1][2], numpy.ndarray)
        assert result[1][2].shape == (4,)
        assert np.allclose(result[1][2], 0)

        tapes, _ = qml.gradients.finite_diff(circuit.tape)
        assert tapes == []

    def test_y0(self, mocker):
        """Test that if first order finite differences is used, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        tapes, fn = finite_diff(tape, approx_order=1)

        # one tape per parameter, plus one global call
        assert len(tapes) == tape.num_params + 1

    def test_y0_provided(self):
        """Test that by providing y0 the number of tapes is equal the number of parameters."""
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        f0 = dev.execute(tape)
        tapes, fn = finite_diff(tape, approx_order=1, f0=f0)

        assert len(tapes) == tape.num_params

    def test_independent_parameters(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(1))

        tapes, fn = finite_diff(tape1, approx_order=1, h=10e-2, shots=many_shots_shot_vector)
        j1 = fn(dev.batch_execute(tapes))

        # We should only be executing the device to differentiate 1 parameter (2 executions)
        assert dev.num_executions == 2

        tapes, fn = finite_diff(tape2, approx_order=1, h=10e-2, shots=default_shot_vector)
        j2 = fn(dev.batch_execute(tapes))

        exp = -np.sin(1)

        assert isinstance(j1, tuple)
        assert len(j1) == len(default_shot_vector)
        assert isinstance(j2, tuple)
        assert len(j2) == len(default_shot_vector)

        for _j1, _j2 in zip(j1, j2):
            assert np.allclose(_j1, [exp, 0], atol=finite_diff_shot_vec_tol)
            assert np.allclose(_j2, [0, exp], atol=finite_diff_shot_vec_tol)

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
            return qml.probs([0, 1]), qml.probs([2, 3])

        x = np.random.rand(3)
        circuits = [qml.QNode(cost, dev) for cost in (cost1, cost2, cost3, cost4, cost5, cost6)]

        transform = [qml.math.shape(qml.gradients.finite_diff(c)(x)) for c in circuits]

        expected = [(3,), (3,), (2, 3), (3, 4), (3, 4), (2, 3, 4)]

        assert all(t == q for t, q in zip(transform, expected))

    # TODO: added with param shift
    # def test_special_observable_qnode_differentiation(self):
    #     """Test differentiation of a QNode on a device supporting a
    #     special observable that returns an object rather than a number."""
    #
    #     class SpecialObject:
    #         """SpecialObject
    #
    #         A special object that conveniently encapsulates the return value of
    #         a special observable supported by a special device and which supports
    #         multiplication with scalars and addition.
    #         """
    #
    #         def __init__(self, val):
    #             self.val = val
    #
    #         def __mul__(self, other):
    #             return SpecialObject(self.val * other)
    #
    #         def __add__(self, other):
    #             = self.val + other.val if isinstance(other, self.__class__) else other
    #             return SpecialObject)
    #
    #     class SpecialObservable(Observable):
    #         """SpecialObservable"""
    #
    #         num_wires = AnyWires
    #
    #         def diagonalizing_gates(self):
    #             """Diagonalizing gates"""
    #             return []
    #
    #     class DeviceSupportingSpecialObservable(DefaultQubit):
    #         name = "Device supporting SpecialObservable"
    #         short_name = "default.qubit.specialobservable"
    #         observables = DefaultQubit.observables.union({"SpecialObservable"})
    #
    #         @staticmethod
    #         def _asarray(arr, dtype=None):
    #             return arr
    #
    #         def __init__(self, *args, **kwargs):
    #             super().__init__(*args, **kwargs)
    #             self.R_DTYPE = SpecialObservable
    #
    #         def expval(self, observable, **kwargs):
    #             if self.analytic and isinstance(observable, SpecialObservable):
    #                 val = super().expval(qml.PauliZ(wires=0), **kwargs)
    #                 return SpecialObject(val)
    #
    #             return super().expval(observable, **kwargs)
    #
    #     dev = DeviceSupportingSpecialObservable(wires=1, shots=None)
    #
    #     @qml.qnode(dev, diff_method="parameter-shift")
    #     def qnode(x):
    #         qml.RY(x, wires=0)
    #         return qml.expval(SpecialObservable(wires=0))
    #
    #     @qml.qnode(dev, diff_method="parameter-shift")
    #     def reference_qnode(x):
    #         qml.RY(x, wires=0)
    #         return qml.expval(qml.PauliZ(wires=0))
    #
    #     par = np.array(0.2, requires_grad=True)
    #     assert np.isclose(qnode(par).item().val, reference_qnode(par))
    #     assert np.isclose(qml.jacobian(qnode)(par).item().val, qml.jacobian(reference_qnode)(par))


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

        with qml.tape.QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1, 2])

        tapes, fn = finite_diff(
            tape, approx_order=approx_order, strategy=strategy, validate_params=validate
        )
        res = fn(dev.batch_execute(tapes))

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
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tapes, fn = finite_diff(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=10e-2,
            shots=default_shot_vector,
        )
        all_res = fn(dev.batch_execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], numpy.ndarray)
            assert res[0].shape == ()

            assert isinstance(res[1], numpy.ndarray)
            assert res[1].shape == ()

            assert np.allclose(res, expected, atol=finite_diff_shot_vec_tol, rtol=0)

    def test_single_expectation_value_with_argnum_all(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where all parameters are chosen to compute
        the jacobian"""
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        # we choose both trainable parameters
        tapes, fn = finite_diff(
            tape,
            argnum=[0, 1],
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=10e-2,
            shots=many_shots_shot_vector,
        )
        all_res = fn(dev.batch_execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], numpy.ndarray)
            assert res[0].shape == ()

            assert isinstance(res[1], numpy.ndarray)
            assert res[1].shape == ()

            assert np.allclose(res, expected, atol=finite_diff_shot_vec_tol, rtol=0)

    def test_single_expectation_value_with_argnum_one(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output where only one parameter is chosen to
        estimate the jacobian.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        # we choose only 1 trainable parameter
        tapes, fn = finite_diff(
            tape,
            argnum=1,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=10e-2,
            shots=many_shots_shot_vector,
        )
        all_res = fn(dev.batch_execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        expected = [0, np.cos(y) * np.cos(x)]

        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], numpy.ndarray)
            assert res[0].shape == ()

            assert isinstance(res[1], numpy.ndarray)
            assert res[1].shape == ()

            assert np.allclose(res, expected, atol=finite_diff_shot_vec_tol, rtol=0)

    def test_multiple_expectation_value_with_argnum_one(
        self, approx_order, strategy, validate, tol
    ):
        """Tests correct output shape and evaluation for a tape
        with a multiple measurement, where only one parameter is chosen to
        be trainable.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.probs(wires=[0, 1])

        # we choose only 1 trainable parameter
        tapes, fn = finite_diff(
            tape,
            argnum=1,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=10e-2,
            shots=many_shots_shot_vector,
        )
        res = fn(dev.batch_execute(tapes))

        all_res = fn(dev.batch_execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        expected = [0, np.cos(y) * np.cos(x)]

        for res in all_res:
            assert isinstance(res, tuple)
            assert isinstance(res[0], tuple)
            assert np.allclose(res[0][0], 0)
            assert isinstance(res[1], tuple)
            assert np.allclose(res[1][0], 0)

    def test_multiple_expectation_values(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tapes, fn = finite_diff(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=10e-2,
            shots=many_shots_shot_vector,
        )
        all_res = fn(dev.batch_execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        for res in all_res:

            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], tuple)
            assert len(res[0]) == 2
            assert np.allclose(res[0], [-np.sin(x), 0], atol=finite_diff_shot_vec_tol, rtol=0)
            assert isinstance(res[0][0], numpy.ndarray)
            assert isinstance(res[0][1], numpy.ndarray)

            assert isinstance(res[1], tuple)
            assert len(res[1]) == 2
            assert np.allclose(res[1], [0, np.cos(y)], atol=finite_diff_shot_vec_tol, rtol=0)
            assert isinstance(res[1][0], numpy.ndarray)
            assert isinstance(res[1][1], numpy.ndarray)

    def test_var_expectation_values(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tapes, fn = finite_diff(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=10e-2,
            shots=many_shots_shot_vector,
        )
        all_res = fn(dev.batch_execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        for res in all_res:

            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], tuple)
            assert len(res[0]) == 2
            assert np.allclose(res[0], [-np.sin(x), 0], atol=finite_diff_shot_vec_tol, rtol=0)
            assert isinstance(res[0][0], numpy.ndarray)
            assert isinstance(res[0][1], numpy.ndarray)

            assert isinstance(res[1], tuple)
            assert len(res[1]) == 2
            assert np.allclose(
                res[1], [0, -2 * np.cos(y) * np.sin(y)], atol=finite_diff_shot_vec_tol, rtol=0
            )
            assert isinstance(res[1][0], numpy.ndarray)
            assert isinstance(res[1][1], numpy.ndarray)

    def test_prob_expectation_values(self, approx_order, strategy, validate, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tapes, fn = finite_diff(
            tape,
            approx_order=approx_order,
            strategy=strategy,
            validate_params=validate,
            h=10e-2,
            shots=many_shots_shot_vector,
        )
        all_res = fn(dev.batch_execute(tapes))

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        for res in all_res:

            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], tuple)
            assert len(res[0]) == 2
            assert np.allclose(res[0][0], -np.sin(x), atol=finite_diff_shot_vec_tol, rtol=0)
            assert isinstance(res[0][0], numpy.ndarray)
            assert np.allclose(res[0][1], 0, atol=finite_diff_shot_vec_tol, rtol=0)
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
                atol=finite_diff_shot_vec_tol,
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
                atol=finite_diff_shot_vec_tol,
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
        dev = qml.device("default.qubit.autograd", wires=2, shots=many_shots_shot_vector)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=10e-2,
                shots=many_shots_shot_vector,
            )
            jac = np.array(fn(dev.batch_execute(tapes)))
            return jac

        all_res = qml.jacobian(cost_fn)(params)

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        for res in all_res:
            x, y = params
            expected = np.array(
                [
                    [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                    [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
                ]
            )

            assert np.allclose(res, expected, atol=finite_diff_shot_vec_tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd_ragged(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        of a ragged tape can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit.autograd", wires=2, shots=many_shots_shot_vector)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=10e-2,
                shots=many_shots_shot_vector,
            )
            jac = fn(dev.batch_execute(tapes))
            return jac[1][0]

        x, y = params
        all_res = qml.jacobian(cost_fn)(params)[0]

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        for res in all_res:
            expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])
            assert np.allclose(res, expected, atol=finite_diff_shot_vec_tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_tf(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using TF, yielding second derivatives."""
        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=2, shots=many_shots_shot_vector)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape(persistent=True) as t:
            with qml.tape.QuantumTape() as tape:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=10e-2,
                shots=many_shots_shot_vector,
            )
            jac_0, jac_1 = fn(dev.batch_execute(tapes))

        x, y = 1.0 * params

        res_0 = t.jacobian(jac_0, params)
        res_1 = t.jacobian(jac_1, params)

        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )
        assert np.allclose([res_0, res_1], expected, atol=finite_diff_shot_vec_tol, rtol=0)

    @pytest.mark.tf
    def test_tf_ragged(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        of a ragged tape can be differentiated using TF, yielding second derivatives."""
        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=2, shots=many_shots_shot_vector)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape(persistent=True) as t:
            with qml.tape.QuantumTape() as tape:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=10e-2,
                shots=many_shots_shot_vector,
            )

            jac_01 = fn(dev.batch_execute(tapes))[1][0]

        x, y = 1.0 * params

        res_01 = t.jacobian(jac_01, params)

        expected = np.array([-np.cos(x) * np.cos(y) / 2, np.sin(x) * np.sin(y) / 2])

        assert np.allclose(res_01[0], expected, atol=finite_diff_shot_vec_tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using Torch, yielding second derivatives."""
        import torch

        dev = qml.device("default.qubit.torch", wires=2, shots=many_shots_shot_vector)
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)

        def cost_fn(params):
            with qml.tape.QuantumTape() as tape:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=10e-2,
                shots=many_shots_shot_vector,
            )
            jac = fn(dev.batch_execute(tapes))
            return jac

        hess = torch.autograd.functional.jacobian(cost_fn, params)

        x, y = params.detach().numpy()

        expected = np.array(
            [
                [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
            ]
        )

        assert np.allclose(
            hess[0].detach().numpy(), expected[0], atol=finite_diff_shot_vec_tol, rtol=0
        )
        assert np.allclose(
            hess[1].detach().numpy(), expected[1], atol=finite_diff_shot_vec_tol, rtol=0
        )

    @pytest.mark.jax
    def test_jax(self, approx_order, strategy, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using JAX, yielding second derivatives."""
        import jax
        from jax import numpy as jnp
        from jax.config import config

        config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit.jax", wires=2, shots=many_shots_shot_vector)
        params = jnp.array([0.543, -0.654])

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = finite_diff(
                tape,
                n=1,
                approx_order=approx_order,
                strategy=strategy,
                h=10e-2,
                shots=many_shots_shot_vector,
            )
            jac = fn(dev.batch_execute(tapes))
            return jac

        all_res = jax.jacobian(cost_fn)(params)

        assert isinstance(all_res, tuple)
        assert len(all_res) == len(default_shot_vector)

        for res in all_res:
            assert isinstance(res, tuple)
            x, y = params
            expected = np.array(
                [
                    [-np.cos(x) * np.sin(y), -np.cos(y) * np.sin(x)],
                    [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)],
                ]
            )
            assert np.allclose(res, expected, atol=finite_diff_shot_vec_tol, rtol=0)

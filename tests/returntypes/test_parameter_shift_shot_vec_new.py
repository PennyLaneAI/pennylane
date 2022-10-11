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
"""Tests for the gradients.parameter_shift module using the new return types and devices that define a shot vector."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift
from pennylane.gradients.parameter_shift import _get_operation_recipe, _put_zeros_in_pdA2_involutory
from pennylane.devices import DefaultQubit
from pennylane.operation import Observable, AnyWires


shot_vec_tol = 10e-3
default_shot_vector = (1000, 100)
many_shots_shot_vector = (1000000, 10000000)


class TestParamShift:
    """Unit tests for the param_shift function"""

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == None

        res = fn(dev.batch_execute(tapes))
        for r in res:
            assert isinstance(r, tuple)
            assert len(r) == 2
            assert r[0].shape == ()
            assert r[1].shape == ()

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, [0])

    # TODO: uncomment and port to shot-vectors when QNode decorator uses new qml.execute pipeline
    # @pytest.mark.autograd
    # def test_no_trainable_params_qnode_autograd(self, mocker):
    #     """Test that the correct ouput and warning is generated in the absence of any trainable
    #     parameters"""
    #     dev = qml.device("default.qubit", wires=2)
    #     spy = mocker.spy(dev, "expval")

    #     @qml.qnode(dev, interface="autograd")
    #     def circuit(weights):
    #         qml.RX(weights[0], wires=0)
    #         qml.RY(weights[1], wires=0)
    #         return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    #     weights = [0.1, 0.2]
    #     with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
    #         res = qml.gradients.param_shift(circuit)(weights)

    #     assert res == ()
    #     spy.assert_not_called()

    # @pytest.mark.torch
    # def test_no_trainable_params_qnode_torch(self, mocker):
    #     """Test that the correct ouput and warning is generated in the absence of any trainable
    #     parameters"""
    #     dev = qml.device("default.qubit", wires=2)
    #     spy = mocker.spy(dev, "expval")

    #     @qml.qnode(dev, interface="torch")
    #     def circuit(weights):
    #         qml.RX(weights[0], wires=0)
    #         qml.RY(weights[1], wires=0)
    #         return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    #     weights = [0.1, 0.2]
    #     with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
    #         res = qml.gradients.param_shift(circuit)(weights)

    #     assert res == ()
    #     spy.assert_not_called()

    # @pytest.mark.tf
    # def test_no_trainable_params_qnode_tf(self, mocker):
    #     """Test that the correct ouput and warning is generated in the absence of any trainable
    #     parameters"""
    #     dev = qml.device("default.qubit", wires=2)
    #     spy = mocker.spy(dev, "expval")

    #     @qml.qnode(dev, interface="tf")
    #     def circuit(weights):
    #         qml.RX(weights[0], wires=0)
    #         qml.RY(weights[1], wires=0)
    #         return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    #     weights = [0.1, 0.2]
    #     with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
    #         res = qml.gradients.param_shift(circuit)(weights)

    #     assert res == ()
    #     spy.assert_not_called()

    # @pytest.mark.jax
    # def test_no_trainable_params_qnode_jax(self, mocker):
    #     """Test that the correct ouput and warning is generated in the absence of any trainable
    #     parameters"""
    #     dev = qml.device("default.qubit", wires=2)
    #     spy = mocker.spy(dev, "expval")

    #     @qml.qnode(dev, interface="jax")
    #     def circuit(weights):
    #         qml.RX(weights[0], wires=0)
    #         qml.RY(weights[1], wires=0)
    #         return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    #     weights = [0.1, 0.2]
    #     with pytest.warns(UserWarning, match="gradient of a QNode with no trainable parameters"):
    #         res = qml.gradients.param_shift(circuit)(weights)

    #     assert res == ()
    #     spy.assert_not_called()

    @pytest.mark.parametrize("broadcast", [True, False])
    def test_no_trainable_params_tape(self, broadcast):
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
            g_tapes, post_processing = qml.gradients.param_shift(tape, broadcast=broadcast)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, np.ndarray)
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
            g_tapes, post_processing = qml.gradients.param_shift(tape)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, tuple)
        for r in res:
            assert isinstance(r, np.ndarray)
            assert r.shape == (0,)

    def test_all_zero_diff_methods_tape(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4, shots=default_shot_vector)

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.Rot(*params, wires=0)
            qml.probs([2, 3])

        g_tapes, post_processing = qml.gradients.param_shift(tape)
        assert g_tapes == []

        result = post_processing(qml.execute(g_tapes, dev, None))

        assert isinstance(result, tuple)

        assert len(result) == 3

        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (4,)
        assert np.allclose(result[0], 0)

        assert isinstance(result[1], np.ndarray)
        assert result[1].shape == (4,)
        assert np.allclose(result[1], 0)

        assert isinstance(result[2], np.ndarray)
        assert result[2].shape == (4,)
        assert np.allclose(result[2], 0)

    def test_all_zero_diff_methods_multiple_returns_tape(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""

        dev = qml.device("default.qubit", wires=4, shots=default_shot_vector)

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.Rot(*params, wires=0)
            qml.expval(qml.PauliZ(wires=2))
            qml.probs([2, 3])

        g_tapes, post_processing = qml.gradients.param_shift(tape)
        assert g_tapes == []

        result = post_processing(qml.execute(g_tapes, dev, None))

        assert isinstance(result, tuple)

        assert len(result) == 2

        # First elem
        assert len(result[0]) == 3

        assert isinstance(result[0][0], np.ndarray)
        assert result[0][0].shape == (1,)
        assert np.allclose(result[0][0], 0)

        assert isinstance(result[0][1], np.ndarray)
        assert result[0][1].shape == (1,)
        assert np.allclose(result[0][1], 0)

        assert isinstance(result[0][2], np.ndarray)
        assert result[0][2].shape == (1,)
        assert np.allclose(result[0][2], 0)

        # Second elem
        assert len(result[0]) == 3

        assert isinstance(result[1][0], np.ndarray)
        assert result[1][0].shape == (4,)
        assert np.allclose(result[1][0], 0)

        assert isinstance(result[1][1], np.ndarray)
        assert result[1][1].shape == (4,)
        assert np.allclose(result[1][1], 0)

        assert isinstance(result[1][2], np.ndarray)
        assert result[1][2].shape == (4,)
        assert np.allclose(result[1][2], 0)

        tapes, _ = qml.gradients.param_shift(tape)
        assert tapes == []

    # TODO: uncomment when QNode decorator uses new qml.execute pipeline
    # @pytest.mark.parametrize("broadcast", [True, False])
    # def test_all_zero_diff_methods(self, broadcast):
    #     """Test that the transform works correctly when the diff method for every parameter is
    #     identified to be 0, and that no tapes were generated."""
    #     dev = qml.device("default.qubit", wires=4)

    #     @qml.qnode(dev)
    #     def circuit(params):
    #         qml.Rot(*params, wires=0)
    #         return qml.probs([2, 3])

    #     params = np.array([0.5, 0.5, 0.5], requires_grad=True)

    #     result = qml.gradients.param_shift(circuit)(params)
    #     assert np.allclose(result, np.zeros((4, 3)), atol=0, rtol=0)

    #     tapes, _ = qml.gradients.param_shift(circuit.tape, broadcast=broadcast)
    #     assert tapes == []

    @pytest.mark.parametrize("ops_with_custom_recipe", [[0], [1], [0, 1]])
    def test_recycled_unshifted_tape(self, ops_with_custom_recipe):
        """Test that if the gradient recipe has a zero-shift component, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)
        x = [0.543, -0.654]

        with qml.tape.QuantumTape() as tape:
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[0])
            qml.expval(qml.PauliZ(0))

        gradient_recipes = tuple(
            [[-1e7, 1, 0], [1e7, 1, 1e-7]] if i in ops_with_custom_recipe else None
            for i in range(2)
        )
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)

        # two tapes per parameter that doesn't use a custom recipe,
        # one tape per parameter that uses custom recipe,
        # plus one global call if at least one uses the custom recipe
        num_ops_standard_recipe = tape.num_params - len(ops_with_custom_recipe)
        assert len(tapes) == 2 * num_ops_standard_recipe + len(ops_with_custom_recipe) + 1
        # Test that executing the tapes and the postprocessing function works
        grad = fn(qml.execute(tapes, dev, None))
        assert qml.math.allclose(grad, -np.sin(x[0] + x[1]), atol=1e-5)

    @pytest.mark.parametrize("y_wire", [0, 1])
    def test_f0_provided(self, y_wire):
        """Test that if the original tape output is provided, then
        the tape is not executed additionally at the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=y_wire)
            qml.expval(qml.PauliZ(0))

        gradient_recipes = ([[-1e7, 1, 0], [1e7, 1, 1e7]],) * 2
        f0 = dev.execute(tape)
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes, f0=f0)

        # one tape per parameter that impacts the expval
        assert len(tapes) == 2 if y_wire == 0 else 1

        fn(dev.batch_execute(tapes))

    def test_op_with_custom_unshifted_term(self):
        """Test that an operation with a gradient recipe that depends on
        its instantiated parameter values works correctly within the parameter
        shift rule. Also tests that grad_recipes supersedes paramter_frequencies.
        """
        s = np.pi / 2

        class RX(qml.RX):
            """RX operation with an additional term in the grad recipe.
            The grad_recipe no longer yields the derivative, but we account for this.
            For this test, the presence of the unshifted term (with non-vanishing coefficient)
            is essential."""

            grad_recipe = ([[0.5, 1, s], [-0.5, 1, -s], [0.2, 1, 0]],)

        x = np.array([-0.361, 0.654], requires_grad=True)
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        with qml.tape.QuantumTape() as tape:
            qml.RX(x[0], wires=0)
            RX(x[1], wires=0)
            qml.expval(qml.PauliZ(0))

        tapes, fn = qml.gradients.param_shift(tape)

        # Unshifted tapes always are first within the tapes created for one operation;
        # They are not batched together because we trust operation recipes to be condensed already
        expected_shifts = [[0, 0], [s, 0], [-s, 0], [0, s], [0, -s]]
        assert len(tapes) == 5
        for tape, expected in zip(tapes, expected_shifts):
            assert tape.operations[0].data[0] == x[0] + expected[0]
            assert tape.operations[1].data[0] == x[1] + expected[1]

        grad = fn(dev.batch_execute(tapes))
        exp = np.stack([-np.sin(x[0] + x[1]), -np.sin(x[0] + x[1]) + 0.2 * np.cos(x[0] + x[1])])
        assert len(grad) == len(exp)
        for (
            a,
            b,
        ) in zip(grad, exp):
            assert np.allclose(a, b)

    @pytest.mark.slow
    def test_independent_parameters_analytic(self):
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

        tapes, fn = qml.gradients.param_shift(tape1)
        j1 = fn(dev.batch_execute(tapes))

        # We should only be executing the device twice: Two shifted evaluations to differentiate
        # one parameter overall, as the other parameter does not impact the returned measurement.

        assert dev.num_executions == 2

        tapes, fn = qml.gradients.param_shift(tape2)
        j2 = fn(dev.batch_execute(tapes))

        exp = -np.sin(1)

        assert np.allclose(j1[0][0], exp, atol=shot_vec_tol)
        assert np.allclose(j1[0][1], exp, atol=shot_vec_tol)

        assert np.allclose(j1[1][0], 0, atol=shot_vec_tol)
        assert np.allclose(j1[1][1], 0, atol=shot_vec_tol)

        assert np.allclose(j2[0][0], 0, atol=shot_vec_tol)
        assert np.allclose(j2[0][1], 0, atol=shot_vec_tol)

        assert np.allclose(j2[1][0], exp, atol=shot_vec_tol)
        assert np.allclose(j2[1][1], exp, atol=shot_vec_tol)

    def test_grad_recipe_parameter_dependent(self, monkeypatch):
        """Test that an operation with a gradient recipe that depends on
        its instantiated parameter values works correctly within the parameter
        shift rule. Also tests that `grad_recipe` supersedes `parameter_frequencies`.
        """

        class RX(qml.RX):
            @property
            def grad_recipe(self):
                # The gradient is given by [f(2x) - f(0)] / (2 sin(x)), by subsituting
                # shift = x into the two term parameter-shift rule.
                x = self.data[0]
                c = 0.5 / np.sin(x)
                return ([[c, 0.0, 2 * x], [-c, 0.0, 0.0]],)

        x = np.array(0.654, requires_grad=True)
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)

        with qml.tape.QuantumTape() as tape:
            RX(x, wires=0)
            qml.expval(qml.PauliZ(0))

        tapes, fn = qml.gradients.param_shift(tape)

        assert len(tapes) == 2
        assert [t.batch_size for t in tapes] == [None, None]
        assert qml.math.allclose(tapes[0].operations[0].data[0], 0)
        assert qml.math.allclose(tapes[1].operations[0].data[0], 2 * x)

        grad = fn(dev.batch_execute(tapes))
        for g in grad:
            assert np.allclose(grad, -np.sin(x), atol=shot_vec_tol)

    def test_error_no_diff_info(self):
        """Test that an error is raised if no grad_recipe, no parameter_frequencies
        and no generator are found."""

        class RX(qml.RX):
            """This copy of RX overwrites parameter_frequencies to report
            missing information, disabling its differentiation."""

            @property
            def parameter_frequencies(self):
                raise qml.operation.ParameterFrequenciesUndefinedError

        class NewOp(qml.operation.Operation):
            """This new operation does not overwrite parameter_frequencies
            but does not have a generator, disabling its differentiation."""

            num_params = 1
            grad_method = "A"
            num_wires = 1

        x = np.array(0.654, requires_grad=True)
        dev = qml.device("default.qubit", wires=2, shots=default_shot_vector)

        for op in [RX, NewOp]:
            with qml.tape.QuantumTape() as tape:
                op(x, wires=0)
                qml.expval(qml.PauliZ(0))

            with pytest.raises(
                qml.operation.OperatorPropertyUndefined, match="does not have a grad_recipe"
            ):
                qml.gradients.param_shift(tape)


@pytest.mark.slow
class TestParamShiftShotVector:
    """Unit tests for the param_shift function used with a device that has a
    shot vector defined"""

    def test_multi_measure_probs_expval(self, tol):
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

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

        res = fn(dev.batch_execute(tapes))
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

        for r in res:

            # Expvals
            r_to_check = r[0][0]
            exp = expval_expected[0]
            assert np.allclose(r_to_check, exp, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == ()

            r_to_check = r[0][1]
            exp = expval_expected[1]
            assert np.allclose(r_to_check, exp, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == ()

            # Probs

            r_to_check = r[1][0]
            exp = probs_expected[:, 0]
            assert np.allclose(r_to_check, exp, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == (4,)

            r_to_check = r[1][1]
            exp = probs_expected[:, 1]
            assert np.allclose(r_to_check, exp, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == (4,)

    def test_involutory_variance(self, tol):
        """Tests qubit observables that are involutory"""
        dev = qml.device("default.qubit", wires=1, shots=many_shots_shot_vector)
        a = 0.54

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.PauliZ(0))

        res = dev.execute(tape)
        expected = 1 - np.cos(a) ** 2
        for r in res:
            assert np.allclose(r, expected, atol=shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        for _gA in gradA:
            assert isinstance(_gA, np.ndarray)
            assert _gA.shape == ()

        assert len(tapes) == 1 + 2 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 2

        expected = 2 * np.sin(a) * np.cos(a)

        # TODO: finite diff shot-vector update
        # assert gradF == pytest.approx(expected, abs=tol)
        for _gA in gradA:
            assert _gA == pytest.approx(expected, abs=shot_vec_tol)

    def test_non_involutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory"""
        dev = qml.device("default.qubit", wires=1, shots=many_shots_shot_vector)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        herm_shot_vec_tol = shot_vec_tol * 100
        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.Hermitian(A, 0))

        tape.trainable_params = {0}

        res = dev.execute(tape)
        expected = (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)
        for r in res:
            assert np.allclose(r, expected, atol=herm_shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 4 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 2

        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        for _gA in gradA:
            assert _gA == pytest.approx(expected, abs=herm_shot_vec_tol)
            assert isinstance(_gA, np.ndarray)
            assert _gA.shape == ()
            # TODO: finite diff shot-vector update
            # assert gradF == pytest.approx(expected, abs=tol)

    def test_involutory_and_noninvolutory_variance_single_param(self, tol):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable when there's a single trainable parameter."""
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        herm_shot_vec_tol = shot_vec_tol * 100
        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        # Note: only the first param is trainable
        tape.trainable_params = {0}

        res = dev.execute(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        for r in res:
            assert np.allclose(r, expected, atol=herm_shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 4

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 1

        expected = [2 * np.sin(a) * np.cos(a), 0]

        # Param-shift
        for shot_vec_result in gradA:
            for param_res in shot_vec_result:
                assert isinstance(param_res, np.ndarray)
                assert param_res.shape == ()

            assert shot_vec_result[0] == pytest.approx(expected[0], abs=herm_shot_vec_tol)
            assert shot_vec_result[1] == pytest.approx(expected[1], abs=herm_shot_vec_tol)

        # TODO: finite diff shot-vector update
        # for shot_vec_result in gradF:
        #     for param_res in shot_vec_result:
        #         assert isinstance(param_res, np.ndarray)
        #         assert param_res.shape == ()

        #     assert shot_vec_result[0] == pytest.approx(expected[0], abs=herm_shot_vec_tol)
        #     assert shot_vec_result[1] == pytest.approx(expected[1], abs=herm_shot_vec_tol)

    def test_involutory_and_noninvolutory_variance_multi_param(self, tol):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable."""
        dev = qml.device("default.qubit", wires=2, shots=many_shots_shot_vector)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        tape.trainable_params = {0, 1}
        herm_shot_vec_tol = shot_vec_tol * 100

        res = dev.execute(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        for res_shot_item in res:
            assert np.allclose(res_shot_item, expected, atol=herm_shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))

        assert isinstance(gradA, tuple)
        assert len(gradA) == 2
        for shot_vec_res in gradA:
            for meas_res in shot_vec_res:
                for param_res in meas_res:
                    assert isinstance(param_res, np.ndarray)
                    assert param_res.shape == ()

        assert len(tapes) == 1 + 2 * 4

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 2

        expected = [2 * np.sin(a) * np.cos(a), 0, 0, -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]

        # Param-shift
        for shot_vec_result in gradA:
            assert isinstance(shot_vec_result[0][0], np.ndarray)
            assert shot_vec_result[0][0].shape == ()
            assert shot_vec_result[0][0] == pytest.approx(expected[0], abs=herm_shot_vec_tol)

            assert isinstance(shot_vec_result[0][1], np.ndarray)
            assert shot_vec_result[0][1].shape == ()
            assert shot_vec_result[0][1] == pytest.approx(expected[1], abs=herm_shot_vec_tol)

            assert isinstance(shot_vec_result[1][0], np.ndarray)
            assert shot_vec_result[1][0].shape == ()
            assert shot_vec_result[1][0] == pytest.approx(expected[2], abs=herm_shot_vec_tol)

            assert isinstance(shot_vec_result[1][1], np.ndarray)
            assert shot_vec_result[1][1].shape == ()
            assert shot_vec_result[1][1] == pytest.approx(expected[3], abs=herm_shot_vec_tol)

        # TODO: finite diff shot-vector update
        # for shot_vec_result in gradF:
        #     for param_res in shot_vec_result:
        #         assert isinstance(param_res, np.ndarray)
        #         assert param_res.shape == ()

        #     assert shot_vec_result[0] == pytest.approx(expected[0], abs=herm_shot_vec_tol)
        #     assert shot_vec_result[1] == pytest.approx(expected[1], abs=herm_shot_vec_tol)

    # TODO: finite diff shot-vector update
    @pytest.mark.xfail(reason="Uses finite diff")
    def test_expval_and_variance(self, tol):
        """Test that the qnode works for a combination of expectation
        values and variances"""
        dev = qml.device("default.qubit", wires=3, shots=many_shots_shot_vector)

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

        res = dev.execute(tape)
        expected = np.array(
            [
                np.sin(a) ** 2,
                np.cos(a) * np.cos(b),
                0.25 * (3 - 2 * np.cos(b) ** 2 * np.cos(2 * c) - np.cos(2 * b)),
            ]
        )

        assert isinstance(res, tuple)
        assert np.allclose(res, expected, atol=shot_vec_tol, rtol=0)

        # # circuit jacobians
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
        assert isinstance(gradA, tuple)
        for a, e in zip(gradA, expected):
            for a_comp, e_comp in zip(a, e):
                assert isinstance(a_comp, np.ndarray)
                assert a_comp.shape == ()
                assert np.allclose(a_comp, e_comp, atol=shot_vec_tol, rtol=0)
        assert gradF == pytest.approx(expected, abs=shot_vec_tol)

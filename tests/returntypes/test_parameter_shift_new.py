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


class TestGetOperationRecipe:
    """Test the helper function `_get_operation_recipe` that obtains the
    `grad_recipe` for a given operation in a tape."""

    @pytest.mark.parametrize(
        "orig_op, frequencies, shifts",
        [
            (qml.RX, (1.0,), None),
            (qml.RX, (1.0,), (np.pi / 2,)),
            (qml.CRY, (0.5, 1), None),
            (qml.CRY, (0.5, 1), (0.4, 0.8)),
        ],
    )
    def test_custom_recipe_first_order(self, orig_op, frequencies, shifts):
        """Test that a custom recipe is returned correctly for first-order derivatives."""
        c, s = qml.gradients.generate_shift_rule(frequencies, shifts=shifts).T
        recipe = list(zip(c, np.ones_like(c), s))

        class DummyOp(orig_op):
            grad_recipe = (recipe,)

        with qml.tape.QuantumTape() as tape:
            DummyOp(0.2, wires=list(range(DummyOp.num_wires)))

        out_recipe = _get_operation_recipe(tape, 0, shifts=shifts, order=1)
        assert qml.math.allclose(out_recipe[:, 0], c)
        assert qml.math.allclose(out_recipe[:, 1], np.ones_like(c))

        if shifts is None:
            assert qml.math.allclose(out_recipe[:, 2], s)
        else:
            exp_out_shifts = [-s for s in shifts[::-1]] + list(shifts)
            assert qml.math.allclose(np.sort(s), exp_out_shifts)
            assert qml.math.allclose(np.sort(out_recipe[:, 2]), np.sort(exp_out_shifts))

    @pytest.mark.parametrize(
        "orig_op, frequencies, shifts",
        [
            (qml.RX, (1.0,), None),
            (qml.RX, (1.0,), (np.pi / 2,)),
            (qml.CRY, (0.5, 1), None),
            (qml.CRY, (0.5, 1), (0.4, 0.8)),
        ],
    )
    def test_custom_recipe_second_order(self, orig_op, frequencies, shifts):
        """Test that a custom recipe is returned correctly for second-order derivatives."""
        c, s = qml.gradients.generate_shift_rule(frequencies, shifts=shifts).T
        recipe = list(zip(c, np.ones_like(c), s))

        class DummyOp(orig_op):
            grad_recipe = (recipe,)

        with qml.tape.QuantumTape() as tape:
            DummyOp(0.2, wires=list(range(DummyOp.num_wires)))

        out_recipe = _get_operation_recipe(tape, 0, shifts=shifts, order=2)
        c2, s2 = qml.gradients.generate_shift_rule(frequencies, shifts=shifts, order=2).T
        assert qml.math.allclose(out_recipe[:, 0], c2)
        assert qml.math.allclose(out_recipe[:, 1], np.ones_like(c2))
        assert qml.math.allclose(out_recipe[:, 2], s2)

    @pytest.mark.parametrize("order", [0, 3])
    def test_error_wrong_order(self, order):
        """Test that get_operation_recipe raises an error for orders other than 1 and 2"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.2, wires=0)

        with pytest.raises(NotImplementedError, match="only is implemented for orders 1 and 2."):
            _get_operation_recipe(tape, 0, shifts=None, order=order)


def grad_fn(tape, dev, fn=qml.gradients.param_shift, **kwargs):
    """Utility function to automate execution and processing of gradient tapes"""
    tapes, fn = fn(tape, **kwargs)
    return fn(dev.batch_execute_new(tapes))


class TestParamShift:
    """Unit tests for the param_shift function"""

    def test_empty_circuit(self):
        """Test that an empty circuit works correctly"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            tapes, _ = qml.gradients.param_shift(tape)
        assert not tapes

    def test_all_parameters_independent(self):
        """Test that a circuit where all parameters do not affect the output"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.expval(qml.PauliZ(1))

        tapes, _ = qml.gradients.param_shift(tape)
        assert not tapes

    def test_state_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a state"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.state()

        with pytest.raises(ValueError, match=r"return the state is not supported"):
            qml.gradients.param_shift(tape)

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == None

        res = fn(dev.batch_execute_new(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, [0])

    # TODO: uncomment when QNode decorator uses new qml.execute pipeline
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
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        # TODO: remove once #2155 is resolved
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.param_shift(tape, broadcast=broadcast)
        res = post_processing(qml.execute_new(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, np.ndarray)
        assert res.shape == (1, 0)

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

    def test_with_gradient_recipes(self):
        """Test that the function behaves as expected"""

        with qml.tape.QuantumTape() as tape:
            qml.PauliZ(0)
            qml.RX(1.0, wires=0)
            qml.CNOT(wires=[0, 2])
            qml.Rot(2.0, 3.0, 4.0, wires=0)
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 2}
        gradient_recipes = ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        tapes, _ = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)

        assert len(tapes) == 5
        assert [t.batch_size for t in tapes] == [None] * 5
        assert tapes[0].get_parameters(trainable_only=False) == [0.2 * 1.0 + 0.3, 2.0, 3.0, 4.0]
        assert tapes[1].get_parameters(trainable_only=False) == [0.5 * 1.0 + 0.6, 2.0, 3.0, 4.0]
        assert tapes[2].get_parameters(trainable_only=False) == [1.0, 2.0, 1 * 3.0 + 1, 4.0]
        assert tapes[3].get_parameters(trainable_only=False) == [1.0, 2.0, 2 * 3.0 + 2, 4.0]
        assert tapes[4].get_parameters(trainable_only=False) == [1.0, 2.0, 3 * 3.0 + 3, 4.0]

    @pytest.mark.parametrize("ops_with_custom_recipe", [[0], [1], [0, 1]])
    def test_recycled_unshifted_tape(self, ops_with_custom_recipe):
        """Test that if the gradient recipe has a zero-shift component, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)
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
        grad = fn(qml.execute_new(tapes, dev, None))
        assert qml.math.allclose(grad, -np.sin(x[0] + x[1]), atol=1e-5)

    @pytest.mark.parametrize("y_wire", [0, 1])
    def test_f0_provided(self, y_wire):
        """Test that if the original tape output is provided, then
        the tape is not executed additionally at the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=y_wire)
            qml.expval(qml.PauliZ(0))

        gradient_recipes = ([[-1e7, 1, 0], [1e7, 1, 1e7]],) * 2
        f0 = dev.execute_new(tape)
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes, f0=f0)

        # one tape per parameter that impacts the expval
        assert len(tapes) == 2 if y_wire == 0 else 1

        fn(dev.batch_execute_new(tapes))

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
        dev = qml.device("default.qubit", wires=2)

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

        grad = fn(dev.batch_execute_new(tapes))
        exp = np.stack([-np.sin(x[0] + x[1]), -np.sin(x[0] + x[1]) + 0.2 * np.cos(x[0] + x[1])])
        assert len(grad) == len(exp)
        for (
            a,
            b,
        ) in zip(grad, exp):
            assert np.allclose(a, b)

    def test_independent_parameters_analytic(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(1))

        tapes, fn = qml.gradients.param_shift(tape1)
        j1 = fn(dev.batch_execute_new(tapes))

        # We should only be executing the device twice: Two shifted evaluations to differentiate
        # one parameter overall, as the other parameter does not impact the returned measurement.

        # TODO: add dev.num_executions update logic to execute_new
        # assert dev.num_executions == 2

        tapes, fn = qml.gradients.param_shift(tape2)
        j2 = fn(dev.batch_execute_new(tapes))

        exp = -np.sin(1)

        assert np.allclose(j1[0], exp)
        assert np.allclose(j1[1], 0)
        assert np.allclose(j2[0], 0)
        assert np.allclose(j2[1], exp)

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
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            RX(x, wires=0)
            qml.expval(qml.PauliZ(0))

        tapes, fn = qml.gradients.param_shift(tape)

        assert len(tapes) == 2
        assert [t.batch_size for t in tapes] == [None, None]
        assert qml.math.allclose(tapes[0].operations[0].data[0], 0)
        assert qml.math.allclose(tapes[1].operations[0].data[0], 2 * x)

        grad = fn(dev.batch_execute_new(tapes))
        assert np.allclose(grad, -np.sin(x))

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
        dev = qml.device("default.qubit", wires=2)

        for op in [RX, NewOp]:
            with qml.tape.QuantumTape() as tape:
                op(x, wires=0)
                qml.expval(qml.PauliZ(0))

            with pytest.raises(
                qml.operation.OperatorPropertyUndefined, match="does not have a grad_recipe"
            ):
                qml.gradients.param_shift(tape)


class TestParamShiftBroadcast:
    """Tests for the `param_shift` function using broadcasting."""

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        assert len(tapes) == 1
        assert tapes[0].batch_size == 2

        res = fn(dev.batch_execute_new(tapes))
        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, [0])

    def test_with_gradient_recipes(self):
        """Test that the function behaves as expected"""

        x, z0, y, z1 = 1.0, 2.0, 3.0, 4.0
        with qml.tape.QuantumTape() as tape:
            qml.PauliZ(0)
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 2])
            qml.Rot(z0, y, z1, wires=0)
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 2}
        gradient_recipes = ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        tapes, _ = qml.gradients.param_shift(
            tape, gradient_recipes=gradient_recipes, broadcast=True
        )

        assert len(tapes) == 2
        assert [t.batch_size for t in tapes] == [2, 3]
        assert all(
            qml.math.allclose(p, exp)
            for p, exp in zip(
                tapes[0].get_parameters(trainable_only=False),
                [[m * x + s for _, m, s in gradient_recipes[0]], z0, y, z1],
            )
        )
        assert all(
            qml.math.allclose(p, exp)
            for p, exp in zip(
                tapes[1].get_parameters(trainable_only=False),
                [x, z0, [m * y + s for _, m, s in gradient_recipes[1]], z1],
            )
        )

    def test_recycled_unshifted_tape(self):
        """Test that if the gradient recipe has a zero-shift component, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        gradient_recipes = ([[-1e7, 1, 0], [1e7, 1, 1e7]],) * 2
        tapes, fn = qml.gradients.param_shift(
            tape, gradient_recipes=gradient_recipes, broadcast=True
        )

        # one tape per parameter, plus one global call
        assert len(tapes) == tape.num_params + 1

    def test_independent_parameters_analytic(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(1))

        tapes, fn = qml.gradients.param_shift(tape1, broadcast=True)
        j1 = fn(dev.batch_execute_new(tapes))

        # We should only be executing the device to differentiate 1 parameter
        # (1 broadcasted execution)

        # TODO: add dev.num_executions update logic to execute_new
        # assert dev.num_executions == 1

        tapes, fn = qml.gradients.param_shift(tape2, broadcast=True)
        j2 = fn(dev.batch_execute_new(tapes))

        exp = -np.sin(1)

        assert np.allclose(j1[0], exp)
        assert np.allclose(j1[1], 0)
        assert np.allclose(j2[0], 0)
        assert np.allclose(j2[1], exp)

    def test_grad_recipe_parameter_dependent(self, monkeypatch):
        """Test that an operation with a gradient recipe that depends on
        its instantiated parameter values works correctly within the parameter
        shift rule. Also tests that grad_recipes supersedes parameter_frequencies.
        """

        def fail(*args, **kwargs):
            raise qml.operation.ParameterFrequenciesUndefinedError

        monkeypatch.setattr(qml.RX, "parameter_frequencies", fail)

        class RX(qml.RX):
            @property
            def grad_recipe(self):
                # The gradient is given by [f(2x) - f(0)] / (2 sin(x)), by subsituting
                # shift = x into the two term parameter-shift rule.
                x = self.data[0]
                c = 0.5 / np.sin(x)
                return ([[c, 0.0, 2 * x], [-c, 0.0, 0.0]],)

        x = np.array(0.654, requires_grad=True)
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            RX(x, wires=0)
            qml.expval(qml.PauliZ(0))

        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)

        assert len(tapes) == 1
        assert tapes[0].batch_size == 2
        assert qml.math.allclose(tapes[0].operations[0].data[0], [0, 2 * x])

        grad = fn(dev.batch_execute_new(tapes))
        assert np.allclose(grad, -np.sin(x))


class TestParameterShiftRule:
    """Tests for the parameter shift implementation"""

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift])
    def test_pauli_rotation_gradient(self, mocker, G, theta, shift, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""

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

        # TODO: finite diff update
        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute_new(tapes))
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    def test_Rot_gradient(self, mocker, theta, shift, tol):
        """Tests that the automatic gradient of an arbitrary Euler-angle-parameterized gate is correct."""
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

        # TODO: finite diff update
        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = np.squeeze(fn(dev.batch_execute_new(tapes)))
        for a_val, n_val in zip(autograd_val, numeric_val):
            assert np.allclose(a_val, n_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, tol):
        """Test gradient of controlled rotation gates"""
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

        # TODO: finite diff update
        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute_new(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parameterized
        gate is correct."""
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

        # TODO: finite diff update
        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = np.squeeze(fn(dev.batch_execute_new(tapes)))
        for idx, g in enumerate(grad):
            assert np.allclose(g, numeric_val[idx], atol=tol, rtol=0)

    def test_gradients_agree_finite_differences(self, tol):
        """Tests that the parameter-shift rule agrees with the first and second
        order finite differences"""
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

        # TODO: finite diff update
        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1)
        grad_F2 = grad_fn(
            tape, dev, fn=qml.gradients.finite_diff, approx_order=2, strategy="center"
        )
        grad_A = grad_fn(tape, dev)

        # gradients computed with different methods must agree
        assert np.allclose(grad_A, grad_F1, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F2, atol=tol, rtol=0)

    # TODO: remove xfail when finite diff works
    @pytest.mark.xfail
    def test_variance_gradients_agree_finite_differences(self, tol):
        """Tests that the variance parameter-shift rule agrees with the first and second
        order finite differences"""
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

    # TODO: remove xfail when finite diff works
    @pytest.mark.autograd
    @pytest.mark.xfail
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

    @pytest.mark.autograd
    @pytest.mark.xfail
    def test_all_fallback(self, mocker, tol):
        """Test that *only* the fallback logic is called if no parameters
        support the parameter-shift rule"""
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

        # TODO: finite diff update
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

    def test_multiple_expectation_values(self, tol):
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

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

        res = fn(dev.batch_execute_new(tapes))
        assert len(res) == 2
        assert len(res[0]) == 2
        assert len(res[1]) == 2

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_var_expectation_values(self, tol):
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

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 5

        res = fn(dev.batch_execute_new(tapes))
        assert len(res) == 2
        assert len(res[0]) == 2
        assert len(res[1]) == 2

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])

        for a, e in zip(res, expected):
            assert np.allclose(np.squeeze(np.stack(a)), e, atol=tol, rtol=0)

    def test_prob_expectation_values(self, tol):
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
        assert np.allclose(res[0][1], expval_expected[1])

        # Probs
        assert np.allclose(res[1][0], probs_expected[:, 0])
        assert np.allclose(res[1][1], probs_expected[:, 1])

    def test_involutory_variance(self, tol):
        """Tests qubit observables that are involutory"""
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

    def test_non_involutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory"""
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

    def test_involutory_and_noninvolutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable."""
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

    def test_expval_and_variance(self, tol):
        """Test that the qnode works for a combination of expectation
        values and variances"""
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

    def test_projector_variance(self, tol):
        """Test that the variance of a projector is correctly returned"""
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

        # TODO: check when finite diff ready:
        # tapes, fn = qml.gradients.finite_diff(tape)
        # gradF = fn(dev.batch_execute_new(tapes))

        expected = np.array(
            [
                [
                    0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                    -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
                ]
            ]
        )
        assert np.allclose(gradA, expected, atol=tol, rtol=0)
        # TODO: check when finite diff ready:
        # assert gradF == pytest.approx(expected, abs=tol)

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
        (cost3, [2, 3], True),
    ]

    @pytest.mark.parametrize("cost, expected_shape, list_output", costs_and_expected_expval)
    def test_output_shape_matches_qnode_expval(self, cost, expected_shape, list_output):
        """Test that the transform output shape matches that of the QNode."""
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

    costs_and_expected_probs = [
        (cost4, [3, 4], False),
        (cost5, [3, 4], True),
        # The output shape of transforms for 2D qnode outputs (cost6) is currently
        # transposed, e.g. (2, 3, 4) instead of (2, 4, 3).
        # TODO: fix qnode/expected once #2296 is resolved
        (cost6, [2, 3, 4], True),
    ]

    @pytest.mark.parametrize("cost, expected_shape, list_output", costs_and_expected_probs)
    def test_output_shape_matches_qnode_probs(self, cost, expected_shape, list_output):
        """Test that the transform output shape matches that of the QNode."""
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

    # TODO: revisit the following test when the Autograd interface supports
    # parameter-shift with the new return type system
    # def test_special_observable_qnode_differentiation(self):
    #     """Test differentiation of a QNode on a device supporting a
    #     special observable that returns an object rather than a number."""

    #     class SpecialObject:
    #         """SpecialObject

    #         A special object that conveniently encapsulates the return value of
    #         a special observable supported by a special device and which supports
    #         multiplication with scalars and addition.
    #         """

    #         def __init__(self, val):
    #             self.val = val

    #         def __mul__(self, other):
    #             return SpecialObject(self.val * other)

    #         def __add__(self, other):
    #             newval = self.val + other.val if isinstance(other, self.__class__) else other
    #             return SpecialObject(newval)

    #     class SpecialObservable(Observable):
    #         """SpecialObservable"""

    #         num_wires = AnyWires

    #         def diagonalizing_gates(self):
    #             """Diagonalizing gates"""
    #             return []

    #     class DeviceSupporingSpecialObservable(DefaultQubit):
    #         name = "Device supporting SpecialObservable"
    #         short_name = "default.qubit.specialobservable"
    #         observables = DefaultQubit.observables.union({"SpecialObservable"})

    #         @staticmethod
    #         def _asarray(arr, dtype=None):
    #             return arr

    #         def init(self, *args, **kwargs):
    #             super().__init__(*args, **kwargs)
    #             self.R_DTYPE = SpecialObservable

    #         def expval(self, observable, **kwargs):
    #             if self.analytic and isinstance(observable, SpecialObservable):
    #                 val = super().expval(qml.PauliZ(wires=0), **kwargs)
    #                 return SpecialObject(val)

    #             return super().expval(observable, **kwargs)

    #     dev = DeviceSupporingSpecialObservable(wires=1, shots=None)

    #     @qml.qnode(dev, diff_method="parameter-shift")
    #     def qnode(x):
    #         qml.RY(x, wires=0)
    #         return qml.expval(SpecialObservable(wires=0))

    #     @qml.qnode(dev, diff_method="parameter-shift")
    #     def reference_qnode(x):
    #         qml.RY(x, wires=0)
    #         return qml.expval(qml.PauliZ(wires=0))

    #     par = np.array(0.2, requires_grad=True)
    #     assert np.isclose(qnode(par).item().val, reference_qnode(par))
    #     assert np.isclose(qml.jacobian(qnode)(par).item().val, qml.jacobian(reference_qnode)(par))

    def test_multi_measure_no_warning(self):
        """Test computing the gradient of a tape that contains multiple
        measurements omits no warnings."""
        dev = qml.device("default.qubit", wires=4)

        par1 = qml.numpy.array(0.3)
        par2 = qml.numpy.array(0.1)

        with qml.tape.QuantumTape() as tape:
            qml.RY(par1, wires=0)
            qml.RX(par2, wires=1)
            qml.probs(wires=[1, 2])
            qml.expval(qml.PauliZ(0))

        with pytest.warns(None) as record:
            tapes, fn = qml.gradients.param_shift(tape)
            fn(dev.batch_execute_new(tapes))

        assert len(record) == 0


class TestParameterShiftRuleBroadcast:
    """Tests for the parameter shift implementation using broadcasting"""

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift])
    def test_pauli_rotation_gradient(self, mocker, G, theta, shift, tol):
        """Tests that the automatic gradients of Pauli rotations are correct with broadcasting."""
        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        dev = qml.device("default.qubit", wires=1)

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)], broadcast=True)
        assert len(tapes) == 1
        assert tapes[0].batch_size == 2

        autograd_val = fn(dev.batch_execute_new(tapes))

        tape_fwd, tape_bwd = tape.copy(copy_operations=True), tape.copy(copy_operations=True)
        tape_fwd.set_parameters([theta + np.pi / 2])
        tape_bwd.set_parameters([theta - np.pi / 2])

        manualgrad_val = np.subtract(*dev.batch_execute_new([tape_fwd, tape_bwd])) / 2
        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)

        assert spy.call_args[1]["shifts"] == (shift,)

        # TODO: finite diff update
        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute_new(tapes))
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    def test_Rot_gradient(self, mocker, theta, shift, tol):
        """Tests that the automatic gradient of an arbitrary Euler-angle-parameterized gate is correct."""
        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        dev = qml.device("default.qubit", wires=1)
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)] * 3, broadcast=True)
        assert len(tapes) == len(tape.trainable_params)
        assert [t.batch_size for t in tapes] == [2, 2, 2]

        autograd_val = fn(dev.batch_execute_new(tapes))
        manualgrad_val = np.zeros_like(autograd_val)

        for idx in list(np.ndindex(*params.shape)):
            s = np.zeros_like(params)
            s[idx] += np.pi / 2

            tape.set_parameters(params + s)
            forward = dev.execute_new(tape)

            tape.set_parameters(params - s)
            backward = dev.execute_new(tape)

            manualgrad_val[idx] = (forward - backward) / 2

        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)
        assert spy.call_args[1]["shifts"] == (shift,)

        # compare to finite differences
        # TODO: update when finite diff available
        # tapes, fn = qml.gradients.finite_diff(tape)
        # numeric_val = fn(dev.batch_execute_new(tapes))
        # assert len(autograd_val) == len(numeric_val[0])
        # for a, n in zip(autograd_val, numeric_val):
        #     assert np.allclose(a, n, atol=tol, rtol=0)

    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, tol):
        """Test gradient of controlled rotation gates"""
        dev = qml.device("default.qubit", wires=2)
        b = 0.123

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(b, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {1}

        res = dev.execute_new(tape)
        assert np.allclose(res, -np.cos(b / 2), atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        grad = fn(dev.batch_execute_new(tapes))
        expected = np.sin(b / 2) / 2
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        # TODO: finite diff update
        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute_new(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parameterized
        gate is correct."""
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

        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        assert len(tapes) == len(tape.trainable_params)
        assert [t.batch_size for t in tapes] == [4, 4, 4]

        grad = fn(dev.batch_execute_new(tapes))
        expected = np.array(
            [
                0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                0.5 * np.sin(b / 2) * np.cos(0.5 * (a + c)),
                0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
            ]
        )
        assert len(grad) == len(expected)
        for g, e in zip(grad, expected):
            assert np.allclose(g, e, atol=tol, rtol=0)

        # compare to finite differences
        # TODO: update when finite diff available
        # tapes, fn = qml.gradients.finite_diff(tape)
        # numeric_val = fn(dev.batch_execute_new(tapes))
        # assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    def test_gradients_agree_finite_differences(self, tol):
        """Tests that the parameter-shift rule agrees with the first and second
        order finite differences"""
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

        # TODO: finite diff update
        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1)
        grad_F2 = grad_fn(
            tape, dev, fn=qml.gradients.finite_diff, approx_order=2, strategy="center"
        )
        grad_A = grad_fn(tape, dev)

        # gradients computed with different methods must agree
        assert np.allclose(grad_A, grad_F1, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F2, atol=tol, rtol=0)

    # TODO: update when finite diff available
    @pytest.mark.xfail
    def test_variance_gradients_agree_finite_differences(self, tol):
        """Tests that the variance parameter-shift rule agrees with the first and second
        order finite differences"""
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

        # TODO: finite diff update
        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1)
        grad_F2 = grad_fn(
            tape, dev, fn=qml.gradients.finite_diff, approx_order=2, strategy="center"
        )
        grad_A = grad_fn(tape, dev)

        # gradients computed with different methods must agree
        assert np.allclose(grad_A, grad_F1, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F2, atol=tol, rtol=0)

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
            with qml.tape.QuantumTape() as tape:
                qml.RX(params[0], wires=[0])
                RY(params[1], wires=[1])  # Use finite differences for this op
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.var(qml.PauliX(1))

            # TODO: finite diff update
            tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff, broadcast=True)
            assert len(tapes) == 4

            # check that the fallback method was called for the specified argnums
            spy.assert_called()
            assert spy.call_args[1]["argnum"] == {1}

            return fn(dev.batch_execute_new(tapes))

        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            res = cost_fn(params)
        """
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # double check the derivative
        jac = qml.jacobian(cost_fn)(params)
        assert np.allclose(jac[0, 0, 0], -np.cos(x), atol=tol, rtol=0)
        assert np.allclose(jac[1, 1, 1], -2 * np.cos(2 * y), atol=tol, rtol=0)
        """

    @pytest.mark.autograd
    def test_all_fallback(self, mocker, tol):
        """Test that *only* the fallback logic is called if no parameters
        support the parameter-shift rule"""
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

        # TODO: finite diff update
        tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff, broadcast=True)
        assert len(tapes) == 1 + 2

        # check that the fallback method was called for all argnums
        spy_fd.assert_called()
        spy_ps.assert_not_called()

        res = fn(dev.batch_execute_new(tapes))
        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value(self, tol):
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

        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == 2

        res = fn(dev.batch_execute_new(tapes))
        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        assert len(res) == len(expected)
        for r, e in zip(res, expected):
            assert np.allclose(r, e, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol):
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

        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        """
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == 2

        res = fn(dev.batch_execute_new(tapes))
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)
        """

    def test_var_expectation_values(self, tol):
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

        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        """
        assert len(tapes) == 3  # One unshifted, two broadcasted shifted tapes
        assert tapes[0].batch_size is None
        assert tapes[1].batch_size == tapes[2].batch_size == 2

        res = fn(dev.batch_execute_new(tapes))
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)
        """

    def test_prob_expectation_values(self, tol):
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

        dev.execute_new(tape)

        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        """
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == 2

        res = fn(dev.batch_execute_new(tapes))
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
        """

    def test_involutory_variance(self, tol):
        """Tests qubit observables that are involutory"""
        dev = qml.device("default.qubit", wires=1)
        a = 0.54

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.PauliZ(0))

        res = dev.execute_new(tape)
        expected = 1 - np.cos(a) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        gradA = fn(dev.batch_execute_new(tapes))
        assert len(tapes) == 2
        assert tapes[0].batch_size is None
        assert tapes[1].batch_size == 2

        # TODO: finite diff update
        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute_new(tapes))
        assert len(tapes) == 2

        expected = 2 * np.sin(a) * np.cos(a)

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_non_involutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory"""
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
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        gradA = fn(dev.batch_execute_new(tapes))
        assert len(tapes) == 1 + 2 * 1
        assert tapes[0].batch_size is None
        assert tapes[1].batch_size == tapes[2].batch_size == 2

        # TODO: finite diff update
        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute_new(tapes))
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
        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        """
        gradA = fn(dev.batch_execute_new(tapes))
        assert len(tapes) == 1 + 2 * 4

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute_new(tapes))
        assert len(tapes) == 1 + 2

        expected = [2 * np.sin(a) * np.cos(a), -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]
        assert np.diag(gradA) == pytest.approx(expected, abs=tol)
        assert np.diag(gradF) == pytest.approx(expected, abs=tol)
        """

    def test_expval_and_variance(self, tol):
        """Test that the qnode works for a combination of expectation
        values and variances"""
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
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        """
        gradA = fn(dev.batch_execute_new(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute_new(tapes))

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
        """

    def test_projector_variance(self, tol):
        """Test that the variance of a projector is correctly returned"""
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

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        gradA = fn(dev.batch_execute_new(tapes))

        # TODO: finite diff update
        # tapes, fn = qml.gradients.finite_diff(tape)
        # gradF = fn(dev.batch_execute_new(tapes))

        expected = np.array(
            [
                0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
            ]
        )
        assert len(gradA) == len(expected)
        for a, e in zip(gradA, expected):
            assert np.allclose(a, e, atol=tol, rtol=0)
        # TODO: finite diff update
        # assert gradF == pytest.approx(expected, abs=tol)

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

        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            transform = [
                qml.math.shape(qml.gradients.param_shift(c, broadcast=True)(x)) for c in circuits
            ]


@pytest.mark.parametrize(
    "broadcast, expected", [(False, (5, [None] * 5)), (True, (3, [None, 2, 2]))]
)
class TestParamShiftGradients:
    """Test that the transform is differentiable"""

    @pytest.mark.autograd
    # TODO: support Hessian with the new return types
    @pytest.mark.skip
    def test_autograd(self, tol, broadcast, expected):
        """Tests that the output of the parameter-shift transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)
        exp_num_tapes, exp_batch_sizes = expected

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
            assert len(tapes) == exp_num_tapes
            assert [t.batch_size for t in tapes] == exp_batch_sizes
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


@pytest.mark.parametrize("broadcast", [True, False])
class TestHamiltonianExpvalGradients:
    """Test that tapes ending with expval(H) can be
    differentiated"""

    def test_not_expval_error(self, broadcast):
        """Test that if the variance of the Hamiltonian is requested,
        an error is raised"""
        dev = qml.device("default.qubit", wires=2)

        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3])
        H = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])

        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(H)

        tape.trainable_params = {2, 3, 4}

        with pytest.raises(ValueError, match="for expectations, not var"):
            qml.gradients.param_shift(tape, broadcast=broadcast)

    def test_no_trainable_coeffs(self, mocker, tol, broadcast):
        """Test no trainable Hamiltonian coefficients"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.gradients, "hamiltonian_grad")

        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3])
        H = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])

        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        a, b, c = coeffs
        x, y = weights
        tape.trainable_params = {0, 1}

        res = dev.batch_execute_new([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        # two (broadcasted if broadcast=True) shifts per rotation gate
        assert len(tapes) == (2 if broadcast else 2 * 2)
        assert [t.batch_size for t in tapes] == ([2, 2] if broadcast else [None] * 4)
        spy.assert_not_called()

        res = fn(dev.batch_execute_new(tapes))
        assert isinstance(res, tuple)

        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        expected = [
            -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
            b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
        ]
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_trainable_coeffs(self, mocker, tol, broadcast):
        """Test trainable Hamiltonian coefficients"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.gradients, "hamiltonian_grad")

        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3])
        H = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])

        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        a, b, c = coeffs
        x, y = weights
        tape.trainable_params = {0, 1, 2, 4}

        res = dev.batch_execute_new([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        # two (broadcasted if broadcast=True) shifts per rotation gate
        # one circuit per trainable H term
        assert len(tapes) == (2 + 2 if broadcast else 2 * 2 + 2)
        assert [t.batch_size for t in tapes] == ([2, 2, None, None] if broadcast else [None] * 6)
        spy.assert_called()

        res = fn(dev.batch_execute_new(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 4
        assert res[0].shape == ()
        assert res[1].shape == ()
        assert res[2].shape == ()
        assert res[3].shape == ()

        expected = [
            -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
            b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
            np.cos(x),
            -(np.sin(x) * np.sin(y)),
        ]
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)
        assert np.allclose(res[2], expected[2], atol=tol, rtol=0)
        assert np.allclose(res[3], expected[3], atol=tol, rtol=0)

    def test_multiple_hamiltonians(self, mocker, tol, broadcast):
        """Test multiple trainable Hamiltonian coefficients"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.gradients, "hamiltonian_grad")

        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3])
        a, b, c = coeffs
        H1 = qml.Hamiltonian(coeffs, obs)

        obs = [qml.PauliZ(0)]
        coeffs = np.array([0.7])
        d = coeffs[0]
        H2 = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])
        x, y = weights

        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H1)
            qml.expval(H2)

        tape.trainable_params = {0, 1, 2, 4, 5}

        res = dev.batch_execute_new([tape])
        expected = [-c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y)), d * np.cos(x)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        if broadcast:
            with pytest.raises(
                NotImplementedError, match="Broadcasting with multiple measurements"
            ):
                tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
            return
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        # two shifts per rotation gate, one circuit per trainable H term
        assert len(tapes) == 2 * 2 + 3
        spy.assert_called()

        res = fn(dev.batch_execute_new(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert len(res[0]) == 5
        assert len(res[1]) == 5

        expected = [
            [
                -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
                b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
                np.cos(x),
                -(np.sin(x) * np.sin(y)),
                0,
            ],
            [-d * np.sin(x), 0, 0, 0, np.cos(x)],
        ]

        assert np.allclose(np.stack(res), expected, atol=tol, rtol=0)

    @staticmethod
    def cost_fn(weights, coeffs1, coeffs2, dev=None, broadcast=False):
        """Cost function for gradient tests"""
        obs1 = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        H1 = qml.Hamiltonian(coeffs1, obs1)

        obs2 = [qml.PauliZ(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        with qml.tape.QuantumTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H1)
            qml.expval(H2)

        tape.trainable_params = {0, 1, 2, 3, 4, 5}
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        jac = fn(dev.batch_execute_new(tapes))
        return jac

    @staticmethod
    def cost_fn_expected(weights, coeffs1, coeffs2):
        """Analytic jacobian of cost_fn above"""
        a, b, c = coeffs1
        d = coeffs2[0]
        x, y = weights
        return [
            [
                -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
                b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
                np.cos(x),
                np.cos(x) * np.sin(y),
                -(np.sin(x) * np.sin(y)),
                0,
            ],
            [-d * np.sin(x), 0, 0, 0, 0, np.cos(x)],
        ]

    @pytest.mark.autograd
    def test_autograd(self, tol, broadcast):
        """Test gradient of multiple trainable Hamiltonian coefficients
        using autograd"""
        coeffs1 = np.array([0.1, 0.2, 0.3], requires_grad=True)
        coeffs2 = np.array([0.7], requires_grad=True)
        weights = np.array([0.4, 0.5], requires_grad=True)
        dev = qml.device("default.qubit.autograd", wires=2)

        if broadcast:
            with pytest.raises(
                NotImplementedError, match="Broadcasting with multiple measurements"
            ):
                res = self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)
            return
        res = self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, np.array(expected), atol=tol, rtol=0)

        # TODO: test when Hessians are supported with the new return types
        # second derivative wrt to Hamiltonian coefficients should be zero
        # ---
        # res = qml.jacobian(self.cost_fn)(weights, coeffs1, coeffs2, dev=dev)
        # assert np.allclose(res[1][:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
        # assert np.allclose(res[2][:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol, broadcast):
        """Test gradient of multiple trainable Hamiltonian coefficients
        using tf"""
        import tensorflow as tf

        coeffs1 = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)
        coeffs2 = tf.Variable([0.7], dtype=tf.float64)
        weights = tf.Variable([0.4, 0.5], dtype=tf.float64)

        dev = qml.device("default.qubit.tf", wires=2)

        if broadcast:
            with pytest.raises(
                NotImplementedError, match="Broadcasting with multiple measurements"
            ):
                with tf.GradientTape() as t:
                    jac = self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)
            return
        with tf.GradientTape() as t:
            jac = self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)

        expected = self.cost_fn_expected(weights.numpy(), coeffs1.numpy(), coeffs2.numpy())
        assert np.allclose(jac[0], np.array(expected)[0], atol=tol, rtol=0)
        assert np.allclose(jac[1], np.array(expected)[1], atol=tol, rtol=0)

        # TODO: test when Hessians are supported with the new return types
        # second derivative wrt to Hamiltonian coefficients should be zero
        # ---
        # hess = t.jacobian(jac, [coeffs1, coeffs2])
        # assert np.allclose(hess[0][:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
        # assert np.allclose(hess[1][:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)

    # TODO: Torch support for param-shift
    @pytest.mark.torch
    @pytest.mark.xfail
    def test_torch(self, tol, broadcast):
        """Test gradient of multiple trainable Hamiltonian coefficients
        using torch"""
        import torch

        coeffs1 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
        coeffs2 = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([0.4, 0.5], dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.qubit.torch", wires=2)

        if broadcast:
            with pytest.raises(
                NotImplementedError, match="Broadcasting with multiple measurements"
            ):
                res = self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)
            return
        res = self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)
        expected = self.cost_fn_expected(
            weights.detach().numpy(), coeffs1.detach().numpy(), coeffs2.detach().numpy()
        )
        assert np.allclose(res.detach(), expected, atol=tol, rtol=0)

        # second derivative wrt to Hamiltonian coefficients should be zero
        hess = torch.autograd.functional.jacobian(
            lambda *args: self.cost_fn(*args, dev, broadcast), (weights, coeffs1, coeffs2)
        )
        assert np.allclose(hess[1][:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
        assert np.allclose(hess[2][:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol, broadcast):
        """Test gradient of multiple trainable Hamiltonian coefficients
        using JAX"""
        import jax

        jnp = jax.numpy

        coeffs1 = jnp.array([0.1, 0.2, 0.3])
        coeffs2 = jnp.array([0.7])
        weights = jnp.array([0.4, 0.5])
        dev = qml.device("default.qubit.jax", wires=2)

        if broadcast:
            with pytest.raises(
                NotImplementedError, match="Broadcasting with multiple measurements"
            ):
                res = self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)
            return
        res = self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, np.array(expected), atol=tol, rtol=0)

        # TODO: test when Hessians are supported with the new return types
        # second derivative wrt to Hamiltonian coefficients should be zero
        # ---
        # second derivative wrt to Hamiltonian coefficients should be zero
        # res = jax.jacobian(self.cost_fn, argnums=1)(weights, coeffs1, coeffs2, dev, broadcast)
        # assert np.allclose(res[:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)

        # res = jax.jacobian(self.cost_fn, argnums=1)(weights, coeffs1, coeffs2, dev, broadcast)
        # assert np.allclose(res[:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)

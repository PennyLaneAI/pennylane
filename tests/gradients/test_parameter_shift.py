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
"""Tests for the gradients.parameter_shift module."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift
from pennylane.gradients.parameter_shift import _gradient_analysis


class TestGradAnalysis:
    """Tests for parameter gradient methods"""

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is
        correctly marked"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.tape.JacobianTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        _gradient_analysis(tape)

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "A"
        assert tape._par_info[2]["grad_method"] == "A"

    def test_analysis_caching(self, mocker):
        """Test that the gradient analysis is only executed once per tape"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.tape.JacobianTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        spy = mocker.spy(qml.operation, "has_grad_method")
        _gradient_analysis(tape)
        spy.assert_called()

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "A"
        assert tape._par_info[2]["grad_method"] == "A"

        spy = mocker.spy(qml.operation, "has_grad_method")
        _gradient_analysis(tape)
        spy.assert_not_called()

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliY(0))

        _gradient_analysis(tape)

        assert tape._par_info[0]["grad_method"] == "A"
        assert tape._par_info[1]["grad_method"] == "0"

    def test_independent_no_graph_mode(self):
        """In non-graph mode, it is impossible to determine
        if a parameter is independent or not"""

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliY(0))

        _gradient_analysis(tape, use_graph=False)

        assert tape._par_info[0]["grad_method"] == "A"
        assert tape._par_info[1]["grad_method"] == "A"

    def test_finite_diff(self, monkeypatch):
        """If an op has grad_method=F, this should be respected"""
        monkeypatch.setattr(qml.RX, "grad_method", "F")

        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.tape.JacobianTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        _gradient_analysis(tape)

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "F"
        assert tape._par_info[2]["grad_method"] == "A"


def grad_fn(tape, dev, fn=qml.gradients.param_shift, **kwargs):
    """Utility function to automate execution and processing of gradient tapes"""
    tapes, fn = fn(tape, **kwargs)
    return fn(dev.batch_execute(tapes))


class TestShiftedTapes:
    """Tests for the generation of shifted tapes"""

    def test_behaviour(self):
        """Test that the function behaves as expected"""

        with qml.tape.JacobianTape() as tape:
            qml.PauliZ(0)
            qml.RX(1.0, wires=0)
            qml.CNOT(wires=[0, 2])
            qml.Rot(2.0, 3.0, 4.0, wires=0)
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 2}
        gradient_recipes = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        tapes, _ = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)

        assert len(tapes) == 5
        assert tapes[0].get_parameters(trainable_only=False) == [0.2 * 1.0 + 0.3, 2.0, 3.0, 4.0]
        assert tapes[1].get_parameters(trainable_only=False) == [0.5 * 1.0 + 0.6, 2.0, 3.0, 4.0]
        assert tapes[2].get_parameters(trainable_only=False) == [1.0, 2.0, 1 * 3.0 + 1, 4.0]
        assert tapes[3].get_parameters(trainable_only=False) == [1.0, 2.0, 2 * 3.0 + 2, 4.0]
        assert tapes[4].get_parameters(trainable_only=False) == [1.0, 2.0, 3 * 3.0 + 3, 4.0]


class TestParamShift:
    """Unit tests for the param_shift function"""

    def test_empty_circuit(self):
        """Test that an empty circuit works correctly"""
        with qml.tape.JacobianTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            tapes, _ = qml.gradients.param_shift(tape)
        assert not tapes

    def test_all_parameters_independent(self):
        """Test that a circuit where all parameters do not affect the output"""
        with qml.tape.JacobianTape() as tape:
            qml.RX(0.4, wires=0)
            qml.expval(qml.PauliZ(1))

        tapes, _ = qml.gradients.param_shift(tape)
        assert not tapes

    def test_state_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a state"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.state()

        with pytest.raises(ValueError, match=r"return the state is not supported"):
            qml.gradients.param_shift(tape)

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 2

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, [0])

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
            res = qml.gradients.param_shift(circuit)(weights)

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
            g_tapes, post_processing = qml.gradients.param_shift(tape)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
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

        result = qml.gradients.param_shift(circuit)(params)
        assert np.allclose(result, np.zeros((4, 3)), atol=0, rtol=0)

        tapes, _ = qml.gradients.param_shift(circuit.tape)
        assert tapes == []

    def test_y0(self):
        """Test that if the gradient recipe has a zero-shift component, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        gradient_recipes = [[[-1e7, 1, 0], [1e7, 1, 1e7]], [[-1e7, 1, 0], [1e7, 1, 1e7]]]
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)

        # one tape per parameter, plus one global call
        assert len(tapes) == tape.num_params + 1

    def test_y0_provided(self):
        """Test that if the original tape output is provided, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        gradient_recipes = [[[-1e7, 1, 0], [1e7, 1, 1e7]], [[-1e7, 1, 0], [1e7, 1, 1e7]]]
        f0 = dev.execute(tape)
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes, f0=f0)

        # one tape per parameter, plus one global call
        assert len(tapes) == tape.num_params

        fn(dev.batch_execute(tapes))

    def test_independent_parameters_analytic(self):
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

        tapes, fn = qml.gradients.param_shift(tape1)
        j1 = fn(dev.batch_execute(tapes))

        # We should only be executing the device to differentiate 1 parameter (2 executions)
        assert dev.num_executions == 2

        tapes, fn = qml.gradients.param_shift(tape2)
        j2 = fn(dev.batch_execute(tapes))

        exp = -np.sin(1)

        assert np.allclose(j1, [exp, 0])
        assert np.allclose(j2, [0, exp])

    def test_grad_recipe_parameter_dependent(self, monkeypatch):
        """Test that an operation with a gradient recipe that depends on
        its instantiated parameter values works correctly within the parameter
        shift rule. Also tests that grad_recipes supersedes paramter_frequencies.
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

        with qml.tape.JacobianTape() as tape:
            RX(x, wires=0)
            qml.expval(qml.PauliZ(0))

        tapes, fn = qml.gradients.param_shift(tape)

        assert len(tapes) == 2
        assert tapes[0].operations[0].data[0] == 0
        assert tapes[1].operations[0].data[0] == 2 * x

        grad = fn(dev.batch_execute(tapes))
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
            with qml.tape.JacobianTape() as tape:
                op(x, wires=0)
                qml.expval(qml.PauliZ(0))

            with pytest.raises(
                qml.operation.OperatorPropertyUndefined, match="does not have a grad_recipe"
            ):
                qml.gradients.param_shift(tape)


class TestParameterShiftRule:
    """Tests for the parameter shift implementation"""

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift])
    def test_pauli_rotation_gradient(self, mocker, G, theta, shift, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        dev = qml.device("default.qubit", wires=1)

        with qml.tape.JacobianTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)])
        assert len(tapes) == 2

        autograd_val = fn(dev.batch_execute(tapes))
        manualgrad_val = (
            tape.execute(dev, params=[theta + np.pi / 2])
            - tape.execute(dev, params=[theta - np.pi / 2])
        ) / 2
        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)

        assert spy.call_args[1]["shifts"] == (shift,)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    def test_Rot_gradient(self, mocker, theta, shift, tol):
        """Tests that the automatic gradient of an arbitrary Euler-angle-parameterized gate is correct."""
        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        dev = qml.device("default.qubit", wires=1)
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.JacobianTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)] * 3)
        assert len(tapes) == 2 * len(tape.trainable_params)

        autograd_val = fn(dev.batch_execute(tapes))
        manualgrad_val = np.zeros_like(autograd_val)

        for idx in list(np.ndindex(*params.shape)):
            s = np.zeros_like(params)
            s[idx] += np.pi / 2

            forward = tape.execute(dev, params=params + s)
            backward = tape.execute(dev, params=params - s)

            manualgrad_val[0, idx] = (forward - backward) / 2

        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)
        assert spy.call_args[1]["shifts"] == (shift,)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, tol):
        """Test gradient of controlled rotation gates"""
        dev = qml.device("default.qubit", wires=2)
        b = 0.123

        with qml.tape.JacobianTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(b, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {1}

        res = tape.execute(dev)
        assert np.allclose(res, -np.cos(b / 2), atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape)
        grad = fn(dev.batch_execute(tapes))
        expected = np.sin(b / 2) / 2
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parameterized
        gate is correct."""
        dev = qml.device("default.qubit", wires=2)
        a, b, c = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.tape.JacobianTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.CRot(a, b, c, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {1, 2, 3}

        res = tape.execute(dev)
        expected = -np.cos(b / 2) * np.cos(0.5 * (a + c))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4 * len(tape.trainable_params)

        grad = fn(dev.batch_execute(tapes))
        expected = np.array(
            [
                [
                    0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                    0.5 * np.sin(b / 2) * np.cos(0.5 * (a + c)),
                    0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                ]
            ]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    def test_gradients_agree_finite_differences(self, tol):
        """Tests that the parameter-shift rule agrees with the first and second
        order finite differences"""
        params = np.array([0.1, -1.6, np.pi / 5])

        with qml.tape.JacobianTape() as tape:
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

    def test_variance_gradients_agree_finite_differences(self, tol):
        """Tests that the variance parameter-shift rule agrees with the first and second
        order finite differences"""
        params = np.array([0.1, -1.6, np.pi / 5])

        with qml.tape.JacobianTape() as tape:
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
            with qml.tape.JacobianTape() as tape:
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

            return fn(dev.batch_execute(tapes))

        res = cost_fn(params)
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # double check the derivative
        jac = qml.jacobian(cost_fn)(params)
        assert np.allclose(jac[0, 0, 0], -np.cos(x), atol=tol, rtol=0)
        assert np.allclose(jac[1, 1, 1], -2 * np.cos(2 * y), atol=tol, rtol=0)

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

        with qml.tape.JacobianTape() as tape:
            RX(x, wires=[0])
            RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff)
        assert len(tapes) == 1 + 2

        # check that the fallback method was called for all argnums
        spy_fd.assert_called()
        spy_ps.assert_not_called()

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_single_expectation_value(self, tol):
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

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

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

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

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

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 5

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

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

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

        with qml.tape.JacobianTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.PauliZ(0))

        res = tape.execute(dev)
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

        with qml.tape.JacobianTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.Hermitian(A, 0))

        tape.trainable_params = {0}

        res = tape.execute(dev)
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

        with qml.tape.JacobianTape() as tape:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        tape.trainable_params = {0, 1}

        res = tape.execute(dev)
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

        with qml.tape.JacobianTape() as tape:
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

    def test_projector_variance(self, tol):
        """Test that the variance of a projector is correctly returned"""
        dev = qml.device("default.qubit", wires=2)
        P = np.array([1])
        x, y = 0.765, -0.654

        with qml.tape.JacobianTape() as tape:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.Projector(P, wires=0) @ qml.PauliX(1))

        tape.trainable_params = {0, 1}

        res = tape.execute(dev)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # # circuit jacobians
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


class TestParamShiftGradients:
    """Test that the transform is differentiable"""

    def test_autograd(self, tol):
        """Tests that the output of the parameter-shift transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x):
            with qml.tape.JacobianTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.param_shift(tape)
            jac = fn(dev.batch_execute(tapes))
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

    @pytest.mark.slow
    def test_tf(self, tol):
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
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.param_shift(tape)
            jac = fn(dev.batch_execute(tapes))

        x, y = 1.0 * params

        expected = np.array([np.sin(2 * x) * np.sin(y) ** 2, -np.cos(x) ** 2 * np.sin(2 * y)])
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        res = t.jacobian(jac, params)
        expected = np.array(
            [
                [2 * np.cos(2 * x) * np.sin(y) ** 2, np.sin(2 * x) * np.sin(2 * y)],
                [np.sin(2 * x) * np.sin(2 * y), -2 * np.cos(x) ** 2 * np.cos(2 * y)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_torch(self, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using Torch, yielding second derivatives."""
        torch = pytest.importorskip("torch")

        dev = qml.device("default.qubit.torch", wires=2)
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)

        with qml.tape.JacobianTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        tapes, fn = qml.gradients.param_shift(tape)
        jac = fn(dev.batch_execute(tapes))
        cost = jac[0, 0]
        cost.backward()
        hess = params.grad

        x, y = params.detach().numpy()

        expected = np.array([np.sin(2 * x) * np.sin(y) ** 2, -np.cos(x) ** 2 * np.sin(2 * y)])
        assert np.allclose(jac.detach().numpy(), expected, atol=tol, rtol=0)

        expected = np.array([2 * np.cos(2 * x) * np.sin(y) ** 2, np.sin(2 * x) * np.sin(2 * y)])
        assert np.allclose(hess.detach().numpy(), expected, atol=0.1, rtol=0)

    def test_jax(self, tol):
        """Tests that the output of the finite-difference transform
        can be differentiated using JAX, yielding second derivatives."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp
        from jax.config import config

        config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit.jax", wires=2)
        params = jnp.array([0.543, -0.654])

        def cost_fn(x):
            with qml.tape.JacobianTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.param_shift(tape)
            jac = fn(dev.batch_execute(tapes))
            return jac

        res = jax.jacobian(cost_fn)(params)
        x, y = params
        expected = np.array(
            [
                [2 * np.cos(2 * x) * np.sin(y) ** 2, np.sin(2 * x) * np.sin(2 * y)],
                [np.sin(2 * x) * np.sin(2 * y), -2 * np.cos(x) ** 2 * np.cos(2 * y)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestHamiltonianExpvalGradients:
    """Test that tapes ending with expval(H) can be
    differentiated"""

    def test_not_expval_error(self):
        """Test that if the variance of the Hamiltonian is requested,
        an error is raised"""
        dev = qml.device("default.qubit", wires=2)

        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3])
        H = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])

        with qml.tape.JacobianTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(H)

        tape.trainable_params = {2, 3, 4}

        with pytest.raises(ValueError, match="for expectations, not var"):
            qml.gradients.param_shift(tape)

    def test_no_trainable_coeffs(self, mocker, tol):
        """Test no trainable Hamiltonian coefficients"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.gradients, "hamiltonian_grad")

        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3])
        H = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])

        with qml.tape.JacobianTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        a, b, c = coeffs
        x, y = weights
        tape.trainable_params = {0, 1}

        res = dev.batch_execute([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape)
        # two shifts per rotation gate, one circuit per trainable H term
        assert len(tapes) == 2 * 2
        spy.assert_not_called()

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 2)

        expected = [
            -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
            b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_trainable_coeffs(self, mocker, tol):
        """Test trainable Hamiltonian coefficients"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.gradients, "hamiltonian_grad")

        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3])
        H = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])

        with qml.tape.JacobianTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        a, b, c = coeffs
        x, y = weights
        tape.trainable_params = {0, 1, 2, 4}

        res = dev.batch_execute([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape)
        # two shifts per rotation gate, one circuit per trainable H term
        assert len(tapes) == 2 * 2 + 2
        spy.assert_called()

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (1, 4)

        expected = [
            -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
            b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
            np.cos(x),
            -(np.sin(x) * np.sin(y)),
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_hamiltonians(self, mocker, tol):
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

        with qml.tape.JacobianTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H1)
            qml.expval(H2)

        tape.trainable_params = {0, 1, 2, 4, 5}

        res = dev.batch_execute([tape])
        expected = [-c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y)), d * np.cos(x)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape)
        # two shifts per rotation gate, one circuit per trainable H term
        assert len(tapes) == 2 * 2 + 3
        spy.assert_called()

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2, 5)

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

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @staticmethod
    def cost_fn(weights, coeffs1, coeffs2, dev=None):
        """Cost function for gradient tests"""
        obs1 = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        H1 = qml.Hamiltonian(coeffs1, obs1)

        obs2 = [qml.PauliZ(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        with qml.tape.JacobianTape() as tape:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H1)
            qml.expval(H2)

        tape.trainable_params = {0, 1, 2, 3, 4, 5}
        tapes, fn = qml.gradients.param_shift(tape)
        jac = fn(dev.batch_execute(tapes))
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

    def test_autograd(self, tol):
        """Test gradient of multiple trainable Hamiltonian coefficients
        using autograd"""
        coeffs1 = np.array([0.1, 0.2, 0.3], requires_grad=True)
        coeffs2 = np.array([0.7], requires_grad=True)
        weights = np.array([0.4, 0.5], requires_grad=True)
        dev = qml.device("default.qubit.autograd", wires=2)

        res = self.cost_fn(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # second derivative wrt to Hamiltonian coefficients should be zero
        res = qml.jacobian(self.cost_fn)(weights, coeffs1, coeffs2, dev=dev)
        assert np.allclose(res[1][:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
        assert np.allclose(res[2][:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)

    @pytest.mark.slow
    def test_tf(self, tol):
        """Test gradient of multiple trainable Hamiltonian coefficients
        using tf"""
        tf = pytest.importorskip("tensorflow")

        coeffs1 = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)
        coeffs2 = tf.Variable([0.7], dtype=tf.float64)
        weights = tf.Variable([0.4, 0.5], dtype=tf.float64)

        dev = qml.device("default.qubit.tf", wires=2)

        with tf.GradientTape() as t:
            jac = self.cost_fn(weights, coeffs1, coeffs2, dev=dev)

        expected = self.cost_fn_expected(weights.numpy(), coeffs1.numpy(), coeffs2.numpy())
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        # second derivative wrt to Hamiltonian coefficients should be zero
        hess = t.jacobian(jac, [coeffs1, coeffs2])
        assert np.allclose(hess[0][:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
        assert np.allclose(hess[1][:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)

    def test_torch(self, tol):
        """Test gradient of multiple trainable Hamiltonian coefficients
        using torch"""
        torch = pytest.importorskip("torch")

        coeffs1 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
        coeffs2 = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([0.4, 0.5], dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.qubit.torch", wires=2)

        res = self.cost_fn(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_expected(
            weights.detach().numpy(), coeffs1.detach().numpy(), coeffs2.detach().numpy()
        )
        assert np.allclose(res.detach(), expected, atol=tol, rtol=0)

        # second derivative wrt to Hamiltonian coefficients should be zero
        hess = torch.autograd.functional.jacobian(
            lambda *args: self.cost_fn(*args, dev=dev), (weights, coeffs1, coeffs2)
        )
        assert np.allclose(hess[1][:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)
        assert np.allclose(hess[2][:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)

    @pytest.mark.slow
    def test_jax(self, tol):
        """Test gradient of multiple trainable Hamiltonian coefficients
        using JAX"""
        jax = pytest.importorskip("jax")
        jnp = jax.numpy

        coeffs1 = jnp.array([0.1, 0.2, 0.3])
        coeffs2 = jnp.array([0.7])
        weights = jnp.array([0.4, 0.5])
        dev = qml.device("default.qubit.jax", wires=2)

        res = self.cost_fn(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # second derivative wrt to Hamiltonian coefficients should be zero
        res = jax.jacobian(self.cost_fn, argnums=1)(weights, coeffs1, coeffs2, dev=dev)
        assert np.allclose(res[:, 2:5], np.zeros([2, 3, 3]), atol=tol, rtol=0)

        res = jax.jacobian(self.cost_fn, argnums=1)(weights, coeffs1, coeffs2, dev=dev)
        assert np.allclose(res[:, -1], np.zeros([2, 1, 1]), atol=tol, rtol=0)

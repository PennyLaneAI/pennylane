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
# pylint:disable=use-implicit-booleaness-not-comparison,abstract-method
from functools import partial

import pytest
from default_qubit_legacy import DefaultQubitLegacy
from device_shots_to_analytic import shots_to_analytic

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift
from pennylane.measurements import Shots

shot_vec_tol = 10e-3
herm_shot_vec_tol = 0.5
finite_diff_tol = 0.1
h_val = 0.1

default_shot_vector = (1000, 2000, 3000)
many_shots_shot_vector = tuple([1000000] * 3)
fallback_shot_vec = tuple([1000000] * 4)

# Pick 4 angles in the [-2 * np.pi, np.pi] interval
angles = (-6.28318531, -3.92699082, 0.78539816, 3.14159265)


def grad_fn(tape, dev, fn=qml.gradients.param_shift, **kwargs):
    """Utility function to automate execution and processing of gradient tapes"""
    tapes, fn = fn(tape, **kwargs)
    return fn(dev.execute(tapes))


# pylint: disable=too-few-public-methods
class RY_with_F(qml.RY):
    """Custom variant of qml.RY with grad_method "F"."""

    grad_method = "F"


# pylint: disable=too-few-public-methods
class RX_with_F(qml.RX):
    """Custom variant of qml.RX with grad_method "F"."""

    grad_method = "F"


# pylint: disable=too-few-public-methods
class RX_par_dep_recipe(qml.RX):
    """RX operation with a parameter-dependent grad recipe."""

    @property
    def grad_recipe(self):
        """The gradient is given by [f(2x) - f(0)] / (2 sin(x)), by subsituting
        shift = x into the two term parameter-shift rule."""
        x = self.data[0]
        c = 0.5 / np.sin(x)
        return ([[c, 0.0, 2 * x], [-c, 0.0, 0.0]],)


class TestParamShift:
    """Unit tests for the param_shift function"""

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        shot_vec = default_shot_vector
        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == None

        res = fn(dev.execute(tapes))
        for r in res:
            assert isinstance(r, tuple)
            assert len(r) == 2
            assert r[0].shape == ()
            assert r[1].shape == ()

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, [0])

    @pytest.mark.parametrize("broadcast", [True, False])
    def test_no_trainable_params_tape(self, broadcast):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        shot_vec = default_shot_vector
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        # TODO: remove once #2155 is resolved
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        all_res = fn(qml.execute(g_tapes, dev, None))
        assert isinstance(all_res, tuple)
        assert len(all_res) == len(shot_vec)

        assert g_tapes == []
        for res in all_res:
            assert isinstance(res, np.ndarray)
            assert res.shape == (0,)

    def test_no_trainable_params_multiple_return_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters with multiple returns."""
        shot_vec = default_shot_vector
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = []
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.param_shift(tape)
        all_res = post_processing(qml.execute(g_tapes, dev, None))
        assert isinstance(all_res, tuple)
        assert len(all_res) == len(shot_vec)

        assert g_tapes == []
        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == len(tape.measurements)
            for r in res:
                assert isinstance(r, np.ndarray)
                assert r.shape == (0,)

    def test_all_zero_diff_methods_tape(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        shot_vec = default_shot_vector
        dev = qml.device("default.qubit", wires=4)

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(*params, wires=0)
            qml.probs([2, 3])

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        g_tapes, post_processing = qml.gradients.param_shift(tape)
        assert g_tapes == []

        all_res = post_processing(qml.execute(g_tapes, dev, None))
        assert isinstance(all_res, tuple)
        assert len(all_res) == len(shot_vec)

        assert g_tapes == []
        for result in all_res:
            assert isinstance(result, tuple)

            assert len(result) == len(tape.trainable_params)

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
        shot_vec = default_shot_vector
        dev = qml.device("default.qubit", wires=4)

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(*params, wires=0)
            qml.expval(qml.PauliZ(wires=2))
            qml.probs([2, 3])

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        g_tapes, post_processing = qml.gradients.param_shift(tape)
        assert g_tapes == []

        all_result = post_processing(dev.execute(g_tapes))

        assert isinstance(all_result, tuple)

        assert len(all_result) == len(shot_vec)

        # First elem
        for result in all_result:
            assert len(result[0]) == 3

            assert isinstance(result[0][0], np.ndarray)
            assert result[0][0].shape == ()
            assert np.allclose(result[0][0], 0)

            assert isinstance(result[0][1], np.ndarray)
            assert result[0][1].shape == ()
            assert np.allclose(result[0][1], 0)

            assert isinstance(result[0][2], np.ndarray)
            assert result[0][2].shape == ()
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

    @pytest.mark.parametrize("broadcast", [True, False])
    def test_all_zero_diff_methods(self, broadcast):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4)

        @qml.set_shots(default_shot_vector)
        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        tape = qml.workflow.construct_tape(circuit)(params)

        result = qml.gradients.param_shift(circuit)(params)
        assert np.allclose(result, np.zeros((4, 3)), atol=0, rtol=0)

        tapes, _ = qml.gradients.param_shift(tape, broadcast=broadcast)
        assert tapes == []

    @pytest.mark.parametrize("ops_with_custom_recipe", [[0], [1], [0, 1]])
    def test_recycled_unshifted_tape(self, ops_with_custom_recipe):
        """Test that if the gradient recipe has a zero-shift component, then
        the tape is executed only once using the current parameter
        values."""
        shot_vec = (100, 10, 1)
        dev = qml.device("default.qubit", wires=2)
        x = [0.543, -0.654]

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        gradient_recipes = tuple(
            [[-1e3, 1, 0], [1e3, 1, 1e-3]] if i in ops_with_custom_recipe else None
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

        assert isinstance(grad, tuple)
        assert len(grad) == len(shot_vec)
        for shot_comp_grad in grad:
            assert isinstance(grad, tuple)

            # Two trainable params
            assert len(shot_comp_grad) == 2
            for g in shot_comp_grad:
                assert isinstance(g, np.ndarray)
                assert g.shape == ()

        # Due to shot-based stochasticity the analytic values are not checked (multiplier are significant and the
        # slightest of differences can cause major deviations from the analytic values

    @pytest.mark.parametrize("y_wire", [0, 1])
    def test_f0_provided(self, y_wire):
        """Test that if the original tape output is provided, then
        the tape is not executed additionally at the current parameter
        values."""
        shot_vec = default_shot_vector
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=y_wire)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        gradient_recipes = ([[-1e7, 1, 0], [1e7, 1, 1e7]],) * 2
        f0 = dev.execute(tape)
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes, f0=f0)

        # one tape per parameter that impacts the expval
        assert len(tapes) == 2 if y_wire == 0 else 1

        fn(dev.execute(tapes))

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
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=0)
            RX(x[1], wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tapes, fn = qml.gradients.param_shift(tape)

        # Unshifted tapes always are first within the tapes created for one operation;
        # They are not batched together because we trust operation recipes to be condensed already
        expected_shifts = [[0, 0], [s, 0], [-s, 0], [0, s], [0, -s]]
        assert len(tapes) == 5
        for tape, expected in zip(tapes, expected_shifts):
            assert tape.operations[0].data[0] == x[0] + expected[0]
            assert tape.operations[1].data[0] == x[1] + expected[1]

        grad = fn(dev.execute(tapes))
        _expected = np.stack(
            [-np.sin(x[0] + x[1]), -np.sin(x[0] + x[1]) + 0.2 * np.cos(x[0] + x[1])]
        )
        assert isinstance(grad, tuple)
        assert len(grad) == len(default_shot_vector)
        for g in grad:
            assert isinstance(g, tuple)
            assert len(g) == len(tape.trainable_params)
            for (
                a,
                b,
            ) in zip(g, _expected):
                assert np.allclose(a, b, atol=shot_vec_tol)

    @pytest.mark.slow
    def test_independent_parameters_analytic(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1, shots=shot_vec)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(1.0, wires=[0])
            qml.RX(1.0, wires=[1])
            qml.expval(qml.PauliZ(1))

        tape2 = qml.tape.QuantumScript.from_queue(q2, shots=shot_vec)
        tapes, fn = qml.gradients.param_shift(tape1)
        with qml.Tracker(dev) as tracker:
            j1 = fn(dev.execute(tapes))

        # We should only be executing the device twice: Two shifted evaluations to differentiate
        # one parameter overall, as the other parameter does not impact the returned measurement.

        assert tracker.totals["executions"] == 2

        tapes, fn = qml.gradients.param_shift(tape2)
        j2 = fn(dev.execute(tapes))

        _expected = -np.sin(1)

        assert isinstance(j1, tuple)
        assert len(j1) == len(many_shots_shot_vector)
        for j in j1:
            assert isinstance(j, tuple)
            assert len(j) == len(tape1.trainable_params)
            assert np.allclose(j[0], _expected, atol=shot_vec_tol)
            assert np.allclose(j[1], 0, atol=shot_vec_tol)

        for j in j2:
            assert isinstance(j, tuple)
            assert len(j) == len(tape1.trainable_params)
            assert np.allclose(j[0], 0, atol=shot_vec_tol)
            assert np.allclose(j[1], _expected, atol=shot_vec_tol)

    def test_grad_recipe_parameter_dependent(self):
        """Test that an operation with a gradient recipe that depends on
        its instantiated parameter values works correctly within the parameter
        shift rule. Also tests that `grad_recipe` supersedes `parameter_frequencies`.
        """

        x = np.array(0.654, requires_grad=True)
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            RX_par_dep_recipe(x, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tapes, fn = qml.gradients.param_shift(tape)

        assert len(tapes) == 2
        assert [t.batch_size for t in tapes] == [None, None]
        assert qml.math.allclose(tapes[0].operations[0].data[0], 0)
        assert qml.math.allclose(tapes[1].operations[0].data[0], 2 * x)

        grad = fn(dev.execute(tapes))
        assert np.allclose(grad, -np.sin(x), atol=shot_vec_tol)

    def test_error_no_diff_info(self):
        """Test that an error is raised if no grad_recipe, no parameter_frequencies
        and no generator are found."""

        class RX(qml.RX):
            """This copy of RX overwrites parameter_frequencies to report
            missing information, disabling its differentiation."""

            @property
            def parameter_frequencies(self):
                """Raise an error instead of returning frequencies."""
                raise qml.operation.ParameterFrequenciesUndefinedError

        class NewOp(qml.operation.Operation):
            """This new operation does not overwrite parameter_frequencies
            but does not have a generator, disabling its differentiation."""

            num_params = 1
            grad_method = "A"
            num_wires = 1

        x = np.array(0.654, requires_grad=True)
        shot_vec = many_shots_shot_vector

        for op in [RX, NewOp]:
            with qml.queuing.AnnotatedQueue() as q:
                op(x, wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
            with pytest.raises(
                qml.exceptions.OperatorPropertyUndefined, match="does not have a grad_recipe"
            ):
                qml.gradients.param_shift(tape)


# TODO: add test class for parameter broadcasting - when using the transform on an already broadcasted
# tape with `broadcast=False` (as double broadcasting is not supported).


# The first line of pylint disable is for cost1 through cost6
# pylint: disable=no-self-argument, not-an-iterable
# pylint: disable=too-many-public-methods
@pytest.mark.slow
@pytest.mark.parametrize("broadcast", [False, True])
class TestParameterShiftRule:
    """Unit tests for the param_shift function used with a device that has a
    shot vector defined"""

    @pytest.mark.parametrize("theta", angles)
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3])
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift])
    def test_pauli_rotation_gradient(self, mocker, G, theta, shift, broadcast, seed):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        # pylint: disable=too-many-arguments

        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=1, seed=seed)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {1}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)], broadcast=broadcast)
        assert len(tapes) == (1 if broadcast else 2)

        autograd_val = fn(dev.execute(tapes))

        tape_fwd = tape.bind_new_parameters([theta + np.pi / 2], [1])
        tape_bwd = tape.bind_new_parameters([theta - np.pi / 2], [1])

        shot_vec_manual_res = dev.execute([tape_fwd, tape_bwd])

        # Parameter axis is the first - reorder the results from execute
        shot_vec_len = len(shot_vec)
        shot_vec_manual_res = [
            tuple(comp[l] for comp in shot_vec_manual_res) for l in range(shot_vec_len)
        ]
        for r1, r2 in zip(autograd_val, shot_vec_manual_res):
            manualgrad_val = np.subtract(*r2) / 2
            assert np.allclose(r1, manualgrad_val, atol=shot_vec_tol, rtol=0)

            assert isinstance(r1, np.ndarray)
            assert r1.shape == ()

        assert spy.call_args[1]["shifts"] == (shift,)

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        numeric_val = fn(dev.execute(tapes))
        for a_val, n_val in zip(autograd_val, numeric_val):
            assert np.allclose(a_val, n_val, atol=finite_diff_tol, rtol=0)

    @pytest.mark.parametrize("theta", angles)
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3])
    def test_Rot_gradient(self, mocker, theta, shift, broadcast):
        """Tests that the automatic gradient of an arbitrary Euler-angle-parametrized gate is correct."""
        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")

        shot_vec = tuple([1000000] * 2)
        dev = qml.device("default.qubit", wires=1)
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {1, 2, 3}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)] * 3, broadcast=broadcast)
        num_params = len(tape.trainable_params)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == tapes_per_param * num_params

        autograd_val = fn(dev.execute(tapes))
        assert isinstance(autograd_val, tuple)
        assert len(autograd_val) == len(shot_vec)

        manualgrad_val = []
        for idx in list(np.ndindex(*params.shape)):
            s = np.zeros_like(params)
            s[idx] += np.pi / 2

            tape = tape.bind_new_parameters(params + s, [1, 2, 3])
            forward = dev.execute(tape)

            tape = tape.bind_new_parameters(params - s, [1, 2, 3])
            backward = dev.execute(tape)

            shot_vec_comp = []
            for f, b in zip(forward, backward):
                shot_vec_comp.append((f - b) / 2)

            manualgrad_val.append(tuple(shot_vec_comp))

        # Parameter axis is the first - reorder the results
        shot_vec_len = len(shot_vec)
        manualgrad_val = [tuple(comp[l] for comp in manualgrad_val) for l in range(shot_vec_len)]
        assert len(autograd_val) == len(manualgrad_val)

        for a_val, m_val in zip(autograd_val, manualgrad_val):
            assert isinstance(a_val, tuple)
            assert len(a_val) == num_params
            assert np.allclose(a_val, m_val, atol=shot_vec_tol, rtol=0)
            assert spy.call_args[1]["shifts"] == (shift,)

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        numeric_val = fn(dev.execute(tapes))
        for a_val, n_val in zip(autograd_val, numeric_val):
            assert np.allclose(a_val, n_val, atol=finite_diff_tol, rtol=0)

    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, broadcast):
        """Test gradient of controlled rotation gates"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)
        b = 0.123

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(b, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {1}

        res = dev.execute(tape)
        assert np.allclose(res, -np.cos(b / 2), atol=shot_vec_tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        assert len(tapes) == (1 if broadcast else 4)
        grad = fn(dev.execute(tapes))
        expected = np.sin(b / 2) / 2
        assert isinstance(grad, tuple)
        assert len(grad) == len(shot_vec)
        assert np.allclose(grad, expected, atol=shot_vec_tol, rtol=0)

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        numeric_val = fn(dev.execute(tapes))
        for a_val, n_val in zip(grad, numeric_val):
            assert np.allclose(a_val, n_val, atol=finite_diff_tol, rtol=0)

    @pytest.mark.parametrize("theta", angles)
    def test_CRot_gradient(self, theta, broadcast):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parametrized
        gate is correct."""
        shot_vec = tuple([1000000] * 2)
        dev = qml.device("default.qubit", wires=2)
        a, b, c = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.CRot(a, b, c, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {1, 2, 3}

        res = dev.execute(tape)
        expected = -np.cos(b / 2) * np.cos(0.5 * (a + c))
        assert np.allclose(res, expected, atol=shot_vec_tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 4
        assert len(tapes) == tapes_per_param * len(tape.trainable_params)

        grad = fn(dev.execute(tapes))
        expected = np.array(
            [
                0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                0.5 * np.sin(b / 2) * np.cos(0.5 * (a + c)),
                0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
            ]
        )
        assert isinstance(grad, tuple)
        assert len(grad) == len(shot_vec)

        for shot_vec_res in grad:
            assert isinstance(shot_vec_res, tuple)
            assert len(shot_vec_res) == len(tape.trainable_params)
            for idx, g in enumerate(shot_vec_res):
                assert np.allclose(g, expected[idx], atol=shot_vec_tol, rtol=0)

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        numeric_val = fn(dev.execute(tapes))
        for a_val, n_val in zip(grad, numeric_val):
            assert np.allclose(a_val, n_val, atol=finite_diff_tol, rtol=0)

    def test_gradients_agree_finite_differences(self, broadcast):
        """Tests that the parameter-shift rule agrees with the first and second
        order finite differences"""
        params = np.array([0.1, -1.6, np.pi / 5])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(params[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(params[2], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        shot_vec = many_shots_shot_vector
        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {0, 2, 3}
        dev = qml.device("default.qubit", wires=2)

        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1, h=h_val)
        grad_F2 = grad_fn(
            tape,
            dev,
            fn=qml.gradients.finite_diff,
            approx_order=2,
            strategy="center",
            h=h_val,
        )
        grad_A = grad_fn(tape, dev, broadcast=broadcast)

        # gradients computed with different methods must agree
        for a_val, n_val in zip(grad_A, grad_F1):
            assert np.allclose(a_val, n_val, atol=finite_diff_tol, rtol=0)
        for a_val, n_val in zip(grad_A, grad_F2):
            assert np.allclose(a_val, n_val, atol=finite_diff_tol, rtol=0)

    def test_variance_gradients_agree_finite_differences(self, broadcast):
        """Tests that the variance parameter-shift rule agrees with the first and second
        order finite differences"""
        params = np.array([0.1, -1.6, np.pi / 5])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(params[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(params[2], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        shot_vec = many_shots_shot_vector
        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {0, 2, 3}
        dev = qml.device("default.qubit", wires=2)

        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1, h=h_val)
        grad_F2 = grad_fn(
            tape,
            dev,
            fn=qml.gradients.finite_diff,
            approx_order=2,
            strategy="center",
            h=h_val,
        )
        grad_A = grad_fn(tape, dev, broadcast=broadcast)

        # gradients computed with different methods must agree
        for idx1, _grad_A in enumerate(grad_A):
            for idx2, g in enumerate(_grad_A):
                assert np.allclose(g, grad_F1[idx1][idx2], atol=finite_diff_tol, rtol=0)
                assert np.allclose(g, grad_F2[idx1][idx2], atol=finite_diff_tol, rtol=0)

    @pytest.mark.autograd
    def test_fallback(self, mocker, broadcast):
        """Test that fallback gradient functions are correctly used"""
        spy = mocker.spy(qml.gradients, "finite_diff")
        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        params = np.array([x, y], requires_grad=True)

        def cost_fn(params):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                RY_with_F(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.var(qml.PauliX(1))
                qml.expval(qml.PauliZ(2))

            tape = qml.tape.QuantumScript.from_queue(q, shots=fallback_shot_vec)
            finite_diff = partial(qml.gradients.finite_diff, h=h_val)
            tapes, fn = param_shift(tape, fallback_fn=finite_diff, broadcast=broadcast)
            # Saving one tape in the parameter-shift part of the computation
            assert len(tapes) == (4 if broadcast else 5)

            # check that the fallback method was called for the specified argnums
            spy.assert_called()
            assert spy.call_args[1]["argnum"] == {1}

            return fn(dev.execute(tapes))

        all_res = cost_fn(params)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)], [0, 0]])
        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 3

            for r in res:
                assert isinstance(r, tuple)
                assert len(r) == 2

                assert isinstance(r[0], np.ndarray)
                assert r[0].shape == ()
                assert isinstance(r[1], np.ndarray)
                assert r[1].shape == ()

            assert np.allclose(res, expected, atol=finite_diff_tol, rtol=0)

            # TODO: support Hessian with the new return types
            # check the second derivative
            # hessian = qml.jacobian(lambda params: np.stack(cost_fn(params)).T)(params)
            # hessian = qml.jacobian(cost_fn(params))(params)

            # assert np.allclose(jac[0, 0, 0], -np.cos(x), atol=shot_vec_tol, rtol=0)
            # assert np.allclose(jac[1, 1, 1], -2 * np.cos(2 * y), atol=shot_vec_tol, rtol=0)

    @pytest.mark.autograd
    def test_fallback_single_meas(self, mocker, broadcast):
        """Test that fallback gradient functions are correctly used for a single measurement."""
        spy = mocker.spy(qml.gradients, "finite_diff")
        shot_vec = tuple([1000000] * 4)
        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        params = np.array([x, y], requires_grad=True)

        def cost_fn(params):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                RX_with_F(params[1], wires=[0])
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
            finite_diff = partial(qml.gradients.finite_diff, h=h_val)
            tapes, fn = param_shift(tape, fallback_fn=finite_diff, broadcast=broadcast)
            assert len(tapes) == (3 if broadcast else 4)

            # check that the fallback method was called for the specified argnums
            spy.assert_called()
            assert spy.call_args[1]["argnum"] == {1}

            return fn(dev.execute(tapes))

        all_res = cost_fn(params)

        expval_expected = [-np.sin(x + y), -np.sin(x + y)]
        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            for r in res:
                assert isinstance(r, np.ndarray)
                assert r.shape == ()

            assert np.allclose(res[0], expval_expected[0], atol=finite_diff_tol)
            assert np.allclose(res[1], expval_expected[1], atol=finite_diff_tol)

    @pytest.mark.parametrize("RX, RY, argnum", [(RX_with_F, qml.RY, 0), (qml.RX, RY_with_F, 1)])
    def test_fallback_probs(
        self, RX, RY, argnum, mocker, broadcast
    ):  # pylint:disable=too-many-arguments
        """Test that fallback gradient functions are correctly used with probs"""
        spy = mocker.spy(qml.gradients, "finite_diff")
        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        params = np.array([x, y], requires_grad=True)

        def cost_fn(params):
            with qml.queuing.AnnotatedQueue() as q:
                RX(params[0], wires=[0])
                RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[0, 1])

            tape = qml.tape.QuantumScript.from_queue(q, shots=fallback_shot_vec)
            finite_diff = partial(qml.gradients.finite_diff, h=h_val)
            tapes, fn = param_shift(tape, fallback_fn=finite_diff, broadcast=broadcast)
            assert len(tapes) == (3 if broadcast else 4)

            # check that the fallback method was called for the specified argnums
            spy.assert_called()

            assert spy.call_args[1]["argnum"] == {argnum}
            return fn(dev.execute(tapes))

        all_res = cost_fn(params)
        assert isinstance(all_res, tuple)

        assert len(all_res) == len(fallback_shot_vec)

        for res in all_res:
            assert isinstance(res, tuple)

            assert len(res) == 2

            expval_res = res[0]
            assert isinstance(expval_res, tuple)
            assert len(expval_res) == 2

            for param_r in expval_res:
                assert isinstance(param_r, np.ndarray)
                assert param_r.shape == ()

            probs_res = res[1]
            assert isinstance(probs_res, tuple)
            assert len(probs_res) == 2
            for param_r in probs_res:
                assert isinstance(param_r, np.ndarray)
                assert param_r.shape == (4,)

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
            assert np.allclose(res[0][0], expval_expected[0], atol=finite_diff_tol)
            assert np.allclose(res[0][1], expval_expected[1], atol=finite_diff_tol)

            # Probs
            assert np.allclose(res[1][0], probs_expected[:, 0], atol=finite_diff_tol)
            assert np.allclose(res[1][1], probs_expected[:, 1], atol=finite_diff_tol)

    @pytest.mark.autograd
    def test_all_fallback(self, mocker, broadcast):
        """Test that *only* the fallback logic is called if no parameters
        support the parameter-shift rule"""
        if broadcast:
            pytest.skip("No need to test fallback with broadcasting")
        spy_fd = mocker.spy(qml.gradients, "finite_diff")
        spy_ps = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            RX_with_F(x, wires=[0])
            RY_with_F(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=fallback_shot_vec)
        finite_diff = partial(qml.gradients.finite_diff, h=h_val)
        tapes, fn = param_shift(tape, fallback_fn=finite_diff)
        assert len(tapes) == 1 + 2

        # check that the fallback method was called for all argnums
        spy_fd.assert_called()
        spy_ps.assert_not_called()

        all_res = fn(dev.execute(tapes))
        assert len(all_res) == len(fallback_shot_vec)
        assert isinstance(all_res, tuple)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2
            assert res[0].shape == ()
            assert res[1].shape == ()

            assert np.allclose(res[0], expected[0], atol=fallback_shot_vec, rtol=0)
            assert np.allclose(res[1], expected[1], atol=fallback_shot_vec, rtol=0)

    def test_single_expectation_value(self, broadcast):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == tapes_per_param * 2

        all_res = fn(dev.execute(tapes))

        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        for res in all_res:
            assert len(res) == 2
            assert not isinstance(res[0], tuple)
            assert not isinstance(res[1], tuple)

            assert np.allclose(res[0], expected[0], atol=shot_vec_tol, rtol=0)
            assert np.allclose(res[1], expected[1], atol=shot_vec_tol, rtol=0)

    def test_multiple_expectation_values(self, broadcast):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == tapes_per_param * 2

        all_res = fn(dev.execute(tapes))
        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        for res in all_res:
            assert len(res) == 2
            assert isinstance(res, tuple)
            assert len(res[0]) == 2
            assert len(res[1]) == 2

            assert np.allclose(res[0], expected[0], atol=shot_vec_tol, rtol=0)
            assert np.allclose(res[1], expected[1], atol=shot_vec_tol, rtol=0)

    def test_var_expectation_values(self, broadcast):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        assert len(tapes) == (3 if broadcast else 5)

        all_res = fn(dev.execute(tapes))
        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2
            assert len(res[0]) == 2
            assert len(res[1]) == 2

            for a, e in zip(res, expected):
                assert np.allclose(np.squeeze(np.stack(a)), e, atol=shot_vec_tol, rtol=0)

    def test_prob_expectation_values(self, broadcast):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == 2 * tapes_per_param

        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == len(many_shots_shot_vector)

        for shot_comp_res in res:
            assert isinstance(shot_comp_res, tuple)
            assert len(shot_comp_res) == len(tape.trainable_params)
            for r in shot_comp_res:
                assert isinstance(r, tuple)
                assert len(r) == len(tape.measurements)

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
            assert isinstance(r[0], tuple)
            assert len(r[0]) == len(tape.trainable_params)

            r_to_check = r[0][0]
            _expected = expval_expected[0]
            assert np.allclose(r_to_check, _expected, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == ()

            r_to_check = r[0][1]
            _expected = expval_expected[1]
            assert np.allclose(r_to_check, _expected, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == ()

            # Probs
            assert isinstance(r[1], tuple)
            assert len(r[1]) == len(tape.trainable_params)

            r_to_check = r[1][0]
            _expected = probs_expected[:, 0]
            assert np.allclose(r_to_check, _expected, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == (4,)

            r_to_check = r[1][1]
            _expected = probs_expected[:, 1]
            assert np.allclose(r_to_check, _expected, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == (4,)

    def test_involutory_variance_single_param(self, broadcast):
        """Tests qubit observables that are involutory with a single trainable param"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=1)
        a = 0.54

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.var(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        res = dev.execute(tape)
        expected = 1 - np.cos(a) ** 2
        for r in res:
            assert np.allclose(r, expected, atol=shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        gradA = fn(dev.execute(tapes))
        for _gA in gradA:
            assert isinstance(_gA, np.ndarray)
            assert _gA.shape == ()

        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == 1 + tapes_per_param * 1

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        all_gradF = fn(dev.execute(tapes))
        assert len(tapes) == 2

        expected = 2 * np.sin(a) * np.cos(a)

        for gradF in all_gradF:
            assert gradF == pytest.approx(expected, abs=finite_diff_tol)

        for _gA in gradA:
            assert _gA == pytest.approx(expected, abs=shot_vec_tol)

    def test_involutory_variance_multi_param(self, broadcast):
        """Tests qubit observables that are involutory with multiple trainable params"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=1)
        a = 0.34
        b = 0.20

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.RX(b, wires=0)
            qml.var(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {0, 1}

        res = dev.execute(tape)
        expected = 1 - np.cos(a + b) ** 2
        assert np.allclose(res, expected, atol=shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == 1 + tapes_per_param * 2

        all_res = fn(dev.execute(tapes))
        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        for gradA in all_res:
            assert isinstance(gradA[0], np.ndarray)
            assert gradA[0].shape == ()

            assert isinstance(gradA[1], np.ndarray)
            assert gradA[1].shape == ()

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        assert len(tapes) == 3

        all_Fres = fn(dev.execute(tapes))
        for gradF, gradA in zip(all_Fres, all_res):

            expected = 2 * np.sin(a + b) * np.cos(a + b)
            assert gradF[0] == pytest.approx(expected, abs=finite_diff_tol)
            assert gradA[0] == pytest.approx(expected, abs=finite_diff_tol)

            assert gradF[1] == pytest.approx(expected, abs=finite_diff_tol)
            assert gradA[1] == pytest.approx(expected, abs=finite_diff_tol)

    def test_non_involutory_variance_single_param(self, broadcast, seed):
        """Tests a qubit Hermitian observable that is not involutory with a single trainable parameter"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=1, seed=seed)
        a = 0.54

        _herm_shot_vec_tol = shot_vec_tol * 100
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.var(qml.Hermitian(A, 0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {0}

        res = dev.execute(tape)
        expected = (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)
        for r in res:
            assert np.allclose(r, expected, atol=_herm_shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        # forward pass + tapes_per_param * (expval+expval^2) * num_params
        assert len(tapes) == 1 + tapes_per_param * 2 * 1
        gradA = fn(dev.execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        all_gradF = fn(dev.execute(tapes))
        assert len(tapes) == 2

        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        for _gA in gradA:
            assert _gA == pytest.approx(expected, abs=_herm_shot_vec_tol)
            assert isinstance(_gA, np.ndarray)
            assert _gA.shape == ()
        for gradF in all_gradF:
            assert isinstance(gradF, np.ndarray)
            assert gradF.shape == ()
            assert qml.math.allclose(gradF, expected, atol=2 * _herm_shot_vec_tol)

    def test_non_involutory_variance_multi_param(self, broadcast, seed):
        """Tests a qubit Hermitian observable that is not involutory with multiple trainable parameters"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=1, seed=seed)
        a = 0.34
        b = 0.20

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.RX(b, wires=0)
            qml.var(qml.Hermitian(A, 0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {0, 1}

        all_res = dev.execute(tape)
        expected = (39 / 2) - 6 * np.sin(2 * (a + b)) + (35 / 2) * np.cos(2 * (a + b))
        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)
        for res in all_res:
            assert np.allclose(res, expected, atol=herm_shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        # forward pass + tapes_per_param * (expval+expval^2) * num_params
        assert len(tapes) == 1 + tapes_per_param * 2 * 2

        all_res = fn(dev.execute(tapes))
        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        expected = -35 * np.sin(2 * (a + b)) - 12 * np.cos(2 * (a + b))
        for gradA in all_res:
            assert isinstance(gradA, tuple)

            assert isinstance(gradA[0], np.ndarray)
            assert gradA[0].shape == ()

            assert isinstance(gradA[1], np.ndarray)
            assert gradA[1].shape == ()
            assert gradA[0] == pytest.approx(expected, abs=herm_shot_vec_tol)
            assert gradA[1] == pytest.approx(expected, abs=herm_shot_vec_tol)

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        assert len(tapes) == 3

        all_gradF = fn(dev.execute(tapes))
        assert len(all_gradF) == len(many_shots_shot_vector)
        assert isinstance(all_gradF, tuple)
        for gradF in all_gradF:

            # Note: the tolerances here are significantly higher than in usual tests
            # due to the stochasticity of the test case
            assert gradF[0] == pytest.approx(expected, abs=2)
            assert qml.math.allclose(gradF[1], expected, atol=1.5)

    @pytest.mark.local_salt(42)
    def test_involutory_and_noninvolutory_variance_single_param(self, broadcast, seed):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable when there's a single trainable parameter."""
        shot_vec = tuple([1000000] * 3)
        # NOTE: this test failed multiple times at https://github.com/PennyLaneAI/pennylane/pull/7306
        # even after tweeking the salt. We fixed the seed to ensure its stability and track it in [sc-91487]
        dev = qml.device("default.qubit", wires=2, seed=seed)
        a = 0.54

        if not broadcast:
            pytest.xfail("This test fails with broadcasting disabled. See [sc-91487] for tracking.")

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        # Note: only the first param is trainable
        tape.trainable_params = {0}

        res = dev.execute(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        for r in res:
            assert qml.math.allclose(r, expected, atol=5e-2)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == 1 + 2 * tapes_per_param

        gradA = fn(dev.execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        gradF = fn(dev.execute(tapes))
        assert len(tapes) == 1 + 1

        expected = [2 * np.sin(a) * np.cos(a), 0]

        # Param-shift
        for shot_vec_result in gradA:
            for param_res in shot_vec_result:
                assert isinstance(param_res, np.ndarray)
                assert param_res.shape == ()

            assert qml.math.allclose(shot_vec_result[0], expected[0], atol=0.1)
            assert qml.math.allclose(shot_vec_result[1], expected[1], atol=0.15)

        # Finite-diff
        for shot_vec_result in gradF:
            for param_res in shot_vec_result:
                assert isinstance(param_res, np.ndarray)
                assert param_res.shape == ()
            # Tolerance came from multi_param test below
            assert qml.math.allclose(shot_vec_result[0], expected[0], atol=0.1)
            assert qml.math.allclose(shot_vec_result[1], expected[1], atol=1.5)

    def test_involutory_and_noninvolutory_variance_multi_param(self, broadcast, seed):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable."""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2, seed=seed)
        a = 0.54

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {0, 1}
        _herm_shot_vec_tol = shot_vec_tol * 100

        res = dev.execute(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        for res_shot_item in res:
            assert np.allclose(res_shot_item, expected, atol=_herm_shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == 1 + 2 * 2 * tapes_per_param

        gradA = fn(dev.execute(tapes))

        assert isinstance(gradA, tuple)
        assert len(gradA) == len(many_shots_shot_vector)
        for shot_vec_res in gradA:
            assert isinstance(shot_vec_res, tuple)
            assert len(shot_vec_res) == len(tape.measurements)
            for meas_res in shot_vec_res:
                assert isinstance(meas_res, tuple)
                assert len(meas_res) == len(tape.trainable_params)
                for param_res in meas_res:
                    assert isinstance(param_res, np.ndarray)
                    assert param_res.shape == ()

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        gradF = fn(dev.execute(tapes))
        assert len(tapes) == 1 + 2

        expected = [2 * np.sin(a) * np.cos(a), 0, 0, -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]

        # Param-shift
        for shot_vec_result in gradA:
            assert isinstance(shot_vec_result[0][0], np.ndarray)
            assert shot_vec_result[0][0].shape == ()
            assert shot_vec_result[0][0] == pytest.approx(expected[0], abs=_herm_shot_vec_tol)

            assert isinstance(shot_vec_result[0][1], np.ndarray)
            assert shot_vec_result[0][1].shape == ()
            assert shot_vec_result[0][1] == pytest.approx(expected[1], abs=_herm_shot_vec_tol)

            assert isinstance(shot_vec_result[1][0], np.ndarray)
            assert shot_vec_result[1][0].shape == ()
            assert shot_vec_result[1][0] == pytest.approx(expected[2], abs=_herm_shot_vec_tol)

            assert isinstance(shot_vec_result[1][1], np.ndarray)
            assert shot_vec_result[1][1].shape == ()
            assert shot_vec_result[1][1] == pytest.approx(expected[3], abs=_herm_shot_vec_tol)

        for shot_vec_result in gradF:
            for param_res in shot_vec_result:
                for meas_res in param_res:
                    assert isinstance(meas_res, np.ndarray)
                    assert meas_res.shape == ()

            assert np.allclose(shot_vec_result[0][0], expected[0], atol=1)
            assert np.allclose(shot_vec_result[0][1], expected[1], atol=1)
            assert np.allclose(shot_vec_result[1][0], expected[2], atol=1.5)
            assert np.allclose(shot_vec_result[1][1], expected[3], atol=1.5)

    @pytest.mark.parametrize("ind", [0, 1])
    def test_var_and_probs_single_param(self, ind, broadcast):
        """Tests a qubit Hermitian observable that is not involutory alongside an involutory observable and probs when
        there's one trainable parameter."""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=4)
        a = 0.54

        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            # Ops influencing var res
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)

            # Ops influencing probs res
            qml.RX(x, wires=[2])
            qml.RY(y, wires=[3])
            qml.CNOT(wires=[2, 3])

            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

            qml.probs(wires=[2, 3])

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {ind}

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == 1 + 2 * tapes_per_param

        all_res = fn(dev.execute(tapes))
        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        for gradA in all_res:
            assert isinstance(gradA, tuple)
            assert len(gradA) == 3
            assert gradA[0].shape == ()
            assert gradA[1].shape == ()
            assert gradA[2].shape == (4,)

            # Vars
            vars_expected = [2 * np.sin(a) * np.cos(a), -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]
            assert isinstance(gradA[0], np.ndarray)
            assert np.allclose(
                gradA[0], vars_expected[0] if ind == 0 else 0, atol=shot_vec_tol, rtol=0
            )

            assert isinstance(gradA[1], np.ndarray)
            assert np.allclose(
                gradA[1], vars_expected[1] if ind == 1 else 0, atol=herm_shot_vec_tol, rtol=0
            )

            # Probs
            assert isinstance(gradA[2], np.ndarray)
            assert np.allclose(gradA[2], 0, atol=shot_vec_tol, rtol=0)

    def test_var_and_probs_multi_params(self, broadcast):
        """Tests a qubit Hermitian observable that is not involutory alongside an involutory observable and probs when
        there are more trainable parameters."""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=4)
        a = 0.54

        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            # Ops influencing var res
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)

            # Ops influencing probs res
            qml.RX(x, wires=[2])
            qml.RY(y, wires=[3])
            qml.CNOT(wires=[2, 3])

            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

            qml.probs(wires=[2, 3])

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {0, 1, 2, 3}

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        assert len(tapes) == 1 + 4 * 2 * (1 if broadcast else 2)

        all_res = fn(dev.execute(tapes))
        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        for gradA in all_res:
            assert isinstance(gradA, tuple)
            assert len(gradA) == 3
            var1_res = gradA[0]
            for param_res in var1_res:
                assert isinstance(param_res, np.ndarray)
                assert param_res.shape == ()

            var2_res = gradA[1]
            for param_res in var2_res:
                assert isinstance(param_res, np.ndarray)
                assert param_res.shape == ()

            probs_res = gradA[2]
            for param_res in probs_res:
                assert isinstance(param_res, np.ndarray)
                assert param_res.shape == (4,)

            # Vars
            vars_expected = [2 * np.sin(a) * np.cos(a), -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]
            assert isinstance(gradA[0], tuple)
            assert np.allclose(gradA[0][0], vars_expected[0], atol=shot_vec_tol, rtol=0)
            assert np.allclose(gradA[0][1], 0, atol=shot_vec_tol, rtol=0)
            assert np.allclose(gradA[0][2], 0, atol=shot_vec_tol, rtol=0)
            assert np.allclose(gradA[0][3], 0, atol=shot_vec_tol, rtol=0)

            assert isinstance(gradA[1], tuple)
            assert np.allclose(gradA[1][0], 0, atol=herm_shot_vec_tol, rtol=0)
            assert np.allclose(gradA[1][1], vars_expected[1], atol=herm_shot_vec_tol, rtol=0)
            assert np.allclose(gradA[1][2], 0, atol=herm_shot_vec_tol, rtol=0)
            assert np.allclose(gradA[1][3], 0, atol=herm_shot_vec_tol, rtol=0)

            # Probs
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
            assert isinstance(gradA[2], tuple)
            assert np.allclose(gradA[2][0], 0, atol=shot_vec_tol, rtol=0)
            assert np.allclose(gradA[2][1], 0, atol=shot_vec_tol, rtol=0)
            assert np.allclose(gradA[2][2], probs_expected[:, 0], atol=shot_vec_tol, rtol=0)
            assert np.allclose(gradA[2][3], probs_expected[:, 1], atol=shot_vec_tol, rtol=0)

    def test_expval_and_variance_single_param(self, broadcast):
        """Test an expectation value and the variance of involutory and non-involutory observables work well with a
        single trainable parameter"""
        shot_vec = many_shots_shot_vector
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

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {0}

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
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == 1 + tapes_per_param

        all_res = fn(dev.execute(tapes))

        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        expected = np.array([2 * np.cos(a) * np.sin(a), -np.cos(b) * np.sin(a), 0])
        for gradA in all_res:
            assert isinstance(gradA, tuple)
            for a_comp, e_comp in zip(gradA, expected):
                assert isinstance(a_comp, np.ndarray)
                assert a_comp.shape == ()
                assert np.allclose(a_comp, e_comp, atol=shot_vec_tol, rtol=0)

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        all_gradF = fn(dev.execute(tapes))
        assert isinstance(all_gradF, tuple)

        for gradF in all_gradF:
            assert isinstance(gradF, tuple)
            assert gradF == pytest.approx(expected, abs=finite_diff_tol)

    @pytest.mark.local_salt(42)
    def test_expval_and_variance_multi_param(self, broadcast, seed):
        """Test an expectation value and the variance of involutory and non-involutory observables work well with
        multiple trainable parameters"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=3, seed=seed)

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

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
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
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == 1 + 3 * tapes_per_param
        all_res = fn(dev.execute(tapes))

        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

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
        for gradA in all_res:
            assert isinstance(gradA, tuple)
            for a, e in zip(gradA, expected):
                for a_comp, e_comp in zip(a, e):
                    assert isinstance(a_comp, np.ndarray)
                    assert a_comp.shape == ()
                    assert np.allclose(a_comp, e_comp, atol=shot_vec_tol, rtol=0)

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        all_gradF = fn(dev.execute(tapes))
        for gradF in all_gradF:
            assert gradF == pytest.approx(expected, abs=finite_diff_tol)

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector_variance(self, state, broadcast):
        """Test that the variance of a projector is correctly returned"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)
        x, y = 0.765, -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.Projector(state, wires=0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {0, 1}

        res = dev.execute(tape)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)

        assert len(res) == len(many_shots_shot_vector)
        assert isinstance(res, tuple)
        for r in res:
            assert np.allclose(r, expected, atol=shot_vec_tol, rtol=0)

        # # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == 1 + 2 * 2 * tapes_per_param
        all_res = fn(dev.execute(tapes))

        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        expected = np.array(
            [
                0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
            ]
        )
        for gradA in all_res:
            assert np.allclose(gradA, expected, atol=shot_vec_tol, rtol=0)

        tapes, fn = qml.gradients.finite_diff(tape, h=h_val)
        all_gradF = fn(dev.execute(tapes))
        for gradF in all_gradF:
            assert gradF == pytest.approx(expected, abs=finite_diff_tol)

    def cost1(x):
        """Perform rotation and return a scalar expectation value."""
        qml.Rot(*x, wires=0)
        return qml.expval(qml.PauliZ(0))

    def cost2(x):
        """Perform rotation and return an expectation value in a 1d array."""
        qml.Rot(*x, wires=0)
        return [qml.expval(qml.PauliZ(0))]

    def cost3(x):
        """Perform rotation and return two expectation value in a 1d array."""
        qml.Rot(*x, wires=0)
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    def cost4(x):
        """Perform rotation and return probabilities."""
        qml.Rot(*x, wires=0)
        return qml.probs([0, 1])

    def cost5(x):
        """Perform rotation and return probabilities in a 2d object."""
        qml.Rot(*x, wires=0)
        return [qml.probs([0, 1])]

    def cost6(x):
        """Perform rotation and return two sets of probabilities in a 2d object."""
        qml.Rot(*x, wires=0)
        return [qml.probs([0, 1]), qml.probs([2, 3])]

    costs_and_expected_expval = [
        (cost1, (3,), np.ndarray),
        (cost2, (1, 3), list),
        (cost3, (2, 3), list),
    ]

    @pytest.mark.parametrize("cost, expected_shape, exp_type", costs_and_expected_expval)
    def test_output_shape_matches_qnode_expval(self, cost, expected_shape, exp_type, broadcast):
        """Test that the transform output shape matches that of the QNode."""

        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.set_shots(qml.QNode(cost, dev), shots=shot_vec)

        all_res = qml.gradients.param_shift(circuit, broadcast=broadcast)(x)
        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        for res in all_res:
            assert isinstance(res, exp_type)
            assert qml.math.shape(res) == expected_shape

    costs_and_expected_probs = [
        (cost4, (4, 3), np.ndarray),
        (cost5, (1, 4, 3), list),
        (cost6, (2, 4, 3), list),
    ]

    @pytest.mark.parametrize("cost, expected_shape, exp_type", costs_and_expected_probs)
    def test_output_shape_matches_qnode_probs(self, cost, expected_shape, exp_type, broadcast):
        """Test that the transform output shape matches that of the QNode."""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.set_shots(qml.QNode(cost, dev), shots=shot_vec)

        all_res = qml.gradients.param_shift(circuit, broadcast=broadcast)(x)
        assert len(all_res) == len(many_shots_shot_vector)
        assert isinstance(all_res, tuple)

        for res in all_res:
            assert isinstance(res, exp_type)
            assert qml.math.shape(res) == expected_shape

    # TODO: revisit the following test when the Autograd interface supports
    # parameter-shift with the new return type system
    def test_special_observable_qnode_differentiation(self, broadcast):
        """Test differentiation of a QNode on a device supporting a
        special observable that returns an object rather than a number."""
        if broadcast:
            pytest.skip("SpecialObject is not compatible with non-scalar objects.")

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

        class SpecialObservable(qml.operation.Operator):
            """SpecialObservable"""

            def diagonalizing_gates(self):
                """Diagonalizing gates"""
                return []

        class DeviceSupporingSpecialObservable(DefaultQubitLegacy):
            """A custom device that supports the above SpecialObservable."""

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

        dev = DeviceSupporingSpecialObservable(wires=1, shots=None)

        @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs={"broadcast": broadcast})
        def qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(SpecialObservable(wires=0))

        @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs={"broadcast": broadcast})
        def reference_qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        par = np.array(0.2, requires_grad=True)
        assert np.isclose(qnode(par).item().val, reference_qnode(par))
        assert np.isclose(qml.jacobian(qnode)(par).item().val, qml.jacobian(reference_qnode)(par))

    def test_multi_measure_no_warning(self, broadcast):
        """Test computing the gradient of a tape that contains multiple
        measurements omits no warnings."""
        import warnings

        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=4)

        par1 = qml.numpy.array(0.3)
        par2 = qml.numpy.array(0.1)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(par1, wires=0)
            qml.RX(par2, wires=1)
            qml.probs(wires=[1, 2])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        with warnings.catch_warnings(record=True) as record:
            tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
            fn(dev.execute(tapes))

        assert len(record) == 0


@pytest.mark.parametrize("broadcast", [False, True])
class TestHamiltonianExpvalGradients:
    """Test that tapes ending with expval(H) can be differentiated."""

    def test_not_expval_error(self, broadcast):
        """Test that if the variance of the Hamiltonian is requested,
        an error is raised"""
        shot_vec = many_shots_shot_vector

        weights = np.array([0.4, 0.5])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
            coeffs = np.array([0.1, 0.2, 0.3])
            H = qml.dot(coeffs, obs)
            qml.var(H)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        tape.trainable_params = {2, 3, 4}

        with pytest.raises(ValueError, match="for expectations, not"):
            qml.gradients.param_shift(tape, broadcast=broadcast)

    def test_no_trainable_coeffs(self, broadcast, tol):
        """Test no trainable Hamiltonian coefficients"""
        shot_vec = many_shots_shot_vector
        dev = qml.device("default.qubit", wires=2)

        weights = np.array([0.4, 0.5])

        coeffs = np.array([0.1, 0.2, 0.3])
        a, b, c = coeffs
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            op1 = qml.s_prod(a, qml.PauliZ(0))
            op2 = qml.s_prod(b, qml.prod(qml.PauliZ(0), qml.PauliX(1)))
            op3 = qml.s_prod(c, qml.PauliY(0))
            H = qml.sum(op1, op2, op3)
            qml.expval(H)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        x, y = weights
        tape.trainable_params = {0, 1}

        res = dev.execute([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=shot_vec_tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        # two (broadcasted if broadcast=True) shifts per rotation gate
        assert len(tapes) == (2 if broadcast else 2 * 2)
        assert [t.batch_size for t in tapes] == ([2, 2] if broadcast else [None] * 4)

        all_res = fn(dev.execute(tapes))
        assert isinstance(all_res, tuple)
        assert len(all_res) == len(many_shots_shot_vector)

        expected = [
            -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
            b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
        ]
        for res in all_res:
            assert len(res) == 2
            assert res[0].shape == ()
            assert res[1].shape == ()

            assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
            assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_trainable_coeffs(self, broadcast, tol):
        """Test trainable Hamiltonian coefficients"""
        shot_vec = many_shots_shot_vector
        dev = shots_to_analytic(qml.device("default.qubit", wires=2))

        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = qml.numpy.array([0.1, 0.2, 0.3])
        H = qml.Hamiltonian(coeffs, obs)

        weights = qml.numpy.array([0.4, 0.5])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shot_vec)
        a, b, c = coeffs
        x, y = weights
        tape.trainable_params = {0, 1, 2, 4}

        res = dev.execute([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        # two (broadcasted if broadcast=True) shifts per rotation gate
        # one circuit per trainable H term
        assert len(tapes) == (2 if broadcast else 2 * 2)
        assert [t.batch_size for t in tapes] == ([2, 2] if broadcast else [None] * 4)

        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)
        assert qml.math.shape(res) == (3, 2)

        expected = [
            -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
            b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
        ]
        for r in res:
            assert qml.math.allclose(r, expected, atol=shot_vec_tol)


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

"""
Shot vectors may have some edge cases:

1. All different, "random order", e.g., (100,1,10)
2. At least 1 shot value repeated, e.g., (1,1,10)
3. All same, e.g., (1,1,1)
"""


@pytest.mark.parametrize(
    "shot_vec", [(100, 1, 10), (1, 1, 10), ((1, 2), 10), (10, (1, 2)), (1, 1, 1)]
)
class TestReturn:
    """Class to test the shape of Jacobian with different return types.

    The parameter-shift pipeline has at least 4 major logical paths:

    1. Expval
    2. Probs
    3. Var - involutory observable
    4. Var - non-involutory observable

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

        tapes, fn = qml.gradients.param_shift(tape)
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

        tapes, fn = qml.gradients.param_shift(tape)
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

        tapes, fn = qml.gradients.param_shift(tape)
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

        tapes, fn = qml.gradients.param_shift(tape)
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

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
from pennylane.gradients.parameter_shift import (
    _get_operation_recipe,
    _put_zeros_in_pdA2_involutory,
    _make_zero_rep,
)
from pennylane.devices import DefaultQubitLegacy
from pennylane.operation import Observable, AnyWires


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
            (qml.TRX, (0.5, 1), None),
            (qml.TRX, (0.5, 1), (0.4, 0.8)),
        ],
    )
    def test_custom_recipe_first_order(self, orig_op, frequencies, shifts):
        """Test that a custom recipe is returned correctly for first-order derivatives."""
        c, s = qml.gradients.generate_shift_rule(frequencies, shifts=shifts).T
        recipe = list(zip(c, np.ones_like(c), s))

        # pylint: disable=too-few-public-methods
        class DummyOp(orig_op):
            """Custom version of original operation with different gradient recipe."""

            grad_recipe = (recipe,)

        with qml.queuing.AnnotatedQueue() as q:
            DummyOp(0.2, wires=list(range(DummyOp.num_wires)))

        tape = qml.tape.QuantumScript.from_queue(q)
        out_recipe = _get_operation_recipe(tape, 0, shifts=shifts, order=1)
        assert qml.math.allclose(out_recipe[:, 0], c)
        assert qml.math.allclose(out_recipe[:, 1], np.ones_like(c))

        if shifts is None:
            assert qml.math.allclose(out_recipe[:, 2], s)
        else:
            exp_out_shifts = [-s for s in shifts[::-1]] + list(shifts)
            assert qml.math.allclose(np.sort(s), exp_out_shifts)
            assert qml.math.allclose(np.sort(out_recipe[:, 2]), np.sort(exp_out_shifts))

    def test_qnode_custom_recipe(self):
        """Test a custom recipe using a QNode."""
        dev = qml.device("default.qubit", wires=2)

        x = np.array(0.4, requires_grad=True)
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, 0)
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # Incorrect gradient recipe, but this test only checks execution with an unshifted term.
        recipes = ([[-1e7, 1, 0], [1e7, 1, 1e7]],)
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=recipes)
        assert len(tapes) == 2

        res = fn(qml.execute(tapes, dev, None))
        assert len(res) == 2
        assert isinstance(res, tuple)

    @pytest.mark.parametrize(
        "orig_op, frequencies, shifts",
        [
            (qml.RX, (1.0,), None),
            (qml.RX, (1.0,), (np.pi / 2,)),
            (qml.CRY, (0.5, 1), None),
            (qml.CRY, (0.5, 1), (0.4, 0.8)),
            (qml.TRX, (0.5, 1), None),
            (qml.TRX, (0.5, 1), (0.4, 0.8)),
        ],
    )
    def test_custom_recipe_second_order(self, orig_op, frequencies, shifts):
        """Test that a custom recipe is returned correctly for second-order derivatives."""
        c, s = qml.gradients.generate_shift_rule(frequencies, shifts=shifts).T
        recipe = list(zip(c, np.ones_like(c), s))

        # pylint: disable=too-few-public-methods
        class DummyOp(orig_op):
            """Custom version of original operation with different gradient recipe."""

            grad_recipe = (recipe,)

        with qml.queuing.AnnotatedQueue() as q:
            DummyOp(0.2, wires=list(range(DummyOp.num_wires)))

        tape = qml.tape.QuantumScript.from_queue(q)
        out_recipe = _get_operation_recipe(tape, 0, shifts=shifts, order=2)
        c2, s2 = qml.gradients.generate_shift_rule(frequencies, shifts=shifts, order=2).T
        assert qml.math.allclose(out_recipe[:, 0], c2)
        assert qml.math.allclose(out_recipe[:, 1], np.ones_like(c2))
        assert qml.math.allclose(out_recipe[:, 2], s2)

    @pytest.mark.parametrize("order", [0, 3])
    def test_error_wrong_order(self, order):
        """Test that get_operation_recipe raises an error for orders other than 1 and 2"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.2, wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.raises(NotImplementedError, match="only is implemented for orders 1 and 2."):
            _get_operation_recipe(tape, 0, shifts=None, order=order)


class TestMakeZeroRep:
    """Test that producing a zero-gradient representative with ``_make_zero_rep`` works."""

    # mimic an expectation value or variance, and a probs vector
    @pytest.mark.parametrize("g", [np.array(0.6), np.array([0.6, 0.9])])
    def test_single_measure_no_partitioned_shots(self, g):
        """Test the zero-gradient representative with a single measurement and single shots."""
        rep = _make_zero_rep(g, single_measure=True, has_partitioned_shots=False)
        assert isinstance(rep, np.ndarray) and rep.shape == g.shape
        assert qml.math.allclose(rep, 0.0)

    # mimic an expectation value or variance, and a probs vector
    @pytest.mark.parametrize(
        "g", [(np.array(0.6), np.array(0.4)) * 3, (np.array([0.3, 0.1]), np.array([0.6, 0.9]))]
    )
    def test_single_measure_partitioned_shots(self, g):
        """Test the zero-gradient representative with a single measurement and a shot vector."""
        rep = _make_zero_rep(g, single_measure=True, has_partitioned_shots=True)
        assert isinstance(rep, tuple) and len(rep) == len(g)
        for r, _g in zip(rep, g):
            assert isinstance(r, np.ndarray) and r.shape == _g.shape
            assert qml.math.allclose(r, 0.0)

    # mimic an expectation value, a probs vector, or a mixture of them
    @pytest.mark.parametrize(
        "g",
        [
            (np.array(0.6), np.array(0.4)) * 3,
            (np.array([0.3, 0.1]), np.array([0.6, 0.9])),
            (np.array(0.5), np.ones(4), np.array(0.2)),
        ],
    )
    def test_multi_measure_no_partitioned_shots(self, g):
        """Test the zero-gradient representative with multiple measurements and single shots."""
        rep = _make_zero_rep(g, single_measure=False, has_partitioned_shots=False)

        assert isinstance(rep, tuple) and len(rep) == len(g)
        for r, _g in zip(rep, g):
            assert isinstance(r, np.ndarray) and r.shape == _g.shape
            assert qml.math.allclose(r, 0.0)

    # mimic an expectation value, a probs vector, or a mixture of them
    @pytest.mark.parametrize(
        "g",
        [
            ((np.array(0.6), np.array(0.4)),) * 3,
            ((np.array([0.3, 0.1]), np.array([0.6, 0.9])),) * 2,
            ((np.array(0.5), np.ones(4), np.array(0.2)),) * 4,
        ],
    )
    def test_multi_measure_partitioned_shots(self, g):
        """Test the zero-gradient representative with multiple measurements and a shot vector."""
        rep = _make_zero_rep(g, single_measure=False, has_partitioned_shots=True)

        assert isinstance(rep, tuple) and len(rep) == len(g)
        for _rep, _g in zip(rep, g):
            assert isinstance(_rep, tuple) and len(_rep) == len(_g)
            for r, __g in zip(_rep, _g):
                assert isinstance(r, np.ndarray) and r.shape == __g.shape
                assert qml.math.allclose(r, 0.0)

    # mimic an expectation value or variance, and a probs vector, but with 1d arguments
    @pytest.mark.parametrize(
        "g", [np.array([0.6, 0.2, 0.1]), np.outer([0.4, 0.2, 0.1], [0.6, 0.9])]
    )
    @pytest.mark.parametrize(
        "par_shapes", [((), ()), ((), (2,)), ((), (3, 1)), ((3,), ()), ((3,), (3,)), ((3,), (4, 5))]
    )
    def test_single_measure_no_partitioned_shots_par_shapes(self, g, par_shapes):
        """Test the zero-gradient representative with a single measurement and single shots
        as well as provided par_shapes."""
        old_shape, new_shape = par_shapes
        exp_shape = new_shape + g.shape[len(old_shape) :]
        rep = _make_zero_rep(
            g, single_measure=True, has_partitioned_shots=False, par_shapes=par_shapes
        )
        assert isinstance(rep, np.ndarray) and rep.shape == exp_shape
        assert qml.math.allclose(rep, 0.0)

    # mimic an expectation value or variance, and a probs vector, but with 1d arguments
    @pytest.mark.parametrize(
        "g", [(np.array(0.6), np.array(0.4)) * 3, (np.array([0.3, 0.1]), np.array([0.6, 0.9]))]
    )
    @pytest.mark.parametrize(
        "par_shapes", [((), ()), ((), (2,)), ((), (3, 1)), ((3,), ()), ((3,), (3,)), ((3,), (4, 5))]
    )
    def test_single_measure_partitioned_shots_par_shapes(self, g, par_shapes):
        """Test the zero-gradient representative with a single measurement and a shot vector
        as well as provided par_shapes."""
        old_shape, new_shape = par_shapes
        rep = _make_zero_rep(
            g, single_measure=True, has_partitioned_shots=True, par_shapes=par_shapes
        )
        assert isinstance(rep, tuple) and len(rep) == len(g)
        for r, _g in zip(rep, g):
            exp_shape = new_shape + _g.shape[len(old_shape) :]
            assert isinstance(r, np.ndarray) and r.shape == exp_shape
            assert qml.math.allclose(r, 0.0)

    # mimic an expectation value, a probs vector, or a mixture of them, but with 1d arguments
    @pytest.mark.parametrize(
        "g",
        [
            (np.array(0.6), np.array(0.4)) * 3,
            (np.array([0.3, 0.1]), np.array([0.6, 0.9])),
            (np.array(0.5), np.ones(4), np.array(0.2)),
        ],
    )
    @pytest.mark.parametrize(
        "par_shapes", [((), ()), ((), (2,)), ((), (3, 1)), ((3,), ()), ((3,), (3,)), ((3,), (4, 5))]
    )
    def test_multi_measure_no_partitioned_shots_par_shapes(self, g, par_shapes):
        """Test the zero-gradient representative with multiple measurements and single shots
        as well as provided par_shapes."""
        old_shape, new_shape = par_shapes
        rep = _make_zero_rep(
            g, single_measure=False, has_partitioned_shots=False, par_shapes=par_shapes
        )

        assert isinstance(rep, tuple) and len(rep) == len(g)
        for r, _g in zip(rep, g):
            exp_shape = new_shape + _g.shape[len(old_shape) :]
            assert isinstance(r, np.ndarray) and r.shape == exp_shape
            assert qml.math.allclose(r, 0.0)

    # mimic an expectation value, a probs vector, or a mixture of them, but with 1d arguments
    @pytest.mark.parametrize(
        "g",
        [
            ((np.array(0.6), np.array(0.4)),) * 3,
            ((np.array([0.3, 0.1]), np.array([0.6, 0.9])),) * 2,
            ((np.array(0.5), np.ones(4), np.array(0.2)),) * 4,
        ],
    )
    @pytest.mark.parametrize(
        "par_shapes", [((), ()), ((), (2,)), ((), (3, 1)), ((3,), ()), ((3,), (3,)), ((3,), (4, 5))]
    )
    def test_multi_measure_partitioned_shots_par_shapes(self, g, par_shapes):
        """Test the zero-gradient representative with multiple measurements and a shot vector
        as well as provided par_shapes."""
        old_shape, new_shape = par_shapes
        rep = _make_zero_rep(
            g, single_measure=False, has_partitioned_shots=True, par_shapes=par_shapes
        )

        assert isinstance(rep, tuple) and len(rep) == len(g)
        for _rep, _g in zip(rep, g):
            assert isinstance(_rep, tuple) and len(_rep) == len(_g)
            for r, __g in zip(_rep, _g):
                exp_shape = new_shape + __g.shape[len(old_shape) :]
                assert isinstance(r, np.ndarray) and r.shape == exp_shape
                assert qml.math.allclose(r, 0.0)


def grad_fn(tape, dev, fn=qml.gradients.param_shift, **kwargs):
    """Utility function to automate execution and processing of gradient tapes"""
    tapes, fn = fn(tape, **kwargs)
    return fn(dev.batch_execute(tapes))


class TestParamShift:
    """Unit tests for the param_shift function"""

    def test_empty_circuit(self):
        """Test that an empty circuit works correctly"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            tapes, _ = qml.gradients.param_shift(tape)
        assert not tapes

    def test_all_parameters_independent(self):
        """Test that a circuit where all parameters do not affect the output"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.4, wires=0)
            qml.expval(qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.gradients.param_shift(tape)
        assert not tapes

    def test_state_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a state"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.state()

        tape = qml.tape.QuantumScript.from_queue(q)
        _match = r"return the state with the parameter-shift rule gradient transform"
        with pytest.raises(ValueError, match=_match):
            qml.gradients.param_shift(tape)

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == None

        res = fn(dev.batch_execute(tapes))
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
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
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
        dev = qml.device("default.qubit", wires=4)

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(*params, wires=0)
            qml.probs([2, 3])

        tape = qml.tape.QuantumScript.from_queue(q)
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

        dev = qml.device("default.qubit", wires=4)

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(*params, wires=0)
            qml.expval(qml.PauliZ(wires=2))
            qml.probs([2, 3])

        tape = qml.tape.QuantumScript.from_queue(q)
        g_tapes, post_processing = qml.gradients.param_shift(tape)
        assert g_tapes == []

        result = post_processing(qml.execute(g_tapes, dev, None))

        assert isinstance(result, tuple)

        assert len(result) == 2

        # First elem
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

    def test_with_gradient_recipes(self):
        """Test that the function behaves as expected"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.RX(1.0, wires=0)
            qml.CNOT(wires=[0, 2])
            qml.Rot(2.0, 3.0, 4.0, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
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

    @pytest.mark.parametrize("ops_with_custom_recipe", [[0], [1], [0, 1]])
    @pytest.mark.parametrize("multi_measure", [False, True])
    def test_custom_recipe_unshifted_only(self, ops_with_custom_recipe, multi_measure):
        """Test that if the gradient recipe has a zero-shift component, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)
        x = [0.543, -0.654]

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[0])
            qml.expval(qml.PauliZ(0))
            if multi_measure:
                qml.expval(qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        gradient_recipes = tuple(
            [[-1e7, 1, 0], [1e7, 1, 0]] if i in ops_with_custom_recipe else None for i in range(2)
        )
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)

        # two tapes per parameter that doesn't use a custom recipe,
        # plus one global (unshifted) call if at least one uses the custom recipe
        num_ops_standard_recipe = tape.num_params - len(ops_with_custom_recipe)
        assert len(tapes) == 2 * num_ops_standard_recipe + int(
            tape.num_params != num_ops_standard_recipe
        )
        # Test that executing the tapes and the postprocessing function works
        grad = fn(qml.execute(tapes, dev, None))
        if multi_measure:
            expected = np.array([[-np.sin(x[0] + x[1])] * 2, [0, 0]])
            # The custom recipe estimates gradients to be 0
            for i in ops_with_custom_recipe:
                expected[0, i] = 0
        else:
            expected = [
                -np.sin(x[0] + x[1]) if i not in ops_with_custom_recipe else 0 for i in range(2)
            ]
        assert qml.math.allclose(grad, expected, atol=1e-5)

    @pytest.mark.parametrize("ops_with_custom_recipe", [[0], [1], [0, 1]])
    def test_custom_recipe_mixing_unshifted_shifted(self, ops_with_custom_recipe):
        """Test that if the gradient recipe has a zero-shift component, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)
        x = [0.543, -0.654]

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[0])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        gradient_recipes = tuple(
            [[-1e-7, 1, 0], [1e-7, 1, 0], [-1e5, 1, -5e-6], [1e5, 1, 5e-6]]
            if i in ops_with_custom_recipe
            else None
            for i in range(2)
        )
        tapes, fn = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)

        # two tapes per parameter, independent of recipe
        # plus one global (unshifted) call if at least one uses the custom recipe
        assert len(tapes) == 2 * tape.num_params + int(len(ops_with_custom_recipe) > 0)
        # Test that executing the tapes and the postprocessing function works
        grad = fn(qml.execute(tapes, dev, None))
        assert qml.math.allclose(grad[0], -np.sin(x[0] + x[1]), atol=1e-5)
        assert qml.math.allclose(grad[1], 0, atol=1e-5)

    @pytest.mark.parametrize("y_wire", [0, 1])
    def test_f0_provided(self, y_wire):
        """Test that if the original tape output is provided, then
        the tape is not executed additionally at the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=y_wire)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
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

        # pylint: disable=too-few-public-methods
        class RX(qml.RX):
            """RX operation with an additional term in the grad recipe.
            The grad_recipe no longer yields the derivative, but we account for this.
            For this test, the presence of the unshifted term (with non-vanishing coefficient)
            is essential."""

            grad_recipe = ([[0.5, 1, s], [-0.5, 1, -s], [0.2, 1, 0]],)

        x = np.array([-0.361, 0.654], requires_grad=True)
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=0)
            RX(x[1], wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
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

    def test_independent_parameters_analytic(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.qubit", wires=2)

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
        tapes, fn = qml.gradients.param_shift(tape1)
        j1 = fn(dev.batch_execute(tapes))

        # We should only be executing the device twice: Two shifted evaluations to differentiate
        # one parameter overall, as the other parameter does not impact the returned measurement.

        assert dev.num_executions == 2

        tapes, fn = qml.gradients.param_shift(tape2)
        j2 = fn(dev.batch_execute(tapes))

        exp = -np.sin(1)

        assert np.allclose(j1[0], exp)
        assert np.allclose(j1[1], 0)
        assert np.allclose(j2[0], 0)
        assert np.allclose(j2[1], exp)

    def test_grad_recipe_parameter_dependent(self):
        """Test that an operation with a gradient recipe that depends on
        its instantiated parameter values works correctly within the parameter
        shift rule. Also tests that `grad_recipe` supersedes `parameter_frequencies`.
        """

        x = np.array(0.654, requires_grad=True)
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            RX_par_dep_recipe(x, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = qml.gradients.param_shift(tape)

        assert len(tapes) == 2
        assert [t.batch_size for t in tapes] == [None, None]
        assert qml.math.allclose(tapes[0].operations[0].data[0], 0)
        assert qml.math.allclose(tapes[1].operations[0].data[0], 2 * x)

        grad = fn(dev.batch_execute(tapes))
        assert np.allclose(grad, -np.sin(x))

    def test_error_no_diff_info(self):
        """Test that an error is raised if no grad_recipe, no parameter_frequencies
        and no generator are found."""

        # pylint: disable=too-few-public-methods
        class RX(qml.RX):
            """This copy of RX overwrites parameter_frequencies to report
            missing information, disabling its differentiation."""

            @property
            def parameter_frequencies(self):
                """Raise an error instead of returning frequencies."""
                raise qml.operation.ParameterFrequenciesUndefinedError

        # pylint: disable=too-few-public-methods
        class NewOp(qml.operation.Operation):
            """This new operation does not overwrite parameter_frequencies
            but does not have a generator, disabling its differentiation."""

            num_params = 1
            grad_method = "A"
            num_wires = 1

        x = np.array(0.654, requires_grad=True)
        for op in [RX, NewOp]:
            with qml.queuing.AnnotatedQueue() as q:
                op(x, wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            with pytest.raises(
                qml.operation.OperatorPropertyUndefined, match="does not have a grad_recipe"
            ):
                qml.gradients.param_shift(tape)


# Remove the following and unskip the class below once broadcasted
# tapes are fully supported with gradient transforms. See #4462 for details.
class TestParamShiftRaisesWithBroadcasted:
    """Test that an error is raised with broadcasted tapes."""

    def test_batched_tape_raises(self):
        """Test that an error is raised for a broadcasted/batched tape."""
        tape = qml.tape.QuantumScript([qml.RX([0.4, 0.2], 0)], [qml.expval(qml.PauliZ(0))])
        _match = "Computing the gradient of broadcasted tapes with the parameter-shift rule"
        with pytest.raises(NotImplementedError, match=_match):
            qml.gradients.param_shift(tape)


# Revert the following skip once broadcasted tapes are fully supported with gradient transforms.
# See #4462 for details.
@pytest.mark.skip(reason="Applying gradient transforms to broadcasted tapes is disallowed")
class TestParamShiftWithBroadcasted:
    """Tests for the `param_shift` transform on already broadcasted tapes.
    The tests for `param_shift` using broadcasting itself can be found
    further below."""

    @pytest.mark.parametrize("dim", [1, 3])
    @pytest.mark.parametrize("pos", [0, 1])
    def test_with_single_parameter_broadcasted(self, dim, pos):
        """Test that the parameter-shift transform works with a tape that has
        one of its parameters broadcasted already."""
        x = np.array([0.23, 9.1, 2.3])
        x = x[:dim]
        y = -0.654
        if pos == 1:
            x, y = y, x

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[0])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        assert tape.batch_size == dim
        tapes, fn = qml.gradients.param_shift(tape, argnum=[0, 1])
        assert len(tapes) == 4
        assert np.allclose([t.batch_size for t in tapes], dim)

        dev = qml.device("default.qubit", wires=2)
        res = fn(dev.batch_execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == (dim,)
        assert res[1].shape == (dim,)

    @pytest.mark.parametrize("argnum", [(0, 2), (0, 1), (1,), (2,)])
    @pytest.mark.parametrize("dim", [1, 3])
    def test_with_multiple_parameters_broadcasted(self, dim, argnum):
        """Test that the parameter-shift transform works with a tape that has
        multiple of its parameters broadcasted already."""
        x, y = np.array([[0.23, 9.1, 2.3], [0.2, 1.2, -0.6]])[:, :dim]
        z = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RZ(z, wires=[0])
            qml.RY(y, wires=[0])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        assert tape.batch_size == dim
        tapes, fn = qml.gradients.param_shift(tape, argnum=argnum)
        assert len(tapes) == 2 * len(argnum)
        assert np.allclose([t.batch_size for t in tapes], dim)

        dev = qml.device("default.qubit", wires=2)
        res = fn(dev.batch_execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 3

        assert res[0].shape == res[1].shape == res[2].shape == (dim,)


class TestParamShiftUsingBroadcasting:
    """Tests for the `param_shift` function using broadcasting.
    The tests for `param_shift` on already broadcasted tapes can be found above."""

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        dev = qml.device("default.qubit", wires=2)
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        assert len(tapes) == 1
        assert tapes[0].batch_size == 2

        res = fn(dev.batch_execute(tapes))
        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, [0])

    def test_with_gradient_recipes(self):
        """Test that the function behaves as expected"""

        x, z0, y, z1 = 1.0, 2.0, 3.0, 4.0
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 2])
            qml.Rot(z0, y, z1, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
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
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        gradient_recipes = ([[-1e7, 1, 0], [1e7, 1, 1e7]],) * 2
        tapes, _ = qml.gradients.param_shift(
            tape, gradient_recipes=gradient_recipes, broadcast=True
        )

        # one tape per parameter, plus one global call
        assert len(tapes) == tape.num_params + 1

    def test_independent_parameters_analytic(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.qubit", wires=2)

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
        tapes, fn = qml.gradients.param_shift(tape1, broadcast=True)
        j1 = fn(dev.batch_execute(tapes))

        # We should only be executing the device to differentiate 1 parameter
        # (1 broadcasted execution)

        assert dev.num_executions == 1

        tapes, fn = qml.gradients.param_shift(tape2, broadcast=True)
        j2 = fn(dev.batch_execute(tapes))

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

        x = np.array(0.654, requires_grad=True)
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            RX_par_dep_recipe(x, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)

        assert len(tapes) == 1
        assert tapes[0].batch_size == 2
        assert qml.math.allclose(tapes[0].operations[0].data[0], [0, 2 * x])

        grad = fn(dev.batch_execute(tapes))
        assert np.allclose(grad, -np.sin(x))


# The first of the pylint disable is for cost1 through cost6
# pylint: disable=no-self-argument, not-an-iterable
# pylint: disable=too-many-public-methods
class TestParameterShiftRule:
    """Tests for the parameter shift implementation"""

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift])
    def test_pauli_rotation_gradient(self, mocker, G, theta, shift, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        dev = qml.device("default.qubit", wires=1)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)])
        assert len(tapes) == 2

        autograd_val = fn(dev.batch_execute(tapes))

        tape_fwd = tape.bind_new_parameters([theta + np.pi / 2], [1])
        tape_bwd = tape.bind_new_parameters([theta - np.pi / 2], [1])

        manualgrad_val = np.subtract(*dev.batch_execute([tape_fwd, tape_bwd])) / 2
        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)

        assert isinstance(autograd_val, np.ndarray)
        assert autograd_val.shape == ()

        assert spy.call_args[1]["shifts"] == (shift,)

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

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1, 2, 3}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)] * 3)
        num_params = len(tape.trainable_params)
        assert len(tapes) == 2 * num_params

        autograd_val = fn(dev.batch_execute(tapes))
        assert isinstance(autograd_val, tuple)
        assert len(autograd_val) == num_params

        manualgrad_val = []
        for idx in list(np.ndindex(*params.shape)):
            s = np.zeros_like(params)
            s[idx] += np.pi / 2

            tape = tape.bind_new_parameters(params + s, [1, 2, 3])
            forward = dev.execute(tape)

            tape = tape.bind_new_parameters(params - s, [1, 2, 3])
            backward = dev.execute(tape)

            component = (forward - backward) / 2
            manualgrad_val.append(component)

        assert len(autograd_val) == len(manualgrad_val)

        for a_val, m_val in zip(autograd_val, manualgrad_val):
            assert np.allclose(a_val, m_val, atol=tol, rtol=0)
            assert spy.call_args[1]["shifts"] == (shift,)

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        for a_val, n_val in zip(autograd_val, numeric_val):
            assert np.allclose(a_val, n_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, tol):
        """Test gradient of controlled rotation gates"""
        dev = qml.device("default.qubit", wires=2)
        b = 0.123

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(b, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1}

        res = dev.execute(tape)
        assert np.allclose(res, -np.cos(b / 2), atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape)
        grad = fn(dev.batch_execute(tapes))
        expected = np.sin(b / 2) / 2
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parameterized
        gate is correct."""
        dev = qml.device("default.qubit", wires=2)
        a, b, c = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.CRot(a, b, c, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1, 2, 3}

        res = dev.execute(tape)
        expected = -np.cos(b / 2) * np.cos(0.5 * (a + c))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4 * len(tape.trainable_params)

        grad = fn(dev.batch_execute(tapes))
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

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        for idx, g in enumerate(grad):
            assert np.allclose(g, numeric_val[idx], atol=tol, rtol=0)

    def test_gradients_agree_finite_differences(self, tol):
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

        tape = qml.tape.QuantumScript.from_queue(q)
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

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 2, 3}
        dev = qml.device("default.qubit", wires=2)

        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1)
        grad_F2 = grad_fn(
            tape, dev, fn=qml.gradients.finite_diff, approx_order=2, strategy="center"
        )
        grad_A = grad_fn(tape, dev)

        # gradients computed with different methods must agree
        for idx1, _grad_A in enumerate(grad_A):
            for idx2, g in enumerate(_grad_A):
                assert np.allclose(g, grad_F1[idx1][idx2], atol=tol, rtol=0)
                assert np.allclose(g, grad_F2[idx1][idx2], atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_fallback(self, mocker, tol):
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

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff)
            assert len(tapes) == 5

            # check that the fallback method was called for the specified argnums
            spy.assert_called()
            assert spy.call_args[1]["argnum"] == {1}

            return fn(dev.batch_execute(tapes))

        res = cost_fn(params)

        assert isinstance(res, tuple)

        assert len(res) == 3

        for r in res:
            assert isinstance(r, tuple)
            assert len(r) == 2

            assert isinstance(r[0], np.ndarray)
            assert r[0].shape == ()
            assert isinstance(r[1], np.ndarray)
            assert r[1].shape == ()

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)], [0, 0]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # TODO: support Hessian with the new return types
        # check the second derivative
        # hessian = qml.jacobian(lambda params: np.stack(cost_fn(params)).T)(params)
        # hessian = qml.jacobian(cost_fn(params))(params)

        # assert np.allclose(jac[0, 0, 0], -np.cos(x), atol=tol, rtol=0)
        # assert np.allclose(jac[1, 1, 1], -2 * np.cos(2 * y), atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_fallback_single_meas(self, mocker):
        """Test that fallback gradient functions are correctly used for a single measurement."""
        spy = mocker.spy(qml.gradients, "finite_diff")
        dev = qml.device("default.qubit.autograd", wires=2)
        x = 0.543
        y = -0.654

        params = np.array([x, y], requires_grad=True)

        def cost_fn(params):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                RX_with_F(params[1], wires=[0])
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff)
            assert len(tapes) == 4

            # check that the fallback method was called for the specified argnums
            spy.assert_called()
            assert spy.call_args[1]["argnum"] == {1}

            return fn(dev.batch_execute(tapes))

        res = cost_fn(params)

        assert isinstance(res, tuple)
        assert len(res) == 2

        for r in res:
            assert isinstance(r, np.ndarray)
            assert r.shape == ()

        expval_expected = [-np.sin(x + y), -np.sin(x + y)]
        assert np.allclose(res[0], expval_expected[0])
        assert np.allclose(res[1], expval_expected[1])

    @pytest.mark.autograd
    @pytest.mark.parametrize("RX, RY, argnum", [(RX_with_F, qml.RY, 0), (qml.RX, RY_with_F, 1)])
    def test_fallback_probs(self, RX, RY, argnum, mocker):
        """Test that fallback gradient functions are correctly used with probs"""
        spy = mocker.spy(qml.gradients, "finite_diff")
        dev = qml.device("default.qubit.autograd", wires=2)
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

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff)
            assert len(tapes) == 4

            # check that the fallback method was called for the specified argnums
            spy.assert_called()
            assert spy.call_args[1]["argnum"] == {argnum}

            return fn(dev.batch_execute(tapes))

        res = cost_fn(params)

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
        assert np.allclose(res[0][0], expval_expected[0])
        assert np.allclose(res[0][1], expval_expected[1])

        # Probs
        assert np.allclose(res[1][0], probs_expected[:, 0])
        assert np.allclose(res[1][1], probs_expected[:, 1])

    @pytest.mark.autograd
    def test_all_fallback(self, mocker, tol):
        """Test that *only* the fallback logic is called if no parameters
        support the parameter-shift rule"""
        spy_fd = mocker.spy(qml.gradients, "finite_diff")
        spy_ps = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        dev = qml.device("default.qubit.autograd", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            RX_with_F(x, wires=[0])
            RY_with_F(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff)
        assert len(tapes) == 1 + 2

        # check that the fallback method was called for all argnums
        spy_fd.assert_called()
        spy_ps.assert_not_called()

        res = fn(dev.batch_execute(tapes))

        assert isinstance(res, tuple)
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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

        res = fn(dev.batch_execute(tapes))
        assert len(res) == 2
        assert not isinstance(res[0], tuple)
        assert not isinstance(res[1], tuple)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "par", [0, 1, 2, 3, np.int8(1), np.int16(1), np.int32(1), np.int64(1)]
    )  # integers, zero
    def test_integer_parameters(self, tol, par):
        """Test that the gradient of the RY gate matches the exact analytic formula."""
        dev = qml.device("default.qubit", wires=2)

        tape = qml.tape.QuantumScript([qml.RY(par, wires=[0])], [qml.expval(qml.PauliX(0))])
        tape.trainable_params = {0}

        # gradients
        exact = np.cos(par)
        gtapes, fn = qml.gradients.param_shift(tape)
        grad_PS = fn(qml.execute(gtapes, dev, gradient_fn=None))

        # different methods must agree
        assert np.allclose(grad_PS, exact, atol=tol, rtol=0)

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

        res = fn(dev.batch_execute(tapes))
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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 5

        res = fn(dev.batch_execute(tapes))
        assert len(res) == 2
        assert len(res[0]) == 2
        assert len(res[1]) == 2

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])

        for a, e in zip(res, expected):
            assert np.allclose(np.squeeze(np.stack(a)), e, atol=tol, rtol=0)

    def test_prob_expectation_values(self):
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

        # Expvals
        assert np.allclose(res[0][0], expval_expected[0])
        assert np.allclose(res[0][1], expval_expected[1])

        # Probs
        assert np.allclose(res[1][0], probs_expected[:, 0])
        assert np.allclose(res[1][1], probs_expected[:, 1])

    def test_involutory_variance_single_param(self, tol):
        """Tests qubit observables that are involutory with a single trainable param"""
        dev = qml.device("default.qubit", wires=1)
        a = 0.54

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.var(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0}

        res = dev.execute(tape)
        expected = 1 - np.cos(a) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert isinstance(gradA, np.ndarray)
        assert gradA.shape == ()
        assert len(tapes) == 1 + 2 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 2

        expected = 2 * np.sin(a) * np.cos(a)
        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_involutory_variance_multi_param(self, tol):
        """Tests qubit observables that are involutory with multiple trainable params"""
        dev = qml.device("default.qubit", wires=1)
        a = 0.34
        b = 0.20

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.RX(b, wires=0)
            qml.var(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}

        res = dev.execute(tape)
        expected = 1 - np.cos(a + b) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert isinstance(gradA, tuple)

        assert isinstance(gradA[0], np.ndarray)
        assert gradA[0].shape == ()

        assert isinstance(gradA[1], np.ndarray)
        assert gradA[1].shape == ()

        assert len(tapes) == 1 + 2 * 2

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 3

        expected = 2 * np.sin(a + b) * np.cos(a + b)
        assert gradF[0] == pytest.approx(expected, abs=tol)
        assert gradA[0] == pytest.approx(expected, abs=tol)

        assert gradF[1] == pytest.approx(expected, abs=tol)
        assert gradA[1] == pytest.approx(expected, abs=tol)

    def test_non_involutory_variance_single_param(self, tol):
        """Tests a qubit Hermitian observable that is not involutory with a single trainable parameter"""
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
        assert isinstance(gradA, np.ndarray)
        assert gradA.shape == ()
        assert len(tapes) == 1 + 4 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 2

        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_non_involutory_variance_multi_param(self, tol):
        """Tests a qubit Hermitian observable that is not involutory with multiple trainable parameters"""
        dev = qml.device("default.qubit", wires=1)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.34
        b = 0.20

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.RX(b, wires=0)
            qml.var(qml.Hermitian(A, 0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}

        res = dev.execute(tape)
        expected = (39 / 2) - 6 * np.sin(2 * (a + b)) + (35 / 2) * np.cos(2 * (a + b))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert isinstance(gradA, tuple)

        assert isinstance(gradA[0], np.ndarray)
        assert gradA[0].shape == ()

        assert isinstance(gradA[1], np.ndarray)
        assert gradA[1].shape == ()
        assert len(tapes) == 1 + 4 * 2

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 3

        expected = -35 * np.sin(2 * (a + b)) - 12 * np.cos(2 * (a + b))
        assert gradA[0] == pytest.approx(expected, abs=tol)
        assert gradF[0] == pytest.approx(expected, abs=tol)

        assert gradA[1] == pytest.approx(expected, abs=tol)
        assert gradF[1] == pytest.approx(expected, abs=tol)

    def test_involutory_and_noninvolutory_variance_single_param(self, tol):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable when there's a single trainable parameter."""
        dev = qml.device("default.qubit", wires=2)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # Note: only the first param is trainable
        tape.trainable_params = {0}

        res = dev.execute(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 4

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 1

        expected = [2 * np.sin(a) * np.cos(a), 0]

        assert isinstance(gradA, tuple)
        assert len(gradA) == 2
        for param_res in gradA:
            assert isinstance(param_res, np.ndarray)
            assert param_res.shape == ()

        assert gradA[0] == pytest.approx(expected[0], abs=tol)
        assert gradA[1] == pytest.approx(expected[1], abs=tol)

        assert gradF[0] == pytest.approx(expected[0], abs=tol)
        assert gradF[1] == pytest.approx(expected[1], abs=tol)

    @pytest.mark.parametrize("ind", [0, 1])
    def test_var_and_probs_single_param(self, ind):
        """Tests a qubit Hermitian observable that is not involutory alongside an involutory observable and probs when
        there's one trainable parameter."""
        dev = qml.device("default.qubit", wires=4)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
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

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {ind}

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)

        gradA = fn(dev.batch_execute(tapes))

        assert isinstance(gradA, tuple)
        assert len(gradA) == 3
        assert gradA[0].shape == ()
        assert gradA[1].shape == ()
        assert gradA[2].shape == (4,)

        # Vars
        vars_expected = [2 * np.sin(a) * np.cos(a), -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]
        assert isinstance(gradA[0], np.ndarray)
        assert np.allclose(gradA[0], vars_expected[0] if ind == 0 else 0)

        assert isinstance(gradA[1], np.ndarray)
        assert np.allclose(gradA[1], vars_expected[1] if ind == 1 else 0)

        # Probs
        assert isinstance(gradA[2], np.ndarray)
        assert np.allclose(gradA[2], 0)

    def test_var_and_probs_multi_params(self):
        """Tests a qubit Hermitian observable that is not involutory alongside an involutory observable and probs when
        there are more trainable parameters."""
        dev = qml.device("default.qubit", wires=4)
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
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

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1, 2, 3}

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))

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
        assert np.allclose(gradA[0][0], vars_expected[0])
        assert np.allclose(gradA[0][1], 0)
        assert np.allclose(gradA[0][2], 0)
        assert np.allclose(gradA[0][3], 0)

        assert isinstance(gradA[1], tuple)
        assert np.allclose(gradA[1][0], 0)
        assert np.allclose(gradA[1][1], vars_expected[1])
        assert np.allclose(gradA[1][2], 0)
        assert np.allclose(gradA[1][3], 0)

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
        assert np.allclose(gradA[2][0], 0)
        assert np.allclose(gradA[2][1], 0)
        assert np.allclose(gradA[2][2], probs_expected[:, 0])
        assert np.allclose(gradA[2][3], probs_expected[:, 1])

    def test_put_zeros_in_pdA2_involutory(self):
        """Tests the _process_pdA2_involutory auxiliary function."""
        params = np.array([0.1, -1.6, np.pi / 5])
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))
            qml.var(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}
        involutory_indices = [2]

        pdA2 = (
            (np.array(-0.09983342), np.array(-4.44643859e-16)),
            (np.array(-1.24098015e-15), np.array(6.17263875)),
            (np.array(-1.10652721e-18), np.array(4.44328375e-16)),
        )
        res = _put_zeros_in_pdA2_involutory(tape, pdA2, involutory_indices)
        assert len(res) == len(pdA2)

        # Expval and non-involutory obs parts are the same as in pdA2
        assert res[0] == pdA2[0]
        assert res[1] == pdA2[1]

        # Involutory obs (PauliZ) part is 0
        assert res[2] == (np.array(0), np.array(0))

    def test_expval_and_variance_single_param(self, tol):
        """Test an expectation value and the variance of involutory and non-involutory observables work well with a
        single trainable parameter"""
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
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))

        expected = np.array([2 * np.cos(a) * np.sin(a), -np.cos(b) * np.sin(a), 0])
        assert isinstance(gradA, tuple)
        for a_comp, e_comp in zip(gradA, expected):
            assert isinstance(a_comp, np.ndarray)
            assert a_comp.shape == ()
            assert np.allclose(a_comp, e_comp, atol=tol, rtol=0)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_expval_and_variance_multi_param(self, tol):
        """Test an expectation value and the variance of involutory and non-involutory observables work well with
        multiple trainable parameters"""
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

        assert isinstance(res, tuple)
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
        assert isinstance(gradA, tuple)
        for a, e in zip(gradA, expected):
            for a_comp, e_comp in zip(a, e):
                assert isinstance(a_comp, np.ndarray)
                assert a_comp.shape == ()
                assert np.allclose(a_comp, e_comp, atol=tol, rtol=0)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_recycling_unshifted_tape_result(self):
        """Test that an unshifted term in the used gradient recipe is reused
        for the chain rule computation within the variance parameter shift rule."""
        gradient_recipes = ([[-1e-5, 1, 0], [1e-5, 1, 0], [-1e5, 1, -5e-6], [1e5, 1, 5e-6]], None)
        x = [0.543, -0.654]

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[0])
            qml.var(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)
        # 2 operations x 2 shifted positions + 1 unshifted term overall
        assert len(tapes) == 2 * 2 + 1

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[0])
            qml.var(qml.Projector([1], wires=0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = [0, 1]
        tapes, _ = qml.gradients.param_shift(tape, gradient_recipes=gradient_recipes)

        # 2 operations x 2 shifted positions + 1 unshifted term overall    <-- <H>
        # + 2 operations x 2 shifted positions + 1 unshifted term          <-- <H^2>
        assert len(tapes) == (2 * 2 + 1) + (2 * 2 + 1)

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector_variance(self, state, tol):
        """Test that the variance of a projector is correctly returned"""
        dev = qml.device("default.qubit", wires=2)
        x, y = 0.765, -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.Projector(state, wires=0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}

        res = dev.execute(tape)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))

        expected = np.array(
            [
                0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
            ]
        )
        assert np.allclose(gradA, expected, atol=tol, rtol=0)
        assert gradF == pytest.approx(expected, abs=tol)

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
        (cost1, [3]),
        (cost2, [1, 3]),
        (cost3, [2, 3]),
    ]

    @pytest.mark.parametrize("cost, expected_shape", costs_and_expected_expval)
    def test_output_shape_matches_qnode_expval(self, cost, expected_shape):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res = qml.gradients.param_shift(circuit)(x)

        assert len(res) == expected_shape[0]

        if len(expected_shape) > 1:
            for r in res:
                assert isinstance(r, tuple)
                assert len(r) == expected_shape[1]

    costs_and_expected_probs = [
        (cost4, [3, 4]),
        (cost5, [1, 3, 4]),
        (cost6, [2, 3, 4]),
    ]

    @pytest.mark.parametrize("cost, expected_shape", costs_and_expected_probs)
    def test_output_shape_matches_qnode_probs(self, cost, expected_shape):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res = qml.gradients.param_shift(circuit)(x)

        assert len(res) == expected_shape[0]

        if len(expected_shape) > 2:
            for r in res:
                assert isinstance(r, tuple)
                assert len(r) == expected_shape[1]

                for _r in r:
                    assert isinstance(_r, qml.numpy.ndarray)
                    assert len(_r) == expected_shape[2]

        elif len(expected_shape) > 1:
            for r in res:
                assert isinstance(r, qml.numpy.ndarray)
                assert len(r) == expected_shape[1]

    # TODO: revisit the following test when the Autograd interface supports
    # parameter-shift with the new return type system
    def test_special_observable_qnode_differentiation(self):
        """Test differentiation of a QNode on a device supporting a
        special observable that returns an object rather than a number."""

        # pylint: disable=too-few-public-methods
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

        # pylint: disable=too-few-public-methods
        class SpecialObservable(Observable):
            """SpecialObservable"""

            num_wires = AnyWires

            def diagonalizing_gates(self):
                """Diagonalizing gates"""
                return []

        # pylint: disable=too-few-public-methods
        class DeviceSupporingSpecialObservable(DefaultQubitLegacy):
            """A custom device that supports the above special observable."""

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

    def test_multi_measure_no_warning(self):
        """Test computing the gradient of a tape that contains multiple
        measurements omits no warnings."""
        import warnings

        dev = qml.device("default.qubit", wires=4)

        par1 = qml.numpy.array(0.3)
        par2 = qml.numpy.array(0.1)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(par1, wires=0)
            qml.RX(par2, wires=1)
            qml.probs(wires=[1, 2])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        with warnings.catch_warnings(record=True) as record:
            tapes, fn = qml.gradients.param_shift(tape)
            fn(dev.batch_execute(tapes))

        assert len(record) == 0


# The following pylint disable is for cost1 through cost6
# pylint: disable=no-self-argument, not-an-iterable
class TestParameterShiftRuleBroadcast:
    """Tests for the parameter shift implementation using broadcasting"""

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift])
    def test_pauli_rotation_gradient(self, mocker, G, theta, shift, tol):
        """Tests that the automatic gradients of Pauli rotations are correct with broadcasting."""
        spy = mocker.spy(qml.gradients.parameter_shift, "_get_operation_recipe")
        dev = qml.device("default.qubit", wires=1)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)], broadcast=True)
        assert len(tapes) == 1
        assert tapes[0].batch_size == 2

        autograd_val = fn(dev.batch_execute(tapes))

        tape_fwd = tape.bind_new_parameters([theta + np.pi / 2], [1])
        tape_bwd = tape.bind_new_parameters([theta - np.pi / 2], [1])

        manualgrad_val = np.subtract(*dev.batch_execute([tape_fwd, tape_bwd])) / 2
        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)

        assert spy.call_args[1]["shifts"] == (shift,)

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

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1, 2, 3}

        tapes, fn = qml.gradients.param_shift(tape, shifts=[(shift,)] * 3, broadcast=True)
        assert len(tapes) == len(tape.trainable_params)
        assert [t.batch_size for t in tapes] == [2, 2, 2]

        autograd_val = fn(dev.batch_execute(tapes))
        manualgrad_val = np.zeros_like(autograd_val)

        for idx in list(np.ndindex(*params.shape)):
            s = np.zeros_like(params)
            s[idx] += np.pi / 2

            tape = tape.bind_new_parameters(params + s, [1, 2, 3])
            forward = dev.execute(tape)

            tape = tape.bind_new_parameters(params - s, [1, 2, 3])
            backward = dev.execute(tape)

            manualgrad_val[idx] = (forward - backward) / 2

        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)
        assert spy.call_args[1]["shifts"] == (shift,)

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))

        assert len(autograd_val) == len(numeric_val)
        for a, n in zip(autograd_val, numeric_val):
            assert np.allclose(a, n, atol=tol, rtol=0)

    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, tol):
        """Test gradient of controlled rotation gates"""
        dev = qml.device("default.qubit", wires=2)
        b = 0.123

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(b, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1}

        res = dev.execute(tape)
        assert np.allclose(res, -np.cos(b / 2), atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        grad = fn(dev.batch_execute(tapes))
        expected = np.sin(b / 2) / 2
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parameterized
        gate is correct."""
        dev = qml.device("default.qubit", wires=2)
        a, b, c = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.CRot(a, b, c, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1, 2, 3}

        res = dev.execute(tape)
        expected = -np.cos(b / 2) * np.cos(0.5 * (a + c))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        assert len(tapes) == len(tape.trainable_params)
        assert [t.batch_size for t in tapes] == [4, 4, 4]

        grad = fn(dev.batch_execute(tapes))
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

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.batch_execute(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    def test_gradients_agree_finite_differences(self, tol):
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

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 2, 3}
        dev = qml.device("default.qubit", wires=2)

        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1)
        grad_F2 = grad_fn(
            tape, dev, fn=qml.gradients.finite_diff, approx_order=2, strategy="center"
        )
        grad_A = grad_fn(tape, dev, broadcast=True)

        # gradients computed with different methods must agree
        assert np.allclose(grad_A, grad_F1, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F2, atol=tol, rtol=0)

    @pytest.mark.xfail(reason="Broadcasting with multiple measurements is not supported yet")
    def test_variance_gradients_agree_finite_differences(self, tol):
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

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 2, 3}
        dev = qml.device("default.qubit", wires=2)

        grad_F1 = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1)
        grad_F2 = grad_fn(
            tape, dev, fn=qml.gradients.finite_diff, approx_order=2, strategy="center"
        )
        grad_A = grad_fn(tape, dev, broadcast=True)

        # gradients computed with different methods must agree
        for idx1, _grad_A in enumerate(grad_A):
            for idx2, g in enumerate(_grad_A):
                assert np.allclose(g, grad_F1[idx1][idx2], atol=tol, rtol=0)
                assert np.allclose(g, grad_F2[idx1][idx2], atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_fallback(self, mocker):
        """Test that fallback gradient functions are correctly used"""
        spy = mocker.spy(qml.gradients, "finite_diff")
        dev = qml.device("default.qubit.autograd", wires=2)
        x = 0.543
        y = -0.654

        params = np.array([x, y], requires_grad=True)

        def cost_fn(params):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                RY_with_F(params[1], wires=[1])  # Use finite differences for this op
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.var(qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff, broadcast=True)
            assert len(tapes) == 4

            # check that the fallback method was called for the specified argnums
            spy.assert_called()
            assert spy.call_args[1]["argnum"] == {1}

            return fn(dev.batch_execute(tapes))

        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            cost_fn(params)
        # TODO: Uncomment the following when #2693 is resolved. Add test fixture arg `tol`
        # res = cost_fn(params)
        # assert res.shape == (2, 2)
        # expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        # assert np.allclose(res, expected, atol=tol, rtol=0)

        # double check the derivative
        # jac = qml.jacobian(cost_fn)(params)
        # assert np.allclose(jac[0, 0, 0], -np.cos(x), atol=tol, rtol=0)
        # assert np.allclose(jac[1, 1, 1], -2 * np.cos(2 * y), atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_all_fallback(self, mocker, tol):
        """Test that *only* the fallback logic is called if no parameters
        support the parameter-shift rule"""
        spy_fd = mocker.spy(qml.gradients, "finite_diff")
        spy_ps = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        dev = qml.device("default.qubit.autograd", wires=2)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            RX_with_F(x, wires=[0])
            RY_with_F(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = param_shift(tape, fallback_fn=qml.gradients.finite_diff, broadcast=True)
        assert len(tapes) == 1 + 2

        # check that the fallback method was called for all argnums
        spy_fd.assert_called()
        spy_ps.assert_not_called()

        res = fn(dev.batch_execute(tapes))
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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == 2

        res = fn(dev.batch_execute(tapes))
        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        assert len(res) == len(expected)
        for r, e in zip(res, expected):
            assert np.allclose(r, e, atol=tol, rtol=0)

    def test_multiple_expectation_values(self):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            qml.gradients.param_shift(tape, broadcast=True)
        # TODO: Uncomment the following when #2693 is resolved. Add test fixture arg `tol`
        # dev = qml.device("default.qubit", wires=2)
        # tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        # assert len(tapes) == 2
        # assert tapes[0].batch_size == tapes[1].batch_size == 2

        # res = fn(dev.batch_execute(tapes))
        # assert res.shape == (2, 2)

        # expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        # assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_expectation_values(self):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            qml.gradients.param_shift(tape, broadcast=True)
        # TODO: Uncomment the following when #2693 is resolved. Add test fixture arg `tol`
        # dev = qml.device("default.qubit", wires=2)
        # tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        # assert len(tapes) == 3  # One unshifted, two broadcasted shifted tapes
        # assert tapes[0].batch_size is None
        # assert tapes[1].batch_size == tapes[2].batch_size == 2

        # res = fn(dev.batch_execute(tapes))
        # assert res.shape == (2, 2)

        # expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        # assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self):
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

        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            qml.gradients.param_shift(tape, broadcast=True)
        # TODO: Uncomment the following when #2693 is resolved. Add test fixture arg `tol`
        # tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        # assert len(tapes) == 2
        # assert tapes[0].batch_size == tapes[1].batch_size == 2

        # res = fn(dev.batch_execute(tapes))
        # assert res.shape == (5, 2)

        # expected = (
        # np.array(
        # [
        # [-2 * np.sin(x), 0],
        # [
        # -(np.cos(y / 2) ** 2 * np.sin(x)),
        # -(np.cos(x / 2) ** 2 * np.sin(y)),
        # ],
        # [
        # -(np.sin(x) * np.sin(y / 2) ** 2),
        # (np.cos(x / 2) ** 2 * np.sin(y)),
        # ],
        # [
        # (np.sin(x) * np.sin(y / 2) ** 2),
        # (np.sin(x / 2) ** 2 * np.sin(y)),
        # ],
        # [
        # (np.cos(y / 2) ** 2 * np.sin(x)),
        # -(np.sin(x / 2) ** 2 * np.sin(y)),
        # ],
        # ]
        # )
        # / 2
        # )

        # assert np.allclose(res, expected, atol=tol, rtol=0)

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
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        gradA = fn(dev.batch_execute(tapes))
        assert isinstance(gradA, np.ndarray)
        assert gradA.shape == ()

        assert len(tapes) == 2
        assert tapes[0].batch_size is None
        assert tapes[1].batch_size == 2

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
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        gradA = fn(dev.batch_execute(tapes))
        assert isinstance(gradA, np.ndarray)
        assert gradA.shape == ()

        assert len(tapes) == 1 + 2 * 1
        assert tapes[0].batch_size is None
        assert tapes[1].batch_size == tapes[2].batch_size == 2

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
        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            qml.gradients.param_shift(tape, broadcast=True)
        # TODO: Uncomment the following when #2693 is resolved.
        # tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        # gradA = fn(dev.batch_execute(tapes))
        # assert len(tapes) == 1 + 2 * 4

        # tapes, fn = qml.gradients.finite_diff(tape)
        # gradF = fn(dev.batch_execute(tapes))
        # assert len(tapes) == 1 + 2

        # expected = [2 * np.sin(a) * np.cos(a), -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]
        # assert np.diag(gradA) == pytest.approx(expected, abs=tol)
        # assert np.diag(gradF) == pytest.approx(expected, abs=tol)

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
        with pytest.raises(NotImplementedError, match="Broadcasting with multiple measurements"):
            qml.gradients.param_shift(tape, broadcast=True)
        # TODO: Uncomment the following when #2693 is resolved.
        # tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        # gradA = fn(dev.batch_execute(tapes))

        # tapes, fn = qml.gradients.finite_diff(tape)
        # gradF = fn(dev.batch_execute(tapes))
        # ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
        # c2c, s2c = np.cos(2 * c), np.sin(2 * c)
        # expected = np.array(
        # [
        # [2 * ca * sa, -cb * sa, 0],
        # [0, -ca * sb, 0.5 * (2 * cb * c2c * sb + s2c)],
        # [0, 0, cb ** 2 * s2c],
        # ]
        # ).T
        # assert gradA == pytest.approx(expected, abs=tol)
        # assert gradF == pytest.approx(expected, abs=tol)

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector_variance(self, state, tol):
        """Test that the variance of a projector is correctly returned"""
        dev = qml.device("default.qubit", wires=2)
        x, y = 0.765, -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.Projector(state, wires=0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}

        res = dev.execute(tape)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        gradA = fn(dev.batch_execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))

        expected = np.array(
            [
                0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
            ]
        )
        assert len(gradA) == len(expected)
        for a, e in zip(gradA, expected):
            assert np.allclose(a, e, atol=tol, rtol=0)

        assert gradF == pytest.approx(expected, abs=tol)

    def test_output_shape_matches_qnode(self):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=4)

        def cost1(x):
            """Perform rotation and return a scalar expectation value."""
            qml.Rot(*x, wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost2(x):
            """Perform rotation and return an expectation value in a 1d array."""
            qml.Rot(*x, wires=0)
            return [qml.expval(qml.PauliZ(0))]

        def cost3(x):
            """Perform rotation and return two expectation values in a 1d array."""
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

        x = np.random.rand(3)
        single_measure_circuits = [qml.QNode(cost, dev) for cost in (cost1, cost2, cost4, cost5)]
        multi_measure_circuits = [qml.QNode(cost, dev) for cost in (cost3, cost6)]

        for c, exp_shape in zip(single_measure_circuits, [(3,), (1, 3), (3, 4), (1, 3, 4)]):
            grad = qml.gradients.param_shift(c, broadcast=True)(x)
            assert qml.math.shape(grad) == exp_shape

        for c in multi_measure_circuits:
            with pytest.raises(
                NotImplementedError, match="Broadcasting with multiple measurements"
            ):
                qml.gradients.param_shift(c, broadcast=True)(x)


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
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
            assert len(tapes) == exp_num_tapes
            assert [t.batch_size for t in tapes] == exp_batch_sizes
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


@pytest.mark.parametrize("broadcast", [True, False])
class TestHamiltonianExpvalGradients:
    """Test that tapes ending with expval(H) can be
    differentiated"""

    def test_not_expval_error(self, broadcast):
        """Test that if the variance of the Hamiltonian is requested,
        an error is raised"""
        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3])
        H = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(H)

        tape = qml.tape.QuantumScript.from_queue(q)
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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        tape = qml.tape.QuantumScript.from_queue(q)
        a, b, c = coeffs
        x, y = weights
        tape.trainable_params = {0, 1}

        res = dev.batch_execute([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        # two (broadcasted if broadcast=True) shifts per rotation gate
        assert len(tapes) == (2 if broadcast else 2 * 2)
        assert [t.batch_size for t in tapes] == ([2, 2] if broadcast else [None] * 4)
        spy.assert_not_called()

        res = fn(dev.batch_execute(tapes))
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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        tape = qml.tape.QuantumScript.from_queue(q)
        a, b, c = coeffs
        x, y = weights
        tape.trainable_params = {0, 1, 2, 4}

        res = dev.batch_execute([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        # two (broadcasted if broadcast=True) shifts per rotation gate
        # one circuit per trainable H term
        assert len(tapes) == (2 + 2 if broadcast else 2 * 2 + 2)
        assert [t.batch_size for t in tapes] == ([2, 2, None, None] if broadcast else [None] * 6)
        spy.assert_called()

        res = fn(dev.batch_execute(tapes))
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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H1)
            qml.expval(H2)

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1, 2, 4, 5}

        res = dev.batch_execute([tape])
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

        res = fn(dev.batch_execute(tapes))
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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(H1)
            qml.expval(H2)

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1, 2, 3, 4, 5}
        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
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
                with tf.GradientTape() as _:
                    self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)
            return
        with tf.GradientTape() as _:
            jac = self.cost_fn(weights, coeffs1, coeffs2, dev, broadcast)

        expected = self.cost_fn_expected(weights.numpy(), coeffs1.numpy(), coeffs2.numpy())
        assert np.allclose(jac[0], np.array(expected)[0], atol=tol, rtol=0)
        assert np.allclose(jac[1], np.array(expected)[1], atol=tol, rtol=0)

        # TODO: test when Hessians are supported with the new return types
        # second derivative wrt to Hamiltonian coefficients should be zero.
        # When activating the following, rename the GradientTape above from _ to t
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


@pytest.mark.autograd
class TestQnodeAutograd:
    """Class to test the parameter shift transform on QNode with some classical processing."""

    interfaces = ["auto", "autograd"]

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_single_param(self, interface):
        """Test for a single measurement and a single param."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(2 * x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = qml.numpy.array(0.543, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = qml.jacobian(circuit)(x)

        assert res.shape == res_expected.shape
        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_single_param_2(self, interface):
        """Test for a single measurement and a single param."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = qml.numpy.array([0.543, 0.2], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = qml.jacobian(circuit)(x)

        assert res.shape == res_expected.shape
        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_prob_single_param(self, interface):
        """Test for a single measurement (probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(2 * x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = qml.numpy.array(0.543, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = qml.jacobian(circuit)(x)

        assert res.shape == res_expected.shape
        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_prob_single_param_2(self, interface):
        """Test for a single measurement (probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = qml.numpy.array([0.543, 0.2], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)
        res_expected = qml.jacobian(circuit)(x)

        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_measurement_single_param(self, interface):
        """Test for multiple measurement and a single param."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = qml.numpy.array([0.543, 0.2], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        def cost(x):
            return qml.math.stack(circuit(x))

        res_expected = qml.jacobian(cost)(x)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_measurement_expval_probs_single_param(self, interface):
        """Test for multiple measurement (expval, probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        x = qml.numpy.array([0.543, 0.2], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        def cost(x):
            return qml.math.hstack(circuit(x))

        res_expected = qml.jacobian(cost)(x)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1][0], res_expected[1])
        assert np.allclose(res[1][1], res_expected[2])
        assert np.allclose(res[1][2], res_expected[3])
        assert np.allclose(res[1][3], res_expected[4])

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_multiple_params(self, interface):
        """Test for a single measurement and multiple params."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        x = qml.numpy.array([0.543, 0.2], requires_grad=True)
        y = qml.numpy.array(-0.654, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x, y)
        res_expected = qml.jacobian(circuit)(x, y)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_multiple_params(self, interface):
        """Test for a single measurement (probs) and multiple params."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = qml.numpy.array([0.543, 0.2], requires_grad=True)
        y = qml.numpy.array(-0.654, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x, y)
        res_expected = qml.jacobian(circuit)(x, y)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_measurements_expval_multi_params(self, interface):
        """Test for multiple measurements and multiple params."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = qml.numpy.array([0.543, 0.2], requires_grad=True)
        y = qml.numpy.array(-0.654, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x, y)

        def cost(x, y):
            return qml.math.stack(circuit(x, y))

        res_expected = qml.jacobian(cost)(x, y)

        assert np.allclose(res[0][0], res_expected[0][0])
        assert np.allclose(res[0][1], res_expected[1][0])
        assert np.allclose(res[1][0], res_expected[0][1])
        assert np.allclose(res[1][1], res_expected[1][1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_meas_expval_probs__multi_params(self, interface):
        """Test for multiple measurements (expval , probs) and multiple params."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        x = qml.numpy.array([0.543, 0.2], requires_grad=True)
        y = qml.numpy.array(-0.654, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x, y)

        def cost(x, y):
            return qml.math.hstack(circuit(x, y))

        res_expected = qml.jacobian(cost)(x, y)

        assert np.allclose(res[0][0], res_expected[0][0])
        assert np.allclose(res[0][1], res_expected[1][0])
        assert np.allclose(res[1][0][0], res_expected[0][1])
        assert np.allclose(res[1][0][1], res_expected[0][2])
        assert np.allclose(res[1][0][2], res_expected[0][3])
        assert np.allclose(res[1][0][3], res_expected[0][4])
        assert np.allclose(res[1][1][0], res_expected[1][1])
        assert np.allclose(res[1][1][1], res_expected[1][2])
        assert np.allclose(res[1][1][2], res_expected[1][3])
        assert np.allclose(res[1][1][3], res_expected[1][4])

    @pytest.mark.parametrize("interface", interfaces)
    def test_identity_classical_jacobian(self, interface):
        """Test for an identity cjac."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = qml.numpy.array([0.543, -0.654], requires_grad=True)
        res = qml.gradients.param_shift(circuit)(x)
        res_expected = qml.jacobian(circuit)(x)

        assert np.allclose(res, res_expected)


@pytest.mark.torch
class TestQnodeTorch:
    """Class to test the parameter shift transform on QNode with some classical processing."""

    expected_jacs = []
    interfaces = ["auto", "torch"]

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_single_param(self, interface):
        """Test for a single measurement and a single param."""
        dev = qml.device("default.qubit", wires=2)

        import torch

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(2 * x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor(0.543, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = torch.autograd.functional.jacobian(circuit, x)

        assert res.shape == res_expected.shape
        assert np.allclose(res.detach().numpy(), res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_single_param_2(self, interface):
        """Test for a single measurement and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import torch

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor([0.543, 0.2], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = torch.autograd.functional.jacobian(circuit, x)

        assert res.shape == res_expected.shape
        assert np.allclose(res.detach().numpy(), res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_single_param(self, interface):
        """Test for a single measurement (probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import torch

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(2 * x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = torch.tensor(0.543, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = torch.autograd.functional.jacobian(circuit, x)

        assert res.shape == res_expected.shape
        assert np.allclose(res.detach().numpy(), res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_single_param_2(self, interface):
        """Test for a single measurement (probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)

        import torch

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = torch.tensor([0.543, 0.2], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)
        res_expected = torch.autograd.functional.jacobian(circuit, x)

        assert np.allclose(res.detach().numpy(), res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_measurement_single_param(self, interface):
        """Test for multiple measurements (expvals) and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import torch

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = torch.tensor([0.543, 0.2], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = torch.autograd.functional.jacobian(circuit, x)

        assert np.allclose(res[0].detach().numpy(), res_expected[0])
        assert np.allclose(res[1].detach().numpy(), res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_measurement_expval_probs_single_param(self, interface):
        """Test for multiple measurement (with shape) and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import torch

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        x = torch.tensor([0.543, 0.2], requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = torch.autograd.functional.jacobian(circuit, x)

        assert np.allclose(res[0].detach().numpy(), res_expected[0])
        assert np.allclose(res[1].detach().numpy(), res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_multiple_params(self, interface):
        """Test for a single measurement and multiple params."""
        dev = qml.device("default.qubit", wires=2)
        import torch

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        x = torch.tensor([0.543, 0.2], requires_grad=True)
        y = torch.tensor(-0.654, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x, y)
        res_expected = torch.autograd.functional.jacobian(circuit, (x, y))

        assert np.allclose(res[0].detach().numpy(), res_expected[0])
        assert np.allclose(res[1].detach().numpy(), res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_multiple_params(self, interface):
        """Test for a single measurement (probs) and multiple params."""
        dev = qml.device("default.qubit", wires=2)
        import torch

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = torch.tensor([0.543, 0.2], requires_grad=True)
        y = torch.tensor(-0.654, requires_grad=True)

        res = qml.gradients.param_shift(circuit)(x, y)
        res_expected = torch.autograd.functional.jacobian(circuit, (x, y))

        assert np.allclose(res[0].detach().numpy(), res_expected[0])
        assert np.allclose(res[1].detach().numpy(), res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multiple_measurements_multiple_params(self, interface):
        """Test for multiple measurements and multiple params."""
        dev = qml.device("default.qubit", wires=2)
        import torch

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = torch.tensor([0.543, 0.2], requires_grad=True, dtype=torch.float64)
        y = torch.tensor(-0.654, requires_grad=True, dtype=torch.float64)

        res = qml.gradients.param_shift(circuit)(x, y)

        res_expected = torch.autograd.functional.jacobian(circuit, (x, y))

        assert np.allclose(res[0][0].detach().numpy(), res_expected[0][0])
        assert np.allclose(res[0][1].detach().numpy(), res_expected[0][1])
        assert np.allclose(res[1][0].detach().numpy(), res_expected[1][0])
        assert np.allclose(res[1][1].detach().numpy(), res_expected[1][1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multiple_measurements_expval_probs_multiple_params(self, interface):
        """Test for multiple measurements (expval, probs) and multiple params."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        x = torch.tensor([0.543, 0.2], requires_grad=True, dtype=torch.float64)
        y = torch.tensor(-0.654, requires_grad=True, dtype=torch.float64)

        res = qml.gradients.param_shift(circuit)(x, y)

        res_expected = torch.autograd.functional.jacobian(circuit, (x, y))

        assert np.allclose(res[0][0].detach().numpy(), res_expected[0][0])
        assert np.allclose(res[0][1].detach().numpy(), res_expected[0][1])
        assert np.allclose(res[1][0].detach().numpy(), res_expected[1][0])
        assert np.allclose(res[1][1].detach().numpy(), res_expected[1][1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_identity_classical_jacobian(self, interface):
        """Test for an identity cjac."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor([0.543, -0.654], requires_grad=True, dtype=torch.float64)
        res = qml.gradients.param_shift(circuit)(x)
        res_expected = torch.autograd.functional.jacobian(circuit, x)

        assert np.allclose(res[0].detach().numpy(), res_expected[0])
        assert np.allclose(res[1].detach().numpy(), res_expected[1])


@pytest.mark.jax
class TestQnodeJax:
    """Class to the parameter shift transform with some classical processing."""

    expected_jacs = []
    interfaces = ["auto", "jax"]

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_single_param(self, interface):
        """Test for a single measurement and a single param."""
        dev = qml.device("default.qubit", wires=2)

        import jax

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(2 * x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = jax.numpy.array(0.543)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = jax.jacobian(circuit)(x)

        assert res.shape == res_expected.shape
        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_single_param_2(self, interface):
        """Test for a single measurement and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = jax.numpy.array([0.543, 0.2])

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = jax.jacobian(circuit)(x)
        assert res.shape == res_expected.shape
        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_single_param(self, interface):
        """Test for a single measurement (probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(2 * x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = jax.numpy.array(0.543)

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = jax.jacobian(circuit)(x)

        assert res.shape == res_expected.shape
        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_single_param_2(self, interface):
        """Test for a single measurement (probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)

        import jax

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = jax.numpy.array([0.543, 0.2])

        res = qml.gradients.param_shift(circuit)(x)
        res_expected = jax.jacobian(circuit)(x)

        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_measurement_single_param(self, interface):
        """Test for multiple measurements and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = jax.jacobian(circuit)(x)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_measurement_expval_probs_single_param(self, interface):
        """Test for multiple measurement (probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        x = jax.numpy.array([0.543, 0.2])

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = jax.jacobian(circuit)(x)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_multiple_params(self, interface):
        """Test for a single measurement and multiple params."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = qml.gradients.param_shift(circuit, argnums=[0, 1])(x, y)
        res_expected = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_multiple_params(self, interface):
        """Test for a single measurement (probs) and multiple params."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = qml.gradients.param_shift(circuit, argnums=[0, 1])(x, y)
        res_expected = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multiple_measurements_multi_params(self, interface, tol):
        """Test for multiple measurements and multiple params."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = qml.gradients.param_shift(circuit, argnums=[0, 1])(x, y)

        res_expected = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        assert np.allclose(res[0][0], res_expected[0][0], atol=tol)
        assert np.allclose(res[0][1], res_expected[0][1], atol=tol)
        assert np.allclose(res[1][0], res_expected[1][0], atol=tol)
        assert np.allclose(res[1][1], res_expected[1][1], atol=tol)

    @pytest.mark.parametrize("interface", interfaces)
    def test_multiple_measurements_expval_probs_multi_params(self, interface, tol):
        """Test for multiple measurements (with shape) and multiple params."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = qml.gradients.param_shift(circuit, argnums=[0, 1])(x, y)

        res_expected = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        assert np.allclose(res[0][0], res_expected[0][0], atol=tol)
        assert np.allclose(res[0][1], res_expected[0][1], atol=tol)
        assert np.allclose(res[1][0], res_expected[1][0], atol=tol)
        assert np.allclose(res[1][1], res_expected[1][1], atol=tol)

    @pytest.mark.parametrize("interface", interfaces)
    def test_identity_classical_jacobian(self, interface, tol):
        """Test for an identity cjac."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = jax.numpy.array([0.543, -0.654])
        res = qml.gradients.param_shift(circuit)(x)
        res_expected = jax.jacobian(circuit)(x)

        assert np.allclose(res[0], res_expected[0], atol=tol)
        assert np.allclose(res[1], res_expected[1], atol=tol)


@pytest.mark.jax
class TestQnodeJaxJit:
    """Class to the parameter shift transform with some classical processing."""

    expected_jacs = []
    interfaces = ["auto", "jax-jit"]

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_single_param(self, interface):
        """Test for a single measurement and a single param."""
        dev = qml.device("default.qubit", wires=2)

        import jax

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x):
            qml.RX(2 * x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = jax.numpy.array(0.543)

        res = jax.jit(qml.gradients.param_shift(circuit))(x)

        res_expected = jax.jacobian(circuit)(x)

        assert res.shape == res_expected.shape
        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_single_param_2(self, interface):
        """Test for a single measurement and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = jax.numpy.array([0.543, 0.2])

        res = qml.gradients.param_shift(circuit)(x)

        res_expected = jax.jacobian(circuit)(x)

        assert res.shape == res_expected.shape
        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_single_param(self, interface):
        """Test for a single measurement (probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x):
            qml.RX(2 * x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = jax.numpy.array(0.543)

        res = jax.jit(qml.gradients.param_shift(circuit))(x)

        res_expected = jax.jacobian(circuit)(x)

        assert res.shape == res_expected.shape
        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_single_param_2(self, interface):
        """Test for a single measurement (probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)

        import jax

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = jax.numpy.array([0.543, 0.2])

        res = jax.jit(qml.gradients.param_shift(circuit))(x)
        res_expected = jax.jacobian(circuit)(x)

        assert np.allclose(res, res_expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_measurement_single_param(self, interface):
        """Test for multiple measurements and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])

        res = jax.jit(qml.gradients.param_shift(circuit))(x)

        res_expected = jax.jacobian(circuit)(x)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multi_measurement_expval_probs_single_param(self, interface):
        """Test for multiple measurement (expval,probs) and a single param."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        x = jax.numpy.array([0.543, 0.2])

        res = jax.jit(qml.gradients.param_shift(circuit))(x)

        res_expected = jax.jacobian(circuit)(x)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_multiple_params(self, interface):
        """Test for a single measurement and multiple params."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = jax.jit(qml.gradients.param_shift(circuit, argnums=[0, 1]))(x, y)
        res_expected = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_single_measurement_probs_multiple_params(self, interface):
        """Test for a single measurement (probs) and multiple params."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = jax.jit(qml.gradients.param_shift(circuit, argnums=[0, 1]))(x, y)
        res_expected = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        assert np.allclose(res[0], res_expected[0])
        assert np.allclose(res[1], res_expected[1])

    @pytest.mark.parametrize("interface", interfaces)
    def test_multiple_measurement_multi_params(self, interface, tol):
        """Test for multiple measurements and multiple params."""
        dev = qml.device("default.qubit", wires=2)
        import jax

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = jax.jit(qml.gradients.param_shift(circuit, argnums=[0, 1]))(x, y)

        res_expected = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        assert np.allclose(res[0][0], res_expected[0][0], atol=tol)
        assert np.allclose(res[0][1], res_expected[0][1], atol=tol)
        assert np.allclose(res[1][0], res_expected[1][0], atol=tol)
        assert np.allclose(res[1][1], res_expected[1][1], atol=tol)

    @pytest.mark.parametrize("interface", interfaces)
    def test_multiple_measurements_expval_probs_multi_params(self, interface, tol):
        """Test for multiple measurements (expval, probs) and multiple params."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = jax.jit(qml.gradients.param_shift(circuit, argnums=[0, 1]))(x, y)

        res_expected = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        assert np.allclose(res[0][0], res_expected[0][0], atol=tol)
        assert np.allclose(res[0][1], res_expected[0][1], atol=tol)
        assert np.allclose(res[1][0], res_expected[1][0], atol=tol)
        assert np.allclose(res[1][1], res_expected[1][1], atol=tol)

    @pytest.mark.parametrize("interface", interfaces)
    def test_identity_classical_jacobian(self, interface, tol):
        """Test for an identity cjac."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = jax.numpy.array([0.543, -0.654])
        res = jax.jit(qml.gradients.param_shift(circuit))(x)
        res_expected = jax.jacobian(circuit)(x)

        assert np.allclose(res[0], res_expected[0], atol=tol)
        assert np.allclose(res[1], res_expected[1], atol=tol)

    @pytest.mark.parametrize("interface", interfaces)
    def test_identity_classical_jacobian_multi_meas(self, interface, tol):
        """Test for an identity cjac with qjac multiple measurements."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface, cache=False)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        x = jax.numpy.array([0.543, -0.654])
        res = jax.jit(qml.gradients.param_shift(circuit))(x)
        res_expected = jax.jacobian(circuit)(x)

        assert np.allclose(res[0], res_expected[0], atol=tol)
        assert np.allclose(res[1], res_expected[1], atol=tol)


@pytest.mark.parametrize("argnums", [[0], [1], [0, 1]])
@pytest.mark.parametrize("interface", ["jax"])
@pytest.mark.jax
class TestJaxArgnums:
    """Class to test the integration of argnums (Jax) and the parameter shift transform."""

    expected_jacs = []
    interfaces = ["auto", "jax"]

    def test_argnum_error(self, argnums, interface):
        """Test that giving argnum to Jax, raises an error."""
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

        with pytest.raises(
            qml.QuantumFunctionError,
            match="argnum does not work with the Jax interface. You should use argnums instead.",
        ):
            qml.gradients.hadamard_grad(circuit, argnum=argnums)(x, y)

    def test_single_expectation_value(self, argnums, interface):
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

        res = qml.gradients.param_shift(circuit, argnums=argnums)(x, y)

        expected_0 = np.array([-np.sin(y) * np.sin(x[0]), 0])
        expected_1 = np.array(np.cos(y) * np.cos(x[0]))

        if argnums == [0]:
            assert np.allclose(res, expected_0)
        if argnums == [1]:
            assert np.allclose(res, expected_1)
        if argnums == [0, 1]:
            assert np.allclose(res[0], expected_0)
            assert np.allclose(res[1], expected_1)

    def test_multi_expectation_values(self, argnums, interface):
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

        res = qml.gradients.param_shift(circuit, argnums=argnums)(x, y)

        expected_0 = np.array([[-np.sin(x[0]), 0.0], [0.0, 0.0]])
        expected_1 = np.array([0, np.cos(y)])

        if argnums == [0]:
            assert np.allclose(res[0], expected_0[0])
            assert np.allclose(res[1], expected_0[1])
        if argnums == [1]:
            assert np.allclose(res, expected_1)
        if argnums == [0, 1]:
            assert np.allclose(res[0][0], expected_0[0])
            assert np.allclose(res[0][1], expected_0[1])
            assert np.allclose(res[1][0], expected_1[0])
            assert np.allclose(res[1][1], expected_1[1])

    def test_hessian(self, argnums, interface):
        """Test for hessian."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, -0.654])
        y = jax.numpy.array(-0.123)

        res = jax.jacobian(qml.gradients.param_shift(circuit, argnums=argnums), argnums=argnums)(x, y)
        res_expected = jax.hessian(circuit, argnums=argnums)(x, y)

        if argnums == [0]:
            assert np.allclose(res[0][0], res_expected[0][0][0])
            assert np.allclose(res[1][0], res_expected[0][0][1])
        else:
            if len(argnums) != 1:
                res = res[0]

            for r, r_e in zip(res, res_expected[0]):
                assert np.allclose(r, r_e)

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

# pylint: disable=use-implicit-booleaness-not-comparison,abstract-method
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.gradients import param_shift
from pennylane.gradients.parameter_shift import (
    _evaluate_gradient,
    _get_operation_recipe,
    _make_zero_rep,
    _put_zeros_in_pdA2_involutory,
)
from pennylane.measurements.shots import Shots

# Constants for TestEvaluateGradient
# Coefficients and expectation values
X = np.arange(1, 5)
# Expected "shift rule" result
Z = np.sum(-np.arange(1, 5) ** 2)
# Single coefficient/expectation value that leads to the same result as X
w = np.sqrt(30)
# Prefactors to emulate a shot vector
shv = np.array([0.1, 0.4, 0.7])
# Fake probability vector (just a 1d array)
p = np.array([0.01, 0.06, -0.2, 0.5, -0.1, 0.7, -0.09])
# Second fake probability vector (just a 1d array)
p2 = p[1:5]
# shifted probability evaluations
P = np.outer(X, p)
# shifted probability evaluations for p2
P2 = np.outer(X, p2)
# Single unshifted result that lead to the same result as P
v = w * p
# Single unshifted result that lead to the same result as P2
v2 = w * p2
# Prefactors to emulate different shot values and multi measurement
shv_m = np.outer([0.1, 0.4, 0.7], [1, 2])


class TestEvaluateGradient:
    """Test _evaluate_gradient."""

    # pylint: disable=too-many-arguments

    # We could theoretically compute the required res, r0 and expected from the parametrization of coeffs,
    # unshifted_coeff and batch_size, but that turned out to take lots of effort and edge case logic

    test_cases_single_shots_single_meas = [
        # Expectation value
        (X, None, None, tuple(-X), None, Z),
        (X, None, 4, -X, None, Z),
        (X[:-1], X[-1], None, tuple(-X[:-1]), -X[-1], Z),
        (X[:-1], X[-1], 4, -X[:-1], -X[-1], Z),
        (np.ones(0), w, None, (), -w, Z),
        (np.ones(0), w, 4, (), -w, Z),
        # Probability
        (X, None, None, tuple(-P), None, p * Z),
        (X, None, 4, -P, None, p * Z),
        (X[:-1], X[-1], None, tuple(-P[:-1]), -P[-1], p * Z),
        (X[:-1], X[-1], 4, -P[:-1], -P[-1], p * Z),
        (np.ones(0), w, None, (), -v, p * Z),
        (np.ones(0), w, 4, (), -v, p * Z),
    ]

    @pytest.mark.parametrize(
        "coeffs, unshifted_coeff, batch_size, res, r0, expected",
        test_cases_single_shots_single_meas,
    )
    def test_single_shots_single_meas(self, coeffs, unshifted_coeff, batch_size, res, r0, expected):
        """Test that a single shots, single measurement gradient is evaluated correctly."""

        shots = Shots(100)
        tape_specs = (None, None, 1, shots)
        data = [None, coeffs, None, unshifted_coeff, None]
        grad = _evaluate_gradient(tape_specs, res, data, r0, batch_size)

        assert isinstance(grad, np.ndarray)
        assert grad.shape == expected.shape
        assert np.allclose(grad, expected)

    exp_probs = (p2 * Z, 2 * p * Z)
    test_cases_single_shots_multi_meas = [
        # Expectation values
        (X, None, None, tuple(zip(-X, -2 * X)), None, (Z, 2 * Z)),
        (X, None, 4, (-X, -2 * X), None, (Z, 2 * Z)),
        (X[:-1], X[-1], None, tuple(zip(-X[:-1], -2 * X[:-1])), (-X[-1], -2 * X[-1]), (Z, 2 * Z)),
        (X[:-1], X[-1], 4, (-X[:-1], -2 * X[:-1]), (-X[-1], -2 * X[-1]), (Z, 2 * Z)),
        (np.ones(0), w, None, (), (-w, -2 * w), (Z, 2 * Z)),
        (np.ones(0), w, 4, (), (-w, -2 * w), (Z, 2 * Z)),
        # Expval and Probability
        (X, None, None, tuple(zip(-X, -2 * P)), None, (Z, 2 * p * Z)),
        (X, None, 4, (-X, -2 * P), None, (Z, 2 * p * Z)),
        (X[:-1], X[-1], None, tuple(zip(-X, -2 * P))[:-1], (-X[-1], -2 * P[-1]), (Z, 2 * p * Z)),
        (X[:-1], X[-1], 4, (-X[:-1], -2 * P[:-1]), (-X[-1], -2 * P[-1]), (Z, 2 * p * Z)),
        (np.ones(0), w, None, (), (-w, -2 * v), (Z, 2 * p * Z)),
        (np.ones(0), w, 4, (), (-w, -2 * v), (Z, 2 * p * Z)),
        # Probabilities
        (X, None, None, tuple(zip(-P2, -2 * P)), None, exp_probs),
        (X, None, 4, (-P2, -2 * P), None, exp_probs),
        (X[:-1], X[-1], None, tuple(zip(-P2, -2 * P))[:-1], (-P2[-1], -2 * P[-1]), exp_probs),
        (X[:-1], X[-1], 4, (-P2[:-1], -2 * P[:-1]), (-P2[-1], -2 * P[-1]), exp_probs),
        (np.ones(0), w, None, (), (-v2, -2 * v), exp_probs),
        (np.ones(0), w, 4, (), (-v2, -2 * v), exp_probs),
    ]

    @pytest.mark.parametrize(
        "coeffs, unshifted_coeff, batch_size, res, r0, expected",
        test_cases_single_shots_multi_meas,
    )
    def test_single_shots_multi_meas(self, coeffs, unshifted_coeff, batch_size, res, r0, expected):
        """Test that a single shots, multiple measurements gradient is evaluated correctly."""

        shots = Shots(100)
        tape_specs = (None, None, 2, shots)
        data = [None, coeffs, None, unshifted_coeff, None]
        grad = _evaluate_gradient(tape_specs, res, data, r0, batch_size)

        assert isinstance(grad, tuple) and len(grad) == 2
        for g, e in zip(grad, expected):
            assert isinstance(g, np.ndarray) and g.shape == e.shape
            assert np.allclose(g, e)

    shot_vec_X = tuple(zip(*(-c * X for c in shv)))
    shot_vec_P = tuple(zip(*(-c * P for c in shv)))
    shot_vec_P_partial = tuple(-c * P[:-1] for c in shv)

    exp_shot_vec_prob = np.outer(shv, p) * Z
    test_cases_multi_shots_single_meas = [
        # Expectation value
        (X, None, None, shot_vec_X, None, shv * Z),
        (X, None, 4, tuple(-c * X for c in shv), None, shv * Z),
        (X[:-1], X[-1], None, shot_vec_X[:-1], shot_vec_X[-1], shv * Z),
        (X[:-1], X[-1], 4, tuple(-c * X[:-1] for c in shv), tuple(-shv * X[-1]), shv * Z),
        (np.ones(0), w, None, (), tuple(-c * w for c in shv), shv * Z),
        (np.ones(0), w, 4, ((), (), ()), tuple(-c * w for c in shv), shv * Z),
        # Probability
        (X, None, None, shot_vec_P, None, exp_shot_vec_prob),
        (X, None, 4, tuple(-c * P for c in shv), None, exp_shot_vec_prob),
        (X[:-1], X[-1], None, shot_vec_P[:-1], shot_vec_P[-1], exp_shot_vec_prob),
        (X[:-1], X[-1], 4, shot_vec_P_partial, tuple(np.outer(-shv, P[-1])), exp_shot_vec_prob),
        (np.ones(0), w, None, (), tuple(-c * v for c in shv), exp_shot_vec_prob),
        (np.ones(0), w, 4, ((), (), ()), tuple(-c * v for c in shv), exp_shot_vec_prob),
    ]

    @pytest.mark.parametrize(
        "coeffs, unshifted_coeff, batch_size, res, r0, expected",
        test_cases_multi_shots_single_meas,
    )
    def test_multi_shots_single_meas(self, coeffs, unshifted_coeff, batch_size, res, r0, expected):
        """Test that a shot vector, single measurements gradient is evaluated correctly."""

        shots = Shots((100, 101, 102))
        tape_specs = (None, None, 1, shots)
        data = [None, coeffs, None, unshifted_coeff, None]
        grad = _evaluate_gradient(tape_specs, res, data, r0, batch_size)

        assert isinstance(grad, tuple) and len(grad) == 3
        for g, e in zip(grad, expected):
            assert isinstance(g, np.ndarray) and g.shape == e.shape
            assert np.allclose(g, e)

    multi_X = tuple(tuple((-c * x, -2 * c * x) for c in shv) for x in X)
    batched_multi_X = tuple((-c * X, -2 * c * X) for c in shv)
    partial_multi_X = tuple((-c * X[:-1], -2 * c * X[:-1]) for c in shv)
    expvals_r0 = tuple((-c * w, -2 * c * w) for c in shv)

    multi_X_P = tuple(tuple((-c * _p, -2 * c * x) for c in shv) for x, _p in zip(X, P))
    batched_multi_X_P = tuple((-c * P, -2 * c * X) for c in shv)
    partial_multi_X_P = tuple((-c * P[:-1], -2 * c * X[:-1]) for c in shv)
    prob_expval_r0 = tuple((-c * v, -2 * c * w) for c in shv)

    multi_P_P = tuple(tuple((-c * _p, -2 * c * _q) for c in shv) for _q, _p in zip(P2, P))
    batched_multi_P_P = tuple((-c * P, -2 * c * P2) for c in shv)
    partial_multi_P_P = tuple((-c * P[:-1], -2 * c * P2[:-1]) for c in shv)
    probs_r0 = tuple((-c * v, -2 * c * v2) for c in shv)

    exp_shot_vec_prob_expval = tuple((c * p * Z, 2 * c * Z) for c in shv)
    exp_shot_vec_probs = tuple((c * p * Z, 2 * c * p2 * Z) for c in shv)
    test_cases_multi_shots_multi_meas = [
        # Expectation values
        (X, None, None, multi_X, None, shv_m * Z),
        (X, None, 4, batched_multi_X, None, shv_m * Z),
        (X[:-1], X[-1], None, multi_X[:-1], multi_X[-1], shv_m * Z),
        (X[:-1], X[-1], 4, partial_multi_X, multi_X[-1], shv_m * Z),
        (np.ones(0), w, None, (), expvals_r0, shv_m * Z),
        (np.ones(0), w, 4, ((), (), ()), expvals_r0, shv_m * Z),
        # Probability and expectation
        (X, None, None, multi_X_P, None, exp_shot_vec_prob_expval),
        (X, None, 4, batched_multi_X_P, None, exp_shot_vec_prob_expval),
        (X[:-1], X[-1], None, multi_X_P[:-1], multi_X_P[-1], exp_shot_vec_prob_expval),
        (X[:-1], X[-1], 4, partial_multi_X_P, multi_X_P[-1], exp_shot_vec_prob_expval),
        (np.ones(0), w, None, (), prob_expval_r0, exp_shot_vec_prob_expval),
        (np.ones(0), w, 4, ((), (), ()), prob_expval_r0, exp_shot_vec_prob_expval),
        # Probabilities
        (X, None, None, multi_P_P, None, exp_shot_vec_probs),
        (X, None, 4, batched_multi_P_P, None, exp_shot_vec_probs),
        (X[:-1], X[-1], None, multi_P_P[:-1], multi_P_P[-1], exp_shot_vec_probs),
        (X[:-1], X[-1], 4, partial_multi_P_P, multi_P_P[-1], exp_shot_vec_probs),
        (np.ones(0), w, None, (), probs_r0, exp_shot_vec_probs),
        (np.ones(0), w, 4, ((), (), ()), probs_r0, exp_shot_vec_probs),
    ]

    @pytest.mark.parametrize(
        "coeffs, unshifted_coeff, batch_size, res, r0, expected",
        test_cases_multi_shots_multi_meas,
    )
    def test_multi_shots_multi_meas(self, coeffs, unshifted_coeff, batch_size, res, r0, expected):
        """Test that a shot vector, multiple measurements gradient is evaluated correctly."""

        shots = Shots((100, 101, 102))
        tape_specs = (None, None, 2, shots)
        data = [None, coeffs, None, unshifted_coeff, None]
        grad = _evaluate_gradient(tape_specs, res, data, r0, batch_size)

        assert isinstance(grad, tuple) and len(grad) == 3
        for g, e in zip(grad, expected):
            assert isinstance(g, tuple) and len(g) == 2
            for _g, _e in zip(g, e):
                assert isinstance(_g, np.ndarray) and _g.shape == _e.shape
                assert np.allclose(_g, _e)


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
    return fn(dev.execute(tapes))


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

    def test_parameter_shift_non_commuting_observables(self):
        """Test that parameter shift works even if the measurements do not commute with each other."""

        ops = (qml.RX(0.5, wires=0),)
        ms = (qml.expval(qml.X(0)), qml.expval(qml.Z(0)))
        tape = qml.tape.QuantumScript(ops, ms, trainable_params=[0])

        batch, _ = qml.gradients.param_shift(tape)
        assert len(batch) == 2
        tape0 = qml.tape.QuantumScript((qml.RX(0.5 + np.pi / 2, 0),), ms, trainable_params=[0])
        tape1 = qml.tape.QuantumScript((qml.RX(0.5 - np.pi / 2, 0),), ms, trainable_params=[0])
        qml.assert_equal(batch[0], tape0)
        qml.assert_equal(batch[1], tape1)

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

        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, [0])

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

    @pytest.mark.parametrize("broadcast", [True, False])
    def test_all_zero_diff_methods(self, broadcast):
        """Test that the transform works correctly when the diff method for every
        parameter is identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qml.gradients.param_shift(circuit)(params)
        assert np.allclose(result, np.zeros((4, 3)), atol=0)

        tape = qml.workflow.construct_tape(circuit)(params)
        tapes, _ = qml.gradients.param_shift(tape, broadcast=broadcast)
        assert tapes == []

    @pytest.mark.parametrize("broadcast", [True, False])
    def test_with_gradient_recipes(self, broadcast):
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
        tapes, _ = param_shift(tape, gradient_recipes=gradient_recipes, broadcast=broadcast)

        if broadcast:
            assert len(tapes) == 2
            assert [t.batch_size for t in tapes] == [2, 3]

            shifted_batch = [0.2 * 1.0 + 0.3, 0.5 * 1.0 + 0.6]
            tape_par = tapes[0].get_parameters(trainable_only=False)
            assert np.allclose(tape_par[0], shifted_batch)
            assert tape_par[1:] == [2.0, 3.0, 4.0]

            shifted_batch = [1 * 3.0 + 1, 2 * 3.0 + 2, 3 * 3.0 + 3]
            tape_par = tapes[1].get_parameters(trainable_only=False)
            assert tape_par[:2] == [1.0, 2.0]
            assert np.allclose(tape_par[2], shifted_batch)
            assert tape_par[3:] == [4.0]
        else:
            assert len(tapes) == 5
            assert [t.batch_size for t in tapes] == [None] * 5
            assert tapes[0].get_parameters(trainable_only=False) == [0.2 * 1.0 + 0.3, 2.0, 3.0, 4.0]
            assert tapes[1].get_parameters(trainable_only=False) == [0.5 * 1.0 + 0.6, 2.0, 3.0, 4.0]
            assert tapes[2].get_parameters(trainable_only=False) == [1.0, 2.0, 1 * 3.0 + 1, 4.0]
            assert tapes[3].get_parameters(trainable_only=False) == [1.0, 2.0, 2 * 3.0 + 2, 4.0]
            assert tapes[4].get_parameters(trainable_only=False) == [1.0, 2.0, 3 * 3.0 + 3, 4.0]

    @pytest.mark.parametrize("broadcast", [True, False])
    @pytest.mark.parametrize("ops_with_custom_recipe", [[0], [1], [0, 1]])
    def test_recycled_unshifted_tape(self, ops_with_custom_recipe, broadcast):
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
        tapes, fn = param_shift(tape, gradient_recipes=gradient_recipes, broadcast=broadcast)

        # two (one with broadcast) tapes per parameter that doesn't use a custom recipe,
        # one tape per parameter that uses custom recipe,
        # plus one global call if at least one uses the custom recipe
        num_custom = len(ops_with_custom_recipe)
        num_ops_standard_recipe = tape.num_params - num_custom
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == tapes_per_param * num_ops_standard_recipe + num_custom + 1
        # Test that executing the tapes and the postprocessing function works
        grad = fn(qml.execute(tapes, dev, None))
        assert qml.math.allclose(grad, -np.sin(x[0] + x[1]), atol=1e-5)

    @pytest.mark.parametrize("broadcast", [False, True])
    @pytest.mark.parametrize("ops_with_custom_recipe", [[0], [1], [0, 1]])
    @pytest.mark.parametrize("multi_measure", [False, True])
    def test_custom_recipe_unshifted_only(self, ops_with_custom_recipe, multi_measure, broadcast):
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
        tapes, fn = param_shift(tape, gradient_recipes=gradient_recipes, broadcast=broadcast)

        # two (one with broadcast) tapes per parameter that doesn't use a custom recipe,
        # plus one global (unshifted) call if at least one uses the custom recipe
        num_ops_standard_recipe = tape.num_params - len(ops_with_custom_recipe)
        tapes_per_param = 1 if broadcast else 2
        assert len(tapes) == tapes_per_param * num_ops_standard_recipe + int(
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

    @pytest.mark.parametrize("broadcast", [False, True])
    @pytest.mark.parametrize("ops_with_custom_recipe", [[0], [1], [0, 1]])
    def test_custom_recipe_mixing_unshifted_shifted(self, ops_with_custom_recipe, broadcast):
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
        custom_recipe = [[-1e-7, 1, 0], [1e-7, 1, 0], [-1e5, 1, -5e-6], [1e5, 1, 5e-6]]
        gradient_recipes = tuple(
            custom_recipe if i in ops_with_custom_recipe else None for i in range(2)
        )
        tapes, fn = param_shift(tape, gradient_recipes=gradient_recipes, broadcast=broadcast)

        # two tapes per parameter, independent of recipe
        # plus one global (unshifted) call if at least one uses the custom recipe
        tapes_per_param = 1 if broadcast else 2
        num_custom = len(ops_with_custom_recipe)
        assert len(tapes) == tapes_per_param * tape.num_params + (num_custom > 0)

        # Test that executing the tapes and the postprocessing function works
        grad = fn(qml.execute(tapes, dev, None))
        assert qml.math.allclose(grad[0], -np.sin(x[0] + x[1]), atol=1e-5)
        assert qml.math.allclose(grad[1], 0, atol=1e-5)

    @pytest.mark.parametrize("broadcast", [True, False])
    @pytest.mark.parametrize("y_wire", [0, 1])
    def test_f0_provided(self, y_wire, broadcast):
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
        tapes, fn = param_shift(tape, gradient_recipes=gradient_recipes, f0=f0, broadcast=broadcast)

        # one tape per parameter that impacts the expval
        assert len(tapes) == 2 if y_wire == 0 else 1

        fn(dev.execute(tapes))

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

        grad = fn(dev.execute(tapes))
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
        with qml.Tracker(dev) as tracker:
            j1 = fn(dev.execute(tapes))

        # We should only be executing the device twice: Two shifted evaluations to differentiate
        # one parameter overall, as the other parameter does not impact the returned measurement.

        assert tracker.totals["executions"] == 2

        tapes, fn = qml.gradients.param_shift(tape2)
        j2 = fn(dev.execute(tapes))

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

        grad = fn(dev.execute(tapes))
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
                qml.exceptions.OperatorPropertyUndefined, match="does not have a grad_recipe"
            ):
                qml.gradients.param_shift(tape)


# Remove the following and unskip the class below once broadcasted
# tapes are fully supported with gradient transforms. See #4462 for details.
class TestParamShiftRaisesWithBroadcasted:
    """Test that an error is raised with broadcasted tapes."""

    def test_batched_tape_raises(self):
        """Test that an error is raised for a broadcasted/batched tape if the broadcasted
        parameter is differentiated."""
        tape = qml.tape.QuantumScript([qml.RX([0.4, 0.2], 0)], [qml.expval(qml.PauliZ(0))])
        _match = r"Computing the gradient of broadcasted tapes .* using the parameter-shift rule"
        with pytest.raises(NotImplementedError, match=_match):
            qml.gradients.param_shift(tape)


class TestParamShiftWithBroadcasted:
    """Tests for the `param_shift` transform on already broadcasted tapes.
    The tests for `param_shift` using broadcasting itself can be found
    further below."""

    # Revert the following skip once broadcasted tapes are fully supported with gradient transforms.
    # See #4462 for details.
    @pytest.mark.skip(reason="Applying gradient transforms to broadcasted tapes is disallowed")
    @pytest.mark.parametrize("dim", [1, 3])
    @pytest.mark.parametrize("pos", [0, 1])
    def test_with_single_trainable_parameter_broadcasted(self, dim, pos):
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
        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == (dim,)
        assert res[1].shape == (dim,)

    # Revert the following skip once broadcasted tapes are fully supported with gradient transforms.
    # See #4462 for details.
    @pytest.mark.skip(reason="Applying gradient transforms to broadcasted tapes is disallowed")
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
        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 3

        assert res[0].shape == res[1].shape == res[2].shape == (dim,)

    @pytest.mark.parametrize("dim", [1, 3])
    @pytest.mark.parametrize("pos", [0, 1])
    def test_with_single_nontrainable_parameter_broadcasted(self, dim, pos):
        """Test that the parameter-shift transform works with a tape that has
        one of its nontrainable parameters broadcasted."""
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
        tape.trainable_params = [1 - pos]
        assert tape.batch_size == dim
        tapes, fn = qml.gradients.param_shift(tape, argnum=[0])
        assert len(tapes) == 2
        assert np.allclose([t.batch_size for t in tapes], dim)

        dev = qml.device("default.qubit", wires=2)
        res = fn(dev.execute(tapes))
        assert res.shape == (dim,)


class TestParamShiftUsingBroadcasting:
    """Tests for the `param_shift` function using broadcasting.
    The tests for `param_shift` on already broadcasted tapes can be found above."""

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped during the Jacobian computation."""
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

        res = fn(dev.execute(tapes))
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
        tapes, fn = qml.gradients.param_shift(tape1, broadcast=True)
        with qml.Tracker(dev) as tracker:
            j1 = fn(dev.execute(tapes))

        # We should only be executing the device to differentiate 1 parameter
        # (1 broadcasted execution)

        assert tracker.totals["executions"] == 2
        assert tracker.totals["simulations"] == 1

        tapes, fn = qml.gradients.param_shift(tape2, broadcast=True)
        j2 = fn(dev.execute(tapes))

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

        grad = fn(dev.execute(tapes))
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

        autograd_val = fn(dev.execute(tapes))

        tape_fwd = tape.bind_new_parameters([theta + np.pi / 2], [1])
        tape_bwd = tape.bind_new_parameters([theta - np.pi / 2], [1])

        manualgrad_val = np.subtract(*dev.execute([tape_fwd, tape_bwd])) / 2
        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)

        assert isinstance(autograd_val, np.ndarray)
        assert autograd_val.shape == ()

        assert spy.call_args[1]["shifts"] == (shift,)

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.execute(tapes))
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    def test_Rot_gradient(self, mocker, theta, shift, tol):
        """Tests that the automatic gradient of an arbitrary Euler-angle-parametrized gate is correct."""
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

        autograd_val = fn(dev.execute(tapes))
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
        numeric_val = fn(dev.execute(tapes))
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
        grad = fn(dev.execute(tapes))
        expected = np.sin(b / 2) / 2
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.execute(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parametrized
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

        grad = fn(dev.execute(tapes))
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
        numeric_val = fn(dev.execute(tapes))
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

            return fn(dev.execute(tapes))

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
        dev = qml.device("default.qubit", wires=2)
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

            return fn(dev.execute(tapes))

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
        dev = qml.device("default.qubit", wires=2)
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

            return fn(dev.execute(tapes))

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

        dev = qml.device("default.qubit", wires=2)
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

        res = fn(dev.execute(tapes))

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

        res = fn(dev.execute(tapes))
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
        grad_PS = fn(qml.execute(gtapes, dev, diff_method=None))

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

        res = fn(dev.execute(tapes))
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

        res = fn(dev.execute(tapes))
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

        res = fn(dev.execute(tapes))
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
        gradA = fn(dev.execute(tapes))
        assert isinstance(gradA, np.ndarray)
        assert gradA.shape == ()
        assert len(tapes) == 1 + 2 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))
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
        gradA = fn(dev.execute(tapes))
        assert isinstance(gradA, tuple)

        assert isinstance(gradA[0], np.ndarray)
        assert gradA[0].shape == ()

        assert isinstance(gradA[1], np.ndarray)
        assert gradA[1].shape == ()

        assert len(tapes) == 1 + 2 * 2

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))
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
        gradA = fn(dev.execute(tapes))
        assert isinstance(gradA, np.ndarray)
        assert gradA.shape == ()
        assert len(tapes) == 1 + 4 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))
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
        gradA = fn(dev.execute(tapes))
        assert isinstance(gradA, tuple)

        assert isinstance(gradA[0], np.ndarray)
        assert gradA[0].shape == ()

        assert isinstance(gradA[1], np.ndarray)
        assert gradA[1].shape == ()
        assert len(tapes) == 1 + 4 * 2

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))
        assert len(tapes) == 3

        expected = -35 * np.sin(2 * (a + b)) - 12 * np.cos(2 * (a + b))
        assert gradA[0] == pytest.approx(expected, abs=tol)
        assert gradF[0] == pytest.approx(expected, abs=tol)

        assert gradA[1] == pytest.approx(expected, abs=tol)
        assert gradF[1] == pytest.approx(expected, abs=tol)

    def test_involutory_and_noninvolutory_variance_single_param(self, tol, seed):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable when there's a single trainable parameter."""
        dev = qml.device("default.qubit", wires=2, seed=seed)
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
        gradA = fn(dev.execute(tapes))
        assert len(tapes) == 1 + 4

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))
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

        gradA = fn(dev.execute(tapes))

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
        gradA = fn(dev.execute(tapes))

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
        gradA = fn(dev.execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))

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
        gradA = fn(dev.execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))

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
        gradA = fn(dev.execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))

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
        qml.Rot(x[0], 0.3 * x[1], x[2], wires=0)
        return qml.expval(qml.PauliZ(0))

    def cost2(x):
        """Perform rotation and return an expectation value in a 1d array."""
        qml.Rot(*x, wires=0)
        return [qml.expval(qml.PauliZ(0))]

    def cost3(x):
        """Perform rotation and return two expectation value in a 1d array."""
        qml.Rot(*x, wires=0)
        return (qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)))

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
        return (qml.probs([0, 1]), qml.probs([2, 3]))

    def cost7(x):
        """Perform rotation and return a scalar expectation value."""
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    def cost8(x):
        """Perform rotation and return an expectation value in a 1d array."""
        qml.RX(x, wires=0)
        return [qml.expval(qml.PauliZ(0))]

    def cost9(x):
        """Perform rotation and return two expectation value in a 1d array."""
        qml.RX(x, wires=0)
        return (qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)))

    costs_and_expected_expval_scalar = [
        (cost7, (), np.ndarray),
        (cost8, (1,), list),
        (cost9, (2,), tuple),
    ]

    @pytest.mark.parametrize("cost, exp_shape, exp_type", costs_and_expected_expval_scalar)
    def test_output_shape_matches_qnode_expval_scalar(self, cost, exp_shape, exp_type):
        """Test that the transform output shape matches that of the QNode for
        expectation values and a scalar parameter."""
        dev = qml.device("default.qubit", wires=4)

        x = np.array(0.419)
        circuit = qml.QNode(cost, dev)

        res_parshift = qml.gradients.param_shift(circuit)(x)

        assert isinstance(res_parshift, exp_type)
        assert np.array(res_parshift).shape == exp_shape

    costs_and_expected_expval = [
        (cost1, [3], np.ndarray),
        (cost2, [3], list),
        (cost3, [2, 3], tuple),
    ]

    @pytest.mark.parametrize("cost, exp_shape, exp_type", costs_and_expected_expval)
    def test_output_shape_matches_qnode_expval_array(self, cost, exp_shape, exp_type):
        """Test that the transform output shape matches that of the QNode for
        expectation values and an array-valued parameter."""
        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res = qml.gradients.param_shift(circuit)(x)

        assert isinstance(res, exp_type)
        if len(res) == 1:
            res = res[0]
        assert len(res) == exp_shape[0]

        if len(exp_shape) > 1:
            for r in res:
                assert isinstance(r, np.ndarray)
                assert len(r) == exp_shape[1]

    costs_and_expected_probs = [
        (cost4, [4, 3], np.ndarray),
        (cost5, [4, 3], list),
        (cost6, [2, 4, 3], tuple),
    ]

    @pytest.mark.parametrize("cost, exp_shape, exp_type", costs_and_expected_probs)
    def test_output_shape_matches_qnode_probs(self, cost, exp_shape, exp_type):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res_parshift = qml.gradients.param_shift(circuit)(x)

        # Check data types
        assert isinstance(res_parshift, exp_type)
        if len(exp_shape) > 1:
            for r in res_parshift:
                assert isinstance(r, np.ndarray)

        # Check shape, result can be put into a single array by assumption
        assert np.allclose(np.squeeze(np.array(res_parshift)).shape, exp_shape)

    # TODO: revisit the following test when the Autograd interface supports
    #       parameter-shift with the new return type system
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
        class SpecialObservable(qml.operation.Operator):
            """SpecialObservable"""

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
            fn(dev.execute(tapes))

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

        autograd_val = fn(dev.execute(tapes))

        tape_fwd = tape.bind_new_parameters([theta + np.pi / 2], [1])
        tape_bwd = tape.bind_new_parameters([theta - np.pi / 2], [1])

        manualgrad_val = np.subtract(*dev.execute([tape_fwd, tape_bwd])) / 2
        assert np.allclose(autograd_val, manualgrad_val, atol=tol, rtol=0)

        assert spy.call_args[1]["shifts"] == (shift,)

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.execute(tapes))
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("shift", [np.pi / 2, 0.3, np.sqrt(2)])
    def test_Rot_gradient(self, mocker, theta, shift, tol):
        """Tests that the automatic gradient of an arbitrary Euler-angle-parametrized gate is correct."""
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

        autograd_val = fn(dev.execute(tapes))
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
        numeric_val = fn(dev.execute(tapes))

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
        grad = fn(dev.execute(tapes))
        expected = np.sin(b / 2) / 2
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.finite_diff(tape)
        numeric_val = fn(dev.execute(tapes))
        assert np.allclose(grad, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parametrized
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

        grad = fn(dev.execute(tapes))
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
        numeric_val = fn(dev.execute(tapes))
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

    @pytest.mark.jax
    def test_fallback(self, mocker, tol):
        """Test that fallback gradient functions are correctly used"""

        import jax
        from jax import numpy as jnp

        spy = mocker.spy(qml.gradients, "finite_diff")
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        params = jnp.array([x, y])

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

            return fn(dev.execute(tapes))

        res = cost_fn(params)
        assert len(res) == 2 and isinstance(res, tuple)
        assert all(len(r) == 2 and isinstance(r, tuple) for r in res)
        expected = ((-np.sin(x), 0), (0, -2 * np.cos(y) * np.sin(y)))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # double check the derivative
        jac = jax.jacobian(cost_fn)(params)
        assert np.allclose(jac[0][0][0], -np.cos(x), atol=tol, rtol=0)
        assert np.allclose(jac[1][1][1], -2 * np.cos(2 * y), atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_all_fallback(self, mocker, tol):
        """Test that *only* the fallback logic is called if no parameters
        support the parameter-shift rule"""
        spy_fd = mocker.spy(qml.gradients, "finite_diff")
        spy_ps = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")

        dev = qml.device("default.qubit", wires=2)
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

        res = fn(dev.execute(tapes))
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

        res = fn(dev.execute(tapes))
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
        x = 0.543
        y = -0.654
        ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT([0, 1])]
        meas = [qml.expval(qml.Z(0)), qml.expval(qml.X(1))]
        tape = qml.tape.QuantumScript(ops, meas)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == 2

        dev = qml.device("default.qubit", wires=2)
        res = fn(dev.execute(tapes))
        assert len(res) == 2
        assert all(len(r) == 2 for r in res)

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        x = 0.543
        y = -0.654
        ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT([0, 1])]
        meas = [qml.expval(qml.Z(0)), qml.var(qml.X(1))]
        tape = qml.tape.QuantumScript(ops, meas)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        assert len(tapes) == 3  # One unshifted, two broadcasted shifted tapes
        assert tapes[0].batch_size is None
        assert tapes[1].batch_size == tapes[2].batch_size == 2

        dev = qml.device("default.qubit", wires=2)
        res = fn(dev.execute(tapes))
        assert len(res) == 2
        assert all(len(r) == 2 for r in res)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        x = 0.543
        y = -0.654
        ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT([0, 1])]
        meas = [qml.expval(qml.Z(0)), qml.probs([0, 1])]
        tape = qml.tape.QuantumScript(ops, meas)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        assert len(tapes) == 2
        assert tapes[0].batch_size == tapes[1].batch_size == 2

        dev = qml.device("default.qubit", wires=2)
        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple) and len(res) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in res)
        assert all(isinstance(r, np.ndarray) and r.shape == () for r in res[0])
        assert all(isinstance(r, np.ndarray) and r.shape == (4,) for r in res[1])

        expected_expval = (-np.sin(x), 0)
        sx, cx, sy, cy = np.sin(x / 2), np.cos(x / 2), np.sin(y / 2), np.cos(y / 2)
        expected_probs = (
            np.sin(x) / 2 * np.array([-(cy**2), -(sy**2), sy**2, cy**2]),
            np.array([-(cx**2), cx**2, sx**2, -(sx**2)]) * np.sin(y) / 2,
        )

        assert np.allclose(res[0], expected_expval, atol=tol, rtol=0)
        assert np.allclose(res[1], expected_probs, atol=tol, rtol=0)

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
        gradA = fn(dev.execute(tapes))
        assert isinstance(gradA, np.ndarray)
        assert gradA.shape == ()

        assert len(tapes) == 2
        assert tapes[0].batch_size is None
        assert tapes[1].batch_size == 2

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))
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
        gradA = fn(dev.execute(tapes))
        assert isinstance(gradA, np.ndarray)
        assert gradA.shape == ()

        assert len(tapes) == 1 + 2 * 1
        assert tapes[0].batch_size is None
        assert tapes[1].batch_size == tapes[2].batch_size == 2

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))
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

        meas = [qml.var(qml.Z(0)), qml.var(qml.Hermitian(A, 1))]
        tape = qml.tape.QuantumScript([qml.RX(a, 0), qml.RX(a, 1)], meas)
        tape.trainable_params = {0, 1}

        res = dev.execute(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        gradA = fn(dev.execute(tapes))
        # 1 unshifted tape and 4 broadcasted shifted tapes
        assert len(tapes) == 1 + 4

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))
        assert len(tapes) == 1 + 2

        expected = [2 * np.sin(a) * np.cos(a), -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]
        assert np.diag(gradA) == pytest.approx(expected, abs=tol)
        assert np.diag(gradF) == pytest.approx(expected, abs=tol)

    def test_expval_and_variance(self, tol):
        """Test that the gradient transform works for a combination of expectation
        values and variances"""
        dev = qml.device("default.qubit", wires=3)

        a = 0.54
        b = -0.423
        c = 0.123
        ops = [qml.RX(a, 0), qml.RY(b, 1), qml.CNOT([1, 2]), qml.RX(c, 2), qml.CNOT([0, 1])]
        meas = [qml.var(qml.Z(0)), qml.expval(qml.Z(1)), qml.var(qml.Z(2))]

        tape = qml.tape.QuantumScript(ops, meas)
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
        tapes, fn = qml.gradients.param_shift(tape, broadcast=True)
        gradA = fn(dev.execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))
        ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
        c2c, s2c, s2b = np.cos(2 * c), np.sin(2 * c), np.sin(2 * b)
        expected = np.array(
            [
                [2 * ca * sa, -cb * sa, 0],
                [0, -ca * sb, 0.5 * (2 * cb * c2c * sb + s2b)],
                [0, 0, cb**2 * s2c],
            ]
        ).T
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

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
        gradA = fn(dev.execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.execute(tapes))

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
            qml.Rot(x[0], 0.3 * x[1], x[2], wires=0)
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
        single_measure_circuits = [
            qml.QNode(cost, dev) for cost in (cost1, cost2, cost4, cost5)
        ] + [qml.QNode(cost, dev) for cost in (cost3, cost6)]
        expected_shapes = [(3,), (1, 3), (4, 3), (1, 4, 3), (2, 3), (2, 4, 3)]

        for c, exp_shape in zip(single_measure_circuits, expected_shapes):
            grad = qml.gradients.param_shift(c, broadcast=True)(x)
            assert qml.math.shape(grad) == exp_shape


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
        dev = qml.device("default.qubit", wires=2)
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
            jac = fn(dev.execute(tapes))
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

    def test_not_var_or_exp_val_error(self, broadcast):
        """Tests error raised when the counts of the Hamiltonian is requested"""
        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3])
        H = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.counts(H)

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {2, 3, 4}

        with pytest.raises(ValueError, match="Can only differentiate Hamiltonian coefficients"):
            qml.gradients.param_shift(tape, broadcast=broadcast)

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

        with pytest.raises(ValueError, match="for expectations, not"):
            qml.gradients.param_shift(tape, broadcast=broadcast)

    def test_not_expval_pass_if_not_trainable_hamiltonian(self, broadcast):
        """Test that if the variance of a non-trainable Hamiltonian is requested,
        no error is raised"""
        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3], requires_grad=False)
        H = qml.Hamiltonian(coeffs, obs)

        weights = np.array([0.4, 0.5])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.var(H)

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}  # different from previous test

        tapes, _ = qml.gradients.param_shift(tape, broadcast=broadcast)
        assert len(tapes) == (3 if broadcast else 5)

    def test_no_trainable_coeffs(self, tol, broadcast):
        """Test no trainable Hamiltonian coefficients"""
        dev = qml.device("default.qubit", wires=2)

        obs = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
        coeffs = np.array([0.1, 0.2, 0.3], requires_grad=False)
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

        res = dev.execute([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        # two (broadcasted if broadcast=True) shifts per rotation gate
        assert len(tapes) == (2 if broadcast else 2 * 2)
        assert [t.batch_size for t in tapes] == ([2, 2] if broadcast else [None] * 4)

        res = fn(dev.execute(tapes))
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

    def test_trainable_coeffs(self, tol, broadcast):
        """Test trainable Hamiltonian coefficients"""
        dev = qml.device("default.qubit", wires=2)

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

        res = dev.execute([tape])
        expected = -c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        tapes, fn = qml.gradients.param_shift(tape, broadcast=broadcast)
        # two (broadcasted if broadcast=True) shifts per rotation gate
        # one circuit per trainable H term
        assert len(tapes) == (2 if broadcast else 2 * 2)
        assert [t.batch_size for t in tapes] == ([2, 2] if broadcast else [None] * 4)

        res = fn(dev.execute(tapes))

        expected = [
            -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
            b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
        ]
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

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
        jac = fn(dev.execute(tapes))
        return jac


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
    def test_single_measurement_single_param_not_hybrid(self, interface):
        """Test for a single measurement and a single param with hybrid False."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(2 * x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = qml.numpy.array(0.543, requires_grad=True)

        res = qml.gradients.param_shift(circuit, hybrid=False)(x)

        res_expected = qml.jacobian(circuit)(x)
        assert res.shape == res_expected.shape
        assert np.allclose(2 * res, res_expected)

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
            QuantumFunctionError,
            match="argnum does not work with the Jax interface. You should use argnums instead.",
        ):
            qml.gradients.param_shift(circuit, argnum=argnums)(x, y)

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

    def test_single_probs(self, argnums, interface):
        """Test for single probs."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs()

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = qml.gradients.param_shift(circuit, argnums=argnums)(x, y)

        c_x, s_x = np.cos(x / 2), np.sin(x / 2)
        c_y, s_y = np.cos(y / 2), np.sin(y / 2)
        sqrt_probs = np.array([c_x * c_y, c_x * s_y, s_x * s_y, s_x * c_y])
        dsqrt_probs_0 = 0.5 * np.array([-s_x * c_y, -s_x * s_y, c_x * s_y, c_x * c_y])
        dsqrt_probs_0[:, 1] = 0.0  # Second parameter in x is not being used
        dsqrt_probs_1 = 0.5 * np.array([-c_x * s_y, c_x * c_y, s_x * c_y, -s_x * s_y])[:, 0]
        expected_0 = 2 * sqrt_probs * dsqrt_probs_0
        expected_1 = 2 * sqrt_probs[:, 0] * dsqrt_probs_1

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
            assert np.allclose(res[0][0], expected_1[0])
            assert np.allclose(res[1][0], expected_1[1])
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
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, -0.654])
        y = jax.numpy.array(-0.123)

        res = jax.jacobian(qml.gradients.param_shift(circuit, argnums=argnums), argnums=argnums)(
            x, y
        )
        res_expected = jax.hessian(circuit, argnums=argnums)(x, y)

        if len(argnums) == 1:
            # jax.hessian produces an additional tuple axis, which we have to index away here
            assert np.allclose(res, res_expected[0])
        else:
            # The Hessian is a 2x2 nested tuple "matrix" for argnums=[0, 1]
            for r, r_e in zip(res, res_expected):
                for r_, r_e_ in zip(r, r_e):
                    assert np.allclose(r_, r_e_)

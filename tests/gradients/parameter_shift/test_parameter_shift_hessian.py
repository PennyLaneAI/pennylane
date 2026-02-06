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
"""Tests for the gradients.param_shift_hessian module."""

from itertools import product

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.gradients.parameter_shift_hessian import (
    _collect_recipes,
    _generate_offdiag_tapes,
    _process_argnum,
)


def test_preprocessing_expansion():
    """Test that the parameter-shift Hessian correctly expands templates into supported gates."""

    dev = qp.device("default.qubit")

    @qp.qnode(
        device=dev,
    )
    def circuit(params):
        qp.StronglyEntanglingLayers(params, wires=[0, 1])
        return qp.expval(qp.PauliZ(0))

    hessian_qnode = qp.gradients.param_shift_hessian(circuit)

    params = qp.numpy.array(
        [[[0.37237552, 0.12791554, 0.52721226], [-0.3707729, 1.75044345, 0.37902089]]],
        requires_grad=True,
    )
    result = hessian_qnode(params)
    assert result.shape == (1, 2, 3, 1, 2, 3)


class TestProcessArgnum:
    """Tests for the helper method _process_argnum."""

    with qp.queuing.AnnotatedQueue() as q:
        qp.RX(0.2, wires=0)
        qp.CRZ(0.9, wires=[1, 0])
        qp.RX(0.2, wires=0)
    tape = qp.tape.QuantumScript.from_queue(q)
    tape.trainable_params = {0, 1, 2}

    def test_none(self):
        """Test that for argnum=None all parameters are marked in the returned Boolean mask."""
        argnum = _process_argnum(None, self.tape)
        assert qp.math.allclose(argnum, qp.math.ones((3, 3), dtype=bool))

    @pytest.mark.parametrize("argnum", [0, 2])
    def test_int(self, argnum):
        """Test that an integer argnum correctly is transformed into a Boolean mask
        with the corresponding diagonal element set to True."""
        new_argnum = _process_argnum(argnum, self.tape)
        expected = qp.math.zeros((3, 3), dtype=bool)
        expected[argnum, argnum] = True
        assert qp.math.allclose(new_argnum, expected)

    def test_index_sequence(self):
        """Test that a sequence argnum with indices correctly is transformed into a Boolean mask."""
        new_argnum = _process_argnum([1, 2], self.tape)
        expected = qp.math.zeros((3, 3), dtype=bool)
        expected[1:3, 1:3] = True
        assert qp.math.allclose(new_argnum, expected)

    def test_bool_sequence(self):
        """Test that a Boolean sequence argnum correctly is transformed into a Boolean mask."""
        new_argnum = _process_argnum([True, False, True], self.tape)
        expected = qp.math.zeros((3, 3), dtype=bool)
        expected[0, 0] = expected[2, 2] = expected[2, 0] = expected[0, 2] = True
        assert qp.math.allclose(new_argnum, expected)

    @pytest.mark.parametrize(
        "argnum",
        [
            [[True, False, True], [False, False, False], [True, False, False]],
            [[False, True, True], [True, False, True], [True, True, False]],
        ],
    )
    def test_boolean_mask(self, argnum):
        """Test that a Boolean mask argnum correctly isn't changed."""
        new_argnum = _process_argnum(argnum, self.tape)
        assert qp.math.allclose(new_argnum, argnum)

    def test_error_single_index_too_big(self):
        """Test that an error is raised if a (single) index is too large."""
        with pytest.raises(ValueError, match="The index 10 exceeds the number"):
            _process_argnum(10, self.tape)

    def test_error_max_index_too_big(self):
        """Test that an error is raised if the largest index is too large."""
        with pytest.raises(ValueError, match="The index 10 exceeds the number"):
            _process_argnum([0, 1, 10], self.tape)

    @pytest.mark.parametrize("length", (2, 5))
    def test_error_1D_bool_wrong_length(self, length):
        """Test that an error is raised if a 1D boolean array with wrong length is provided."""
        argnum = qp.math.ones(length, dtype=bool)
        with pytest.raises(ValueError, match="One-dimensional Boolean array argnum"):
            _process_argnum(argnum, self.tape)

    def test_error_wrong_ndim(self):
        """Test that an error is raised if an nD boolean array with wrong number of dimensions
        is provided."""
        argnum = qp.math.ones((3, 3, 3), dtype=bool)
        with pytest.raises(ValueError, match="Expected a symmetric 2D Boolean array"):
            _process_argnum(argnum, self.tape)

    @pytest.mark.parametrize("shape", [(4, 4), (3, 2)])
    def test_error_wrong_shape(self, shape):
        """Test that an error is raised if an nD boolean array with wrong shape is provided."""
        argnum = qp.math.ones(shape, dtype=bool)
        with pytest.raises(ValueError, match="Expected a symmetric 2D Boolean array"):
            _process_argnum(argnum, self.tape)

    @pytest.mark.parametrize("dtype", [float, int])
    def test_error_wrong_dtype(self, dtype):
        """Test that an error is raised if a 2D array with wrong data type is provided."""
        argnum = qp.math.ones((3, 3), dtype=dtype)
        with pytest.raises(ValueError, match="Expected a symmetric 2D Boolean array"):
            _process_argnum(argnum, self.tape)

    def test_error_asymmetric(self):
        """Test that an error is raised if an asymmetric 2D boolean array type is provided."""
        argnum = [[True, False, False], [True, True, False], [False, False, True]]
        with pytest.raises(ValueError, match="Expected a symmetric 2D Boolean array"):
            _process_argnum(argnum, self.tape)


class TestCollectRecipes:
    """Test that gradient recipes are collected/generated correctly based
    on provided shift values, hard-coded recipes of operations, and argnum."""

    with qp.queuing.AnnotatedQueue() as q:
        qp.RX(0.4, wires=0)
        qp.CRZ(-0.9, wires=[1, 0])
        qp.Hadamard(wires=0)
        qp.SingleExcitation(-1.2, wires=[1, 3])

    tape = qp.tape.QuantumScript.from_queue(q)

    def test_with_custom_recipes(self):
        """Test that custom gradient recipes are used correctly."""
        dummy_recipe = [(-0.3, 1.0, 0.0), (0.3, 1.0, 0.4)]
        dummy_recipe_2nd_order = [(0.09, 1.0, 0.0), (-0.18, 1.0, 0.4), (0.09, 1.0, 0.8)]
        channel_recipe = [(-1, 0, 0), (1, 0, 1)]
        channel_recipe_2nd_order = [(0, 0, 0), (0, 0, 1)]

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.RX):
            """A custom RX variant with dummy gradient recipe."""

            grad_recipe = (dummy_recipe,)

        with qp.queuing.AnnotatedQueue() as q:
            qp.DepolarizingChannel(0.2, wires=0)
            DummyOp(0.3, wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        argnum = qp.math.ones((tape.num_params, tape.num_params), dtype=bool)
        diag, offdiag = _collect_recipes(tape, argnum, ("A", "A"), None, None)
        assert qp.math.allclose(diag[0], channel_recipe_2nd_order)
        assert qp.math.allclose(diag[1], qp.math.array(dummy_recipe_2nd_order))
        assert qp.math.allclose(offdiag[0], qp.math.array(channel_recipe))
        assert qp.math.allclose(offdiag[1], qp.math.array(dummy_recipe))

    two_term_recipe = [(0.5, 1.0, np.pi / 2), (-0.5, 1.0, -np.pi / 2)]
    c0 = (np.sqrt(2) + 1) / (4 * np.sqrt(2))
    c1 = (np.sqrt(2) - 1) / (4 * np.sqrt(2))
    four_term_recipe = [
        (c0, 1.0, np.pi / 2),
        (-c0, 1.0, -np.pi / 2),
        (-c1, 1.0, 3 * np.pi / 2),
        (c1, 1.0, -3 * np.pi / 2),
    ]
    two_term_2nd_order = [(-0.5, 1.0, 0.0), (0.5, 1.0, -np.pi)]
    four_term_2nd_order = [
        (-0.375, 1.0, 0),
        (0.25, 1.0, np.pi),
        (0.25, 1.0, -np.pi),
        (-0.125, 1.0, -2 * np.pi),
    ]

    expected_diag_recipes = [two_term_2nd_order, four_term_2nd_order, four_term_2nd_order]

    def test_with_diag_argnum(self):
        """Test that a diagonal boolean argnum is considered correctly."""
        argnum = qp.math.eye(3, dtype=bool)
        diag, offdiag = _collect_recipes(self.tape, argnum, ("A",) * 3, None, None)
        for res, exp in zip(diag, self.expected_diag_recipes):
            assert qp.math.allclose(res, exp)
        assert all(entry == (None, None, None) for entry in offdiag)

    def test_with_block_diag_argnum(self):
        """Test that a block diagonal boolean argnum is considered correctly."""
        argnum = qp.math.array([[True, True, False], [True, True, False], [False, False, True]])
        diag, offdiag = _collect_recipes(self.tape, argnum, ("A",) * 3, None, None)
        for res, exp in zip(diag, self.expected_diag_recipes):
            assert qp.math.allclose(res, exp)
        assert qp.math.allclose(offdiag[0], self.two_term_recipe)
        assert qp.math.allclose(offdiag[1], self.four_term_recipe)
        assert offdiag[2] == (None, None, None)

    def test_with_other_argnum(self):
        """Test that a custom boolean argnum is considered correctly."""
        argnum = qp.math.array([[True, True, False], [True, False, True], [False, True, True]])
        diag, offdiag = _collect_recipes(self.tape, argnum, ("A",) * 3, None, None)
        for i, (res, exp) in enumerate(zip(diag, self.expected_diag_recipes)):
            if i == 1:
                assert res is None
            else:
                assert qp.math.allclose(res, exp)
        assert qp.math.allclose(offdiag[0], self.two_term_recipe)
        assert qp.math.allclose(offdiag[1], self.four_term_recipe)
        assert qp.math.allclose(offdiag[2], self.four_term_recipe)


# pylint: disable=too-few-public-methods
class TestGenerateOffDiagTapes:
    """Test some special features of `_generate_offdiag_tapes`."""

    @pytest.mark.parametrize("add_unshifted", [True, False])
    def test_with_zero_shifts(self, add_unshifted):
        """Test that zero shifts are taken into account in _generate_offdiag_tapes."""
        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(np.array(0.2), wires=[0])
            qp.RY(np.array(0.9), wires=[0])

        tape = qp.tape.QuantumScript.from_queue(q)
        recipe_0 = np.array([[-0.5, 1.0, 0.0], [0.5, 1.0, np.pi]])
        recipe_1 = np.array([[-0.25, 1.0, 0.0], [0.25, 1.0, np.pi]])
        t, c = [], []
        new_add_unshifted, unshifted_coeff = _generate_offdiag_tapes(
            tape, (0, 1), [recipe_0, recipe_1], add_unshifted, t, c
        )

        assert len(t) == 3 + int(add_unshifted)  # Four tapes of which the first is not shifted
        assert np.allclose(c, [-0.125, -0.125, 0.125])
        assert np.isclose(unshifted_coeff, 0.125)
        assert not new_add_unshifted

        orig_cls = [orig_op.__class__ for orig_op in tape.operations]
        expected_pars = list(product([0.2, 0.2 + np.pi], [0.9, 0.9 + np.pi]))
        if not add_unshifted:
            expected_pars = expected_pars[1:]
        for exp_par, h_tape in zip(expected_pars, t):
            assert len(h_tape.operations) == 2
            assert all(op.__class__ == cls for op, cls in zip(h_tape.operations, orig_cls))
            assert np.allclose(h_tape.get_parameters(), exp_par)


class TestParameterShiftHessian:
    """Test the general functionality of the param_shift_hessian method
    on the tape level"""

    def test_single_expval(self):
        """Test that the correct hessian is calculated for a tape with single RX operator
        and single expectation value output"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array(0.1, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = -np.cos(x)

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, np.ndarray)
        assert hessian.shape == ()
        assert np.allclose(expected, hessian)

    def test_single_probs(self):
        """Test that the correct hessian is calculated for a tape with single RX operator
        and single probability output"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array(0.1, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.probs(wires=[0, 1])

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = 0.5 * np.cos(x) * np.array([-1, 0, 0, 1])

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, np.ndarray)
        assert hessian.shape == (4,)
        assert np.allclose(expected, hessian)

    def test_multi_expval(self):
        """Test that the correct hessian is calculated for a tape with single RX operator
        and multiple expval outputs"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array(0.1, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))
            qp.expval(qp.Hadamard(1))

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = (-np.cos(x), -np.cos(x) / np.sqrt(2))

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, tuple)
        assert len(hessian) == 2

        for hess, exp in zip(hessian, expected):
            assert isinstance(hess, np.ndarray)
            assert hess.shape == ()
            assert np.allclose(hess, exp)

    def test_multi_expval_probs(self):
        """Test that the correct hessian is calculated for a tape with single RX operator
        and both expval and probability outputs"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array(0.1, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))
            qp.probs(wires=[0, 1])

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = (-np.cos(x), 0.5 * np.cos(x) * np.array([-1, 0, 0, 1]))

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, tuple)
        assert len(hessian) == 2

        for hess, exp in zip(hessian, expected):
            assert isinstance(hess, np.ndarray)
            assert hess.shape == exp.shape
            assert np.allclose(hess, exp)

    def test_multi_probs(self):
        """Test that the correct hessian is calculated for a tape with single RX operator
        and multiple probability outputs"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array(0.1, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.probs(wires=[0])
            qp.probs(wires=[0, 1])

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = (0.5 * np.cos(x) * np.array([-1, 1]), 0.5 * np.cos(x) * np.array([-1, 0, 0, 1]))

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, tuple)
        assert len(hessian) == 2

        for hess, exp in zip(hessian, expected):
            assert isinstance(hess, np.ndarray)
            assert hess.shape == exp.shape
            assert np.allclose(hess, exp)

    def test_single_expval_multi_params(self):
        """Test that the correct hessian is calculated for a tape with multiple operators
        and single expectation value output"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array([0.1, 0.4], requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x[0], wires=0)
            qp.RY(x[1], wires=1)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = ((-np.cos(x[0]), 0), (0, 0))

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, tuple)
        assert len(hessian) == 2
        assert all(isinstance(hess, tuple) for hess in hessian)
        assert all(len(hess) == 2 for hess in hessian)
        assert all(
            all(isinstance(h, np.ndarray) and h.shape == () for h in hess) for hess in hessian
        )

        assert np.allclose(hessian, expected)

    def test_single_probs_multi_params(self):
        """Test that the correct hessian is calculated for a tape with multiple operators
        and single probability output"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array([0.1, 0.4], requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x[0], wires=0)
            qp.RY(x[1], wires=1)
            qp.CNOT(wires=[0, 1])
            qp.probs(wires=[0, 1])

        tape = qp.tape.QuantumScript.from_queue(q)
        a = [
            np.cos(x[0] / 2) ** 2,
            np.sin(x[0] / 2) ** 2,
            np.cos(x[1] / 2) ** 2,
            np.sin(x[1] / 2) ** 2,
        ]
        expected = (
            (
                0.5 * np.cos(x[0]) * np.array([-a[2], -a[3], a[3], a[2]]),
                0.25 * np.sin(x[0]) * np.sin(x[1]) * np.array([1, -1, 1, -1]),
            ),
            (
                0.25 * np.sin(x[0]) * np.sin(x[1]) * np.array([1, -1, 1, -1]),
                0.5 * np.cos(x[1]) * np.array([-a[0], a[0], a[1], -a[1]]),
            ),
        )

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, tuple)
        assert len(hessian) == 2
        assert all(isinstance(hess, tuple) for hess in hessian)
        assert all(len(hess) == 2 for hess in hessian)
        assert all(
            all(isinstance(h, np.ndarray) and h.shape == (4,) for h in hess) for hess in hessian
        )

        assert np.allclose(hessian, expected)

    def test_multi_expval_multi_params(self):
        """Test that the correct hessian is calculated for a tape with multiple operators
        and multiple expval outputs"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array([0.1, 0.4], requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x[0], wires=0)
            qp.RY(x[1], wires=1)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))
            qp.expval(qp.Hadamard(1))

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = (
            ((-np.cos(x[0]), 0), (0, 0)),
            (
                (
                    -np.cos(x[0]) * np.cos(x[1]) / np.sqrt(2),
                    np.sin(x[0]) * np.sin(x[1]) / np.sqrt(2),
                ),
                (
                    np.sin(x[0]) * np.sin(x[1]) / np.sqrt(2),
                    (-np.sin(x[1]) - np.cos(x[0]) * np.cos(x[1])) / np.sqrt(2),
                ),
            ),
        )

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, tuple)
        assert len(hessian) == 2

        for hess, exp in zip(hessian, expected):
            assert isinstance(hess, tuple)
            assert len(hess) == 2
            assert all(isinstance(h, tuple) for h in hess)
            assert all(len(h) == 2 for h in hess)
            assert all(all(isinstance(h_, np.ndarray) and h_.shape == () for h_ in h) for h in hess)
            assert np.allclose(hess, exp)

    def test_multi_expval_probs_multi_params(self):
        """Test that the correct hessian is calculated for a tape with multiple operators
        and both expval and probability outputs"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array([0.1, 0.4], requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x[0], wires=0)
            qp.RY(x[1], wires=1)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))
            qp.probs(wires=[0, 1])

        tape = qp.tape.QuantumScript.from_queue(q)
        a = [
            np.cos(x[0] / 2) ** 2,
            np.sin(x[0] / 2) ** 2,
            np.cos(x[1] / 2) ** 2,
            np.sin(x[1] / 2) ** 2,
        ]
        expected = (
            ((-np.cos(x[0]), 0), (0, 0)),
            (
                (
                    0.5 * np.cos(x[0]) * np.array([-a[2], -a[3], a[3], a[2]]),
                    0.25 * np.sin(x[0]) * np.sin(x[1]) * np.array([1, -1, 1, -1]),
                ),
                (
                    0.25 * np.sin(x[0]) * np.sin(x[1]) * np.array([1, -1, 1, -1]),
                    0.5 * np.cos(x[1]) * np.array([-a[0], a[0], a[1], -a[1]]),
                ),
            ),
        )

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, tuple)
        assert len(hessian) == 2

        for hess, exp in zip(hessian, expected):
            assert isinstance(hess, tuple)
            assert len(hess) == 2
            assert all(isinstance(h, tuple) for h in hess)
            assert all(len(h) == 2 for h in hess)
            assert all(
                all(isinstance(h_, np.ndarray) and h_.shape == exp[0][0].shape for h_ in h)
                for h in hess
            )
            assert np.allclose(hess, exp)

    def test_multi_probs_multi_params(self):
        """Test that the correct hessian is calculated for a tape with multiple operators
        and multiple probability outputs"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array([0.1, 0.4], requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x[0], wires=0)
            qp.RY(x[1], wires=1)
            qp.CNOT(wires=[0, 1])
            qp.probs(wires=[1])
            qp.probs(wires=[0, 1])

        tape = qp.tape.QuantumScript.from_queue(q)
        a = [
            np.cos(x[0] / 2) ** 2,
            np.sin(x[0] / 2) ** 2,
            np.cos(x[1] / 2) ** 2,
            np.sin(x[1] / 2) ** 2,
        ]
        expected = (
            (
                (
                    0.5 * np.cos(x[0]) * np.cos(x[1]) * np.array([-1, 1]),
                    0.5 * np.sin(x[0]) * np.sin(x[1]) * np.array([1, -1]),
                ),
                (
                    0.5 * np.sin(x[0]) * np.sin(x[1]) * np.array([1, -1]),
                    0.5 * np.cos(x[0]) * np.cos(x[1]) * np.array([-1, 1]),
                ),
            ),
            (
                (
                    0.5 * np.cos(x[0]) * np.array([-a[2], -a[3], a[3], a[2]]),
                    0.25 * np.sin(x[0]) * np.sin(x[1]) * np.array([1, -1, 1, -1]),
                ),
                (
                    0.25 * np.sin(x[0]) * np.sin(x[1]) * np.array([1, -1, 1, -1]),
                    0.5 * np.cos(x[1]) * np.array([-a[0], a[0], a[1], -a[1]]),
                ),
            ),
        )

        tapes, fn = qp.gradients.param_shift_hessian(tape)
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, tuple)
        assert len(hessian) == 2

        for hess, exp in zip(hessian, expected):
            assert isinstance(hess, tuple)
            assert len(hess) == 2
            assert all(isinstance(h, tuple) for h in hess)
            assert all(len(h) == 2 for h in hess)
            assert all(
                all(isinstance(h_, np.ndarray) and h_.shape == exp[0][0].shape for h_ in h)
                for h in hess
            )
            assert np.allclose(hess, exp)

    def test_multi_params_argnum(self):
        """Test that the correct hessian is calculated for a tape with multiple operators
        but not all parameters trainable"""
        dev = qp.device("default.qubit", wires=2)

        x = np.array([0.1, 0.4, 0.7], requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x[0], wires=0)
            qp.RY(x[1], wires=1)
            qp.RY(x[2], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        expected = ((0, 0, 0), (0, 0, 0), (0, 0, -np.cos(x[2] + x[0])))

        tapes, fn = qp.gradients.param_shift_hessian(tape, argnum=(1, 2))
        hessian = fn(qp.execute(tapes, dev, diff_method=None))

        assert isinstance(hessian, tuple)
        assert len(hessian) == 3
        assert all(isinstance(hess, tuple) for hess in hessian)
        assert all(len(hess) == 3 for hess in hessian)
        assert all(
            all(isinstance(h, np.ndarray) and h.shape == () for h in hess) for hess in hessian
        )

        assert np.allclose(hessian, expected)

    def test_state_error(self):
        """Test that an error is raised when computing the gradient of a tape
        that returns state"""

        x = np.array(0.1, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.state()

        tape = qp.tape.QuantumScript.from_queue(q)
        msg = "Computing the Hessian of circuits that return the state is not supported"
        with pytest.raises(ValueError, match=msg):
            qp.gradients.param_shift_hessian(tape)

    def test_variance_error(self):
        """Test that an error is raised when computing the gradient of a tape
        that returns variance"""

        x = np.array(0.1, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.var(qp.PauliZ(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        msg = "Computing the Hessian of circuits that return variances is currently not supported"
        with pytest.raises(ValueError, match=msg):
            qp.gradients.param_shift_hessian(tape)

    @pytest.mark.parametrize("num_measurements", [1, 2])
    def test_no_trainable_params(self, num_measurements):
        """Test that the correct output and warning is generated in the absence of any trainable
        parameters"""
        dev = qp.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(weights[0], wires=0)
            qp.RY(weights[1], wires=0)
            for _ in range(num_measurements):
                qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape.trainable_params = []

        msg = "Attempted to compute the Hessian of a tape with no trainable parameters"
        with pytest.warns(UserWarning, match=msg):
            tapes, fn = qp.gradients.param_shift_hessian(tape)

        res = fn(qp.execute(tapes, dev, None))

        if num_measurements == 1:
            res = (res,)

        assert tapes == []
        assert isinstance(res, tuple)
        assert len(res) == num_measurements
        assert all(isinstance(r, np.ndarray) and r.shape == (0,) for r in res)

    @pytest.mark.parametrize("num_measurements", [1, 2])
    def test_all_zero_grads(self, num_measurements):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qp.device("default.qubit", wires=2)

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.CRZ):
            """A custom variant of qp.CRZ with zero grad_method."""

            grad_method = "0"

        x = np.array(0.1, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            DummyOp(x, wires=[0, 1])
            for _ in range(num_measurements):
                qp.probs(wires=[0, 1])

        tape = qp.tape.QuantumScript.from_queue(q)
        tapes, fn = qp.gradients.param_shift_hessian(tape)
        res = fn(qp.execute(tapes, dev, None))

        if num_measurements == 1:
            res = (res,)

        assert tapes == []
        assert isinstance(res, tuple)
        assert len(res) == num_measurements
        assert all(
            isinstance(r, np.ndarray) and np.allclose(r, np.array([0, 0, 0, 0])) for r in res
        )

    def test_error_unsupported_op(self):
        """Test that the correct error is thrown for unsupported operations"""

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.CRZ):
            """A custom variant of qp.CRZ with grad_method "F"."""

            grad_method = "F"

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            DummyOp(x[2], wires=[0, 1])
            qp.probs(wires=1)

        tape = qp.tape.QuantumScript.from_queue(q)
        msg = "The parameter-shift Hessian currently does not support the operations"
        with pytest.raises(ValueError, match=msg):
            qp.gradients.param_shift_hessian(tape, argnum=[0, 1, 2])(x)

    @pytest.mark.parametrize("argnum", [None, (0,)])
    def test_error_wrong_diagonal_shifts(self, argnum):
        """Test that an error is raised if the number of diagonal shifts does
        not match the required number (`len(trainable_params)` or `len(argnum)`)."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(0.4, wires=0)
            qp.CRY(0.9, wires=[0, 1])

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.raises(ValueError, match="sets of shift values for diagonal entries"):
            qp.gradients.param_shift_hessian(tape, argnum=argnum, diagonal_shifts=[])

    @pytest.mark.parametrize("argnum", [None, (0, 1)])
    def test_error_wrong_offdiagonal_shifts(self, argnum):
        """Test that an error is raised if the number of offdiagonal shifts does
        not match the required number (`len(trainable_params)` or `len(argnum)`)."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(0.4, wires=0)
            qp.CRY(0.9, wires=[0, 1])
            qp.RX(-0.4, wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.raises(ValueError, match="sets of shift values for off-diagonal entries"):
            qp.gradients.param_shift_hessian(tape, argnum=argnum, off_diagonal_shifts=[])


# pylint: disable=too-many-public-methods
class TestParameterShiftHessianQNode:
    """Test the general functionality of the param_shift_hessian method
    with QNodes on the default interface (autograd)"""

    def test_single_two_term_gate(self):
        """Test that the correct hessian is calculated for a QNode with single RX operator
        and single expectation value output (0d -> 0d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        expected = qp.jacobian(qp.grad(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_fixed_params(self):
        """Test that the correct hessian is calculated for a QNode with single RX operator
        and single expectation value output (0d -> 0d) where some fixed parameters gate are added"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RZ(0.1, wires=0)
            qp.RZ(-0.1, wires=0)
            qp.RX(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        expected = qp.jacobian(qp.grad(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_gate_without_impact(self):
        """Test that the correct hessian is calculated for a QNode with an operator
        that does not have any impact on the QNode output."""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RX(x[1], wires=1)
            return qp.expval(qp.PauliZ(0))

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qp.jacobian(qp.grad(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    @pytest.mark.filterwarnings("ignore:Output seems independent of input.")
    def test_no_gate_with_impact(self):
        """Test that the correct hessian is calculated for a QNode without any
        operators that have an impact on the QNode output."""

        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=2)
            qp.RX(x[1], wires=1)
            return qp.expval(qp.PauliZ(0))

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qp.jacobian(qp.grad(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_single_multi_term_gate(self):
        """Test that the correct hessian is calculated for a QNode with single operation
        with more than two terms in the shift rule, parameter frequencies defined,
        and single expectation value output (0d -> 0d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.Hadamard(wires=1)
            qp.CRX(x, wires=[1, 0])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        expected = qp.jacobian(qp.grad(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_single_gate_custom_recipe(self):
        """Test that the correct hessian is calculated for a QNode with single operation
        with more than two terms in the shift rule, parameter frequencies defined,
        and single expectation value output (0d -> 0d)"""

        dev = qp.device("default.qubit", wires=2)

        c, s = qp.gradients.generate_shift_rule((0.5, 1)).T
        recipe = list(zip(c, np.ones_like(c), s))

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.CRX):
            """A custom variant of qp.CRX with a specific gradient recipe."""

            grad_recipe = (recipe,)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.Hadamard(wires=1)
            DummyOp(x, wires=[1, 0])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        expected = qp.jacobian(qp.grad(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_single_two_term_gate_vector_output(self):
        """Test that the correct hessian is calculated for a QNode with single RY operator
        and probabilies as output (0d -> 1d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        x = np.array(0.1, requires_grad=True)

        expected = qp.jacobian(qp.jacobian(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates(self):
        """Test that the correct hessian is calculated for a QNode with two rotation operators
        and one expectation value output (1d -> 0d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RY(x[2], wires=1)
            return qp.expval(qp.PauliZ(1))

        x = np.array([0.1, 0.2, -0.8], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(expected, hessian)

    def test_multiple_two_term_gates_vector_output(self):
        """Test that the correct hessian is calculated for a QNode with two rotation operators
        and probabilities output (1d -> 1d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RY(x[2], wires=1)
            return qp.probs(wires=1)

        x = np.array([0.1, 0.2, -0.8], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(qp.math.transpose(expected, (2, 1, 0)), hessian)

    def test_quantum_hessian_shape_vector_input_vector_output(self):
        """Test that the purely "quantum" hessian has the correct shape (1d -> 1d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RZ(x[2], wires=1)
            qp.Rot(x[0], x[1], x[2], wires=1)
            return qp.probs(wires=[0, 1])

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)
        shape = (3, 3, 4)  # (num_args, num_args, num_output_vals)

        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert qp.math.shape(hessian) == shape

    def test_multiple_two_term_gates_reusing_parameters(self):
        """Test that the correct hessian is calculated when reusing parameters (1d -> 1d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RZ(x[2], wires=1)
            qp.Rot(x[0], x[1], x[2], wires=1)
            return qp.probs(wires=[0, 1])

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(qp.math.transpose(expected, (2, 1, 0)), hessian)

    def test_multiple_two_term_gates_classical_processing(self):
        """Test that the correct hessian is calculated when manipulating parameters (1d -> 1d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[0] + x[1] + x[2], wires=0)
            qp.RY(x[1] - x[0] + 3 * x[2], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RZ(x[2] / x[0] - x[1], wires=1)
            return qp.probs(wires=[0, 1])

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(qp.math.transpose(expected, (2, 1, 0)), hessian)

    def test_multiple_two_term_gates_matrix_output(self):
        """Test that the correct hessian is calculated for higher dimensional QNode outputs
        (1d -> 2d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=0), qp.probs(wires=1)

        def cost(x):
            return qp.math.stack(circuit(x))

        x = np.ones([2], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(cost))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(qp.math.transpose(expected, (0, 3, 2, 1)), hessian)

    def test_multiple_two_term_gates_matrix_input(self):
        """Test that the correct hessian is calculated for higher dimensional cl. jacobians
        (2d -> 2d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x):
            qp.RX(x[0, 0], wires=0)
            qp.RY(x[0, 1], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RX(x[0, 2], wires=0)
            qp.RY(x[0, 0], wires=0)
            return qp.probs(wires=0), qp.probs(wires=1)

        def cost(x):
            return qp.math.stack(circuit(x))

        x = np.ones([1, 3], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(cost))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(qp.math.transpose(expected, (0, 2, 3, 4, 5, 1)), hessian)

    def test_multiple_qnode_arguments_scalar(self):
        """Test that the correct Hessian is calculated with multiple QNode arguments (0D->1D)"""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x, y, z):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            qp.SingleExcitation(z, wires=[1, 0])
            qp.RY(y, wires=0)
            qp.RX(x, wires=0)
            return qp.probs(wires=[0, 1])

        def wrapper(X):
            return circuit(*X)

        x = np.array(0.1, requires_grad=True)
        y = np.array(0.5, requires_grad=True)
        z = np.array(0.3, requires_grad=True)
        X = qp.math.stack([x, y, z])

        expected = qp.jacobian(qp.jacobian(wrapper))(X)
        expected = tuple(expected[:, i, i] for i in range(3))
        circuit.interface = "autograd"
        hessian = qp.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(expected, hessian)

    def test_multiple_qnode_arguments_vector(self):
        """Test that the correct Hessian is calculated with multiple QNode arguments (1D->1D)"""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x, y, z):
            qp.RX(x[0], wires=1)
            qp.RY(y[0], wires=0)
            qp.CRZ(z[0] + z[1], wires=[1, 0])
            qp.RY(y[1], wires=1)
            qp.RX(x[1], wires=0)
            return qp.probs(wires=[0, 1])

        def wrapper(X):
            return circuit(*X)

        x = np.array([0.1, 0.3], requires_grad=True)
        y = np.array([0.5, 0.7], requires_grad=True)
        z = np.array([0.3, 0.2], requires_grad=True)
        X = qp.math.stack([x, y, z])

        expected = qp.jacobian(qp.jacobian(wrapper))(X)
        expected = tuple(expected[:, i, :, i] for i in range(3))

        circuit.interface = "autograd"
        hessian = qp.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(qp.math.transpose(expected, (0, 2, 3, 1)), hessian)

    def test_multiple_qnode_arguments_matrix(self):
        """Test that the correct Hessian is calculated with multiple QNode arguments (2D->1D)"""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x, y, z):
            qp.RX(x[0, 0], wires=0)
            qp.RY(y[0, 0], wires=1)
            qp.CRZ(z[0, 0] + z[1, 1], wires=[1, 0])
            qp.RY(y[1, 0], wires=0)
            qp.RX(x[1, 0], wires=1)
            return qp.probs(wires=[0, 1])

        def wrapper(X):
            return circuit(*X)

        x = np.array([[0.1, 0.3], [0.2, 0.4]], requires_grad=True)
        y = np.array([[0.5, 0.7], [0.2, 0.4]], requires_grad=True)
        z = np.array([[0.3, 0.2], [0.2, 0.4]], requires_grad=True)
        X = qp.math.stack([x, y, z])

        expected = qp.jacobian(qp.jacobian(wrapper))(X)
        expected = tuple(expected[:, i, :, :, i] for i in range(3))

        circuit.interface = "autograd"
        hessian = qp.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(qp.math.transpose(expected, (0, 2, 3, 4, 5, 1)), hessian)

    def test_multiple_qnode_arguments_mixed(self):
        """Test that the correct Hessian is calculated with multiple mixed-shape QNode arguments"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x, y, z):
            qp.RX(x, wires=0)
            qp.RY(z[0] + z[1], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RX(y[1, 0], wires=0)
            qp.CRY(y[0, 1], wires=[0, 1])
            return qp.probs(wires=0), qp.probs(wires=1)

        def cost(x, y, z):
            return qp.math.stack(circuit(x, y, z))

        x = np.array(0.1, requires_grad=True)
        y = np.array([[0.5, 0.6], [0.2, 0.1]], requires_grad=True)
        z = np.array([0.3, 0.4], requires_grad=True)

        expected = tuple(
            qp.jacobian(qp.jacobian(cost, argnums=i), argnums=i)(x, y, z) for i in range(3)
        )
        hessian = qp.gradients.param_shift_hessian(circuit)(x, y, z)

        assert np.allclose(expected[0], hessian[0])
        assert np.allclose(qp.math.transpose(expected[1], (0, 2, 3, 4, 5, 1)), hessian[1])
        assert np.allclose(qp.math.transpose(expected[2], (0, 2, 3, 1)), hessian[2])

    @pytest.mark.xfail(
        reason=r"ProbsMP.process_density_matrix issue. See https://github.com/PennyLaneAI/pennylane/pull/6684#issuecomment-2552123064"
    )
    def test_with_channel(self):
        """Test that the Hessian is correctly computed for circuits
        that contain quantum channels."""

        dev = qp.device("default.mixed", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x):
            qp.RX(x[1], wires=0)
            qp.RY(x[0], wires=0)
            qp.DepolarizingChannel(x[2], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        x = np.array([-0.4, 0.9, 0.1], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(circuit))(x)
        hessian = qp.gradients.param_shift_hessian(circuit)(x)

        assert np.allclose(qp.math.transpose(expected, (2, 1, 0)), hessian)

    def test_hessian_transform_is_differentiable(self):
        """Test that the 3rd derivate can be calculated via auto-differentiation (1d -> 1d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=3)
        def circuit(x):
            qp.RX(x[1], wires=0)
            qp.RY(x[0], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(qp.jacobian(circuit)))(x)

        def cost_fn(x):
            hess = qp.gradients.param_shift_hessian(circuit)(x)
            hess = qp.math.stack([qp.math.stack(row) for row in hess])
            return hess

        derivative = qp.jacobian(cost_fn)(x)

        assert np.allclose(qp.math.transpose(expected, (1, 2, 0, 3)), derivative)

    # Some bounds on the efficiency (device executions) of the hessian for 2-term shift rules:
    # - < jacobian(jacobian())
    # - <= 2^d * (m+d-1)C(d)      see arXiv:2008.06517 p. 4
    # - <= 3^m                    see arXiv:2008.06517 p. 4
    # here d=2 is the derivative order, m is the number of variational parameters (w.r.t. gate args)

    def test_fewer_device_invocations_scalar_input(self):
        """Test that the hessian invokes less hardware executions than double differentiation
        (0d -> 0d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(1))

        x = np.array(0.1, requires_grad=True)

        with qp.Tracker(dev) as tracker:
            hessian = qp.gradients.param_shift_hessian(circuit)(x)
            hessian_qruns = tracker.totals["executions"]
            expected = qp.jacobian(qp.jacobian(circuit))(x)
            jacobian_qruns = tracker.totals["executions"] - hessian_qruns

        assert np.allclose(hessian, expected)
        assert hessian_qruns < jacobian_qruns
        assert hessian_qruns <= 2**2 * 1  # 1 = (1+2-1)C(2)
        assert hessian_qruns <= 3**1

    def test_fewer_device_invocations_vector_input(self):
        """Test that the hessian invokes less hardware executions than double differentiation
        (1d -> 0d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RY(x[1], wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))

        x = np.array([0.1, 0.2], requires_grad=True)

        with qp.Tracker(dev) as tracker:
            hessian = qp.gradients.param_shift_hessian(circuit)(x)
            hessian_qruns = tracker.totals["executions"]
            expected = qp.jacobian(qp.jacobian(circuit))(x)
            jacobian_qruns = tracker.totals["executions"] - hessian_qruns

        assert np.allclose(hessian, expected)
        assert hessian_qruns < jacobian_qruns
        assert hessian_qruns <= 2**2 * 3  # 3 = (2+2-1)C(2)
        assert hessian_qruns <= 3**2

    @pytest.mark.xfail(reason="Update tracker for new return types")
    def test_fewer_device_invocations_vector_output(self):
        """Test that the hessian invokes less hardware executions than double differentiation
        (1d -> 1d)"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RY(x[1], wires=0)
            qp.RZ(x[2], wires=1)
            return qp.probs(wires=[0, 1])

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with qp.Tracker(dev) as tracker:
            hessian = qp.gradients.param_shift_hessian(circuit)(x)
            hessian_qruns = tracker.totals["executions"]
            expected = qp.jacobian(qp.jacobian(circuit))(x)
            jacobian_qruns = tracker.totals["executions"] - hessian_qruns

        assert np.allclose(hessian, expected)
        assert hessian_qruns < jacobian_qruns
        assert hessian_qruns <= 2**2 * 6  # 6 = (3+2-1)C(2)
        assert hessian_qruns <= 3**3

    def test_error_unsupported_operation_without_argnum(self):
        """Test that the correct error is thrown for unsupported operations when
        no argnum is given."""

        dev = qp.device("default.qubit", wires=2)

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.CRZ):
            """A custom variant of qp.CRZ with grad_method "F"."""

            grad_method = "F"

        @qp.qnode(dev, max_diff=2, diff_method="parameter-shift")
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            DummyOp(x[2], wires=[0, 1])
            return qp.probs(wires=1)

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with pytest.raises(
            ValueError,
            match=r"The analytic gradient method cannot be used with the parameter\(s\)",
        ):
            qp.gradients.param_shift_hessian(circuit)(x)

    def test_error_unsupported_variance_measurement(self):
        """Test that the correct error is thrown for variance measurements"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2, diff_method="parameter-shift")
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            qp.CRZ(x[2], wires=[0, 1])
            return qp.var(qp.PauliZ(1))

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with pytest.raises(
            ValueError,
            match="Computing the Hessian of circuits that return variances is currently not supported.",
        ):
            qp.gradients.param_shift_hessian(circuit)(x)

    def test_error_unsupported_state_measurement(self):
        """Test that the correct error is thrown for state measurements"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2, diff_method="parameter-shift")
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            qp.CRZ(x[2], wires=[0, 1])
            return qp.state()

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with pytest.raises(
            ValueError,
            match="Computing the Hessian of circuits that return the state is not supported.",
        ):
            qp.gradients.param_shift_hessian(circuit)(x)

    def test_no_error_nondifferentiable_unsupported_operation(self):
        """Test that no error is thrown for operations that are not marked differentiable"""

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.CRZ):
            """A custom variant of qp.CRZ with grad_method "F"."""

            grad_method = "F"

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2, diff_method="parameter-shift")
        def circuit(x, y, z):
            qp.RX(x, wires=0)
            qp.RY(y, wires=0)
            DummyOp(z, wires=[0, 1])
            return qp.probs(wires=1)

        x = np.array(0.1, requires_grad=True)
        y = np.array(0.2, requires_grad=True)
        z = np.array(0.3, requires_grad=False)

        qp.gradients.param_shift_hessian(circuit)(x, y, z)

    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface="autograd", diff_method="parameter-shift")
        def circuit(weights):
            qp.RX(weights[0], wires=0)
            qp.RY(weights[1], wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qp.gradients.param_shift_hessian(circuit)(weights)

    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(weights):
            qp.RX(weights[0], wires=0)
            qp.RY(weights[1], wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qp.gradients.param_shift_hessian(circuit)(weights)

    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface="tf", diff_method="parameter-shift")
        def circuit(weights):
            qp.RX(weights[0], wires=0)
            qp.RY(weights[1], wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qp.gradients.param_shift_hessian(circuit)(weights)

    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface="jax", diff_method="parameter-shift")
        def circuit(weights):
            qp.RX(weights[0], wires=0)
            qp.RY(weights[1], wires=0)
            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qp.gradients.param_shift_hessian(circuit)(weights)

    def test_all_zero_diff_methods(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev, diff_method="parameter-shift")
        def circuit(params):
            qp.Rot(*params, wires=0)
            return qp.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qp.gradients.param_shift_hessian(circuit)(params)
        assert np.allclose(result, np.zeros((3, 3, 4)), atol=0, rtol=0)

        tapes, _ = qp.gradients.param_shift_hessian(qp.workflow.construct_tape(circuit)(params))
        assert tapes == []

    @pytest.mark.xfail(reason="Update tracker for new return types")
    def test_f0_argument(self):
        """Test that we can provide the results of a QNode to save on quantum invocations"""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=1)

        x = np.array([0.1, 0.2], requires_grad=True)

        res = circuit(x)

        with qp.Tracker(dev) as tracker:
            hessian1 = qp.gradients.param_shift_hessian(circuit, f0=res)(x)
            qruns1 = tracker.totals["executions"]
            hessian2 = qp.gradients.param_shift_hessian(circuit)(x)
            qruns2 = tracker.totals["executions"] - qruns1

        assert np.allclose(hessian1, hessian2)
        assert qruns1 < qruns2

    def test_output_shape_matches_qnode(self):
        """Test that the transform output shape matches that of the QNode."""
        dev = qp.device("default.qubit", wires=4)

        def cost1(x):
            qp.Rot(*x, wires=0)
            return qp.expval(qp.PauliZ(0))

        def cost2(x):
            qp.Rot(*x, wires=0)
            return [qp.expval(qp.PauliZ(0))]

        def cost3(x):
            qp.Rot(*x, wires=0)
            return [qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliZ(1))]

        def cost4(x):
            qp.Rot(*x, wires=0)
            return qp.probs([0, 1])

        def cost5(x):
            qp.Rot(*x, wires=0)
            return [qp.probs([0, 1])]

        def cost6(x):
            qp.Rot(*x, wires=0)
            return [qp.probs([0, 1]), qp.probs([2, 3])]

        x = np.random.rand(3)
        circuits = [qp.QNode(cost, dev) for cost in (cost1, cost2, cost3, cost4, cost5, cost6)]

        transform = [qp.math.shape(qp.gradients.param_shift_hessian(c)(x)) for c in circuits]
        expected = [(3, 3), (3, 3), (2, 3, 3), (3, 3, 4), (3, 3, 4), (2, 3, 3, 4)]

        assert all(t == e for t, e in zip(transform, expected))


class TestParamShiftHessianWithKwargs:
    """Test the parameter-shift Hessian computation when manually
    providing parameter shifts or `argnum`."""

    @pytest.mark.parametrize(
        "diagonal_shifts",
        (
            [(np.pi / 3,), (np.pi / 2,)],
            [(np.pi / 3,), None],
        ),
    )
    def test_with_diagonal_shifts(self, diagonal_shifts):
        """Test that diagonal shifts are used and yield the correct Hessian."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.RY(x[1], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=0)

        x = np.array([0.6, -0.2], requires_grad=True)

        expected = qp.math.transpose(qp.jacobian(qp.jacobian(circuit))(x), (1, 2, 0))
        tapes, fn = qp.gradients.param_shift_hessian(
            qp.workflow.construct_tape(circuit)(x), diagonal_shifts=diagonal_shifts
        )

        # We expect the following tapes:
        # - 1 without shifts (used for second diagonal),
        # - 2 for first diagonal,
        # - 4 for off-diagonal,
        # - 1 for second diagonal.
        assert len(tapes) == 1 + 2 + 4 + 1
        assert np.allclose(tapes[0].get_parameters(), x)
        assert np.allclose(tapes[1].get_parameters(), x + np.array([2 * np.pi / 3, 0.0]))
        assert np.allclose(tapes[2].get_parameters(), x + np.array([-2 * np.pi / 3, 0.0]))
        assert np.allclose(tapes[-1].get_parameters(), x + np.array([0.0, -np.pi]))
        expected_shifts = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * (np.pi / 2)
        for _tape, exp_shift in zip(tapes[3:-1], expected_shifts):
            assert np.allclose(_tape.get_parameters(), x + exp_shift)

        hessian = fn(qp.execute(tapes, dev, diff_method=qp.gradients.param_shift))

        assert np.allclose(expected, hessian)

    @pytest.mark.parametrize(
        "off_diagonal_shifts",
        (
            [(np.pi / 2,), (0.3, 0.6)],
            [None, (0.3, 0.6)],
        ),
    )
    def test_with_offdiagonal_shifts(self, off_diagonal_shifts):
        """Test that off-diagonal shifts are used and yield the correct Hessian."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x):
            qp.RX(x[0], wires=0)
            qp.CRY(x[1], wires=[0, 1])
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=0)

        x = np.array([0.6, -0.2], requires_grad=True)

        expected = qp.math.transpose(qp.jacobian(qp.jacobian(circuit))(x), (1, 2, 0))
        tapes, fn = qp.gradients.param_shift_hessian(
            qp.workflow.construct_tape(circuit)(x), off_diagonal_shifts=off_diagonal_shifts
        )

        # We expect the following tapes:
        # - 1 without shifts (used for diagonals),
        # - 1 for first diagonal,
        # - 8 for off-diagonal,
        # - 3 for second diagonal.
        assert len(tapes) == 1 + 1 + 8 + 3
        assert np.allclose(tapes[0].get_parameters(), x)

        # Check that the vanilla diagonal rule is used for the first diagonal entry
        assert np.allclose(tapes[1].get_parameters(), x + np.array([-np.pi, 0.0]))

        # Check that the provided off-diagonal shift values are used
        expected_shifts = np.array(
            [[1, 1], [1, -1], [1, 2], [1, -2], [-1, 1], [-1, -1], [-1, 2], [-1, -2]]
        ) * np.array([[np.pi / 2, 0.3]])
        for _tape, exp_shift in zip(tapes[2:10], expected_shifts):
            assert np.allclose(_tape.get_parameters(), x + exp_shift)

        # Check that the vanilla diagonal rule is used for the second diagonal entry
        shift_order = [1, -1, -2]
        for mult, _tape in zip(shift_order, tapes[10:]):
            assert np.allclose(_tape.get_parameters(), x + np.array([0.0, np.pi * mult]))

        hessian = fn(qp.execute(tapes, dev, diff_method=qp.gradients.param_shift))
        assert np.allclose(expected, hessian)

    @pytest.mark.parametrize("argnum", [(0,), (1,), (0, 1)])
    def test_with_1d_argnum(self, argnum):
        """Test that providing an argnum to indicate differentiable parameters works."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2, diff_method="parameter-shift")
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.CRY(y, wires=[0, 1])
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        def wrapper(X):
            return circuit(*X)

        X = np.array([0.6, -0.2], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(wrapper))(X)
        # Extract "diagonal" across arguments
        expected = np.array([np.diag(sub) for i, sub in enumerate(expected)])
        # Set non-argnum argument entries to 0
        for i in range(len(X)):
            if i not in argnum:
                expected[:, i] = 0.0
        hessian = qp.gradients.param_shift_hessian(circuit, argnum=argnum)(*X)
        assert np.allclose(hessian, expected.T)

    @pytest.mark.parametrize(
        "argnum",
        [
            qp.math.eye(3, dtype=bool),
            qp.math.array([[True, False, False], [False, False, True], [False, True, True]]),
            qp.math.array([[False, False, True], [False, False, False], [True, False, False]]),
        ],
    )
    def test_with_2d_argnum(self, argnum):
        """Test that providing an argnum to indicated differentiable parameters works."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(par):
            qp.RX(par[0], wires=0)
            qp.CRY(par[1], wires=[0, 1])
            qp.CNOT(wires=[0, 1])
            qp.CRY(par[2], wires=[0, 1])
            return qp.probs(wires=[0, 1])

        par = np.array([0.6, -0.2, 0.8], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(circuit))(par)
        # Set non-argnum argument entries to 0
        expected = qp.math.transpose(
            expected * qp.math.array(argnum, dtype=float)[None], (1, 2, 0)
        )
        hessian = qp.gradients.param_shift_hessian(circuit, argnum=argnum)(par)
        assert np.allclose(hessian, expected)

    @pytest.mark.parametrize("argnum", [(0,), (1,), (0, 1)])
    def test_with_argnum_and_shifts(self, argnum):
        """Test that providing an argnum to indicated differentiable parameters works."""
        diagonal_shifts = [(0.1,), (0.9, 1.1)]
        off_diagonal_shifts = [(0.4,), (0.3, 2.1)]
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2, diff_method="parameter-shift")
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.CRY(y, wires=[0, 1])
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        def wrapper(X):
            return circuit(*X)

        X = np.array([0.6, -0.2], requires_grad=True)

        expected = qp.jacobian(qp.jacobian(wrapper))(X)
        # Extract "diagonal" across arguments
        expected = np.array([np.diag(sub) for i, sub in enumerate(expected)])
        # Set non-argnum argument entries to 0
        for i in range(len(X)):
            if i not in argnum:
                expected[:, i] = 0.0
        d_shifts = [diagonal_shifts[arg] for arg in argnum]
        od_shifts = [off_diagonal_shifts[arg] for arg in argnum if len(argnum) > 1]
        hessian = qp.gradients.param_shift_hessian(
            circuit, argnum=argnum, diagonal_shifts=d_shifts, off_diagonal_shifts=od_shifts
        )(*X)
        assert np.allclose(hessian, expected.T)


class TestInterfaces:
    """Test the param_shift_hessian method on different interfaces"""

    @pytest.mark.skip("Requires Torch integration for new return types")
    @pytest.mark.torch
    def test_hessian_transform_with_torch(self):
        """Test that the Hessian transform can be used with Torch (1d -> 1d)"""
        import torch

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qp.RX(x[1], wires=0)
            qp.RY(x[0], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        x_np = np.array([0.1, 0.2], requires_grad=True)
        x_torch = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)

        expected = qp.jacobian(qp.jacobian(circuit))(x_np)
        circuit.interface = "torch"
        hess = qp.gradients.param_shift_hessian(circuit)(x_torch)[0]

        assert np.allclose(expected, hess.detach())

    @pytest.mark.skip("Requires Torch integration for new return types")
    @pytest.mark.torch
    def test_hessian_transform_is_differentiable_torch(self):
        """Test that the 3rd derivate can be calculated via auto-differentiation in Torch
        (1d -> 1d)"""
        import torch

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method="parameter-shift", max_diff=3)
        def circuit(x):
            qp.RX(x[1], wires=0)
            qp.RY(x[0], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)
        x_torch = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)

        expected = qp.jacobian(qp.jacobian(qp.jacobian(circuit)))(x)
        circuit.interface = "torch"
        jacobian_fn = torch.autograd.functional.jacobian
        torch_deriv = jacobian_fn(qp.gradients.param_shift_hessian(circuit), x_torch)[0]

        assert np.allclose(expected, torch_deriv)

    @pytest.mark.jax
    @pytest.mark.slow
    def test_hessian_transform_with_jax(self):
        """Test that the Hessian transform can be used with JAX (1d -> 1d)"""
        import jax

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x):
            qp.RX(x[1], wires=0)
            qp.RY(x[0], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        x_np = np.array([0.1, 0.2], requires_grad=True)
        x_jax = jax.numpy.array([0.1, 0.2])

        expected = qp.jacobian(qp.jacobian(circuit))(x_np)
        circuit.interface = "jax"
        hess = qp.gradients.param_shift_hessian(circuit, argnums=[0])(x_jax)

        assert np.allclose(qp.math.transpose(expected, (1, 2, 0)), hess)

    @pytest.mark.jax
    @pytest.mark.slow
    def test_hessian_transform_is_differentiable_jax(self):
        """Test that the 3rd derivate can be calculated via auto-differentiation in JAX
        (1d -> 1d)"""
        import jax

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=3)
        def circuit(x):
            qp.RX(x[1], wires=0)
            qp.RY(x[0], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)
        x_jax = jax.numpy.array([0.1, 0.2])

        expected = qp.jacobian(qp.jacobian(qp.jacobian(circuit)))(x)

        def cost_fn(x):
            hess = qp.gradients.param_shift_hessian(circuit)(x)
            hess = qp.math.stack([qp.math.stack(row) for row in hess])
            return hess

        circuit.interface = "jax"
        jax_deriv = jax.jacobian(cost_fn)(x_jax)

        assert np.allclose(qp.math.transpose(expected, (1, 2, 0, 3)), jax_deriv)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_hessian_transform_with_tensorflow(self):
        """Test that the Hessian transform can be used with TensorFlow (1d -> 1d)"""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=2)
        def circuit(x):
            qp.RX(x[1], wires=0)
            qp.RY(x[0], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        x_np = np.array([0.1, 0.2], requires_grad=True)
        x_tf = tf.Variable([0.1, 0.2], dtype=tf.float64)

        expected = qp.jacobian(qp.jacobian(circuit))(x_np)
        circuit.interface = "tf"
        with tf.GradientTape():
            hess = qp.gradients.param_shift_hessian(circuit)(x_tf)

        assert np.allclose(qp.math.transpose(expected, (1, 2, 0)), hess)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_hessian_transform_is_differentiable_tensorflow(self):
        """Test that the 3rd derivate can be calculated via auto-differentiation in Tensorflow
        (1d -> 1d)"""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, max_diff=3)
        def circuit(x):
            qp.RX(x[1], wires=0)
            qp.RY(x[0], wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.probs(wires=[0, 1])

        x = np.array([0.1, 0.2], requires_grad=True)
        x_tf = tf.Variable([0.1, 0.2], dtype=tf.float64)

        expected = qp.jacobian(qp.jacobian(qp.jacobian(circuit)))(x)
        circuit.interface = "tf"
        with tf.GradientTape() as tf_tape:
            hessian = qp.gradients.param_shift_hessian(circuit)(x_tf)[0]
            hessian = qp.math.stack([qp.math.stack(row) for row in hessian])

        tensorflow_deriv = tf_tape.jacobian(hessian, x_tf)

        assert np.allclose(qp.math.transpose(expected, (1, 2, 0, 3)), tensorflow_deriv)

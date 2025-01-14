"""Tests for Vibronic Hamiltonian"""

from itertools import product

import numpy as np
import pytest

from pennylane.labs.vibronic import (
    Node,
    VibronicHamiltonian,
    VibronicMatrix,
    VibronicTerm,
    VibronicWord,
    coeffs,
    commutator,
    momentum_operator,
    position_operator,
)

# pylint: disable=protected-access,import-outside-toplevel


class TestVibronic:
    """Test the vibronic hamiltonians"""

    @pytest.mark.parametrize("modes", range(5))
    def test_epsilon(self, modes):
        """Test that epsilon is correct for 2 states"""
        states = 2
        delta = 0.72
        scalar = -(delta**2) / 24
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))

        terms = [
            vham._commute_fragments(0, 0, 1),
            2 * vham._commute_fragments(1, 0, 1),
            2 * vham._commute_fragments(2, 0, 1),
            vham._commute_fragments(0, 0, 2),
            2 * vham._commute_fragments(1, 0, 2),
            2 * vham._commute_fragments(2, 0, 2),
            vham._commute_fragments(1, 1, 2),
            2 * vham._commute_fragments(2, 1, 2),
        ]

        actual = vham.epsilon(delta)
        expected = scalar * sum(terms, VibronicMatrix(states, modes))

        assert actual == expected

    states = [2**i for i in range(1, 3)]
    states_modes = list(product(states, range(5)))

    @pytest.mark.parametrize("states, modes", states_modes)
    def test_fragment_against_v(self, states, modes):
        """Test that the fragments have correct block structure"""
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))

        zero_word = VibronicWord([VibronicTerm(tuple(), Node.tensor_node(np.array(0)))])

        for i in range(states):
            h = vham.fragment(i)
            for j, k in zip(range(states), range(states)):
                if k == i ^ j:
                    assert h.block(j, k) == vham.v_word(j, k)
                else:
                    assert h.block(j, k) == zero_word

    @pytest.mark.parametrize("modes", range(5))
    def test_commute_h0_h1_on_two_states(self, modes):
        """Test that Y = [H_0, H_1] is correct"""
        states = 2

        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))
        h0 = vham.fragment(0)
        h1 = vham.fragment(1)
        y = commutator(h0, h1)

        word1 = (vham.v_word(0, 0) @ vham.v_word(0, 1)) - (vham.v_word(0, 1) @ vham.v_word(1, 1))
        word2 = (vham.v_word(1, 1) @ vham.v_word(0, 1)) - (vham.v_word(0, 1) @ vham.v_word(0, 0))

        assert y.block(0, 0).is_zero
        assert y.block(0, 1) == word1
        assert y.block(1, 0) == word2
        assert y.block(1, 1).is_zero


class TestMatrix:
    """Test thet matrix representation"""

    states = [2**i for i in range(1, 3)]
    states_modes = list(product(states, range(1, 6)))

    @pytest.mark.parametrize("states, modes", states_modes)
    @pytest.mark.parametrize("gridpoints", range(1, 3))
    def test_matrix_add(self, states, modes, gridpoints):
        """Test that the matrix representation is consistent with addition"""
        vham1 = VibronicHamiltonian(states, modes, *coeffs(states, modes))
        vham2 = VibronicHamiltonian(states, modes, *coeffs(states, modes))

        mat1 = vham1.matrix(gridpoints)
        mat2 = vham2.matrix(gridpoints)

        assert np.allclose((mat1 + mat2), (vham1 + vham2).matrix(gridpoints))

    @pytest.mark.parametrize("states, modes", states_modes)
    @pytest.mark.parametrize("gridpoints", range(1, 3))
    @pytest.mark.parametrize("scalar", [0, 1, -1, 2.2, -1.7, 5j, -2 + 1j])
    def test_matrix_scalar_mul(self, states, modes, gridpoints, scalar):
        """Test that the matrix representation is consistent with addition"""
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))

        assert np.allclose((scalar * vham).matrix(gridpoints), scalar * vham.matrix(gridpoints))

    single_term_params = [
        ((3, 4, (0,), [position_operator(4, 1)]), np.kron(position_operator(4, 1), np.eye(4**2))),
        (
            (4, 4, (0, 1), [position_operator(4, 1), momentum_operator(4, 1)]),
            np.kron(position_operator(4, 1), np.kron(momentum_operator(4, 1), np.eye(4**2))),
        ),
        (
            (4, 4, (1, 1), [position_operator(4, 1), momentum_operator(4, 1)]),
            np.kron(
                np.eye(4), np.kron(position_operator(4, 1) @ momentum_operator(4, 1), np.eye(4**2))
            ),
        ),
    ]

    @pytest.mark.parametrize("params, expected", single_term_params)
    def test_single_term(self, params, expected):
        """Test the _single_term_matrix function"""
        from pennylane.labs.vibronic.utils.matrix_ops import _single_term_matrix

        actual = _single_term_matrix(*params)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

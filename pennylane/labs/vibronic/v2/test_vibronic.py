"""Tests for Vibronic Hamiltonian"""

from itertools import product

import pytest
from utils import coeffs
from vibronic_hamiltonian import VibronicHamiltonian
from vibronic_matrix import VibronicMatrix, commutator
from vibronic_term import VibronicWord

# pylint: disable=protected-access


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

        for i in range(states):
            h = vham.fragment(i)
            for j, k in zip(range(states), range(states)):
                if k == i ^ j:
                    assert h.block(j, k) == vham.v_word(j, k)
                else:
                    assert h.block(j, k) == VibronicWord(tuple())

    @pytest.mark.parametrize("states, modes", states_modes)
    def test_commute_h0_h1(self, states, modes):
        """Test that Y = [H_0, H_1] is correct"""

        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))
        h0 = vham.fragment(0)
        h1 = vham.fragment(1)
        y = commutator(h0, h1)

        mat1 = (vham.v_word(0, 0) @ vham.v_word(0, 1)) - (vham.v_word(0, 1) @ vham.v_word(1, 1))
        mat2 = (vham.v_word(1, 1) @ vham.v_word(0, 1)) - (vham.v_word(0, 1) @ vham.v_word(0, 0))

        assert y.block(0, 0) == VibronicWord({})
        assert y.block(0, 1) == mat1
        assert y.block(1, 0) == mat2
        assert y.block(1, 1) == VibronicWord({})

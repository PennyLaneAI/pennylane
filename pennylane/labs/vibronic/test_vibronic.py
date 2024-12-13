"""Tests for the VibronicHamiltonian class"""
from itertools import product

import numpy as np
import pytest
from vibronic import VibronicBlockMatrix, VibronicHamiltonian, VibronicWord, commutator


def coeffs(states, modes):
    """Produce random coefficients for input"""

    alphas = np.random.random((states, states, modes))
    betas = np.random.random((states, states, modes, modes))
    lambdas = np.random.random((states, states))
    omegas = np.random.random((modes,))

    symmetric_alphas = np.zeros(alphas.shape)
    symmetric_betas = np.zeros(betas.shape)
    symmetric_lambdas = np.zeros(lambdas.shape)

    for i in range(states):
        for j in range(states):
            betas[i, j] = (betas[i, j] + betas[i, j].T) / 2

    for i in range(states):
        for j in range(states):
            symmetric_alphas[i, j] = (alphas[i, j] + alphas[j, i]) / 2
            symmetric_betas[i, j] = (betas[i, j] + betas[j, i]) / 2
            symmetric_lambdas[i, j] = (lambdas[i, j] + lambdas[j, i]) / 2

    return symmetric_alphas, symmetric_betas, symmetric_lambdas, omegas


class TestVibronic:
    """Test vibronic arithmetic"""

    states = [2**i for i in range(5)]
    states_modes = list(product(states, range(6)))

    @pytest.mark.parametrize("states, modes", states_modes)
    def test_v_symmetry(self, states, modes):
        """Test that each V matrix is symmetric"""
        # pylint: disable=arguments-out-of-order
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))

        for i in range(states):
            for j in range(states):
                assert vham.v_matrix(i, j) == vham.v_matrix(j, i)

    @pytest.mark.parametrize("states, modes", states_modes)
    def test_fragment_against_v(self, states, modes):
        """Test that the fragments have correct block structure"""
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))

        for i in range(states):
            h = vham.fragment(i)
            for j, k in zip(range(states), range(states)):
                if k == i ^ j:
                    assert h.get_block(j, k) == vham.v_matrix(j, k)
                else:
                    assert h.get_block(j, k) == VibronicWord({})

    states = [2**i for i in range(1, 6)]
    states_modes = list(product(states, range(5)))

    @pytest.mark.parametrize("states, modes", states_modes)
    def test_y(self, states, modes):
        """Test that Y = [H_0, H_1] is correct"""
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))
        h0 = vham.fragment(0)
        h1 = vham.fragment(1)
        y = commutator(h0, h1)

        mat1 = (vham.v_matrix(0, 0) @ vham.v_matrix(0, 1)) - (
            vham.v_matrix(0, 1) @ vham.v_matrix(1, 1)
        )
        mat2 = (vham.v_matrix(1, 1) @ vham.v_matrix(0, 1)) - (
            vham.v_matrix(0, 1) @ vham.v_matrix(0, 0)
        )

        assert y.get_block(0, 0) == VibronicWord({})
        assert y.get_block(0, 1) == mat1
        assert y.get_block(1, 0) == mat2
        assert y.get_block(1, 1) == VibronicWord({})

    @pytest.mark.parametrize("modes", range(5))
    def test_epsilon(self, modes):
        """Test that epsilon is correct for N=2"""
        states = 2
        delta = 0.72
        scalar = -(delta**2) / 24
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))

        terms = [
            vham.commute_fragments(0, 0, 1),
            2 * vham.commute_fragments(1, 0, 1),
            2 * vham.commute_fragments(2, 0, 1),
            vham.commute_fragments(0, 0, 2),
            2 * vham.commute_fragments(1, 0, 2),
            2 * vham.commute_fragments(2, 0, 2),
            vham.commute_fragments(1, 1, 2),
            2 * vham.commute_fragments(2, 1, 2),
        ]

        actual = vham.epsilon(delta)
        expected = scalar * sum(terms, VibronicBlockMatrix(dim=states))

        assert actual == expected

"""Tests for the VibronicHamiltonian class"""
from itertools import product

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from vibronic import VibronicBlockMatrix, VibronicHamiltonian, VibronicWord, commutator, momentum_operator, position_operator, _term_to_matrix, _matrix_from_op, _tensor_with_id


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

    states = [2**i for i in range(2)]
    states_modes = list(product(states, range(2)))

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
    states_modes = list(product(states, range(3)))

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

    @pytest.mark.parametrize("states, modes", states_modes)
    @pytest.mark.parametrize("gridpoints", range(1,3))
    def test_matrix_add(self, states, modes, gridpoints):
        """Test that the matrix representation is consistent with addition"""
        vham1 = VibronicHamiltonian(states, modes, *coeffs(states, modes))
        vham2 = VibronicHamiltonian(states, modes, *coeffs(states, modes))

        mat1 = vham1.matrix(gridpoints)
        mat2 = vham2.matrix(gridpoints)

        assert np.allclose((mat1 + mat2).todense(), (vham1 + vham2).matrix(gridpoints).todense())


    #@pytest.mark.parametrize("states, modes", states_modes)
    #@pytest.mark.parametrize("gridpoints", range(1,5))
    #def test_matrix_mul(self, states, modes, gridpoints):
    #    """Test that the matrix representation is consistent with addition"""
    #    vham1 = VibronicHamiltonian(states, modes, *coeffs(states, modes))
    #    vham2 = VibronicHamiltonian(states, modes, *coeffs(states, modes))

    #    mat1 = vham1.matrix(gridpoints)
    #    mat2 = vham2.matrix(gridpoints)

    #    assert (mat1 @ mat2) == (vham1 @ vham2).matrix(gridpoints)

    @pytest.mark.parametrize("states, modes", states_modes)
    @pytest.mark.parametrize("gridpoints", range(1,3))
    @pytest.mark.parametrize("scalar", [0, 1, -1, 2.2, -1.7, 5j, -2 + 1j])
    def test_matrix_scalar_mul(self, states, modes, gridpoints, scalar):
        """Test that the matrix representation is consistent with addition"""
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes))

        assert np.allclose((scalar*vham).matrix(gridpoints).todense(), scalar*vham.matrix(gridpoints).todense())

    params = [
        ("P", 2, momentum_operator(2, 1)),
        ("Q", 2, position_operator(2, 1)),
        ("P", 3, momentum_operator(3, 1)),
        ("Q", 3, position_operator(3, 1)),
        ("PQ", 2, momentum_operator(2, 1) @ position_operator(2, 1)),
        ("QP", 2, position_operator(2, 1) @ momentum_operator(2, 1)),
        ("PQ", 3, momentum_operator(3, 1) @ position_operator(3, 1)),
        ("QP", 3, position_operator(3, 1) @ momentum_operator(3, 1)),
    ]


    @pytest.mark.parametrize("term, gridpoints, expected", params)
    def test_term_to_matrix(self, term, gridpoints, expected):
        assert np.allclose(_term_to_matrix(term, gridpoints).todense(), expected.todense())


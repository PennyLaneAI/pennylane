import pytest

import numpy as np
from vibronic import VibronicBlockMatrix, VibronicWord, VibronicHamiltonian, commutator

def coeffs(states, modes):
    alphas = np.random.random((states, states, modes))
    betas = np.random.random((states, states, modes, modes))
    lambdas = np.random.random((states, states))
    omegas = np.random.random((modes,))

    symmetric_alphas = np.zeros(alphas.shape)
    symmetric_betas = np.zeros(betas.shape)
    symmetric_lambdas = np.zeros(lambdas.shape)

    for i in range(states):
        for j in range(states):
            betas[i, j] = (betas[i, j] + betas[i, j].T)/2

    for i in range(states):
        for j in range(states):
            symmetric_alphas[i, j] = (alphas[i, j] + alphas[j, i])/2
            assert np.allclose(betas[i, j], betas[i, j].T)
            assert np.allclose(betas[j, i], betas[j, i].T)
            symmetric_betas[i, j] = (betas[i, j] + betas[j, i])/2
            symmetric_lambdas[i, j] = (lambdas[i, j] + lambdas[j, i])/2

    return symmetric_alphas, symmetric_betas, symmetric_lambdas, omegas

class TestVibronic:
    """Test vibronic arithmetic"""

    def test_v_symmetry(self):
        n = 3
        m = 4

        vham = VibronicHamiltonian(n, m, *coeffs(n, m))

        for i in range(n):
            for j in range(n):
                assert vham.v_matrix(i, j) == vham.v_matrix(j, i) #pylint: disable=arguments-out-of-order


    def test_h0(self):
        n = 2
        m = 4
        vham = VibronicHamiltonian(n, m, *coeffs(n, m))
        h0 = vham.fragment(0)

        assert h0.get_block(0, 0) == vham.v_matrix(0, 0)
        assert h0.get_block(1, 1) == vham.v_matrix(1, 1)
        assert h0.get_block(0, 1) == VibronicWord({})
        assert h0.get_block(1, 0) == VibronicWord({})

    def test_h1(self):
        n = 2
        m = 4

        vham = VibronicHamiltonian(n, m, *coeffs(n, m))
        h1 = vham.fragment(1)

        assert h1.get_block(0, 1) == vham.v_matrix(0, 1)
        assert h1.get_block(1, 0) == vham.v_matrix(1, 0)
        assert h1.get_block(0, 0) == VibronicWord({})
        assert h1.get_block(1, 1) == VibronicWord({})

    def test_y(self):
        n = 2
        m = 2

        vham = VibronicHamiltonian(n, m, *coeffs(n, m))
        h0 = vham.fragment(0)
        h1 = vham.fragment(1)
        y = commutator(h0, h1)

        mat1 = (vham.v_matrix(0, 0) @ vham.v_matrix(0, 1)) - (vham.v_matrix(0, 1) @ vham.v_matrix(1, 1))
        mat2 = (vham.v_matrix(1, 1) @ vham.v_matrix(0, 1)) - (vham.v_matrix(0, 1) @ vham.v_matrix(0, 0))

        assert y.get_block(0, 0) == VibronicWord({})
        assert y.get_block(0, 1) == mat1
        assert y.get_block(1, 0) == mat2
        assert y.get_block(1, 1) == VibronicWord({})

    def test_epsilon(self):
        n = 2
        m = 4

        delta = 0.72
        scalar = -(delta**2)/24

        vham = VibronicHamiltonian(n, m, *coeffs(n, m))
        h0 = vham.fragment(0)
        h1 = vham.fragment(1)
        h2 = vham.fragment(2)

        #terms = [
        #    commutator(h0, commutator(h0, h1)),
        #    2*commutator(h1, commutator(h0, h1)),
        #    2*commutator(h2, commutator(h0, h1)),
        #    commutator(h0, commutator(h0, h2)),
        #    2*commutator(h1, commutator(h0, h2)),
        #    2*commutator(h2, commutator(h0, h2)),
        #    commutator(h1, commutator(h1, h2)),
        #    2*commutator(h2, commutator(h1, h2)),
        #]

        terms = [
            vham.commute_fragments(0, 0, 1),
            2*vham.commute_fragments(1, 0, 1),
            2*vham.commute_fragments(2, 0, 1),
            vham.commute_fragments(0, 0, 2),
            2*vham.commute_fragments(1, 0, 2),
            2*vham.commute_fragments(2, 0, 2),
            vham.commute_fragments(1, 1, 2),
            2*vham.commute_fragments(2, 1, 2),
        ]

        actual = vham.epsilon(delta)
        expected = scalar*sum(terms, VibronicBlockMatrix(dim=n))

        assert actual == expected

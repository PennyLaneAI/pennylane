"""Common utilities for the Vibronic classes"""

import numpy as np


def coeffs(states: int, modes: int):
    """Produce random coefficients for input"""

    alphas = np.random.random(size=(states, states, modes))
    betas = np.random.random(size=(states, states, modes, modes))
    lambdas = np.random.random(size=(states, states))
    omegas = np.random.random(size=(modes,))

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


def is_pow_2(k: int) -> bool:
    """Test if k is a power of two"""
    if (k & (k - 1) == 0) or k == 0:
        return True

    return False

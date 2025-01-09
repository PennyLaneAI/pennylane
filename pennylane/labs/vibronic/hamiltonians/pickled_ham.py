"""Load picked Hamiltonian on 21 modes"""

import pickle
from math import ceil, log2, pow

import numpy as np

n_states = 6
n_modes = 21
n_blocks = int(pow(2, ceil(log2(n_states))))


def get_coeffs():
    """Return the coefficients of the pickled ham"""

    with open("n4o4a_ts.pkl", "rb") as pkl_file:
        omegas, couplings = pickle.load(pkl_file)

    # Vertical excitation energies
    equilibrium_energies = np.array(
        [
            -0.697135074556189,
            -0.5422856203390646,
            -0.5035165463514143,
            -0.36175762855222365,
            -0.697135074556189,
            0.49521646056560054,
        ]
    )

    # Vertical excitation energies offsetted
    offset = (
        np.min(equilibrium_energies)
        + (np.max(equilibrium_energies) - np.min(equilibrium_energies)) / 2
    )
    equilibrium_energies_offsetted = equilibrium_energies - offset

    lambdas = np.zeros((n_blocks, n_blocks))
    alphas = np.zeros((n_blocks, n_blocks, n_modes))
    betas = np.zeros((n_blocks, n_blocks, n_modes, n_modes))

    for i in range(n_states):
        lambdas[i, i] = equilibrium_energies_offsetted[i]

    for j in range(n_states):
        betas[j][j] = couplings[2][j][j]
        for i in range(n_modes):
            betas[j][j][i][i] += omegas[i] / 2
        for k in range(j):
            betas[j][k] = betas[k][j] = couplings[2][j][k]

    for j in range(n_states):
        alphas[j][j] = couplings[1][j][j]
        for k in range(j):
            alphas[j][k] = alphas[k][j] = couplings[1][j][k]

    return alphas, betas, lambdas, omegas

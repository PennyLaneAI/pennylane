"""Common utilities for the Vibronic classes"""

import numpy as np


def coeffs(states: int, modes: int, order: int):
    """Produce random coefficients for input"""

    phis = []
    symmetric_phis = []
    for i in range(order + 1):
        shape = (states, states) + (modes,) * i
        phi = np.random.random(size=shape)
        phis.append(phi)
        symmetric_phis.append(np.zeros(shape))

    for phi in phis:
        for i in range(states):
            for j in range(states):
                phi[i, j] = (phi[i, j] + phi[i, j].T) / 2

    for phi, symmetric_phi in zip(phis, symmetric_phis):
        for i in range(states):
            for j in range(states):
                symmetric_phi[i, j] = (phi[i, j] + phi[j, i]) / 2

    return np.random.random(size=(modes,)), symmetric_phis


def is_pow_2(k: int) -> bool:
    """Test if k is a power of two"""
    return (k & (k - 1) == 0) or k == 0


def next_pow_2(k: int) -> int:
    """Return the smallest power of 2 greater than or equal to k"""
    return 2 ** (k - 1).bit_length()

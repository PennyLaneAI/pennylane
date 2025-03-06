import pytest
import numpy as np

from pennylane.labs.trotter import nested_commutator
from pennylane.labs.trotter.fragments import vibronic_fragments
from pennylane.labs.trotter.product_formulas import trotter
from pennylane.labs.trotter.realspace import VibronicMatrix

def _coeffs(states: int, modes: int, order: int):
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

@pytest.mark.parametrize("modes", range(5))
def test_epsilon(modes):
    """Test that epsilon is correct for 2 states"""
    states = 2
    delta = 0.72
    scalar = -(delta**2) / 24
    fragments = vibronic_fragments(states, modes, *_coeffs(states, modes, order=2))

    terms = [
        nested_commutator([fragments[0], fragments[0], fragments[1]]),
        2 * nested_commutator([fragments[1], fragments[0], fragments[1]]),
        2 * nested_commutator([fragments[2], fragments[0], fragments[1]]),
        nested_commutator([fragments[0], fragments[0], fragments[2]]),
        2 * nested_commutator([fragments[1], fragments[0], fragments[2]]),
        2 * nested_commutator([fragments[2], fragments[0], fragments[2]]),
        nested_commutator([fragments[1], fragments[1], fragments[2]]),
        2 * nested_commutator([fragments[2], fragments[1], fragments[2]])
    ]

    actual = trotter(fragments, delta, order=2)
    expected = scalar * sum(terms, VibronicMatrix(states, modes))

    assert actual == expected

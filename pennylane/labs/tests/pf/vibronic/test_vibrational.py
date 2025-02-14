"""Test the VibrationalHamiltonian class"""

import numpy as np

from pennylane.labs.pf import HOState, VibrationalHamiltonian


def test_expectation_1_mode():
    """Test the expectation computation against known values"""

    freq = 1.2345
    n_states = 5

    omegas = np.array([freq])
    ham = VibrationalHamiltonian(1, omegas, []).operator()
    states = [HOState.from_dict(1, 10, {(i,): 1}) for i in range(n_states)]

    expected = np.diag(np.arange(n_states) + 0.5) * freq
    actual = np.zeros(shape=(n_states, n_states), dtype=np.complex128)

    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            actual[i, j] = state1.dot(ham.apply(state2))

    assert np.allclose(actual, expected)

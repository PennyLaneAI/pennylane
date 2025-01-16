"""Test the norms against known values"""

import numpy as np

from pennylane.labs.vibronic import VibronicHamiltonian
from pennylane.labs.vibronic.hamiltonians.spin_vibronic_ham import get_coeffs as ten_mode


def test_against_10_mode():
    expected = 1.6621227513071748

    vham = VibronicHamiltonian(8, 10, *ten_mode(), sparse=True)
    ep = vham.epsilon(1)
    actual = ep.norm(4)

    assert np.isclose(actual, expected)

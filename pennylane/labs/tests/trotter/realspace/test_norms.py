"""Test the norms against known values"""

import numpy as np

from pennylane.labs.tests.trotter.realspace.spin_vibronic_ham import get_coeffs as ten_mode
from pennylane.labs.trotter.fragments import vibronic_fragments
from pennylane.labs.trotter.product_formulas import trotter_error


def test_against_10_mode():
    """Test norm against precomputed value"""
    expected = 1.6621227513071748

    frags = vibronic_fragments(6, 10, *ten_mode())
    ep = trotter_error(frags, 1)

    params = {"gridpoints": 4, "sparse": True}

    actual = ep.norm(params)

    assert np.isclose(actual, expected)

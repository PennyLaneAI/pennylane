# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the second order trotter error calculation"""

import numpy as np
import pytest

from pennylane.labs.trotter import nested_commutator
from pennylane.labs.trotter.fragments import vibronic_fragments
from pennylane.labs.trotter.product_formulas import trotter_error
from pennylane.labs.trotter.realspace import VibronicMatrix


def _coeffs(states: int, modes: int, order: int):
    """Produce random coefficients used to construct a Vibronic Hamiltonian. The coefficients are modified to reflect the symmetries obtained by commuting operators."""

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
        2 * nested_commutator([fragments[2], fragments[1], fragments[2]]),
    ]

    actual = trotter_error(fragments, delta)
    expected = scalar * sum(terms, VibronicMatrix(states, modes))

    assert actual == expected

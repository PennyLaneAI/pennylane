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
"""Tests for Vibronic Hamiltonian"""

from itertools import product

import numpy as np
import pytest
import scipy as sp

from pennylane.labs.trotter_error.fragments import vibronic_fragments
from pennylane.labs.trotter_error.realspace import HOState, RealspaceMatrix, VibronicHO

# pylint: disable=no-self-use


def _vibronic_hamiltonian(states, modes, freqs, taylor_coeffs):
    frags = vibronic_fragments(states, modes, freqs, taylor_coeffs)
    return sum(frags, RealspaceMatrix.zero(states, modes))


def test_vibronic_fragments():
    """Test that vibronic_fragments returns RealspaceMatrix objects with the correct number of states and modes."""
    n_states = 5
    n_modes = 5
    freqs = np.array([1, 2, 3, 4, 5])

    frags = vibronic_fragments(n_states, n_modes, freqs, [])

    for frag in frags:
        assert isinstance(frag, RealspaceMatrix)
        assert frag.states == n_states
        assert frag.modes == n_modes


class Test1Mode:
    """Test a simple one mode, one state vibronic Hamiltonian"""

    freq = 1.2345
    n_states = 5
    omegas = np.array([freq])
    ham = _vibronic_hamiltonian(1, 1, omegas, [])
    states = [VibronicHO(1, 1, 10, [HOState(1, 10, {(i,): 1})]) for i in range(n_states)]

    @pytest.mark.parametrize("n_states, freq, ham, states", [(n_states, freq, ham, states)])
    def test_expectation_1_mode(self, n_states, freq, ham, states):
        """Test the expectation computation against known values"""

        expected = np.diag(np.arange(n_states) + 0.5) * freq
        actual = np.zeros(shape=(n_states, n_states), dtype=np.complex128)

        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                actual[i, j] = ham.expectation(state1, state2)

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("n_states, freq, ham, states", [(n_states, freq, ham, states)])
    def test_linear_combination_1_mode(self, n_states, freq, ham, states):
        """Test the expectation of a linear combination of harmonics"""

        rot_log = np.zeros(shape=(n_states, n_states))

        for i in range(n_states):
            for j in range(i):
                rot_log[i, j] = (i + 1) * (j + 1)
                rot_log[j, i] = -(i + 1) * (j + 1)

        rot = sp.linalg.expm(rot_log)

        comb_states = []
        for i in range(n_states):
            state = sum(
                (states[j] * rot[j, i] for j in range(n_states)), VibronicHO.zero_state(1, 1, 10)
            )
            comb_states.append(state)

        expected = rot.T @ (np.diag(np.arange(n_states) + 0.5) * freq) @ rot
        actual = np.zeros(shape=(n_states, n_states), dtype=np.complex128)

        for i, state1 in enumerate(comb_states):
            for j, state2 in enumerate(comb_states):
                actual[i, j] = ham.expectation(state1, state2)

        assert np.allclose(actual, expected)


class TestHarmonic:
    """Test a simple 1 state, 2 mode vibronic Hamiltonian"""

    n_states = 3
    omegas = np.array([1, 2.3])
    ham = _vibronic_hamiltonian(1, 2, omegas, [])
    states = [
        VibronicHO(1, 2, 10, [HOState(2, 10, {(i, j): 1})])
        for i, j in product(range(n_states), repeat=2)
    ]

    excitations = list(product(range(n_states), repeat=2))

    @pytest.mark.parametrize(
        "omegas, ham, states, excitations", [(omegas, ham, states, excitations)]
    )
    def test_harmonic(self, omegas, ham, states, excitations):
        """Test the expectation value of a harmonic"""

        expected = np.zeros((len(states), len(states)), dtype=np.complex128)
        for i in range(len(states)):
            expected[i, i] = (0.5 + excitations[i][0]) * omegas[0] + (
                0.5 + excitations[i][1]
            ) * omegas[1]

        actual = np.zeros((len(states), len(states)), dtype=np.complex128)
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                actual[i, j] = ham.expectation(state1, state2)

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize(
        "omegas, ham, states, excitations", [(omegas, ham, states, excitations)]
    )
    def test_linear_combination_harmonic(self, omegas, ham, states, excitations):
        """Test the expectation value of a linear combintaion of harmonics"""

        rot_log = np.zeros((len(states), len(states)))
        for i in range(len(states)):
            for j in range(i):
                rot_log[i, j] = (i + 1) * (j + 1)
                rot_log[j, i] = -(i + 1) * (j + 1)

        rot = sp.linalg.expm(rot_log)

        comb_states = []
        for i in range(len(states)):
            state = sum(
                (states[j] * rot[j, i] for j in range(len(states))), VibronicHO.zero_state(1, 2, 10)
            )
            comb_states.append(state)

        expected = np.zeros((len(states), len(states)), dtype=np.complex128)
        for i in range(len(states)):
            expected[i, i] = (0.5 + excitations[i][0]) * omegas[0] + (
                0.5 + excitations[i][1]
            ) * omegas[1]

        expected = rot.T @ expected @ rot

        actual = np.zeros((len(states), len(states)), dtype=np.complex128)
        for i, state1 in enumerate(comb_states):
            for j, state2 in enumerate(comb_states):
                actual[i, j] = ham.expectation(state1, state2)

        assert np.allclose(actual, expected)

"""Test the VibrationalHamiltonian class"""

import itertools

import numpy as np
import pytest
import scipy as sp

from pennylane.labs.pf import HOState, VibrationalHamiltonian


class TestFragments:
    """Test properties of the fragments"""

    @pytest.mark.parametrize(
        "n_modes, omegas, phis, gridpoints",
        [
            (5, np.random.random(5), [], 2),
            (5, np.random.random(5), [np.random.random(size=(5,) * i) for i in range(4)], 2),
        ],
    )
    def test_fragementation_schemes_equal(self, n_modes, omegas, phis, gridpoints):
        ham = VibrationalHamiltonian(n_modes, omegas, phis)

        frag1 = (ham.harmonic_fragment() + ham.anharmonic_fragment()).matrix(gridpoints, n_modes)
        frag2 = (ham.kinetic_fragment() + ham.potential_fragment()).matrix(gridpoints, n_modes)

        assert np.allclose(frag1, frag2)


class Test1Mode:
    """Test the vibrational Hamiltonian on a single mode"""

    freq = 1.2345
    n_states = 5
    omegas = np.array([freq])
    ham = VibrationalHamiltonian(1, omegas, []).operator()
    states = [HOState.from_dict(1, 10, {(i,): 1}) for i in range(n_states)]

    @pytest.mark.parametrize("n_states, freq, ham, states", [(n_states, freq, ham, states)])
    def test_expectation_1_mode(self, n_states, freq, ham, states):
        """Test the expectation computation against known values"""

        expected = np.diag(np.arange(n_states) + 0.5) * freq
        actual = np.zeros(shape=(n_states, n_states), dtype=np.complex128)

        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                actual[i, j] = state1.dot(ham.apply(state2))

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("n_states, freq, ham, states", [(n_states, freq, ham, states)])
    def test_linear_combination_harmonic(self, n_states, freq, ham, states):
        """Test the expectation of a linear combination of harmonics"""

        rot_log = np.zeros(shape=(n_states, n_states))

        for i in range(n_states):
            for j in range(i):
                rot_log[i, j] = (i + 1) * (j + 1)
                rot_log[j, i] = -(i + 1) * (j + 1)

        rot = sp.linalg.expm(rot_log)

        comb_states = []
        for i in range(n_states):
            state = sum((states[j] * rot[j, i] for j in range(n_states)), HOState.zero_state(1, 10))
            comb_states.append(state)

        expected = rot.T @ (np.diag(np.arange(n_states) + 0.5) * freq) @ rot
        actual = np.zeros(shape=(n_states, n_states), dtype=np.complex128)

        for i, state1 in enumerate(comb_states):
            for j, state2 in enumerate(comb_states):
                actual[i, j] = state1.dot(ham.apply(state2))

        assert np.allclose(actual, expected)


class TestMultiMode:
    """Test the vibrational Hamiltonian with multiple modes"""

    n_states = 3
    omegas = np.array([1, 2.3])
    ham = VibrationalHamiltonian(2, omegas, []).operator()
    states = [
        HOState.from_dict(2, 10, {(i, j): 1})
        for i, j in itertools.product(range(n_states), repeat=2)
    ]
    excitations = list(itertools.product(range(n_states), repeat=2))

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
                actual[i, j] = state1.dot(ham.apply(state2))

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
                (states[j] * rot[j, i] for j in range(len(states))), HOState.zero_state(2, 10)
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
                actual[i, j] = state1.dot(ham.apply(state2))

        assert np.allclose(actual, expected)

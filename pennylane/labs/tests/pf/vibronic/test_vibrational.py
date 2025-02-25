"""Test the VibrationalHamiltonian class"""

import itertools

import numpy as np
import pytest
import scipy as sp

import pennylane as qml
from pennylane.labs.pf import HOState, VibrationalHamiltonian
from pennylane.labs.vibrational import vibrational_pes


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

class TestExpectation:

    @pytest.mark.parametrize("symbols, charge, geometry", [
        (['H', 'H'], 0, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
        (['H', 'H', 'H'], 1, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]))
    ])
    def test_molecular_ham(self, symbols, charge, geometry):
        """Test the exepctation value against PL"""

        mol = qml.qchem.Molecule(symbols, geometry, charge=charge)
        pes = vibrational_pes(mol, quad_order=5, cubic=True, dipole_level=1)

        phi_one_body, phi_two_body, phi_three_body = qml.qchem.taylor_coeffs(pes)
        omegas = pes.freqs

        print("one")
        print(phi_one_body)
        print("two")
        print(phi_two_body)
        print("three")
        print(phi_three_body)

        taylor_ham_qubit = qml.qchem.taylor_hamiltonian(pes)
        taylor_ham_matrix = taylor_ham_qubit.sparse_matrix()
        taylor_ham_eigdecomp = taylor_ham_qubit.eigendecomposition
        taylor_ham_eigvec = taylor_ham_eigdecomp['eigvec']
        taylor_ham_eigval = taylor_ham_eigdecomp['eigval']

        vham = VibrationalHamiltonian(len(omegas), omegas, [np.array(0), phi_one_body, phi_two_body, phi_three_body])

        for eigvec, eigval, in zip(taylor_ham_eigvec, taylor_ham_eigval):
            print(eigvec, eigval)

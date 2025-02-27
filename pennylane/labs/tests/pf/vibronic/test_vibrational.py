"""Test the VibrationalHamiltonian class"""

import itertools

import numpy as np
import pytest
import scipy as sp

import pennylane as qml
from pennylane.labs.pf import HOState, VibrationalHamiltonian
from pennylane.labs.vibrational import vibrational_pes
from pennylane.qchem.vibrational.taylor_ham import _threebody_degs, _twobody_degs


def _transform_coeffs(phis):
    """Transform the coefficients"""

    taylor_1, taylor_2, taylor_3 = phis

    start_deg = 2
    n_modes, num_1d_coeffs = np.shape(taylor_1)
    taylor_deg = num_1d_coeffs + start_deg - 1

    phis = []
    for i in range(5):
        shape = (n_modes,) * (i + 1)
        phis.append(np.zeros(shape))

    # one-mode
    for m1 in range(n_modes):
        for deg_i in range(start_deg, taylor_deg + 1):
            index = (m1,) * deg_i
            phis[deg_i - 1][index] = taylor_1[m1, deg_i - start_deg]

    # two-mode
    degs_2d = _twobody_degs(taylor_deg, min_deg=start_deg)
    for m1 in range(n_modes):
        for m2 in range(m1):
            for deg_idx, Qs in enumerate(degs_2d):
                q1deg, q2deg = Qs[:2]
                index = ((m1,) * q1deg) + ((m2,) * q2deg)
                phis[q1deg + q2deg - 1][index] = taylor_2[m1, m2, deg_idx]

    # three-mode
    degs_3d = _threebody_degs(taylor_deg, min_deg=start_deg)
    for m1 in range(n_modes):
        for m2 in range(m1):
            for m3 in range(m2):
                for deg_idx, Qs in enumerate(degs_3d):
                    q1deg, q2deg, q3deg = Qs[:3]
                    index = ((m1,) * q1deg) + ((m2,) * q2deg) + ((m3,) * q3deg)
                    phis[q1deg + q2deg + q3deg - 1][index] = taylor_3[m1, m2, m3, deg_idx]

    return phis


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


class TestHarmonic1Mode:
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


class TestHarmonicMultiMode:
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

    mat1 = np.array(
        [[0.01095216 + 0.0j, 0.00018082 + 0.0j], [0.00018082 + 0.0j, 0.01095216 + 0.0j]]
    )
    mat2 = np.array(
        [
            [
                1.85130786e-02 + 0.0j,
                4.00934485e-05 + 0.0j,
                1.10670334e-03 + 0.0j,
                -4.01192823e-04 + 0.0j,
                2.06315706e-03 + 0.0j,
                1.70409266e-04 + 0.0j,
                -5.92787860e-04 + 0.0j,
                -7.53000830e-05 + 0.0j,
            ],
            [
                4.00934485e-05 + 0.0j,
                1.85130786e-02 + 0.0j,
                -4.01192823e-04 + 0.0j,
                1.10670334e-03 + 0.0j,
                1.70409266e-04 + 0.0j,
                2.06315706e-03 + 0.0j,
                -7.53000830e-05 + 0.0j,
                -5.92787860e-04 + 0.0j,
            ],
            [
                1.10670334e-03 + 0.0j,
                -4.01192823e-04 + 0.0j,
                1.85130786e-02 + 0.0j,
                4.00934485e-05 + 0.0j,
                -5.92787860e-04 + 0.0j,
                -7.53000830e-05 + 0.0j,
                2.06315706e-03 + 0.0j,
                1.70409266e-04 + 0.0j,
            ],
            [
                -4.01192823e-04 + 0.0j,
                1.10670334e-03 + 0.0j,
                4.00934485e-05 + 0.0j,
                1.85130786e-02 + 0.0j,
                -7.53000830e-05 + 0.0j,
                -5.92787860e-04 + 0.0j,
                1.70409266e-04 + 0.0j,
                2.06315706e-03 + 0.0j,
            ],
            [
                2.06315706e-03 + 0.0j,
                1.70409266e-04 + 0.0j,
                -5.92787860e-04 + 0.0j,
                -7.53000830e-05 + 0.0j,
                1.85130786e-02 + 0.0j,
                4.00934485e-05 + 0.0j,
                1.10670334e-03 + 0.0j,
                -4.01192823e-04 + 0.0j,
            ],
            [
                1.70409266e-04 + 0.0j,
                2.06315706e-03 + 0.0j,
                -7.53000830e-05 + 0.0j,
                -5.92787860e-04 + 0.0j,
                4.00934485e-05 + 0.0j,
                1.85130786e-02 + 0.0j,
                -4.01192823e-04 + 0.0j,
                1.10670334e-03 + 0.0j,
            ],
            [
                -5.92787860e-04 + 0.0j,
                -7.53000830e-05 + 0.0j,
                2.06315706e-03 + 0.0j,
                1.70409266e-04 + 0.0j,
                1.10670334e-03 + 0.0j,
                -4.01192823e-04 + 0.0j,
                1.85130786e-02 + 0.0j,
                4.00934485e-05 + 0.0j,
            ],
            [
                -7.53000830e-05 + 0.0j,
                -5.92787860e-04 + 0.0j,
                1.70409266e-04 + 0.0j,
                2.06315706e-03 + 0.0j,
                -4.01192823e-04 + 0.0j,
                1.10670334e-03 + 0.0j,
                4.00934485e-05 + 0.0j,
                1.85130786e-02 + 0.0j,
            ],
        ]
    )

    H2_params = (["H", "H"], 0, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), mat1)
    H3_params = (
        ["H", "H", "H"],
        1,
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        mat2,
    )

    @pytest.mark.parametrize("symbols, charge, geometry, expected", [H2_params, H3_params])
    def test_matrix(self, symbols, charge, geometry, expected):
        """Test the matrix"""

        mol = qml.qchem.Molecule(symbols, geometry, charge=charge)
        pes = vibrational_pes(mol, quad_order=5, cubic=True, dipole_level=1)

        phis = [np.array(0)] + _transform_coeffs(qml.qchem.taylor_coeffs(pes))
        omegas = pes.freqs

        ham = VibrationalHamiltonian(len(omegas), omegas, phis)
        actual = ham.operator().matrix(2, len(omegas), basis="harmonic")

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("symbols, charge, geometry, expected", [H2_params, H3_params])
    def test_expectation(self, symbols, charge, geometry, expected):
        """Test the expectation values"""

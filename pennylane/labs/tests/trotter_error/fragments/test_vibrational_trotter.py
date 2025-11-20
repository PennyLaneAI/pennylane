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
"""Test the VibrationalHamiltonian class"""

import itertools

import numpy as np
import pytest
import scipy as sp
from scipy.sparse import csr_array

from pennylane.labs.trotter_error.fragments import vibrational_fragments
from pennylane.labs.trotter_error.realspace import HOState, RealspaceSum

# pylint: disable=no-self-use


def _vibrational_hamiltonian(modes, freqs, taylor_coeffs):
    frags = vibrational_fragments(modes, freqs, taylor_coeffs)
    return sum(frags, RealspaceSum.zero(modes))


@pytest.mark.parametrize(
    "n_modes, omegas, phis, gridpoints",
    [
        (5, np.random.random(5), [], 2),
        (5, np.random.random(5), [np.random.random(size=(5,) * i) for i in range(4)], 2),
    ],
)
def test_fragementation_schemes_equal(n_modes, omegas, phis, gridpoints):
    """Test that harmonic + anharmonic = kinetic + potential"""

    harmonic, anharmonic = vibrational_fragments(n_modes, omegas, phis, frag_method="harmonic")
    kinetic, potential = vibrational_fragments(n_modes, omegas, phis, frag_method="kinetic")

    mat1 = (harmonic + anharmonic).matrix(gridpoints)
    mat2 = (kinetic + potential).matrix(gridpoints)

    assert np.allclose(mat1, mat2)


class TestHarmonic1Mode:
    """Test the vibrational Hamiltonian on a single mode"""

    freq = 1.2345
    n_states = 5
    omegas = np.array([freq])
    ham = _vibrational_hamiltonian(1, omegas, [])
    states = [HOState(1, 10, {(i,): 1}) for i in range(n_states)]

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
    def test_linear_combination_harmonic(self, n_states, freq, ham, states):
        """Test the expectation of a linear combination of harmonics"""

        rot_log = np.zeros(shape=(n_states, n_states))

        for i in range(n_states):
            for j in range(i):
                rot_log[i, j] = (i + 1) * (j + 1)
                rot_log[j, i] = -rot_log[i, j]

        rot = sp.linalg.expm(rot_log)

        comb_states = []
        for i in range(n_states):
            state = sum((states[j] * rot[j, i] for j in range(n_states)), HOState.zero_state(1, 10))
            comb_states.append(state)

        expected = rot.T @ (np.diag(np.arange(n_states) + 0.5) * freq) @ rot
        actual = np.zeros(shape=(n_states, n_states), dtype=np.complex128)

        for i, state1 in enumerate(comb_states):
            for j, state2 in enumerate(comb_states):
                actual[i, j] = ham.expectation(state1, state2)

        assert np.allclose(actual, expected)


class TestHarmonicMultiMode:
    """Test the vibrational Hamiltonian with multiple modes"""

    n_states = 3
    omegas = np.array([1, 2.3])
    ham = _vibrational_hamiltonian(2, omegas, [])
    states = [HOState(2, 10, {(i, j): 1}) for i, j in itertools.product(range(n_states), repeat=2)]
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
                rot_log[j, i] = -rot_log[i, j]

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
                actual[i, j] = ham.expectation(state1, state2)

        assert np.allclose(actual, expected)


class TestExpectation:
    """Test the expectation values against precomputed expected results"""

    phis1 = [np.array(0), np.array([0.0]), np.array([[-0.00306787]]), np.array([[[0.00051144]]])]
    omegas1 = np.array([0.0249722])
    mat1 = np.array(
        [[0.01095216 + 0.0j, 0.00018082 + 0.0j], [0.00018082 + 0.0j, 0.01095216 + 0.0j]]
    )

    phis2 = [
        np.array(0),
        np.array([0.0, 0.0, 0.0]),
        np.array(
            [
                [-6.21741719e-14, 0.00000000e00, 0.00000000e00],
                [1.99211595e-13, 9.46571153e-04, 0.00000000e00],
                [-5.71982634e-14, 8.73074134e-04, -1.31910690e-03],
            ]
        ),
        np.array(
            [
                [
                    [5.53904193e-05, 0.00000000e00, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                ],
                [
                    [-4.98906472e-15, 0.00000000e00, 0.00000000e00],
                    [-3.00847220e-03, 7.84674613e-05, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                ],
                [
                    [5.97655606e-15, 0.00000000e00, 0.00000000e00],
                    [3.88965240e-14, -9.17507861e-05, 0.00000000e00],
                    [-2.99010685e-03, -3.09965385e-03, 1.61908332e-04],
                ],
            ]
        ),
    ]
    omegas2 = np.array([0.00978462, 0.00978489, 0.01663723])
    mat2 = np.array(
        [
            [
                1.79171056e-02,
                2.48044382e-05,
                -1.06815069e-03,
                4.36537067e-04,
                -2.10123449e-03,
                -2.85917137e-14,
                9.96152552e-14,
                1.37476535e-14,
            ],
            [
                2.48044382e-05,
                1.79171056e-02,
                4.36537067e-04,
                -1.06815069e-03,
                -2.85917137e-14,
                -2.10123449e-03,
                1.37476535e-14,
                9.96152552e-14,
            ],
            [
                -1.06815069e-03,
                4.36537067e-04,
                1.79171056e-02,
                2.48044382e-05,
                9.96152552e-14,
                1.37476535e-14,
                -2.10123449e-03,
                -2.85917137e-14,
            ],
            [
                4.36537067e-04,
                -1.06815069e-03,
                2.48044382e-05,
                1.79171056e-02,
                1.37476535e-14,
                9.96152552e-14,
                -2.85917137e-14,
                -2.10123449e-03,
            ],
            [
                -2.10123449e-03,
                -2.85917137e-14,
                9.96152552e-14,
                1.37476535e-14,
                1.79171056e-02,
                2.48044382e-05,
                -1.06815069e-03,
                4.36537067e-04,
            ],
            [
                -2.85917137e-14,
                -2.10123449e-03,
                1.37476535e-14,
                9.96152552e-14,
                2.48044382e-05,
                1.79171056e-02,
                4.36537067e-04,
                -1.06815069e-03,
            ],
            [
                9.96152552e-14,
                1.37476535e-14,
                -2.10123449e-03,
                -2.85917137e-14,
                -1.06815069e-03,
                4.36537067e-04,
                1.79171056e-02,
                2.48044382e-05,
            ],
            [
                1.37476535e-14,
                9.96152552e-14,
                -2.85917137e-14,
                -2.10123449e-03,
                4.36537067e-04,
                -1.06815069e-03,
                2.48044382e-05,
                1.79171056e-02,
            ],
        ]
    )

    @pytest.mark.parametrize(
        "phis, omegas, expected", [(phis1, omegas1, mat1), (phis2, omegas2, mat2)]
    )
    def test_matrix(self, phis, omegas, expected):
        """Test the matrix"""

        ham = _vibrational_hamiltonian(len(omegas), omegas, phis)
        actual = ham.matrix(2, basis="harmonic")

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize(
        "phis, omegas, expected", [(phis1, omegas1, mat1), (phis2, omegas2, mat2)]
    )
    def test_expectation(self, phis, omegas, expected):
        """Test the expectation values"""

        n_modes = len(omegas)
        eigvals, eigvecs = np.linalg.eig(expected)

        ham = _vibrational_hamiltonian(n_modes, omegas, phis)

        for i, eigval in enumerate(eigvals):
            eigvec = eigvecs[:, i]
            ho_state = HOState(n_modes, 2, csr_array(eigvec.reshape(2**n_modes, 1)))
            assert np.isclose(ham.expectation(ho_state, ho_state), eigval)

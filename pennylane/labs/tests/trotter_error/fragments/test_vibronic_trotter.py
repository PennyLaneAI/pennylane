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

import math
from itertools import product

import numpy as np
import pytest
import scipy as sp

from pennylane.labs.trotter_error.fragments import vibronic_fragments
from pennylane.labs.trotter_error.realspace import (
    HOState,
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
    VibronicHO,
)

# pylint: disable=no-self-use


def _vibronic_hamiltonian(states, modes, freqs, taylor_coeffs):
    frags = vibronic_fragments(states, modes, freqs, taylor_coeffs)
    return sum(frags, RealspaceMatrix.zero(states, modes))


@pytest.mark.parametrize("states", range(1, 6))
@pytest.mark.parametrize("modes", range(1, 6))
def test_vibronic_fragments(states, modes):
    """Test that vibronic_fragments returns ``RealspaceMatrix`` objects with the correct number of
    states and modes."""
    freqs = np.random.random(modes)
    lambdas = np.random.random(size=(states, states))
    alphas = np.random.random(size=(states, states, modes))
    betas = np.random.random(size=(states, states, modes, modes))
    taylor_coeffs = [lambdas, alphas, betas]

    frags = vibronic_fragments(states, modes, freqs, taylor_coeffs)

    for frag in frags:
        assert isinstance(frag, RealspaceMatrix)
        assert frag.states == states
        assert frag.modes == modes


@pytest.mark.parametrize("states", range(1, 6))
@pytest.mark.parametrize("modes", range(1, 6))
def test_frag_schemes_equal(states, modes):
    """Test the two fragmentation schemes sum to the same Hamiltonian"""
    freqs = np.random.random(modes)
    lambdas = np.random.random(size=(states, states))
    alphas = np.random.random(size=(states, states, modes))
    betas = np.random.random(size=(states, states, modes, modes))
    taylor_coeffs = [lambdas, alphas, betas]

    frags_og = vibronic_fragments(states, modes, freqs, taylor_coeffs, scheme="og")
    ham_og = sum(frags_og, RealspaceMatrix.zero(states, modes))
    mat_og = ham_og.matrix(2)

    frags_mode = vibronic_fragments(states, modes, freqs, taylor_coeffs, scheme="mode")
    ham_mode = sum(frags_mode, RealspaceMatrix.zero(states, modes))
    mat_mode = ham_mode.matrix(2)

    assert mat_og.shape == mat_mode.shape
    assert np.allclose(mat_og, mat_mode)


def test_mode_based_fragments_1_state():
    """Test the mode based fragmentation scheme against a known example"""

    states = 1
    modes = 1
    blocks = 2 ** math.ceil(math.log2(states))

    idx0 = 0

    omegas = np.array([6.0])

    lambdas = np.zeros((states, states))
    alphas = np.zeros((states, states, modes))
    alphas[0, 0, idx0] = 2.5
    betas = np.zeros((states, states, modes, modes))

    taylor_coeffs = [lambdas.copy(), alphas.copy(), betas.copy()]

    frags = vibronic_fragments(blocks, modes, omegas, taylor_coeffs, scheme="mode")

    # Q^2 term
    exp0 = RealspaceMatrix.zero(blocks, modes)
    M = np.zeros((modes, modes))
    M[idx0, idx0] = omegas[idx0] / 2
    op = RealspaceOperator(modes, ("Q", "Q"), RealspaceCoeffs(M, label=f"beta[{idx0}][0,0]"))
    exp0.set_block(0, 0, RealspaceSum(modes, (op,)))

    # Q linear term
    exp1 = RealspaceMatrix.zero(blocks, modes)
    v = np.zeros(modes)
    v[idx0] = alphas[0, 0, idx0]
    opL = RealspaceOperator(modes, ("Q",), RealspaceCoeffs(v, label=f"alpha[{idx0}][0,0]"))
    exp1.set_block(0, 0, RealspaceSum(modes, (opL,)))

    # FC merges commuting potential fragments (exp0 + exp1)
    exp_potential = exp0 + exp1

    # kinetic term (appended after grouping)
    exp_kinetic = RealspaceMatrix.zero(blocks, modes)
    PP = RealspaceOperator(modes, ("P", "P"), RealspaceCoeffs(np.diag(omegas) / 2, label="omega"))
    exp_kinetic.set_block(0, 0, RealspaceSum(modes, (PP,)))

    assert len(frags) == 2
    assert np.allclose(frags[0].matrix(2), exp_potential.matrix(2))
    assert np.allclose(frags[1].matrix(2), exp_kinetic.matrix(2))


def test_mode_based_fragments_2_states():
    """Test the mode based fragmentation scheme against a known example"""
    states = 2
    modes = 2
    blocks = 2 ** math.ceil(math.log2(states))

    omegas = np.array([6.0, 4.0])

    lambdas = np.zeros((states, states))
    alphas = np.zeros((states, states, modes))

    # unequal diagonal on mode 0 -> coupling mat diag(2.5, 1.5)
    alphas[0, 0, 0] = 2.5
    alphas[1, 1, 0] = 1.5

    # off-diagonal on mode 1 -> coupling mat [[0, 0.7],[0.7, 0]] (does not commute)
    alphas[0, 1, 1] = 0.7
    alphas[1, 0, 1] = 0.7
    betas = np.zeros((states, states, modes, modes))

    taylor_coeffs = [lambdas.copy(), alphas.copy(), betas.copy()]

    frags = vibronic_fragments(states, modes, omegas, taylor_coeffs, scheme="mode")

    # Q^2 per mode (harmonic on electronic diagonal) — coupling ~ I, merges with diagonal Q
    q2_frags = []
    for r in range(modes):
        frag = RealspaceMatrix.zero(blocks, modes)
        for i in range(states):
            M = np.zeros((modes, modes))
            M[r, r] = omegas[r] / 2
            op = RealspaceOperator(
                modes, ("Q", "Q"), RealspaceCoeffs(M, label=f"beta[{r}][{i},{i}]")
            )
            frag.set_block(i, i, RealspaceSum(modes, (op,)))
        q2_frags.append(frag)

    # diagonal Q on mode 0
    exp_Q_diag = RealspaceMatrix.zero(blocks, modes)
    for i in range(states):
        v = np.zeros(modes)
        v[0] = alphas[i, i, 0]
        opL = RealspaceOperator(modes, ("Q",), RealspaceCoeffs(v, label=f"alpha[0][{i},{i}]"))
        exp_Q_diag.set_block(i, i, RealspaceSum(modes, (opL,)))

    # off-diagonal Q on mode 1
    exp_Q_off = RealspaceMatrix.zero(blocks, modes)
    for i in range(states):
        for j in range(states):
            if abs(alphas[i, j, 1]) > 1e-15:
                v = np.zeros(modes)
                v[1] = alphas[i, j, 1]
                opL = RealspaceOperator(
                    modes, ("Q",), RealspaceCoeffs(v, label=f"alpha[1][{i},{j}]")
                )
                exp_Q_off.set_block(i, j, RealspaceSum(modes, (opL,)))

    # FC group 0: Q2 modes + diagonal Q (commuting)
    exp0 = RealspaceMatrix.zero(blocks, modes)
    for frag in q2_frags + [exp_Q_diag]:
        exp0 = exp0 + frag

    # FC group 1: off-diagonal Q
    exp1 = exp_Q_off

    # kinetic
    exp2 = RealspaceMatrix.zero(blocks, modes)
    PP = RealspaceOperator(modes, ("P", "P"), RealspaceCoeffs(np.diag(omegas) / 2, label="omega"))
    for i in range(states):
        exp2.set_block(i, i, RealspaceSum(modes, (PP,)))

    assert len(frags) == 3
    assert frags[0] == exp0
    assert frags[1] == exp1
    assert frags[2] == exp2


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

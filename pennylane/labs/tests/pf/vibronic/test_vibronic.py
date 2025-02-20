"""Tests for Vibronic Hamiltonian"""

from itertools import product

import numpy as np
import pytest
import scipy as sp

from pennylane.labs.pf import (
    HOState,
    Node,
    RealspaceOperator,
    RealspaceSum,
    VibronicHamiltonian,
    VibronicHO,
    VibronicMatrix,
    coeffs,
    commutator,
    momentum_operator,
    position_operator,
)

# pylint: disable=protected-access,import-outside-toplevel


class TestVibronic:
    """Test the vibronic hamiltonians"""

    @pytest.mark.parametrize("modes", range(5))
    def test_epsilon(self, modes):
        """Test that epsilon is correct for 2 states"""
        states = 2
        delta = 0.72
        scalar = -(delta**2) / 24
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes, order=2))

        terms = [
            vham._commute_fragments(0, 0, 1),
            2 * vham._commute_fragments(1, 0, 1),
            2 * vham._commute_fragments(2, 0, 1),
            vham._commute_fragments(0, 0, 2),
            2 * vham._commute_fragments(1, 0, 2),
            2 * vham._commute_fragments(2, 0, 2),
            vham._commute_fragments(1, 1, 2),
            2 * vham._commute_fragments(2, 1, 2),
        ]

        actual = vham.epsilon(delta)
        expected = scalar * sum(terms, VibronicMatrix(states, modes))

        assert actual == expected

    states = [2**i for i in range(1, 3)]
    states_modes = list(product(states, range(5)))

    @pytest.mark.parametrize("states, modes", states_modes)
    def test_fragment_against_v(self, states, modes):
        """Test that the fragments have correct block structure"""
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes, order=2))

        zero_word = RealspaceSum([RealspaceOperator(tuple(), Node.tensor_node(np.array(0)))])

        for i in range(states):
            h = vham.fragment(i)
            for j, k in zip(range(states), range(states)):
                print(vham.v_word(j, k))
                if k == i ^ j:
                    assert h.block(j, k) == vham.v_word(j, k)
                else:
                    assert h.block(j, k) == zero_word

    @pytest.mark.parametrize("modes", range(5))
    def test_commute_h0_h1_on_two_states(self, modes):
        """Test that Y = [H_0, H_1] is correct"""
        states = 2

        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes, order=2))
        h0 = vham.fragment(0)
        h1 = vham.fragment(1)
        y = commutator(h0, h1)

        word1 = (vham.v_word(0, 0) @ vham.v_word(0, 1)) - (vham.v_word(0, 1) @ vham.v_word(1, 1))
        word2 = (vham.v_word(1, 1) @ vham.v_word(0, 1)) - (vham.v_word(0, 1) @ vham.v_word(0, 0))

        assert y.block(0, 0).is_zero
        assert y.block(0, 1) == word1
        assert y.block(1, 0) == word2
        assert y.block(1, 1).is_zero


class TestMatrix:
    """Test thet matrix representation"""

    states = [2**i for i in range(1, 3)]
    states_modes = list(product(states, range(1, 6)))

    @pytest.mark.parametrize("states, modes", states_modes)
    @pytest.mark.parametrize("gridpoints", range(1, 3))
    def test_matrix_add(self, states, modes, gridpoints):
        """Test that the matrix representation is consistent with addition"""
        vham1 = VibronicHamiltonian(states, modes, *coeffs(states, modes, order=2))
        vham2 = VibronicHamiltonian(states, modes, *coeffs(states, modes, order=2))

        mat1 = vham1.matrix(gridpoints)
        mat2 = vham2.matrix(gridpoints)

        assert np.allclose((mat1 + mat2), (vham1 + vham2).matrix(gridpoints))

    @pytest.mark.parametrize("states, modes", states_modes)
    @pytest.mark.parametrize("gridpoints", range(1, 3))
    @pytest.mark.parametrize("scalar", [0, 1, -1, 2.2, -1.7, 5j, -2 + 1j])
    def test_matrix_scalar_mul(self, states, modes, gridpoints, scalar):
        """Test that the matrix representation is consistent with addition"""
        vham = VibronicHamiltonian(states, modes, *coeffs(states, modes, order=2))

        assert np.allclose((scalar * vham).matrix(gridpoints), scalar * vham.matrix(gridpoints))

    single_term_params = [
        ((3, 4, (0,), [position_operator(4)]), np.kron(position_operator(4), np.eye(4**2))),
        (
            (4, 4, (0, 1), [position_operator(4), momentum_operator(4)]),
            np.kron(position_operator(4), np.kron(momentum_operator(4), np.eye(4**2))),
        ),
        (
            (4, 4, (1, 1), [position_operator(4), momentum_operator(4)]),
            np.kron(np.eye(4), np.kron(position_operator(4) @ momentum_operator(4), np.eye(4**2))),
        ),
    ]

    @pytest.mark.parametrize("params, expected", single_term_params)
    def test_single_term(self, params, expected):
        """Test the _single_term_matrix function"""
        from pennylane.labs.vibronic.utils.matrix_ops import _single_term_matrix

        actual = _single_term_matrix(*params)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


class TestApply:
    """Test that the vibronic matrix can be applied to a VibronicHO"""

    freq = 1.2345
    n_states = 5
    omegas = np.array([freq])
    ham = VibronicHamiltonian(1, 1, omegas, []).block_operator()
    states = [VibronicHO(1, 1, 10, [HOState.from_dict(1, 10, {(i,): 1})]) for i in range(n_states)]

    @pytest.mark.parametrize("n_states, freq, ham, states", [(n_states, freq, ham, states)])
    def test_expectation_1_mode(self, n_states, freq, ham, states):
        """Test the expectation computation against known values"""

        expected = np.diag(np.arange(n_states) + 0.5) * freq
        actual = np.zeros(shape=(n_states, n_states), dtype=np.complex128)

        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                actual[i, j] = state1.dot(ham.apply(state2))

        print(expected)
        print(actual)

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
                actual[i, j] = state1.dot(ham.apply(state2))

        assert np.allclose(actual, expected)

    n_states = 3
    omegas = np.array([1, 2.3])
    ham = VibronicHamiltonian(1, 2, omegas, []).block_operator()
    states = [
        VibronicHO(1, 2, 10, [HOState.from_dict(2, 10, {(i, j): 1})])
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
                actual[i, j] = state1.dot(ham.apply(state2))

        assert np.allclose(actual, expected)

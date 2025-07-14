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
"""Test the SparseFragment and SparseState classes"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pennylane.labs.trotter_error.fragments import sparse_fragments, SparseFragment, SparseState

# pylint: disable=no-self-use


def _sparse_hamiltonian(matrices):
    """Create a sparse Hamiltonian from a list of matrices."""
    frags = sparse_fragments(matrices)
    return sum(frags[1:], frags[0]) if frags else SparseFragment(csr_matrix((2, 2)))


def test_sparse_fragments_basic():
    """Test that sparse_fragments returns SparseFragment objects correctly."""

    # Test with identity and Pauli-X matrices
    identity = csr_matrix([[1, 0], [0, 1]])
    pauli_x = csr_matrix([[0, 1], [1, 0]])
    matrices = [identity, pauli_x]

    frags = sparse_fragments(matrices)

    assert len(frags) == 2
    for frag in frags:
        assert isinstance(frag, SparseFragment)


def test_sparse_fragments_empty():
    """Test that sparse_fragments handles empty input correctly."""
    frags = sparse_fragments([])
    assert frags == []


def test_sparse_fragments_type_error():
    """Test that sparse_fragments raises TypeError for non-csr_matrix inputs."""
    with pytest.raises(TypeError, match="Fragments must be csr_matrix objects"):
        sparse_fragments([np.array([[1, 0], [0, 1]])])


class TestBasicOperations:
    """Test basic arithmetic operations on SparseFragments"""

    def test_addition(self):
        """Test addition of SparseFragments"""
        mat1 = csr_matrix([[1, 0], [0, 1]])
        mat2 = csr_matrix([[0, 1], [1, 0]])

        frag1 = SparseFragment(mat1)
        frag2 = SparseFragment(mat2)

        result = frag1 + frag2
        expected = csr_matrix([[1, 1], [1, 1]])

        assert isinstance(result, SparseFragment)
        assert np.allclose(result.fragment.toarray(), expected.toarray())

    def test_subtraction(self):
        """Test subtraction of SparseFragments"""
        mat1 = csr_matrix([[1, 0], [0, 1]])
        mat2 = csr_matrix([[0, 1], [1, 0]])

        frag1 = SparseFragment(mat1)
        frag2 = SparseFragment(mat2)

        result = frag1 - frag2
        expected = csr_matrix([[1, -1], [-1, 1]])

        assert isinstance(result, SparseFragment)
        assert np.allclose(result.fragment.toarray(), expected.toarray())

    def test_scalar_multiplication(self):
        """Test scalar multiplication of SparseFragments"""
        mat = csr_matrix([[1, 2], [3, 4]])
        frag = SparseFragment(mat)

        result = frag * 2.5
        result_rmul = 2.5 * frag
        expected = csr_matrix([[2.5, 5], [7.5, 10]])

        assert isinstance(result, SparseFragment)
        assert isinstance(result_rmul, SparseFragment)
        assert np.allclose(result.fragment.toarray(), expected.toarray())
        assert np.allclose(result_rmul.fragment.toarray(), expected.toarray())

    def test_matrix_multiplication(self):
        """Test matrix multiplication of SparseFragments"""
        mat1 = csr_matrix([[1, 0], [0, 1]])
        mat2 = csr_matrix([[0, 1], [1, 0]])

        frag1 = SparseFragment(mat1)
        frag2 = SparseFragment(mat2)

        result = frag1 @ frag2
        expected = csr_matrix([[0, 1], [1, 0]])

        assert isinstance(result, SparseFragment)
        assert np.allclose(result.fragment.toarray(), expected.toarray())

    def test_norm(self):
        """Test norm computation of SparseFragments"""
        mat = csr_matrix([[3, 4], [0, 0]])
        frag = SparseFragment(mat)

        norm_val = frag.norm()
        expected_norm = 5.0  # sqrt(3^2 + 4^2)

        assert np.isclose(norm_val, expected_norm)


class TestSingleMatrix:
    """Test SparseFragment with a single matrix"""

    identity = csr_matrix([[1, 0], [0, 1]])
    frag = SparseFragment(identity)
    states = [SparseState(csr_matrix([[1], [0]])), SparseState(csr_matrix([[0], [1]]))]

    @pytest.mark.parametrize("frag, states", [(frag, states)])
    def test_expectation_identity(self, frag, states):
        """Test expectation values with identity matrix"""

        # Test that <ψ|I|ψ> = 1 for normalized states
        for state in states:
            expectation = frag.expectation(state, state)
            assert np.isclose(expectation, 1.0)

        # Test that <0|I|1> = 0 and <1|I|0> = 0
        expectation_01 = frag.expectation(states[0], states[1])
        expectation_10 = frag.expectation(states[1], states[0])

        assert np.isclose(expectation_01, 0.0)
        assert np.isclose(expectation_10, 0.0)

    pauli_z = csr_matrix([[1, 0], [0, -1]])
    pauli_z_frag = SparseFragment(pauli_z)

    @pytest.mark.parametrize("frag, states", [(pauli_z_frag, states)])
    def test_expectation_pauli_z(self, frag, states):
        """Test expectation values with Pauli-Z matrix"""

        # Test that <0|Z|0> = 1 and <1|Z|1> = -1
        expectation_00 = frag.expectation(states[0], states[0])
        expectation_11 = frag.expectation(states[1], states[1])

        assert np.isclose(expectation_00, 1.0)
        assert np.isclose(expectation_11, -1.0)

        # Test that <0|Z|1> = 0 and <1|Z|0> = 0
        expectation_01 = frag.expectation(states[0], states[1])
        expectation_10 = frag.expectation(states[1], states[0])

        assert np.isclose(expectation_01, 0.0)
        assert np.isclose(expectation_10, 0.0)


class TestMultipleMatrices:
    """Test SparseFragment with multiple matrices forming a Hamiltonian"""

    # Create a simple 2x2 Hamiltonian: H = σ_x + 0.5 * σ_z
    pauli_x = csr_matrix([[0, 1], [1, 0]])
    pauli_z = csr_matrix([[1, 0], [0, -1]])

    matrices = [pauli_x, 0.5 * pauli_z]
    ham = _sparse_hamiltonian(matrices)

    # Basis states |0⟩ and |1⟩
    states = [SparseState(csr_matrix([[1], [0]])), SparseState(csr_matrix([[0], [1]]))]

    @pytest.mark.parametrize("ham, states", [(ham, states)])
    def test_hamiltonian_expectation(self, ham, states):
        """Test expectation values of the combined Hamiltonian"""

        # The Hamiltonian matrix should be [[0.5, 1], [1, -0.5]]
        expected_matrix = np.array([[0.5, 1], [1, -0.5]])

        # Compute expectation values
        actual = np.zeros((2, 2), dtype=np.complex128)
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                actual[i, j] = ham.expectation(state1, state2)

        assert np.allclose(actual, expected_matrix)

    @pytest.mark.parametrize("ham, states", [(ham, states)])
    def test_hamiltonian_eigenvalues(self, ham, states):
        """Test that eigenvalues are computed correctly"""

        # The expected eigenvalues of [[0.5, 1], [1, -0.5]] are ±√1.25
        expected_eigenvals = np.array([np.sqrt(1.25), -np.sqrt(1.25)])

        # Build the Hamiltonian matrix
        ham_matrix = np.zeros((2, 2), dtype=np.complex128)
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                ham_matrix[i, j] = ham.expectation(state1, state2)

        actual_eigenvals = np.sort(np.linalg.eigvals(ham_matrix))[::-1]

        assert np.allclose(actual_eigenvals, expected_eigenvals)


class TestLinearCombinations:
    """Test SparseFragment with linear combinations of states"""

    pauli_x = csr_matrix([[0, 1], [1, 0]])
    frag = SparseFragment(pauli_x)

    # Basis states |0⟩ and |1⟩
    basis_states = [SparseState(csr_matrix([[1], [0]])), SparseState(csr_matrix([[0], [1]]))]

    @pytest.mark.parametrize("frag, basis_states", [(frag, basis_states)])
    def test_linear_combination_pauli_x(self, frag, basis_states):  # pylint: disable=unused-argument
        """Test expectation values with linear combinations for Pauli-X"""

        # Create superposition states |+⟩ = (|0⟩ + |1⟩)/√2 and |-⟩ = (|0⟩ - |1⟩)/√2
        plus_state = SparseState(csr_matrix([[1/np.sqrt(2)], [1/np.sqrt(2)]]))
        minus_state = SparseState(csr_matrix([[1/np.sqrt(2)], [-1/np.sqrt(2)]]))

        # Test that ⟨+|X|+⟩ = 1 and ⟨-|X|-⟩ = -1
        expectation_plus = frag.expectation(plus_state, plus_state)
        expectation_minus = frag.expectation(minus_state, minus_state)

        assert np.isclose(expectation_plus, 1.0)
        assert np.isclose(expectation_minus, -1.0)

        # Test that ⟨+|X|-⟩ = 0 and ⟨-|X|+⟩ = 0
        expectation_plus_minus = frag.expectation(plus_state, minus_state)
        expectation_minus_plus = frag.expectation(minus_state, plus_state)

        assert np.isclose(expectation_plus_minus, 0.0)
        assert np.isclose(expectation_minus_plus, 0.0)

    @pytest.mark.parametrize("frag, basis_states", [(frag, basis_states)])
    def test_random_rotation_invariance(self, frag, basis_states):
        """Test expectation values under random unitary rotations"""

        # Generate a random 2x2 unitary matrix
        random_matrix = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        u, _, vh = np.linalg.svd(random_matrix)
        unitary = u @ vh

        # Create rotated states
        rotated_states = []
        for i in range(2):
            rotated_coeff = unitary[:, i]
            rotated_state = (basis_states[0] * rotated_coeff[0] +
                           basis_states[1] * rotated_coeff[1])
            rotated_states.append(rotated_state)

        # Compute expectation matrix in rotated basis
        rotated_expectations = np.zeros((2, 2), dtype=np.complex128)
        for i, state1 in enumerate(rotated_states):
            for j, state2 in enumerate(rotated_states):
                rotated_expectations[i, j] = frag.expectation(state1, state2)

        # Original Pauli-X matrix
        original_matrix = np.array([[0, 1], [1, 0]])

        # Expected result: U† X U
        expected_matrix = unitary.conj().T @ original_matrix @ unitary

        assert np.allclose(rotated_expectations, expected_matrix, atol=1e-10)


class TestSparseState:
    """Test SparseState operations"""

    def test_state_arithmetic(self):
        """Test basic arithmetic operations on SparseStates"""

        state1 = SparseState(csr_matrix([[1], [0]]))
        state2 = SparseState(csr_matrix([[0], [1]]))

        # Test addition
        sum_state = state1 + state2
        expected_sum = csr_matrix([[1], [1]])
        assert np.allclose(sum_state.csr_matrix.toarray(), expected_sum.toarray())

        # Test subtraction
        diff_state = state1 - state2
        expected_diff = csr_matrix([[1], [-1]])
        assert np.allclose(diff_state.csr_matrix.toarray(), expected_diff.toarray())

        # Test scalar multiplication
        scaled_state = state1 * 2.0
        scaled_state_rmul = 3.0 * state1
        expected_scaled = csr_matrix([[2], [0]])
        expected_scaled_rmul = csr_matrix([[3], [0]])

        assert np.allclose(scaled_state.csr_matrix.toarray(), expected_scaled.toarray())
        assert np.allclose(scaled_state_rmul.csr_matrix.toarray(), expected_scaled_rmul.toarray())

    def test_dot_product(self):
        """Test dot product between SparseStates"""

        state1 = SparseState(csr_matrix([[1], [0]]))
        state2 = SparseState(csr_matrix([[0], [1]]))
        state3 = SparseState(csr_matrix([[1], [1]]))

        # Test orthogonal states
        dot_12 = state1.dot(state2)
        dot_21 = state2.dot(state1)
        assert np.isclose(dot_12, 0.0)
        assert np.isclose(dot_21, 0.0)

        # Test state with itself
        dot_11 = state1.dot(state1)
        dot_22 = state2.dot(state2)
        assert np.isclose(dot_11, 1.0)
        assert np.isclose(dot_22, 1.0)

        # Test overlapping states
        dot_13 = state1.dot(state3)
        dot_31 = state3.dot(state1)
        assert np.isclose(dot_13, 1.0)
        assert np.isclose(dot_31, 1.0)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_fragment_equality(self):
        """Test equality comparison between SparseFragments"""

        mat1 = csr_matrix([[1, 0], [0, 1]])
        frag1 = SparseFragment(mat1)

        # Note: The current implementation doesn't seem to have proper equality
        # but we test the type error for comparison with wrong types
        with pytest.raises(TypeError):
            _ = frag1 == "not a fragment"  # Using _ to avoid unused variable warning

    def test_large_sparse_matrices(self):
        """Test with larger sparse matrices"""

        # Create a larger diagonal matrix
        size = 100
        diagonal_data = np.arange(1, size + 1)
        large_matrix = csr_matrix((diagonal_data, (range(size), range(size))), shape=(size, size))

        frag = SparseFragment(large_matrix)

        # Test norm
        expected_norm = np.sqrt(np.sum(diagonal_data**2))
        actual_norm = frag.norm()
        assert np.isclose(actual_norm, expected_norm)

        # Test scalar multiplication
        scaled_frag = frag * 2.0
        expected_scaled_norm = 2.0 * expected_norm
        actual_scaled_norm = scaled_frag.norm()
        assert np.isclose(actual_scaled_norm, expected_scaled_norm)

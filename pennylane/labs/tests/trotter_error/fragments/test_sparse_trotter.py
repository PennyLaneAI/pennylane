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
from scipy.sparse import csr_array

from pennylane.labs.trotter_error.fragments import sparse_fragments
from pennylane.labs.trotter_error.fragments.sparse_fragments import SparseFragment, SparseState

identity = SparseFragment(csr_array([[1, 0], [0, 1]]))
pauli_x = SparseFragment(csr_array([[0, 1], [1, 0]]))
pauli_z = SparseFragment(csr_array([[1, 0], [0, -1]]))

state_10 = SparseState(csr_array([[1, 0]]))
state_01 = SparseState(csr_array([[0, 1]]))
state_11 = SparseState(csr_array([[1, 1]]))
state_plus = SparseState(csr_array([[1 / np.sqrt(2), 1 / np.sqrt(2)]]))
state_minus = SparseState(csr_array([[1 / np.sqrt(2), -1 / np.sqrt(2)]]))


def _sparse_hamiltonian(matrices):
    """Create a sparse Hamiltonian from a list of matrices."""
    frags = sparse_fragments(matrices)
    return sum(frags[1:], frags[0]) if frags else SparseFragment(csr_array((2, 2)))


def test_sparse_fragments_basic():
    """Test that sparse_fragments returns SparseFragment objects correctly."""

    matrices = [csr_array([[1, 0], [0, 1]]), csr_array([[0, 1], [1, 0]])]
    frags = sparse_fragments(matrices)

    assert len(frags) == 2
    for frag in frags:
        assert isinstance(frag, SparseFragment)


def test_sparse_fragments_empty():
    """Test that sparse_fragments handles empty input correctly."""
    frags = sparse_fragments([])
    assert frags == []


def test_sparse_fragments_type_error():
    """Test that sparse_fragments raises TypeError for non-csr_array inputs."""
    with pytest.raises(TypeError, match="Fragments must be csr_array objects"):
        sparse_fragments([np.array([[1, 0], [0, 1]])])


@pytest.mark.parametrize(
    "frag1, frag2, expected", [(identity, pauli_x, SparseFragment(csr_array([[1, 1], [1, 1]])))]
)
def test_addition(frag1, frag2, expected):
    """Test addition of SparseFragments"""
    result = frag1 + frag2

    assert isinstance(result, SparseFragment)
    assert result == expected


@pytest.mark.parametrize(
    "frag1, frag2, expected", [(identity, pauli_x, SparseFragment(csr_array([[1, -1], [-1, 1]])))]
)
def test_subtraction(frag1, frag2, expected):
    """Test subtraction of SparseFragments"""
    result = frag1 - frag2

    assert isinstance(result, SparseFragment)
    assert result == expected


@pytest.mark.parametrize(
    "frag, scalar, expected",
    [
        (
            SparseFragment(csr_array([[1, 2], [3, 4]])),
            2.5,
            SparseFragment(csr_array([[2.5, 5], [7.5, 10]])),
        )
    ],
)
def test_scalar_multiplication(frag, scalar, expected):
    """Test scalar multiplication of SparseFragments"""
    result = frag * scalar
    result_rmul = scalar * frag

    assert isinstance(result, SparseFragment)
    assert isinstance(result_rmul, SparseFragment)
    assert result == expected
    assert result_rmul == expected


@pytest.mark.parametrize(
    "frag1, frag2, expected",
    [
        (identity, pauli_x, pauli_x),
    ],
)
def test_matrix_multiplication(frag1, frag2, expected):
    """Test matrix multiplication of SparseFragments"""
    result = frag1 @ frag2

    assert isinstance(result, SparseFragment)
    assert result == expected


@pytest.mark.parametrize(
    "frag, expected",
    [
        (SparseFragment(csr_array([[3, 4], [0, 0]])), 5.0),
    ],
)
def test_norm(frag, expected):
    """Test norm computation of SparseFragments"""
    assert np.isclose(frag.norm(), expected)


@pytest.mark.parametrize(
    "frag, l_state, r_state, expected",
    [
        (identity, state_01, state_01, 1.0),
        (identity, state_10, state_10, 1.0),
        (identity, state_01, state_10, 0.0),
        (identity, state_10, state_01, 0.0),
        (pauli_z, state_01, state_01, -1.0),
        (pauli_z, state_10, state_10, 1.0),
        (pauli_z, state_01, state_10, 0.0),
        (pauli_z, state_10, state_01, 0.0),
        (pauli_x + 0.5 * pauli_z, state_10, state_10, 0.5),
        (pauli_x + 0.5 * pauli_z, state_10, state_01, 1.0),
        (pauli_x + 0.5 * pauli_z, state_01, state_10, 1.0),
        (pauli_x + 0.5 * pauli_z, state_01, state_01, -0.5),
        (pauli_x, state_plus, state_plus, 1.0),
        (pauli_x, state_minus, state_minus, -1.0),
        (pauli_x, state_plus, state_minus, 0.0),
        (pauli_x, state_minus, state_plus, 0.0),
    ],
)
def test_expectation(frag, l_state, r_state, expected):
    """Test expectation values"""

    assert np.isclose(frag.expectation(l_state, r_state), expected)


@pytest.mark.parametrize(
    "frag, state, expected",
    [
        (identity, state_10, state_10),  # Identity should leave state unchanged
        (identity, state_01, state_01),  # Identity should leave state unchanged
        (pauli_x, state_10, state_01),  # Pauli-X flips |0⟩ to |1⟩
        (pauli_x, state_01, state_10),  # Pauli-X flips |1⟩ to |0⟩
    ],
)
def test_apply(frag, state, expected):
    """Test apply method of SparseFragment"""
    result = frag.apply(state)

    assert isinstance(result, SparseState)
    assert result == expected


@pytest.mark.parametrize(
    "state1, state2, expected",
    [
        (state_01, state_10, state_11),
        (state_10, state_01, state_11),
    ],
)
def test_state_addition(state1, state2, expected):
    """Test addition on SparseState"""
    assert state1 + state2 == expected


@pytest.mark.parametrize(
    "state1, state2, expected",
    [
        (state_01, state_10, SparseState(csr_array([[-1, 1]]))),
        (state_10, state_01, SparseState(csr_array([[1, -1]]))),
    ],
)
def test_state_subtraction(state1, state2, expected):
    """Test subtraction on SparseState"""
    assert state1 - state2 == expected


@pytest.mark.parametrize(
    "state, scalar, expected",
    [
        (state_01, 5.0, SparseState(csr_array([[0, 5]]))),
        (state_10, -5.0, SparseState(csr_array([[-5, 0]]))),
    ],
)
def test_state_scalar_multiplication(state, scalar, expected):
    """Test scalar multiplication on SparseState"""
    assert scalar * state == expected
    assert state * scalar == expected


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (state_10, state_01, 0.0),
        (state_01, state_10, 0.0),
        (state_10, state_10, 1.0),
        (state_01, state_01, 1.0),
        (state_10, state_11, 1.0),
        (state_11, state_10, 1.0),
    ],
)
def test_dot_product(x, y, expected):
    """Test dot product between SparseStates"""
    assert np.isclose(x.dot(y), expected)


def test_fragment_equality_type_error():
    """Test equality comparison between SparseFragments"""

    mat1 = csr_array([[1, 0], [0, 1]])
    frag1 = SparseFragment(mat1)

    with pytest.raises(TypeError, match="Cannot compare SparseFragment with type <class 'str'>"):
        frag1 == "not a fragment"  # pylint: disable=pointless-statement

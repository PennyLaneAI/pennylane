# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the base-case flag decompositions (d=1, d=2, d=3)."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates.mps_synthesis.flag import (
    PartiallyMultiplexedFlag,
    d2_generalized_flag_decomp,
    d3_generalized_flag_decomp,
    one_qubit_flag_decomp,
)


def random_unitary(n, seed):
    """Return a Haar-ish random ``n x n`` unitary via a QR decomposition."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    # Fix the phases of the diagonal so ``q`` is uniformly distributed.
    return q @ np.diag(np.diagonal(r) / np.abs(np.diagonal(r)))


def flags_matrix(ops, wires):
    """Matrix of a flag sequence applied in list order (``ops[0]`` first)."""
    mat = np.eye(2 ** len(wires), dtype=complex)
    for op in ops:
        mat = qp.matrix(op, wire_order=wires) @ mat
    return mat


# Structured 2x2 matrices plus random cases spanning the parameter space.
MATRICES = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),  # X
    np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),  # H
    np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),  # T
] + [random_unitary(2, seed) for seed in range(5)]


# ============================================================================
# one_qubit_flag_decomp (d=1)
# ============================================================================


@pytest.mark.parametrize("matrix", MATRICES)
def test_reconstructs_matrix(matrix):
    """The flag times the trailing diagonal reproduces the input matrix exactly."""
    F, Delta = one_qubit_flag_decomp(matrix, wires=[0])

    flag_matrix = qp.matrix(F[0], wire_order=[0])
    reconstructed = np.diag(Delta) @ flag_matrix

    assert np.allclose(reconstructed, matrix, atol=1e-8)


@pytest.mark.parametrize("matrix", MATRICES)
def test_return_structure(matrix):
    """The decomposition returns a single flag on the requested wire and a length-2 diagonal."""
    F, Delta = one_qubit_flag_decomp(matrix, wires=["a"])

    assert isinstance(F, list) and len(F) == 1
    assert isinstance(F[0], PartiallyMultiplexedFlag)
    assert F[0].wires.tolist() == ["a"]
    # A single-qubit flag has no control wires and one (rz, ry) angle pair.
    assert F[0].hyperparameters["control_values"] == ()
    assert len(F[0].parameters[0]) == 1 and len(F[0].parameters[1]) == 1
    assert np.shape(Delta) == (2,)


@pytest.mark.parametrize("matrix", MATRICES)
def test_trailing_diagonal_is_unimodular(matrix):
    """The trailing diagonal entries are pure phases (unit modulus)."""
    _, Delta = one_qubit_flag_decomp(matrix, wires=[0])
    assert np.allclose(np.abs(Delta), 1.0, atol=1e-8)


# ============================================================================
# d2_generalized_flag_decomp (d=2)
# ============================================================================


def controlled_matrix(matrix, control_value):
    """Embed ``matrix`` on the target subspace where the MSB control equals ``control_value``."""
    expected = np.eye(4, dtype=complex)
    block = slice(2, 4) if control_value == 1 else slice(0, 2)
    expected[block, block] = matrix
    return expected


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize("control_value", [0, 1])
def test_d2_reconstructs_controlled_matrix(matrix, control_value):
    """The flag and trailing diagonal reproduce ``matrix`` controlled on ``wires[0]``."""
    ops, controlled = d2_generalized_flag_decomp(matrix, wires=[0, 1], control_value=control_value)

    reconstructed = np.diag(controlled) @ qp.matrix(ops[0], wire_order=[0, 1])

    assert np.allclose(reconstructed, controlled_matrix(matrix, control_value), atol=1e-8)


@pytest.mark.parametrize("matrix", MATRICES)
def test_d2_return_structure(matrix):
    """Returns a single controlled flag on the two wires and a length-4 diagonal."""
    ops, controlled = d2_generalized_flag_decomp(matrix, wires=["a", "b"], control_value=1)

    assert isinstance(ops, list) and len(ops) == 1
    assert ops[0].wires.tolist() == ["a", "b"]
    assert ops[0].hyperparameters["control_values"] == ((1,),)
    assert np.shape(controlled) == (4,)


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize("control_value", [0, 1])
def test_d2_trailing_diagonal_is_unimodular(matrix, control_value):
    """The trailing controlled diagonal entries are pure phases (unit modulus)."""
    _, controlled = d2_generalized_flag_decomp(matrix, wires=[0, 1], control_value=control_value)
    assert np.allclose(np.abs(controlled), 1.0, atol=1e-8)


@pytest.mark.parametrize("num_wires", [1, 3])
def test_d2_requires_two_wires(num_wires):
    """A register that is not two wires (N=4) is rejected."""
    with pytest.raises(ValueError, match="two wires"):
        d2_generalized_flag_decomp(random_unitary(2, 0), wires=list(range(num_wires)))


# ============================================================================
# d3_generalized_flag_decomp (d=3)
# ============================================================================

# Active fractal states for (d=3, N=4): get_fractal_embedding_states(3, 4) -> [1, 2, 3].
ACTIVE_D3 = [1, 2, 3]
INACTIVE_D3 = 0

D3_MATRICES = [np.eye(3, dtype=complex)] + [random_unitary(3, seed) for seed in range(5)]


@pytest.mark.parametrize("matrix", D3_MATRICES)
def test_d3_reconstructs_embedded_matrix(matrix):
    """The flags and trailing diagonal reproduce ``matrix`` on the active fractal subspace."""
    ops, diag = d3_generalized_flag_decomp(matrix, wires=[0, 1])

    full = np.diag(diag) @ flags_matrix(ops, [0, 1])

    assert np.allclose(full[np.ix_(ACTIVE_D3, ACTIVE_D3)], matrix, atol=1e-8)


@pytest.mark.parametrize("matrix", D3_MATRICES)
def test_d3_inactive_state_decoupled(matrix):
    """The inactive fractal state does not couple to the active block."""
    ops, diag = d3_generalized_flag_decomp(matrix, wires=[0, 1])

    full = np.diag(diag) @ flags_matrix(ops, [0, 1])

    assert np.allclose(full[INACTIVE_D3, ACTIVE_D3], 0.0, atol=1e-8)
    assert np.allclose(full[ACTIVE_D3, INACTIVE_D3], 0.0, atol=1e-8)
    assert np.isclose(np.abs(full[INACTIVE_D3, INACTIVE_D3]), 1.0, atol=1e-8)


@pytest.mark.parametrize("matrix", D3_MATRICES)
def test_d3_return_structure(matrix):
    """Returns three flags on the two wires and a length-4 diagonal."""
    ops, diag = d3_generalized_flag_decomp(matrix, wires=["a", "b"])

    assert isinstance(ops, list) and len(ops) == 3
    assert all(isinstance(op, PartiallyMultiplexedFlag) for op in ops)
    assert all(set(op.wires.tolist()) == {"a", "b"} for op in ops)
    assert np.shape(diag) == (4,)


@pytest.mark.parametrize("matrix", D3_MATRICES)
def test_d3_trailing_diagonal_is_unimodular(matrix):
    """The trailing diagonal entries are pure phases (unit modulus)."""
    _, diag = d3_generalized_flag_decomp(matrix, wires=[0, 1])
    assert np.allclose(np.abs(diag), 1.0, atol=1e-8)


@pytest.mark.parametrize("num_wires", [1, 3])
def test_d3_requires_two_wires(num_wires):
    """A register that is not two wires (N=4) is rejected."""
    with pytest.raises(ValueError, match="two wires"):
        d3_generalized_flag_decomp(random_unitary(3, 0), wires=list(range(num_wires)))

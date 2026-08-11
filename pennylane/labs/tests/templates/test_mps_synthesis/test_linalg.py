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
"""Tests for the CSD helpers ``shift_csd_one`` and ``synthesis_csd``."""

import numpy as np
import pytest
from scipy.linalg import cossin

import pennylane as qp
from pennylane.labs.templates.mps_synthesis.linalg import (
    embed_unitary,
    get_controlled_unitary_msq,
    propagate_diagonal_through_unitary,
    shift_csd_one,
    split_d,
    split_diagonal_into_control_branches,
    split_diagonal_into_partially_multiplexed_rz,
    synthesis_csd,
)


def random_unitary(n, seed):
    """Return a Haar-ish random ``n x n`` unitary via a QR decomposition."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    return q @ np.diag(np.diagonal(r) / np.abs(np.diagonal(r)))


def random_phases(n, seed):
    """Return ``2**n`` unit-modulus phases (a valid diagonal of a unitary)."""
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, 2**n))


# ============================================================================
# shift_csd_one
# ============================================================================

ODD_DIMS = [3, 5, 7]


@pytest.mark.parametrize("d", ODD_DIMS)
def test_shift_csd_one_preserves_reconstruction(d):
    """Relocating the uncoupled '1' leaves ``U @ CS @ V_H`` equal to the original unitary."""
    V = random_unitary(d, seed=d)
    p = d // 2
    U, CS, V_H = cossin(V, p=p, q=p, separate=False)

    for target in range(p, d):  # valid targets live in the second block [p, d)
        U_c, CS_c, V_H_c = shift_csd_one(U, CS, V_H, target)
        assert np.allclose(U_c @ CS_c @ V_H_c, V, atol=1e-8)


@pytest.mark.parametrize("d", ODD_DIMS)
def test_shift_csd_one_relocates_uncoupled_one(d):
    """The uncoupled '1' ends up on the diagonal at exactly ``target_index``."""
    V = random_unitary(d, seed=d + 100)
    p = d // 2
    U, CS, V_H = cossin(V, p=p, q=p, separate=False)

    for target in range(p, d):
        _, CS_c, _ = shift_csd_one(U, CS, V_H, target)
        assert np.isclose(CS_c[target, target], 1.0, atol=1e-8)


# ============================================================================
# synthesis_csd
# ============================================================================

# (d, shift) combinations for which the raw factors U/CS/V_H are returned.
# (even d only returns raw factors for shift=False; shift is a no-op there.)
FACTOR_CASES = [
    (2, False),
    (4, False),
    (6, False),
    (3, False),
    (3, True),
    (5, False),
    (5, True),
]


@pytest.mark.parametrize("d, shift", FACTOR_CASES)
def test_synthesis_csd_raw_reconstruction(d, shift):
    """The (possibly shifted) raw CSD factors reconstruct the input unitary."""
    V = random_unitary(d, seed=d + (10 if shift else 0))
    _, _, _, _, _, U, CS, V_H = synthesis_csd(V, shift=shift, return_all=True)
    assert np.allclose(U @ CS @ V_H, V, atol=1e-8)


@pytest.mark.parametrize("d", [2, 4, 6])
def test_synthesis_csd_shift_is_noop_for_even_dims(d):
    """For even ``d`` (p == q) the shift has no effect on the separated outputs."""
    V = random_unitary(d, seed=d + 7)
    no_shift = synthesis_csd(V, shift=False)
    with_shift = synthesis_csd(V, shift=True)
    for a, b in zip(no_shift[:5], with_shift[:5]):  # K00, K01, theta, K10, K11
        assert np.allclose(a, b, atol=1e-8)


# ============================================================================
# split_diagonal_into_partially_multiplexed_rz
# ============================================================================

# wires with the Rz target (wires[-1]) in various sorted positions.
RZ_WIRES = [[0, 1], [0, 1, 2], [2, 0, 1]]


@pytest.mark.parametrize("wires", RZ_WIRES)
def test_split_rz_full_reconstruction(wires):
    """``rz * remaining`` reproduces the diagonal when all control states are covered."""
    n = len(wires)
    full = random_phases(n, seed=n)
    control_states = list(range(2 ** (n - 1)))

    angles, remaining, rz = split_diagonal_into_partially_multiplexed_rz(
        full, wires, control_states
    )

    assert np.allclose(rz * remaining, full, atol=1e-8)
    assert len(angles) == len(control_states)
    assert np.allclose(np.abs(rz), 1.0, atol=1e-8)


@pytest.mark.parametrize("wires", RZ_WIRES)
def test_split_rz_partial_reconstruction(wires):
    """``rz * remaining`` reproduces the diagonal for a partial set of control states."""
    n = len(wires)
    full = random_phases(n, seed=n + 50)
    control_states = list(range(2 ** (n - 1)))[::2]  # every other pattern

    _, remaining, rz = split_diagonal_into_partially_multiplexed_rz(full, wires, control_states)

    assert np.allclose(rz * remaining, full, atol=1e-8)


# ============================================================================
# split_diagonal_into_control_branches
# ============================================================================


@pytest.mark.parametrize("wires", [[0, 1], [0, 1, 2], [1, 0, 2]])
def test_split_control_branches_reconstruction(wires):
    """The |0>- and |1>-controlled diagonals multiply back to the original."""
    n = len(wires)
    diag = random_phases(n, seed=n + 7)

    d0, d1, target_d0, target_d1 = split_diagonal_into_control_branches(diag, wires)

    assert np.allclose(d0 * d1, diag, atol=1e-8)
    assert d0.shape == (2**n,) and d1.shape == (2**n,)
    assert len(target_d0) == 2 ** (n - 1)
    assert len(target_d1) == 2 ** (n - 1)


# ============================================================================
# get_controlled_unitary_msq
# ============================================================================

# (wires, n_target): wires[0] is control, wires[1:] the targets.
CTRL_CASES = [([0, 1], 1), ([0, 1, 2], 2)]


@pytest.mark.parametrize("wires, n_target", CTRL_CASES)
@pytest.mark.parametrize("control_value", [0, 1])
def test_get_controlled_unitary_dense(wires, n_target, control_value):
    """A dense target unitary is lifted to a controlled operation (matches ``qp.ctrl``)."""
    U = random_unitary(2**n_target, seed=n_target + control_value)

    result = get_controlled_unitary_msq(U, wires, control_value)
    expected = qp.matrix(
        qp.ctrl(
            qp.QubitUnitary(U, wires=wires[1:]),
            control=wires[0],
            control_values=control_value,
        ),
        wire_order=sorted(wires),
    )
    assert np.allclose(result, expected, atol=1e-8)


@pytest.mark.parametrize("control_value", [0, 1])
def test_get_controlled_unitary_diagonal(control_value):
    """A 1D diagonal target returns a controlled diagonal (matches ``qp.ctrl``)."""
    wires = [0, 1]
    diag = random_phases(1, seed=control_value)  # length-2 target diagonal

    result = get_controlled_unitary_msq(diag, wires, control_value)
    expected = qp.matrix(
        qp.ctrl(
            qp.DiagonalQubitUnitary(diag, wires=wires[1:]),
            control=wires[0],
            control_values=control_value,
        ),
        wire_order=sorted(wires),
    )
    assert result.ndim == 1
    assert np.allclose(np.diag(result), expected, atol=1e-8)


def test_get_controlled_unitary_active_indices_diagonal():
    """Padding a smaller diagonal with ``active_indices`` matches passing the padded diagonal."""
    wires, target_dim, active, cv = [0, 1, 2], 4, [1, 2, 3], 1
    diag = random_phases(2, seed=11)[: len(active)]

    padded = np.ones(target_dim, dtype=complex)
    padded[active] = diag

    res_padded = get_controlled_unitary_msq(diag, wires, cv, active_indices=active)
    res_full = get_controlled_unitary_msq(padded, wires, cv)
    assert np.allclose(res_padded, res_full, atol=1e-8)


def test_get_controlled_unitary_active_indices_dense():
    """Padding a smaller unitary with ``active_indices`` matches passing the embedded unitary."""
    wires, target_dim, active, cv = [0, 1, 2], 4, [1, 2, 3], 0
    U = random_unitary(len(active), seed=12)

    res_padded = get_controlled_unitary_msq(U, wires, cv, active_indices=active)
    res_full = get_controlled_unitary_msq(embed_unitary(U, target_dim, active), wires, cv)
    assert np.allclose(res_padded, res_full, atol=1e-8)


# ============================================================================
# propagate_diagonal_through_unitary
# ============================================================================

# (wires, active_indices): the target unitary acts on ``active_indices`` of wires[1:].
PROPAGATE_CASES = [
    ([0, 1], [0, 1]),
    ([0, 1, 2], [0, 1, 2, 3]),
    ([0, 1, 2], [1, 2, 3]),
]


@pytest.mark.parametrize("wires, active", PROPAGATE_CASES)
@pytest.mark.parametrize("control_val", [0, 1])
def test_propagate_diagonal_invariant(wires, active, control_val):
    """Absorbing the diagonal preserves ``controlled_U @ diag``: D then U == U' then D'."""
    n = len(wires)
    full_dim = 2**n
    U = random_unitary(len(active), seed=len(active) + control_val)
    full_diag = random_phases(n, seed=n + control_val + 3)

    new_U, new_full_diag, controlled_new_U = propagate_diagonal_through_unitary(
        full_diag, U, wires, control_val, active
    )

    controlled_orig = get_controlled_unitary_msq(U, wires, control_val, active_indices=active)
    lhs = controlled_orig @ np.diag(full_diag)
    rhs = controlled_new_U @ np.diag(new_full_diag)

    assert np.allclose(lhs, rhs, atol=1e-8)
    assert new_U.shape == U.shape
    assert new_full_diag.shape == (full_dim,)
    assert controlled_new_U.shape == (full_dim, full_dim)

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
"""
Tests for the mps_preparation template.
"""

# pylint: disable=no-value-for-parameter
import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates.mps_synthesis.mps_synthesis import (
    mps_preparation,
    mps_synthesis,
)

# (chi, L, seed) test cases spanning power-of-two and non-power-of-two bond dims.
# L is the total number of tensors; the effective bond dim is min(chi, 2 ** (L // 2)),
# so L == 2 * ceil(log2(chi_max)) means no bulk tensors (the boundaries abut directly).
# The (5, 4) case is quirky: chi is capped to 4, so no tensor ever reaches chi = 5.
CASES = [
    (2, 3, 0),
    (3, 5, 1),
    (4, 6, 2),
    (3, 6, 3),
    (5, 7, 4),
    (6, 8, 5),
    (4, 7, 6),
    (2, 2, 7),
    (4, 4, 8),
    (3, 4, 9),
    (5, 4, 10),
]


def random_mps(chi, L, d=2, seed=None):
    """Random right-canonical MPS split into left boundary, bulk, right boundary.

    ``L`` is the total number of tensors in the chain. The effective bond dimension is
    capped at ``chi_max = min(chi, 2 ** (L // 2))``: with only ``L`` tensors the bond can
    at most double out from each open end, so a larger ``chi`` would only introduce
    redundant, unreachable bond dimension (e.g. ``chi, L = 5, 4`` yields bonds
    ``1 -> 2 -> 4 -> 2 -> 1``, never reaching 5). The left boundary grows
    ``1 -> 2 -> 4 -> ... -> chi_max`` (doubling, last step capped at ``chi_max``), the right
    boundary mirrors it, and the remaining tensors form the bulk of shape
    ``chi_max -> chi_max``. Each tensor ``A`` of shape ``(chi_L, d, chi_R)`` is right-canonical.
    """
    rng = np.random.default_rng(seed)

    def rc_tensor(cl, cr):
        cols = d * cr
        G = rng.standard_normal((cols, cl)) + 1j * rng.standard_normal((cols, cl))
        Q, _ = np.linalg.qr(G)
        return Q.conj().T.reshape(cl, d, cr)

    chi_max = min(chi, 2 ** (L // 2))
    S = [1]
    while S[-1] < chi_max:
        S.append(min(2 * S[-1], chi_max))
    left_bonds, right_bonds = S, S[::-1]

    n_boundary = len(S) - 1  # tensors per boundary = ceil(log2(chi_max))
    n_bulk = L - 2 * n_boundary  # guaranteed >= 0 by the chi_max cap

    left = [rc_tensor(cl, cr) for cl, cr in zip(left_bonds[:-1], left_bonds[1:])]
    bulk = [rc_tensor(chi_max, chi_max) for _ in range(n_bulk)]
    right = [rc_tensor(cl, cr) for cl, cr in zip(right_bonds[:-1], right_bonds[1:])]
    return left + bulk + right


def mps_wires(chi, L):
    """Return the combined wire register (auxiliary wires followed by physical wires)
    for ``random_mps(chi, L)``. ``L`` is the total number of tensors, so the register has
    ``L`` wires: ``n = ceil(log2(chi_max))`` auxiliary wires and ``L - n`` physical wires,
    where ``chi_max = min(chi, 2 ** (L // 2))`` is the effective bond dimension."""
    chi_max = min(chi, 2 ** (L // 2))
    n = int(np.ceil(np.log2(chi_max))) if chi_max > 1 else 0
    phys_wires = list(range(L - n))  # L - n physical wires (left + bulk sites)
    aux_wires = list(range(L - n, L))  # n auxiliary (bond) wires
    return aux_wires + phys_wires


def get_vector(tensors):
    """Contract a list of MPS tensors into the full statevector (open boundaries)."""
    alpha = np.zeros(tensors[0].shape[0])
    alpha[0] = 1.0
    tensor = np.einsum("i,iaj->aj", alpha, tensors[0])
    for next_tensor in tensors[1:]:
        tensor = np.einsum("aj,jbk->abk", tensor, next_tensor)
        tensor = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])
    return tensor


def _prepared_state(mps, wires):
    all_wires = sorted(set(wires))
    dev = qp.device("default.qubit", wires=all_wires)

    @qp.qnode(dev)
    def circuit():
        mps_preparation(mps, wires)
        return qp.state()

    return np.asarray(circuit()).reshape(-1)


@pytest.mark.parametrize("chi, L, seed", CASES)
def test_prepared_state_matches_mps(chi, L, seed):
    """The prepared state reproduces the target MPS amplitudes (fidelity 1)."""
    mps = random_mps(chi, L, seed=seed)
    wires = mps_wires(chi, L)

    state = _prepared_state(mps, wires)
    target = np.asarray(get_vector(list(mps))).reshape(-1)
    target = target / np.linalg.norm(target)

    fidelity = np.abs(np.vdot(target, state)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-8)


def test_non_unimodal_profile_raises():
    """A right-canonical but non-unimodal bond profile is rejected, not silently reordered."""
    rng = np.random.default_rng(0)

    def rc_tensor(cl, cr, d=2):
        G = rng.standard_normal((d * cr, cl)) + 1j * rng.standard_normal((d * cr, cl))
        Q, _ = np.linalg.qr(G)
        return Q.conj().T.reshape(cl, d, cr)

    # Bond profile 1 -> 2 -> 1 -> 2 -> 1 (each tensor right-canonical).
    mps = [rc_tensor(1, 2), rc_tensor(2, 1), rc_tensor(1, 2), rc_tensor(2, 1)]
    wires = [0, 1, 2, 3]  # one wire per tensor, so the profile check is what triggers

    with pytest.raises(ValueError, match="single left/bulk/right bond profile"):
        mps_synthesis(mps, wires)


@pytest.mark.parametrize("chi, L, seed", CASES)
def test_synthesis_matches_template(chi, L, seed):
    """The queued template reproduces the state built by mps_synthesis directly."""
    mps = random_mps(chi, L, seed=seed)
    wires = mps_wires(chi, L)
    all_wires = sorted(set(wires))

    circuit, _ = mps_synthesis(mps, wires)
    dev = qp.device("default.qubit", wires=all_wires)

    @qp.qnode(dev)
    def manual():
        for op in circuit:
            qp.apply(op)
        return qp.state()

    manual_state = np.asarray(manual()).reshape(-1)
    template_state = _prepared_state(mps, wires)
    # Equal up to an unobservable global phase.
    assert np.isclose(np.abs(np.vdot(manual_state, template_state)) ** 2, 1.0, atol=1e-8)


# Non-monotonic wire relabelings: these change the ascending order of the wires,
# so a template that (incorrectly) baked in ``sorted(wires)`` internally would give
# different, wrong results under them.
RELABELINGS = [
    pytest.param(lambda used: {w: 1000 - w for w in used}, id="reversed"),
    pytest.param(
        lambda used: dict(zip(used, np.random.default_rng(123).permutation(used).tolist())),
        id="permuted",
    ),
]


@pytest.mark.parametrize("chi, L, seed", CASES)
@pytest.mark.parametrize("make_relabel", RELABELINGS)
def test_prepared_state_is_wire_relabeling_invariant(chi, L, seed, make_relabel):
    """Relabeling the wires only relabels the prepared state (any wire order works)."""
    mps = random_mps(chi, L, seed=seed)
    wires_b = mps_wires(chi, L)

    used = sorted(set(wires_b))
    g = {int(k): int(v) for k, v in make_relabel(used).items()}
    assert len(set(g.values())) == len(used), "relabeling must be a bijection"

    state_b = _prepared_state(mps, wires_b)
    dev_b = sorted(set(wires_b))

    wires_r = [g[w] for w in wires_b]
    state_r = _prepared_state(mps, wires_r)
    dev_r = sorted(set(wires_r))

    # Reorder the relabeled state back into the baseline wire layout, then compare.
    m = len(dev_b)
    axes = [dev_r.index(g[w]) for w in dev_b]
    aligned = np.transpose(np.asarray(state_r).reshape([2] * m), axes).reshape(-1)

    fidelity = np.abs(np.vdot(state_b, aligned)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-8)

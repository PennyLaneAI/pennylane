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
# L=0 cases have no bulk tensors (left boundary abuts the right boundary directly).
CASES = [
    (2, 1, 0),
    (3, 1, 1),
    (4, 2, 2),
    (3, 2, 3),
    (5, 1, 4),
    (6, 2, 5),
    (4, 3, 6),
    (2, 0, 7),
    (4, 0, 8),
    (3, 0, 9),
]


def random_mps(chi, L, d=2, seed=None):
    """Random right-canonical MPS split into left boundary, bulk, right boundary.

    Bond profile (``chi`` need not be a power of two): the left boundary grows
    ``1 -> 2 -> 4 -> ... -> chi`` (doubling, last step capped at ``chi``), the
    bulk is ``L`` tensors of shape ``chi -> chi``, and the right boundary mirrors
    the left. Each tensor ``A`` of shape ``(chi_L, d, chi_R)`` is right-canonical.
    """
    rng = np.random.default_rng(seed)

    def rc_tensor(cl, cr):
        cols = d * cr
        G = rng.standard_normal((cols, cl)) + 1j * rng.standard_normal((cols, cl))
        Q, _ = np.linalg.qr(G)
        return Q.conj().T.reshape(cl, d, cr)

    S = [1]
    while S[-1] < chi:
        S.append(min(2 * S[-1], chi))
    left_bonds, right_bonds = S, S[::-1]

    left = [rc_tensor(cl, cr) for cl, cr in zip(left_bonds[:-1], left_bonds[1:])]
    bulk = [rc_tensor(chi, chi) for _ in range(L)]
    right = [rc_tensor(cl, cr) for cl, cr in zip(right_bonds[:-1], right_bonds[1:])]
    return left + bulk + right


def mps_wires(chi, L):
    """Return ``(phys_wires_left, phys_wires_bulk, aux_wires)`` for ``random_mps(chi, L)``."""
    n = int(np.ceil(np.log2(chi))) if chi > 1 else 0
    phys_wires_left = list(range(n))
    phys_wires_bulk = list(range(n, n + L))
    aux_wires = list(range(n + L, 2 * n + L))
    return phys_wires_left, phys_wires_bulk, aux_wires


def get_vector(tensors):
    """Contract a list of MPS tensors into the full statevector (open boundaries)."""
    alpha = np.zeros(tensors[0].shape[0])
    alpha[0] = 1.0
    tensor = np.einsum("i,iaj->aj", alpha, tensors[0])
    for next_tensor in tensors[1:]:
        tensor = np.einsum("aj,jbk->abk", tensor, next_tensor)
        tensor = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])
    return tensor


def _prepared_state(mps, aux, phys):
    """Run the mps_preparation template and return the full statevector."""
    all_wires = sorted(set(phys) | set(aux))
    dev = qp.device("default.qubit", wires=all_wires)

    @qp.qnode(dev)
    def circuit():
        mps_preparation(mps, aux, phys)
        return qp.state()

    return np.asarray(circuit()).reshape(-1)


@pytest.mark.parametrize("chi, L, seed", CASES)
def test_prepared_state_matches_mps(chi, L, seed):
    """The prepared state reproduces the target MPS amplitudes (fidelity 1)."""
    mps = random_mps(chi, L, seed=seed)
    phys_left, phys_bulk, aux = mps_wires(chi, L)
    phys = phys_left + phys_bulk

    state = _prepared_state(mps, aux, phys)
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
    phys, aux = [0], [1]

    with pytest.raises(ValueError, match="single left/bulk/right bond profile"):
        mps_synthesis(mps, aux, phys)


@pytest.mark.parametrize("chi, L, seed", CASES)
def test_synthesis_matches_template(chi, L, seed):
    """The queued template reproduces the state built by mps_synthesis directly."""
    mps = random_mps(chi, L, seed=seed)
    phys_left, phys_bulk, aux = mps_wires(chi, L)
    phys = phys_left + phys_bulk
    all_wires = sorted(set(phys) | set(aux))

    circuit, _ = mps_synthesis(mps, aux, phys)
    dev = qp.device("default.qubit", wires=all_wires)

    @qp.qnode(dev)
    def manual():
        for op in circuit:
            qp.apply(op)
        return qp.state()

    manual_state = np.asarray(manual()).reshape(-1)
    template_state = _prepared_state(mps, aux, phys)
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
    phys_left, phys_bulk, aux_b = mps_wires(chi, L)
    phys_b = phys_left + phys_bulk

    used = sorted(set(phys_b) | set(aux_b))
    g = {int(k): int(v) for k, v in make_relabel(used).items()}
    assert len(set(g.values())) == len(used), "relabeling must be a bijection"

    state_b = _prepared_state(mps, aux_b, phys_b)
    dev_b = sorted(set(phys_b) | set(aux_b))
    aux_r, phys_r = [g[w] for w in aux_b], [g[w] for w in phys_b]
    state_r = _prepared_state(mps, aux_r, phys_r)
    dev_r = sorted(set(phys_r) | set(aux_r))

    # Reorder the relabeled state back into the baseline wire layout, then compare.
    m = len(dev_b)
    axes = [dev_r.index(g[w]) for w in dev_b]
    aligned = np.transpose(np.asarray(state_r).reshape([2] * m), axes).reshape(-1)

    fidelity = np.abs(np.vdot(state_b, aligned)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-8)

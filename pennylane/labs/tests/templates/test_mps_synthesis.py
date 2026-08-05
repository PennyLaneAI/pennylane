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
CASES = [(2, 1, 0), (3, 1, 1), (4, 2, 2), (3, 2, 3), (5, 1, 4), (6, 2, 5), (4, 3, 6)]


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


def test_non_right_canonical_raises():
    """A non-right-canonical MPS is rejected."""
    mps = random_mps(4, 2, seed=0)
    mps[0] = mps[0] * 2.0  # break the canonical condition
    phys_left, phys_bulk, aux = mps_wires(4, 2)
    with pytest.raises(AssertionError, match="right-canonical"):
        mps_synthesis(mps, aux, phys_left + phys_bulk)

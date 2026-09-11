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
"""Tests for the THC qubitization walk operator."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates import qubitization_thc, qubitization_thc_wires
from pennylane.labs.templates.alias_sampling import _build_alias_tables
from pennylane.labs.templates.alias_sampling_thc import _build_thc_pairs, _lcu_signs


def _reference_V(leaf, N, sector):
    r"""Reference :math:`V = I - 2 c^\dagger c` from Jordan-Wigner, no circuit involved."""
    leaf = np.asarray(leaf, dtype=float)
    leaf = leaf / np.linalg.norm(leaf)
    off = 0 if sector == 0 else N // 2
    cdag = sum(float(leaf[p]) * qp.FermiC(off + p) for p in range(len(leaf)))
    ann = sum(float(leaf[p]) * qp.FermiA(off + p) for p in range(len(leaf)))
    n_op = qp.jordan_wigner(cdag * ann, wire_map={w: w for w in range(N)})
    return np.eye(2**N) - 2.0 * qp.matrix(n_op, wire_order=list(range(N)))


def _realized_lcu(M, N, zeta, t_ell, aleph):  # pylint: disable=too-many-arguments
    """Per-entry probability and sign the alias tables actually realize.

    This replays the integer tables the ``QROM`` loads, so the comparison is exact for
    any ``aleph`` instead of an approximation of the ideal coefficients.
    """
    entries, weights = _build_thc_pairs(M, N, zeta, t_ell)
    signs = _lcu_signs(M, entries, weights)
    alt, keep = _build_alias_tables([abs(w) for w in weights], aleph)

    per_pair = {entry: 0.0 for entry in entries}
    for i, entry in enumerate(entries):
        keep_prob = keep[i] / 2**aleph
        per_pair[entry] += keep_prob / len(entries)
        per_pair[entries[alt[i]]] += (1 - keep_prob) / len(entries)
    return per_pair, {e: (-1.0 if s else 1.0) for e, s in zip(entries, signs)}


def _reference_block(M, N, zeta, t_ell, chi, t_eigenvectors, aleph):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """The operator the walk block-encodes, assembled from the documented LCU."""
    per_pair, sign_of = _realized_lcu(M, N, zeta, t_ell, aleph)
    block = np.zeros((2**N, 2**N), dtype=complex)
    for (mu, nu), prob in per_pair.items():
        if prob == 0.0:
            continue
        if nu == M:  # one-body column: a single V, its spin flag averaged
            term = 0.5 * sum(_reference_V(t_eigenvectors[:, mu], N, a) for a in (0, 1))
        else:  # two-body: the symmetrized product, both spin flags averaged
            term = np.zeros((2**N, 2**N), dtype=complex)
            for a in (0, 1):
                for b in (0, 1):
                    v_mu_a, v_nu_b = _reference_V(chi[mu], N, a), _reference_V(chi[nu], N, b)
                    v_nu_a, v_mu_b = _reference_V(chi[nu], N, a), _reference_V(chi[mu], N, b)
                    term += 0.125 * (v_nu_b @ v_mu_a + v_mu_b @ v_nu_a)
        block += prob * sign_of[(mu, nu)] * term
    return block


def _run(zeta, t_ell, chi, t_eigenvectors, aleph, beth, psi, spare=0):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Apply the walk to ``psi`` and return the system amplitudes with every ancilla on |0>."""
    M, n_half = np.shape(chi)
    sizes = qubitization_thc_wires(M, 2 * n_half, aleph, beth)
    total = sum(sizes.values()) + spare
    wires = qp.registers(sizes)
    system = list(wires["system_wires"])
    ancillas = [w for w in range(total) if w not in system]

    def gradient_state():
        for j, wire in enumerate(wires["gradient_wires"]):
            qp.Hadamard(wire)
            qp.PhaseShift(-2 * np.pi * 2 ** (beth - 1 - j) / 2**beth, wires=wire)

    @qp.transforms.decompose(stopping_condition=lambda op: len(op.wires) <= 3)
    @qp.qnode(qp.device("default.qubit", wires=total))
    def circuit():
        qp.StatePrep(psi, wires=system)
        gradient_state()
        qubitization_thc(
            zeta,
            t_ell,
            chi,
            t_eigenvectors,
            aleph,
            beth,
            system,
            wires["index_wires"],
            wires["prep_wires"],
            wires["gradient_wires"],
            wires["work_wires"],
        )
        qp.adjoint(gradient_state)()
        return qp.state()

    state = np.asarray(circuit()).reshape([2] * total)
    selector = [slice(None)] * total
    for wire in ancillas:
        selector[wire] = 0
    return np.asarray(state[tuple(selector)]).reshape(-1)


@pytest.mark.parametrize(
    "M, N, aleph, beth, expected",
    [
        (1, 2, 1, 1, {"system": 2, "index": 2, "prep": 16, "gradient": 1, "work": 1}),
        (2, 2, 1, 1, {"system": 2, "index": 4, "prep": 19, "gradient": 1, "work": 1}),
        (2, 2, 2, 1, {"system": 2, "index": 4, "prep": 22, "gradient": 1, "work": 1}),
        (2, 4, 2, 3, {"system": 4, "index": 4, "prep": 23, "gradient": 3, "work": 5}),
    ],
)
def test_qubitization_thc_wires(M, N, aleph, beth, expected):
    """Test that the register sizes are the documented ones."""
    assert qubitization_thc_wires(M, N, aleph, beth) == {
        key + "_wires": value for key, value in expected.items()
    }


def test_prepare_cannot_succeed_raises():
    """Test that an M whose valid index set is too small to amplify exactly is rejected."""
    M, N, aleph, beth = 4, 2, 1, 1  # d = 11 < 2 ** (2 * 3 - 2) = 16
    with pytest.raises(ValueError, match="cannot reach unit success probability"):
        qubitization_thc_wires(M, N, aleph, beth)  # sizes are fine, the walk is not
        qubitization_thc(
            np.eye(M),
            np.ones(N // 2),
            np.ones((M, N // 2)),
            np.eye(N // 2),
            aleph,
            beth,
            range(N),
            range(N, N + 6),
            range(N + 6, N + 6 + 20),
            [N + 26],
            [N + 27],
        )


@pytest.mark.parametrize(
    "register, match",
    [
        ("system_wires", "system_wires must have exactly"),
        ("index_wires", "index_wires must have exactly"),
        ("prep_wires", "prep_wires must have exactly"),
        ("gradient_wires", "gradient_wires must have exactly"),
    ],
)
def test_wrong_register_size_raises(register, match):
    """Test that every exact register is checked."""
    M, N, aleph, beth = 1, 2, 1, 1
    sizes = qubitization_thc_wires(M, N, aleph, beth)
    sizes[register] += 1
    wires = qp.registers(sizes)
    with pytest.raises(ValueError, match=match):
        qubitization_thc(
            np.eye(M),
            np.ones(N // 2),
            np.ones((M, N // 2)),
            np.eye(N // 2),
            aleph,
            beth,
            wires["system_wires"],
            wires["index_wires"],
            wires["prep_wires"],
            wires["gradient_wires"],
            wires["work_wires"],
        )


@pytest.mark.parametrize(
    "chi_shape, tev_shape, match",
    [((1, 2), (1, 1), "t_eigenvectors must have shape"), ((2, 1), (1, 1), "chi must have shape")],
)
def test_wrong_array_shape_raises(chi_shape, tev_shape, match):
    """Test that the THC arrays must be mutually consistent."""
    M, N, aleph, beth = 1, 2, 1, 1
    sizes = qubitization_thc_wires(M, N, aleph, beth)
    wires = qp.registers(sizes)
    with pytest.raises(ValueError, match=match):
        qubitization_thc(
            np.eye(M),
            np.ones(chi_shape[1]),
            np.ones(chi_shape),
            np.ones(tev_shape),
            aleph,
            beth,
            wires["system_wires"],
            wires["index_wires"],
            wires["prep_wires"],
            wires["gradient_wires"],
            wires["work_wires"],
        )


class TestBlockEncoding:
    """Checks the |0> block of the walk against a Jordan-Wigner reference.

    Every instance below is a full state-vector simulation of
    ``sum(qubitization_thc_wires(...).values())`` wires, so only the smallest ones are
    enabled. ``N = 2`` leaves the Givens network empty, which is what keeps the wire count
    low; the Givens machinery itself is covered by ``test_select_thc.py``.
    """

    @pytest.mark.parametrize("zeta, t_ell", [([[2.0]], [-1.0]), ([[-2.0]], [1.0])])
    def test_block_matches_reference(self, zeta, t_ell):
        """Test that the |0> block is the signed LCU, and in particular that flipping every
        sign flips the block: the coefficient signs must survive PREPARE^dagger."""
        M, N, aleph, beth = 1, 2, 1, 1
        zeta, t_ell = np.array(zeta), np.array(t_ell)
        chi, tev = np.ones((M, N // 2)), np.eye(N // 2)

        psi = np.random.default_rng(7).standard_normal(2**N) + 0j
        psi /= np.linalg.norm(psi)

        got = _run(zeta, t_ell, chi, tev, aleph, beth, psi)
        expected = _reference_block(M, N, zeta, t_ell, chi, tev, aleph) @ psi
        assert np.allclose(got, expected, atol=1e-8)

    def test_identity_shift_form(self):
        """Test the closed form quoted in the docstring: the block is H / lambda up to a
        multiple of the identity."""
        M, N, aleph = 2, 2, 1
        zeta = np.array([[2.0, 1.0], [1.0, -2.0]])
        t_ell = np.array([1.0])
        chi, tev = np.ones((M, N // 2)), np.eye(N // 2)

        def number_op(leaf):
            return sum(0.5 * (np.eye(2**N) - _reference_V(leaf, N, s)) for s in (0, 1))

        n_mu = [number_op(chi[mu]) for mu in range(M)]
        ham = sum(zeta[mu, nu] * n_mu[mu] @ n_mu[nu] for mu in range(M) for nu in range(M))
        ham = ham - 2 * sum(zeta[mu].sum() * n_mu[mu] for mu in range(M))
        ham = ham + 2 * t_ell[0] * number_op(tev[:, 0])
        lam = np.abs(zeta).sum() + 2 * np.abs(t_ell).sum()

        block = _reference_block(M, N, zeta, t_ell, chi, tev, aleph)
        shift = np.trace(block - ham / lam) / 2**N
        assert np.allclose(block - ham / lam, shift * np.eye(2**N), atol=1e-12)

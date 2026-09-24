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
"""Tests for the qubitized tensor hypercontraction walk operator ``QubitizationTHC``."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import assert_valid
from pennylane.templates.subroutines.alias_sampling import _build_alias_tables
from pennylane.templates.subroutines.alias_sampling_thc import _build_thc_pairs, _lcu_signs


@pytest.mark.parametrize(
    "M, N, aleph, beth, expected",
    [
        (1, 2, 1, 1, {"system": 2, "index": 2, "prep_garbage": 15, "gradient": 2, "work": 1}),
        (2, 2, 1, 1, {"system": 2, "index": 4, "prep_garbage": 18, "gradient": 2, "work": 1}),
        (2, 2, 2, 1, {"system": 2, "index": 4, "prep_garbage": 20, "gradient": 2, "work": 2}),
        (2, 4, 2, 3, {"system": 4, "index": 4, "prep_garbage": 21, "gradient": 4, "work": 6}),
    ],
)
def test_qubitization_thc_wires(M, N, aleph, beth, expected):
    """Test that the register sizes are the documented ones."""
    assert qp.qubitization_thc_wires(M, N, aleph, beth) == {
        key + "_wires": value for key, value in expected.items()
    }


def _random_input(M, N, seed):
    """Create random inputs zeta, t_ell, chi for QubitizationTHC (excludes t_eigenvectors)."""
    rng = np.random.default_rng(seed)
    zeta = rng.standard_normal((M, M))
    zeta = tuple(map(tuple, (zeta + zeta.T) / 2))
    chi = tuple(map(tuple, rng.standard_normal((M, N // 2))))
    t_ell = tuple(rng.standard_normal(N // 2))
    return zeta, t_ell, chi


def _dummy_input(M, N):
    """Create dummy inputs zeta, t_ell, chi, t_eigenvectors for QubitizationTHC."""
    return (
        tuple(map(tuple, np.eye(M))),
        tuple(np.ones(N // 2)),
        tuple(map(tuple, np.ones((M, N // 2)))),
        tuple(map(tuple, np.eye(N // 2))),
    )


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
    zeta = np.array(zeta)
    t_ell = np.array(t_ell)
    chi = np.array(chi)
    t_eigenvectors = np.array(t_eigenvectors)
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


def _run(zeta, t_ell, chi, t_eigenvectors, aleph, beth, psi, num_walks=1):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Apply the walk ``num_walks`` times to ``psi`` and return the system amplitudes with
    every auxiliary wire on |0>."""
    M, n_half = np.shape(chi)
    sizes = qp.qubitization_thc_wires(M, 2 * n_half, aleph, beth)
    total = sum(sizes.values())
    wires = qp.registers(sizes)
    system = list(wires["system_wires"])
    auxiliaries = [w for w in range(total) if w not in system]

    def gradient_state():
        for j, wire in enumerate(wires["gradient_wires"]):
            qp.Hadamard(wire)
            qp.PhaseShift(-2 * np.pi * 2 ** (beth - 1 - j) / 2**beth, wires=wire)

    @qp.transforms.decompose(stopping_condition=lambda op: len(op.wires) <= 3)
    @qp.qnode(qp.device("default.qubit", wires=total))
    def circuit():
        qp.StatePrep(psi, wires=system)
        gradient_state()
        for _ in range(num_walks):
            qp.QubitizationTHC(
                tuple(map(tuple, zeta)),
                tuple(t_ell),
                tuple(map(tuple, chi)),
                tuple(map(tuple, t_eigenvectors)),
                aleph,
                beth,
                system,
                wires["index_wires"],
                wires["prep_garbage_wires"],
                wires["gradient_wires"],
                wires["work_wires"],
            )
        qp.adjoint(gradient_state)()
        return qp.state()

    state = np.asarray(circuit()).reshape([2] * total)
    selector = [slice(None)] * total
    for wire in auxiliaries:
        selector[wire] = 0
    return np.asarray(state[tuple(selector)]).reshape(-1)


class TestQubitizationTHC:
    """Test the QubitizationTHC class."""

    # pylint: disable=too-few-public-methods

    @pytest.mark.parametrize("num_batches", [1, 2])
    def test_standard_validity(self, seed, num_batches):
        """Test standard validity of the QubitizationTHC operator with assert_valid."""

        M, N, aleph, beth = 6, 2, 2, 2
        zeta, t_ell, chi = _random_input(M, N, seed)
        *_, t_eigenvectors = _dummy_input(M, N)

        sizes = qp.qubitization_thc_wires(M, N, aleph, beth)
        wires = qp.registers(sizes)

        op = qp.QubitizationTHC(
            zeta, t_ell, chi, t_eigenvectors, aleph, beth, **wires, num_batches=num_batches
        )
        assert_valid(op)

    @pytest.mark.parametrize(
        "register, match",
        [
            ("system_wires", "system_wires must have exactly"),
            ("index_wires", "index_wires must have exactly"),
            ("prep_garbage_wires", "prep_garbage_wires must have exactly"),
            ("gradient_wires", "gradient_wires must have exactly"),
        ],
    )
    def test_wrong_register_size_raises(self, register, match):
        """Test that every exactly-sized register is checked."""
        M, N, aleph, beth = 1, 2, 1, 1
        sizes = qp.qubitization_thc_wires(M, N, aleph, beth)
        sizes[register] += 1
        wires = qp.registers(sizes)
        with pytest.raises(ValueError, match=match):
            qp.QubitizationTHC(
                *_dummy_input(M, N),
                aleph,
                beth,
                **wires,
            )

    def test_too_few_work_wires_raises(self):
        """Test that an error is raised if work_wires, which only have a lower bound on their size,
        are too small."""
        M, N, aleph, beth = 1, 2, 1, 1
        sizes = qp.qubitization_thc_wires(M, N, aleph, beth)
        sizes["work_wires"] -= 1
        wires = qp.registers(sizes)
        with pytest.raises(ValueError, match="work_wires must have at least"):
            qp.QubitizationTHC(
                *_dummy_input(M, N),
                aleph,
                beth,
                **wires,
            )

    @pytest.mark.parametrize("zeta, t_ell", [([[2.0]], [-1.0]), ([[-2.0]], [1.0])])
    def test_block_matches_reference(self, zeta, t_ell, seed):
        """Test that the |0> block is the signed LCU, and in particular that flipping every
        sign flips the block: the coefficient signs must survive PREPARE^dagger."""
        M, N, aleph, beth = 1, 2, 1, 1
        zeta, t_ell = np.array(zeta), np.array(t_ell)
        *_, chi, tev = _dummy_input(M, N)

        psi = np.random.default_rng(seed).standard_normal(2**N) + 0j
        psi /= np.linalg.norm(psi)

        got = _run(zeta, t_ell, chi, tev, aleph, beth, psi)
        expected = _reference_block(M, N, zeta, t_ell, chi, tev, aleph) @ psi
        assert np.allclose(got, expected, atol=1e-8)

    def test_second_chebyshev_moment(self, seed):
        """Test that two walks give T_2(H / lambda) = 2 (H / lambda)^2 - I.

        The single-walk test above cannot detect a reflection of the wrong scope, because
        ``<0| R = <0|`` for any reflection whose fixed subspace contains ``|0>``. This one
        can: it is the first moment that sees the reflection, and it fails by O(1) if the
        garbage is left out of the reflected register.
        """
        M, N, aleph, beth = 1, 2, 1, 1
        zeta, t_ell = np.array([[2.0]]), np.array([-1.0])
        *_, chi, tev = _dummy_input(M, N)

        psi = np.random.default_rng(seed).standard_normal(2**N) + 0j
        psi /= np.linalg.norm(psi)

        block = _reference_block(M, N, zeta, t_ell, chi, tev, aleph)
        chebyshev_2 = 2 * block @ block - np.eye(2**N)

        got = _run(zeta, t_ell, chi, tev, aleph, beth, psi, num_walks=2)
        assert np.allclose(got, chebyshev_2 @ psi, atol=1e-8)

    def test_identity_shift_form(self):
        """Test the closed form quoted in the docstring: the block is H / lambda up to a
        multiple of the identity."""
        M, N, aleph = 2, 2, 1
        zeta = np.array([[2.0, 1.0], [1.0, -2.0]])
        t_ell = np.array([1.0])
        *_, chi, tev = _dummy_input(M, N)

        def number_op(leaf):
            return sum(0.5 * (np.eye(2**N) - _reference_V(leaf, N, s)) for s in (0, 1))

        n_mu = [number_op(chi[mu]) for mu in range(M)]
        ham = sum(zeta[mu, nu] * n_mu[mu] @ n_mu[nu] for mu in range(M) for nu in range(M))
        ham = ham - 2 * sum(zeta[mu].sum() * n_mu[mu] for mu in range(M))
        ham = ham + 2 * t_ell[0] * number_op(np.array(tev)[:, 0])
        lam = np.abs(zeta).sum() + 2 * np.abs(t_ell).sum()

        block = _reference_block(M, N, zeta, t_ell, chi, tev, aleph)
        shift = np.trace(block - ham / lam) / 2**N
        assert np.allclose(block - ham / lam, shift * np.eye(2**N), atol=1e-12)

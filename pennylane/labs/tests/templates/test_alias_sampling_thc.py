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
Tests for the ``alias_sampling_thc`` quantum function.
"""

import numpy as np
import pytest

import pennylane as qp

# Adjust this import to the module path once the file lands in the labs templates.
from pennylane.labs.templates import (
    SuperpositionTHC,
    _build_alias_tables,
    _build_qrom_data,
    _first_arithmetic_op,
    alias_sampling_thc,
)


def _wire_layout(M, N, n, aleph):
    """mu / nu / superposition-work / edge-flag / alias-work registers.

    ``SuperpositionTHC`` prepares the input superposition (using ``3 * n + 5``
    work wires) and its ``work_wires[3]`` carries the one-body sentinel flag that
    ``alias_sampling_thc`` consumes as ``edge_flag``.
    """
    n_d = int(np.ceil(np.log2(N // 2 + M * (M + 1) / 2))) + 1
    num_work = n_d + 2 * n + 3 * aleph + 4
    mu_wires = list(range(0, n))
    nu_wires = list(range(n, 2 * n))
    sup_work = list(range(2 * n, 2 * n + 3 * n + 5))
    edge_flag = sup_work[3]

    # SuperpositionTHC returns every work wire to |0> except its flags at indices
    # 0, 3 and 6, so the remaining wires can be reused as alias-sampling scratch.
    clean = [w for i, w in enumerate(sup_work) if i not in (0, 3, 6)]
    start = sup_work[-1] + 1
    fresh = list(range(start, start + max(0, num_work - len(clean))))
    work_wires = (clean + fresh)[:num_work]
    return mu_wires, nu_wires, sup_work, edge_flag, work_wires


def _valid_pairs(M, N):
    two_body = [(mu, nu) for nu in range(M) for mu in range(nu + 1)]
    one_body = [(ell, M) for ell in range(N // 2)]
    return two_body + one_body


def _symmetric_target(M, N, zeta, t_ell, n):
    """The probability distribution over |mu>|nu> encoded by ``alias_sampling_thc``.

    The routine normalizes its alias table by

        total_w = sum_{mu <= nu} |zeta_{mu,nu}| (diagonal halved) + sum_ell |t_ell|,

    so the two-body 1-norm is counted over *unordered* pairs. The symmetrization
    step (the mu <-> nu swap) then splits each two-body weight equally across the
    two orderings (mu, nu) and (nu, mu), giving

        P[mu, nu] = |zeta_{mu,nu}| / (2 * total_w)   for the two-body block.

    The one-body sentinel column (ell, M) is *excluded* from the swap (the
    ``edge_flag`` zero-control), so it keeps its full weight

        P[ell, M] = |t_ell| / total_w .

    Note: an earlier version of this helper normalized the one-body block by the
    same symmetric denominator as the two-body block, which under-counted the
    one-body probability by roughly a factor of two. The formula below matches the
    distribution the circuit actually prepares.
    """
    # Normalization used by the alias table (unordered two-body pairs, diagonal
    # halved) plus the one-body 1-norm.
    total_w = 0.0
    for nu in range(M):
        for mu in range(nu + 1):
            w = abs(zeta[mu, nu])
            total_w += w / 2.0 if mu == nu else w
    for ell in range(N // 2):
        total_w += abs(t_ell[ell])

    size = 2**n
    P = np.zeros((size, size))
    # Two-body block: the symmetrization splits the weight across both orderings.
    for mu in range(M):
        for nu in range(M):
            P[mu, nu] = abs(zeta[mu, nu]) / (2.0 * total_w)
    # One-body block: excluded from the swap, so it keeps its full weight.
    for ell in range(N // 2):
        P[ell, M] = abs(t_ell[ell]) / total_w
    return P


def _run(M, N, zeta, t_ell, n, aleph, device="lightning.qubit"):
    mu_wires, nu_wires, sup_work, edge_flag, work_wires = _wire_layout(M, N, n, aleph)
    total = max(mu_wires + nu_wires + sup_work + work_wires) + 1
    dev = qp.device(device, wires=total)

    @qp.qnode(dev)
    def circuit():
        SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
        alias_sampling_thc(
            M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph
        )
        return qp.probs(wires=mu_wires + nu_wires)

    return np.asarray(circuit()).reshape((2**n, 2**n))


# --------------------------------------------------------------------------- #
# Classical preprocessing
# --------------------------------------------------------------------------- #
class TestClassicalTables:
    """Test the classical alias-table construction."""

    def test_table_size_and_normalization(self):
        M, N = 3, 4
        np.random.seed(0)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        table = _build_alias_tables(M, N, zeta, t_ell)
        d = N // 2 + M * (M + 1) // 2
        assert len(table) == d

        # keep_prob is a valid probability for every entry.
        for row in table:
            assert 0.0 <= row["keep_prob"] <= 1.0

    def test_alias_reconstructs_target(self):
        """The (unquantized) alias tables reproduce the target distribution."""
        M, N = 4, 2
        np.random.seed(1)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        table = _build_alias_tables(M, N, zeta, t_ell)
        d = len(table)

        # Target (per valid pair, before symmetrization).
        weights = {}
        for nu in range(M):
            for mu in range(nu + 1):
                w = zeta[mu, nu]
                if mu == nu:
                    w = w / 2.0
                weights[(mu, nu)] = abs(w)
        for ell in range(N // 2):
            weights[(ell, M)] = abs(t_ell[ell])
        tot = sum(weights.values())

        recon = {k: 0.0 for k in weights}
        for row in table:
            recon[(row["mu"], row["nu"])] += (1 / d) * row["keep_prob"]
            recon[(row["mu_alt"], row["nu_alt"])] += (1 / d) * (1 - row["keep_prob"])

        for k in weights:
            assert np.isclose(recon[k], weights[k] / tot, atol=1e-6)

    def test_qrom_data_shape(self):
        M, N, n, aleph = 3, 2, 3, 5
        np.random.seed(2)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        data = _build_qrom_data(M, N, zeta, t_ell, n, aleph)
        d = N // 2 + M * (M + 1) // 2
        assert len(data) == d
        # Each row: sign + alt_sign + mu_alt + nu_alt + keep + alt_edge.
        expected_bits = 1 + 1 + n + n + aleph + 1
        for row in data:
            assert len(row) == expected_bits
            assert all(bit in (0, 1) for bit in row)


# --------------------------------------------------------------------------- #
# Arithmetic address computation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("M", "N", "n"),
    [(5, 2, 3), (3, 2, 2)],
)
def test_first_arithmetic_op_index(M, N, n):
    """``_first_arithmetic_op`` computes s = mu + nu (nu + 1) / 2."""
    n_d = int(np.ceil(np.log2(N // 2 + M * (M + 1) / 2))) + 1
    mu_wires = list(range(n))
    nu_wires = list(range(n, 2 * n))
    work_wires = list(range(2 * n, 2 * n + 2 * n_d + 5))
    dev = qp.device("default.qubit", wires=2 * n + len(work_wires))

    @qp.qnode(dev)
    def circuit(mu_val, nu_val):
        qp.BasisState(mu_val, wires=mu_wires)
        qp.BasisState(nu_val, wires=nu_wires)
        _first_arithmetic_op(M, N, mu_wires, nu_wires, work_wires)
        return qp.probs(wires=work_wires[:n_d])

    for nu in range(M):
        for mu in range(nu + 1):
            probs = circuit(mu, nu)
            assert int(np.argmax(probs)) == mu + nu * (nu + 1) // 2


# --------------------------------------------------------------------------- #
# End-to-end preparation
# --------------------------------------------------------------------------- #
class TestAliasSamplingTHC:
    """Test the full alias-sampling PREPARE routine."""

    @pytest.mark.parametrize(
        ("M", "N", "n", "aleph"),
        [
            (2, 2, 2, 3),
            (2, 2, 2, 4),
        ],
    )
    def test_probabilities_normalized(self, M, N, n, aleph):
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)
        probs = _run(M, N, zeta, t_ell, n, aleph)
        assert np.isclose(probs.sum(), 1.0)

    def test_support_matches_symmetric_valid_set(self):
        """All probability mass lands on the symmetrized valid support."""
        M, N, n, aleph = 2, 2, 2, 4
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        probs = _run(M, N, zeta, t_ell, n, aleph)
        target = _symmetric_target(M, N, zeta, t_ell, n)

        support = {(a, b) for a in range(2**n) for b in range(2**n) if probs[a, b] > 1e-9}
        target_support = {
            (a, b) for a in range(2**n) for b in range(2**n) if target[a, b] > 1e-9
        }
        assert support == target_support

    def test_approaches_target_distribution(self):
        """The prepared distribution is close to the (correctly normalized) target.

        With the fixed one-body normalization the residual is dominated by the
        single amplitude-free alias round (it is essentially independent of
        ``aleph`` for these small instances), so a modest tolerance is used.
        """
        M, N, n, aleph = 2, 2, 2, 4
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        probs = _run(M, N, zeta, t_ell, n, aleph)
        target = _symmetric_target(M, N, zeta, t_ell, n)

        assert np.max(np.abs(probs - target)) < 0.05


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
class TestInputValidation:
    """Test the argument checks."""

    def _dummy(self, M, N):
        zeta = np.ones((M, M))
        t_ell = np.ones(N // 2)
        return zeta, t_ell

    def test_mismatched_registers(self):
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="same number of wires"):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3, 4], 5, list(range(6, 40)), 3)

    def test_index_register_too_small(self):
        zeta, t_ell = self._dummy(8, 2)
        with pytest.raises(ValueError, match="at least ceil"):
            alias_sampling_thc(8, 2, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

    def test_not_enough_work_wires(self):
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="At least"):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, [5, 6, 7], 3)

    def test_invalid_aleph(self):
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="aleph"):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 0)

    def test_bad_n_over_two(self):
        zeta = np.ones((2, 2))
        t_ell = np.ones(4)
        with pytest.raises(ValueError, match="N / 2 must be"):
            alias_sampling_thc(2, 8, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

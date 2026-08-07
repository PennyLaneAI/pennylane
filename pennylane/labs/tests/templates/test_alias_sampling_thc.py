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
from pennylane.labs.templates.alias_sampling_thc import _build_thc_pairs


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


def _reconstruct_distribution(M, N, zeta, t_ell, n, aleph):  # pylint: disable=too-many-arguments
    """Exact distribution over |mu>|nu> prepared by ``alias_sampling_thc``.

    This is the THC analogue of ``_reconstruct_amplitudes`` in
    ``test_alias_sampling.py``: it plays the *same* integer alias tables the circuit
    loads into the QROM back classically, so the comparison is exact (independent of
    ``aleph``) rather than an approximation of the ideal target.

    Each address keeps its original pair with probability ``(keep + 1) / 2 ** aleph``
    (the circuit tests ``keep_thresh < sigma`` against a uniform ``aleph``-bit sample,
    so ``keep`` values ``0 .. keep`` all pass), and routes the remaining mass to its
    alternate. The symmetrization step then splits every two-body weight across the
    two orderings ``(mu, nu)`` and ``(nu, mu)``, while the one-body sentinel column
    ``nu = M`` is excluded from the swap and keeps its full weight.
    """
    entries, weights = _build_thc_pairs(M, N, zeta, t_ell)
    probs = [abs(w) for w in weights]
    alt, keep = _build_alias_tables(probs, aleph)

    d = len(entries)
    n_levels = 2**aleph
    per_pair = {e: 0.0 for e in entries}
    for i, entry in enumerate(entries):
        keep_prob = (keep[i] + 1) / n_levels  # comparator "<" against uniform sample
        per_pair[entry] += (1 / d) * keep_prob
        per_pair[entries[alt[i]]] += (1 / d) * (1 - keep_prob)

    size = 2**n
    P = np.zeros((size, size))
    for (mu, nu), p in per_pair.items():
        if nu == M:  # one-body block: excluded from the symmetrizing swap
            P[mu, nu] += p
        else:  # two-body block: split across both orderings
            P[mu, nu] += p / 2.0
            P[nu, mu] += p / 2.0
    return P


def _run(
    M, N, zeta, t_ell, n, aleph, device="lightning.qubit"
):  # pylint: disable=too-many-arguments
    mu_wires, nu_wires, sup_work, edge_flag, work_wires = _wire_layout(M, N, n, aleph)
    total = max(mu_wires + nu_wires + sup_work + work_wires) + 1
    dev = qp.device(device, wires=total)

    @qp.qnode(dev)
    def circuit():
        SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
        alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)
        return qp.probs(wires=mu_wires + nu_wires)

    return np.asarray(circuit()).reshape((2**n, 2**n))


class TestClassicalTables:
    """Test the classical alias-table construction."""

    def test_table_size_and_normalization(self):
        """The THC pair enumeration has one entry per valid pair with a valid keep."""
        M, N, aleph = 3, 4, 6
        np.random.seed(0)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        entries, weights = _build_thc_pairs(M, N, zeta, t_ell)
        d = N // 2 + M * (M + 1) // 2
        assert len(entries) == d
        assert len(weights) == d

        alt, keep = _build_alias_tables([abs(w) for w in weights], aleph)
        assert all(0 <= a < d for a in alt)
        assert all(0 <= k < 2**aleph for k in keep)

    def test_alias_reconstructs_target(self):
        """The (unquantized) alias tables reproduce the target distribution."""
        M, N, aleph = 4, 2, 12
        np.random.seed(1)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        entries, weights = _build_thc_pairs(M, N, zeta, t_ell)
        d = len(entries)
        probs = [abs(w) for w in weights]
        tot = sum(probs)

        alt, keep = _build_alias_tables(probs, aleph)
        n_levels = 2**aleph
        recon = {e: 0.0 for e in entries}
        for i, entry in enumerate(entries):
            keep_prob = keep[i] / n_levels
            recon[entry] += (1 / d) * keep_prob
            recon[entries[alt[i]]] += (1 / d) * (1 - keep_prob)

        for entry, w in zip(entries, probs):
            # mu-bit alias sampling reproduces the target up to the d / 2 ** aleph bound.
            assert np.abs(recon[entry] - w / tot) <= float(d) / n_levels

    def test_qrom_data_shape(self):
        """Each packed QROM row has the expected number of bits, all binary."""
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


@pytest.mark.parametrize(
    ("M", "N", "n"),
    [
        (3, 2, 2),
        (5, 2, 3),
    ],
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


class TestAliasSamplingTHC:
    """Test the full alias-sampling PREPARE routine."""

    # Each instance below runs a full state-vector simulation whose wire count is
    # 5 * n + 5 + (n_d + 3 * aleph + 4); memory and runtime grow as 2 ** wires, so
    # only the small instances (< 40 s each) are enabled. Larger instances are kept
    # commented out for reference -- uncomment to run them on a bigger machine.
    _INSTANCES = [
        (2, 2, 2, 1),  # 21 wires, ~1 s
        (2, 2, 2, 2),  # 24 wires, ~5 s
        (3, 2, 2, 2),  # 25 wires, ~12 s
        # (2, 2, 2, 3),
        # (2, 2, 2, 4),
        # (3, 2, 3, 5),
    ]

    @pytest.mark.parametrize(("M", "N", "n", "aleph"), _INSTANCES)
    def test_probabilities_normalized(self, M, N, n, aleph):
        """The prepared distribution sums to one."""
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)
        probs = _run(M, N, zeta, t_ell, n, aleph)
        assert np.isclose(probs.sum(), 1.0)

    @pytest.mark.parametrize(("M", "N", "n", "aleph"), _INSTANCES)
    def test_marginal_matches_reconstruction(self, M, N, n, aleph):
        """The prepared distribution matches the classical alias reconstruction exactly."""
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        probs = _run(M, N, zeta, t_ell, n, aleph)
        recon = _reconstruct_distribution(M, N, zeta, t_ell, n, aleph)

        assert np.allclose(probs, recon, atol=1e-9)

    @pytest.mark.parametrize(("M", "N", "n", "aleph"), _INSTANCES)
    def test_support_matches_symmetric_valid_set(self, M, N, n, aleph):
        """All probability mass lands on the symmetrized valid support."""
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        probs = _run(M, N, zeta, t_ell, n, aleph)
        recon = _reconstruct_distribution(M, N, zeta, t_ell, n, aleph)

        support = {(a, b) for a in range(2**n) for b in range(2**n) if probs[a, b] > 1e-9}
        target_support = {(a, b) for a in range(2**n) for b in range(2**n) if recon[a, b] > 1e-9}
        assert support == target_support


class TestInputValidation:
    """Test the argument checks."""

    def _dummy(self, M, N):
        zeta = np.ones((M, M))
        t_ell = np.ones(N // 2)
        return zeta, t_ell

    def test_mismatched_registers(self):
        """mu_wires and nu_wires of different lengths raise an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="same number of wires"):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3, 4], 5, list(range(6, 40)), 3)

    def test_index_register_too_small(self):
        """An index register too small to hold M raises an error."""
        zeta, t_ell = self._dummy(8, 2)
        with pytest.raises(ValueError, match="at least ceil"):
            alias_sampling_thc(8, 2, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

    def test_not_enough_work_wires(self):
        """Too few work wires raise an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="At least"):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, [5, 6, 7], 3)

    def test_invalid_aleph(self):
        """A non-positive aleph raises an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="aleph"):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 0)

    def test_bad_n_over_two(self):
        """A value of N / 2 larger than M + 1 raises an error."""
        zeta = np.ones((2, 2))
        t_ell = np.ones(4)
        with pytest.raises(ValueError, match="N / 2 must be"):
            alias_sampling_thc(2, 8, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

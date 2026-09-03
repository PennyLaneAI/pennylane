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
from pennylane.labs.templates import (
    SuperpositionTHC,
    alias_sampling_thc,
    alias_sampling_thc_wires,
)
from pennylane.labs.templates.alias_sampling_thc import (
    _build_alias_tables,
    _build_qrom_data,
    _build_thc_pairs,
    _compute_contiguous_register,
)


def _wire_layout(M, N, aleph):
    """mu / nu / superposition-work / edge-flag / alias-work registers.

    Register sizes come from ``alias_sampling_thc_wires``, so ``n`` is derived from
    ``M`` rather than passed in. ``SuperpositionTHC`` prepares the input
    superposition and its ``work_wires[3]`` carries the one-body sentinel flag that
    ``alias_sampling_thc`` consumes as ``edge_flag``.
    """
    sizes = alias_sampling_thc_wires(M, N, aleph)
    n = sizes["mu_wires"]
    num_work = sizes["work_wires"]
    mu_wires = list(range(0, n))
    nu_wires = list(range(n, 2 * n))
    sup_work = list(range(2 * n, 2 * n + sizes["superposition_work_wires"]))
    edge_flag = sup_work[3]

    # SuperpositionTHC returns every work wire to |0> except its flags at indices
    # 0, 3 and 6, so the remaining wires can be reused as alias-sampling scratch.
    clean = [w for i, w in enumerate(sup_work) if i not in (0, 3, 6)]
    start = sup_work[-1] + 1
    fresh = list(range(start, start + max(0, num_work - len(clean))))
    work_wires = (clean + fresh)[:num_work]
    return mu_wires, nu_wires, sup_work, edge_flag, work_wires


def _reconstruct_distribution(M, N, zeta, t_ell, aleph):  # pylint: disable=too-many-arguments
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

    size = 2 ** alias_sampling_thc_wires(M, N, aleph)["mu_wires"]
    P = np.zeros((size, size))
    for (mu, nu), p in per_pair.items():
        if nu == M:  # one-body block: excluded from the symmetrizing swap
            P[mu, nu] += p
        else:  # two-body block: split across both orderings
            P[mu, nu] += p / 2.0
            P[nu, mu] += p / 2.0
    return P


_T_GATE_SET = {
    "T",
    "Adjoint(T)",
    "Hadamard",
    "S",
    "Adjoint(S)",
    "CNOT",
    "X",
    "Z",
    "CZ",
    "SWAP",
    "GlobalPhase",
    "RZ",
}


def _t_count(M, N, aleph, extra_work_wires):
    """T-gate count of the template given ``extra_work_wires`` beyond the minimum."""
    sizes = alias_sampling_thc_wires(M, N, aleph)
    n = sizes["mu_wires"]
    mu_wires = list(range(n))
    nu_wires = list(range(n, 2 * n))
    edge_flag = 2 * n
    num_work = sizes["work_wires"] + extra_work_wires
    work_wires = list(range(2 * n + 1, 2 * n + 1 + num_work))

    np.random.seed(3)
    zeta = np.random.randn(M, M)
    zeta = (zeta + zeta.T) / 2
    t_ell = np.random.randn(N // 2)

    def qfunc():
        alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)

    with qp.decomposition.toggle_graph_ctx(True):
        tape = qp.tape.make_qscript(qfunc)()
        [decomposed], _ = qp.transforms.decompose(tape, gate_set=_T_GATE_SET)

    names = [op.name for op in decomposed.operations]
    return names.count("T") + names.count("Adjoint(T)")


def _run(M, N, zeta, t_ell, aleph, device="lightning.qubit"):  # pylint: disable=too-many-arguments
    mu_wires, nu_wires, sup_work, edge_flag, work_wires = _wire_layout(M, N, aleph)
    total = max(mu_wires + nu_wires + sup_work + work_wires) + 1
    dev = qp.device(device, wires=total)

    @qp.qnode(dev)
    def circuit():
        SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
        alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)
        return qp.probs(wires=mu_wires + nu_wires)

    n = len(mu_wires)
    return np.asarray(circuit()).reshape((2**n, 2**n))


class TestClassicalTables:
    """Test the classical alias-table construction."""

    def test_table_size_and_normalization(self):
        """Test that the THC pair enumeration has one entry per valid pair with a valid keep."""
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
        """Test that the (unquantized) alias tables reproduce the target distribution."""
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
        """Test that each packed QROM row has the expected number of bits, all binary."""
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


@pytest.mark.parametrize(("M", "N"), [(3, 2), (5, 2)])
def test_compute_contiguous_register_index(M, N):
    """Test that ``_compute_contiguous_register`` computes s = mu + nu (nu + 1) / 2."""
    n = alias_sampling_thc_wires(M, N, aleph=1)["mu_wires"]
    n_d = int(np.ceil(np.log2(N // 2 + M * (M + 1) // 2))) + 1
    mu_wires = list(range(n))
    nu_wires = list(range(n, 2 * n))
    work_wires = list(range(2 * n, 2 * n + 2 * n_d + 5))
    dev = qp.device("default.qubit", wires=2 * n + len(work_wires))

    @qp.qnode(dev)
    def circuit(mu_val, nu_val):
        qp.BasisState(qp.math.int_to_binary(mu_val, n), wires=mu_wires)
        qp.BasisState(qp.math.int_to_binary(nu_val, n), wires=nu_wires)
        _compute_contiguous_register(M, N, mu_wires, nu_wires, work_wires)
        return qp.probs(wires=work_wires[:n_d])

    for nu in range(M):
        for mu in range(nu + 1):
            probs = circuit(mu, nu)
            s = int(np.argmax(probs))
            assert s == mu + nu * (nu + 1) // 2
            # The leading wire is only needed to hold ``nu ** 2 + nu`` before the
            # division by two; the final address always fits in ``n_d - 1`` wires, which
            # is what ``alias_sampling_thc`` uses to control its QROM.
            assert s < 2 ** (n_d - 1)


class TestAliasSamplingTHC:
    """Test the full alias-sampling PREPARE routine."""

    # Each instance below runs a full state-vector simulation whose wire count is
    # 5 * n + 5 + (n_d + 3 * aleph + 4) with n = ceil(log2(M + 1)); memory and runtime
    # grow as 2 ** wires, so only the small instances (< 40 s each) are enabled. Larger
    # instances are kept commented out for reference -- uncomment to run them on a
    # bigger machine.
    _INSTANCES = [
        (2, 2, 1),  # n = 2, 21 wires, ~1 s
        (2, 2, 2),  # n = 2, 24 wires, ~5 s
        (3, 2, 2),  # n = 2, 25 wires, ~12 s
        # (2, 2, 3),
        # (2, 2, 4),
        # (5, 2, 5),
    ]

    @pytest.mark.parametrize(("M", "N", "aleph"), _INSTANCES)
    def test_probabilities_normalized(self, M, N, aleph):
        """Test that the prepared distribution sums to one."""
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)
        probs = _run(M, N, zeta, t_ell, aleph)
        assert np.isclose(probs.sum(), 1.0)

    @pytest.mark.parametrize(("M", "N", "aleph"), _INSTANCES)
    def test_marginal_matches_reconstruction(self, M, N, aleph):
        """Test that the prepared distribution matches the classical alias reconstruction."""
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        probs = _run(M, N, zeta, t_ell, aleph)
        recon = _reconstruct_distribution(M, N, zeta, t_ell, aleph)

        assert np.allclose(probs, recon, atol=1e-9)

    @pytest.mark.parametrize(("M", "N", "aleph"), _INSTANCES)
    def test_support_matches_symmetric_valid_set(self, M, N, aleph):
        """Test that all probability mass lands on the symmetrized valid support."""
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        probs = _run(M, N, zeta, t_ell, aleph)
        recon = _reconstruct_distribution(M, N, zeta, t_ell, aleph)

        n = alias_sampling_thc_wires(M, N, aleph)["mu_wires"]
        support = {(a, b) for a in range(2**n) for b in range(2**n) if probs[a, b] > 1e-9}
        target_support = {(a, b) for a in range(2**n) for b in range(2**n) if recon[a, b] > 1e-9}
        assert support == target_support

    @pytest.mark.parametrize(("M", "N"), [(2, 2), (3, 2), (5, 2), (8, 4)])
    def test_qrom_uses_minimal_address_space(self, M, N):
        """Test that the QROM is controlled on the minimal number of address wires.

        The contiguous address never exceeds ``d - 1``, so ``ceil(log2(d))`` control
        wires are enough; controlling on the spare high wire of the arithmetic register
        would double the QROM address space and its gate cost.
        """
        aleph = 3
        sizes = alias_sampling_thc_wires(M, N, aleph)
        n = sizes["mu_wires"]
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n + 1, 2 * n + 1 + sizes["work_wires"]))

        zeta = np.ones((M, M))
        t_ell = np.ones(N // 2)

        def qfunc():
            alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, 2 * n, work_wires, aleph)

        tape = qp.tape.make_qscript(qfunc)()
        qroms = [op for op in tape.operations if isinstance(op, qp.QROM)]
        assert len(qroms) == 1

        d = N // 2 + M * (M + 1) // 2
        assert len(qroms[0].control_wires) == int(np.ceil(np.log2(d)))

    def test_ancillas_returned_to_zero(self):
        """Test that the comparator flag and its work wires are left in |0>.

        The inequality test of step 3 is uncomputed with the *same* comparator
        in step 6, so ``alt_flag`` and the comparator work wires end in |0>.
        """
        M, N, aleph = 2, 2, 2
        mu_wires, nu_wires, sup_work, edge_flag, work_wires = _wire_layout(M, N, aleph)
        n = len(mu_wires)
        n_d = int(np.ceil(np.log2(N // 2 + M * (M + 1) // 2))) + 1
        b = n_d + 2 * n + 2 * aleph
        ancillas = [work_wires[b + 2]] + list(work_wires[b + 5 : b + aleph + 4])

        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        total = max(mu_wires + nu_wires + sup_work + work_wires) + 1
        dev = qp.device("default.qubit", wires=total)

        @qp.qnode(dev)
        def circuit():
            SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
            alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)
            return qp.probs(wires=ancillas)

        probs = np.asarray(circuit())
        assert np.isclose(probs[0], 1.0, atol=1e-9)


class TestInputValidation:
    """Test the argument checks."""

    def _dummy(self, M, N):
        zeta = np.ones((M, M))
        t_ell = np.ones(N // 2)
        return zeta, t_ell

    def test_mismatched_registers(self):
        """Test that mu_wires and nu_wires of different lengths raise an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="same number of wires"):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3, 4], 5, list(range(6, 40)), 3)

    @pytest.mark.parametrize("n", [2, 5])
    def test_index_register_wrong_size(self, n):
        """Test that index registers not of size exactly ceil(log2(M + 1)) raise an error."""
        # M = 8 needs ceil(log2(9)) = 4 wires per register: 2 is too few, 5 too many.
        zeta, t_ell = self._dummy(8, 2)
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        with pytest.raises(ValueError, match="exactly ceil"):
            alias_sampling_thc(
                8, 2, zeta, t_ell, mu_wires, nu_wires, 2 * n, list(range(2 * n + 1, 60)), 3
            )

    def test_not_enough_work_wires(self):
        """Test that too few work wires raise an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="At least"):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, [5, 6, 7], 3)

    @pytest.mark.parametrize("aleph", [0, -1, 2.0, 3.5, True, "3", None])
    def test_invalid_aleph(self, aleph):
        """Test that a non-integer or non-positive aleph raises an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="aleph must be a positive integer"):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), aleph)

    @pytest.mark.parametrize(
        ("zeta", "t_ell", "match"),
        [
            (np.ones((3, 3)), np.ones(1), r"zeta must be of shape \(2, 2\)"),
            (np.ones(2), np.ones(1), r"zeta must be of shape \(2, 2\)"),
            (np.ones((2, 2)), np.ones(0), r"t_ell must be of shape \(1,\)"),
            (np.ones((2, 2)), np.ones((1, 1)), r"t_ell must be of shape \(1,\)"),
        ],
    )
    def test_bad_coefficient_shapes(self, zeta, t_ell, match):
        """Test that coefficients indexing out of bounds raise a ValueError, not IndexError."""
        with pytest.raises(ValueError, match=match):
            alias_sampling_thc(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

    def test_bad_n_over_two(self):
        """Test that a value of N // 2 larger than M + 1 raises an error."""
        zeta = np.ones((2, 2))
        t_ell = np.ones(4)
        with pytest.raises(ValueError, match="N // 2 must be"):
            alias_sampling_thc(2, 8, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

    def test_odd_spin_orbitals_allowed(self):
        """Test that an odd N is floor-divided, matching ``SuperpositionTHC``: N = 5, M = 1."""
        # ``N // 2 = 2 <= M + 1 = 2``, so the previous ``N / 2 = 2.5 > 2`` check was wrong.
        zeta = np.ones((1, 1))
        t_ell = np.ones(5 // 2)
        sizes = alias_sampling_thc_wires(1, 5, aleph=3)
        n = sizes["mu_wires"]
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n + 1, 2 * n + 1 + sizes["work_wires"]))
        # Queued without raising; the wire helper agrees with the template's own checks.
        with qp.queuing.AnnotatedQueue():
            alias_sampling_thc(1, 5, zeta, t_ell, mu_wires, nu_wires, 2 * n, work_wires, 3)


class TestWiresHelper:
    """Test ``alias_sampling_thc_wires``."""

    def test_reported_sizes_are_accepted(self):
        """Test that the reported register sizes satisfy every check in ``alias_sampling_thc``."""
        M, N, aleph = 5, 2, 4
        sizes = alias_sampling_thc_wires(M, N, aleph)
        n = sizes["mu_wires"]
        assert n == sizes["nu_wires"] == int(np.ceil(np.log2(M + 1)))
        assert sizes["superposition_work_wires"] == 3 * n + 5
        n_d = int(np.ceil(np.log2(N // 2 + M * (M + 1) // 2))) + 1
        assert sizes["work_wires"] == n_d + 2 * n + 3 * aleph + 4

        zeta = np.ones((M, M))
        t_ell = np.ones(N // 2)
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n + 1, 2 * n + 1 + sizes["work_wires"]))
        with qp.queuing.AnnotatedQueue():
            alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, 2 * n, work_wires, aleph)

    def test_extra_work_wires_reduce_t_count(self):
        """Test that work wires beyond the minimum are forwarded to ``qp.QROM``, which
        uses them for a ``SelectSwap`` decomposition with a lower T-gate count.

        The trade-off is not monotonic: ``qp.QROM`` consumes every work wire it is
        given, so far more wires than the width of a target register can push the count
        back up. Only the documented reduction is asserted here.
        """
        minimum = _t_count(4, 2, 3, extra_work_wires=0)
        with_extra = _t_count(4, 2, 3, extra_work_wires=2)
        assert with_extra < minimum

    @pytest.mark.parametrize(
        ("M", "N", "aleph", "match"),
        [
            (0, 2, 3, "M must be a positive integer"),
            (2.0, 2, 3, "M must be a positive integer"),
            (2, 0, 3, "N must be a positive integer"),
            (2, 2, 0, "aleph must be a positive integer"),
            (2, 2, 1.5, "aleph must be a positive integer"),
            (1, 8, 3, "N // 2 must be"),
        ],
    )
    def test_invalid_arguments(self, M, N, aleph, match):
        """Test that invalid arguments raise an error."""
        with pytest.raises(ValueError, match=match):
            alias_sampling_thc_wires(M, N, aleph)

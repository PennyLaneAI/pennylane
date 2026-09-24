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
"""Tests for AliasSamplingTHC and alias_sampling_thc_wires."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import assert_valid
from pennylane.templates.subroutines.alias_sampling_thc import (
    _build_alias_tables,
    _build_qrom_data,
    _build_thc_pairs,
    _compute_contiguous_register,
    _cswap_pair,
    _symmetrize,
)
from pennylane.typing import AbstractWires


def _wire_layout(M, N, aleph):
    """mu / nu / superposition-work / edge-flag / alias-work registers.

    Register sizes come from ``alias_sampling_thc_wires``, so ``n`` is derived from
    ``M`` rather than passed in. ``SuperpositionTHC`` prepares the input
    superposition and its ``work_wires[3]`` carries the one-body sentinel flag that
    ``AliasSamplingTHC`` consumes as ``edge_flag``.
    """
    sizes = qp.alias_sampling_thc_wires(M, N, aleph)
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


def _static_coeffs(zeta, t_ell):
    """Hashable nested tuples for compilable ``zeta`` / ``t_ell``."""
    return (
        tuple(tuple(map(float, row)) for row in np.asarray(zeta, dtype=float)),
        tuple(map(float, np.asarray(t_ell, dtype=float).ravel())),
    )


def _reconstruct_distribution(M, N, zeta, t_ell, aleph):  # pylint: disable=too-many-arguments
    """Exact distribution over |mu>|nu> prepared by ``AliasSamplingTHC``.

    This is the THC analogue of ``_reconstruct_amplitudes`` in
    ``test_alias_sampling.py``: it plays the *same* integer alias tables the circuit
    loads into the QROM back classically, so the comparison is exact (independent of
    ``aleph``).

    Each address keeps its original pair with probability ``keep / 2 ** aleph`` (the
    circuit tests ``keep_thresh <= sigma`` against a uniform ``aleph``-bit sample, so
    only ``sigma < keep`` keeps the original), and routes the remaining mass to its
    alternate. The symmetrization step then splits every two-body weight across the
    two orderings ``(mu, nu)`` and ``(nu, mu)``, while the one-body sentinel column
    ``nu = M`` is excluded from the swap and keeps its full weight.
    """
    entries, weights = _build_thc_pairs(M, N, zeta, t_ell)
    probs = np.abs(weights)
    alt, keep = _build_alias_tables(probs, aleph)

    d = len(entries)
    n_levels = 2**aleph
    per_pair = {e: 0.0 for e in entries}
    for i, entry in enumerate(entries):
        keep_prob = keep[i] / n_levels  # comparator "<=" against uniform sample
        per_pair[entry] += (1 / d) * keep_prob
        per_pair[entries[alt[i]]] += (1 / d) * (1 - keep_prob)

    size = 2 ** qp.alias_sampling_thc_wires(M, N, aleph)["mu_wires"]
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


def _run(M, N, zeta, t_ell, aleph, device="lightning.qubit"):  # pylint: disable=too-many-arguments
    """Run SuperpositionTHC and AliasSamplingTHC and return the probability distribution on the
    index wires, reshaped into square shape."""

    zeta, t_ell = _static_coeffs(zeta, t_ell)
    mu_wires, nu_wires, sup_work, edge_flag, work_wires = _wire_layout(M, N, aleph)
    total = max(mu_wires + nu_wires + sup_work + work_wires) + 1
    dev = qp.device(device, wires=total)

    @qp.qnode(dev)
    def circuit():
        qp.SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
        qp.AliasSamplingTHC(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)
        return qp.probs(wires=mu_wires + nu_wires)

    n = len(mu_wires)
    probs = np.asarray(circuit())
    return probs.reshape((2**n, 2**n))


@pytest.mark.parametrize(
    "call",
    [
        lambda: _cswap_pair(0, [], [], []),
        lambda: _symmetrize([], [], 1, 2, []),
    ],
)
def test_register_helpers_are_noops_when_degenerate(call):
    """Test that the register helpers queue nothing on empty / single-wire registers."""
    with qp.queuing.AnnotatedQueue() as q:
        call()
    assert not q.queue


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

        data = _build_qrom_data(M, N, tuple(map(tuple, zeta)), tuple(t_ell), n, aleph)
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
    n = qp.alias_sampling_thc_wires(M, N, aleph=1)["mu_wires"]
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
        return qp.probs(wires=work_wires[: n_d - 1])

    for nu in range(M):
        for mu in range(nu + 1):
            probs = circuit(mu, nu)
            s = int(np.argmax(probs))
            assert s == mu + nu * (nu + 1) // 2


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

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_assert_valid_and_decomposition(self):
        """Operator2 validity and decomposition rules, with and without capture."""
        M, N, aleph = 2, 2, 1
        zeta, t_ell = _static_coeffs(np.ones((M, M)), np.ones(N // 2))
        sizes = qp.alias_sampling_thc_wires(M, N, aleph)
        n = sizes["mu_wires"]
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n + 1, 2 * n + 1 + sizes["work_wires"]))
        op = qp.AliasSamplingTHC(M, N, zeta, t_ell, mu_wires, nu_wires, 2 * n, work_wires, aleph)
        assert_valid(op, skip_differentiation=True)
        assert op.wires == qp.wires.Wires(mu_wires + nu_wires + [2 * n] + work_wires)
        with pytest.raises(ValueError, match="must not overlap"):
            qp.ctrl(op, control=work_wires[-1])

    def test_abstract_wires_keep_hashable_coefficients(self):
        """Test that abstract construction stores compilable coefficients as given."""
        M, N, aleph = 2, 2, 1
        sizes = qp.alias_sampling_thc_wires(M, N, aleph)
        n = sizes["mu_wires"]
        zeta, t_ell = _static_coeffs(np.ones((M, M)), np.ones(N // 2))
        op = qp.AliasSamplingTHC(
            M,
            N,
            zeta,
            t_ell,
            AbstractWires(n),
            AbstractWires(n),
            AbstractWires(1),
            AbstractWires(sizes["work_wires"]),
            aleph,
        )

        assert op.zeta == zeta
        assert op.t_ell == t_ell
        assert qp.equal(op, op)

    def test_abstract_wires_validate_coefficient_shapes(self):
        """Test that abstract construction still reports invalid coefficient shapes."""
        sizes = qp.alias_sampling_thc_wires(2, 2, 1)
        zeta, t_ell = _static_coeffs(np.ones((3, 3)), np.ones(1))
        with pytest.raises(ValueError, match=r"zeta must be of shape \(2, 2\)"):
            qp.AliasSamplingTHC(
                2,
                2,
                zeta,
                t_ell,
                AbstractWires(2),
                AbstractWires(2),
                AbstractWires(1),
                AbstractWires(sizes["work_wires"]),
                1,
            )

    @pytest.mark.catalyst
    def test_qjit_operation_result(self):
        """Test the compiler-specific decomposition branches."""
        M, N, aleph = 2, 2, 1
        zeta, t_ell = _static_coeffs(np.ones((M, M)), np.ones(N // 2))
        mu_wires, nu_wires, sup_work, edge_flag, work_wires = _wire_layout(M, N, aleph)
        total_wires = max(mu_wires + nu_wires + sup_work + work_wires) + 1

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=total_wires))
        def circuit():
            qp.SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
            qp.AliasSamplingTHC(
                M,
                N,
                zeta,
                t_ell,
                mu_wires,
                nu_wires,
                edge_flag,
                work_wires,
                aleph,
            )
            return qp.probs(wires=mu_wires + nu_wires)

        n = len(mu_wires)
        probs = np.asarray(circuit()).reshape((2**n, 2**n))
        expected = _reconstruct_distribution(M, N, zeta, t_ell, aleph)
        assert np.allclose(probs, expected)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(("M", "N", "aleph"), _INSTANCES)
    def test_marginal_matches_reconstruction(self, M, N, aleph, seed):
        """Test that the prepared distribution matches the classical alias reconstruction."""
        np.random.seed(seed)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        probs = _run(M, N, zeta, t_ell, aleph)
        recon = _reconstruct_distribution(M, N, zeta, t_ell, aleph)

        # Test that the probabilities sum to one
        assert np.isclose(probs.sum(), 1.0), "Probabilities are not normalized"
        assert np.allclose(probs, recon, atol=1e-9)

    @pytest.mark.parametrize(("M", "N"), [(2, 2), (3, 2), (5, 2), (8, 4)])
    def test_qrom_uses_minimal_address_space(self, M, N):
        """Test that the QROM is controlled on the minimal number of address wires.

        The contiguous address never exceeds ``d - 1``, so ``ceil(log2(d))`` control
        wires are enough; controlling on the spare high wire of the arithmetic register
        would double the QROM address space and its gate cost.
        """
        aleph = 3
        sizes = qp.alias_sampling_thc_wires(M, N, aleph)
        n = sizes["mu_wires"]
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n + 1, 2 * n + 1 + sizes["work_wires"]))

        zeta, t_ell = _static_coeffs(np.ones((M, M)), np.ones(N // 2))

        op = qp.AliasSamplingTHC(M, N, zeta, t_ell, mu_wires, nu_wires, 2 * n, work_wires, aleph)
        with qp.decomposition.toggle_graph_ctx(True):
            qroms = [g for g in op.decomposition() if isinstance(g, qp.QROM)]
        assert len(qroms) == 1

        d = N // 2 + M * (M + 1) // 2
        assert len(qroms[0].control_wires) == int(np.ceil(np.log2(d)))

    def test_ancillas_returned_to_zero(self, seed):
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

        np.random.seed(seed)
        zeta = np.random.randn(M, M)
        zeta, t_ell = _static_coeffs((zeta + zeta.T) / 2, np.random.randn(N // 2))

        total = max(mu_wires + nu_wires + sup_work + work_wires) + 1
        dev = qp.device("default.qubit", wires=total)

        @qp.qnode(dev)
        def circuit():
            qp.SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
            qp.AliasSamplingTHC(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)
            return qp.probs(wires=ancillas)

        probs = np.asarray(circuit())
        assert np.isclose(probs[0], 1.0, atol=1e-9)


class TestInputValidation:
    """Test the argument checks."""

    def _dummy(self, M, N):
        return _static_coeffs(np.ones((M, M)), np.ones(N // 2))

    @pytest.mark.parametrize(
        ("M", "N", "match"),
        [
            (0, 2, "M must be a positive integer"),
            (2.0, 2, "M must be a positive integer"),
            (2, 0, "N must be a positive integer"),
            (2, True, "N must be a positive integer"),
        ],
    )
    def test_invalid_rank_or_orbitals(self, M, N, match):
        """Test that a non-integer or non-positive M or N raises an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match=match):
            qp.AliasSamplingTHC(M, N, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

    @pytest.mark.parametrize("edge_flag", [[], [4, 5]])
    def test_edge_flag_wrong_size(self, edge_flag):
        """Test that an edge_flag register that does not hold exactly one wire raises."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(
            ValueError,
            match="Incorrect number of wires for 'AliasSamplingTHC.edge_flag'. Expected 1",
        ):
            qp.AliasSamplingTHC(2, 2, zeta, t_ell, [0, 1], [2, 3], edge_flag, list(range(6, 40)), 3)

    def test_mismatched_registers(self):
        """Test that mu_wires and nu_wires of different lengths raise an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="same number of wires"):
            qp.AliasSamplingTHC(2, 2, zeta, t_ell, [0, 1], [2, 3, 4], 5, list(range(6, 40)), 3)

    @pytest.mark.parametrize("n", [2, 5])
    def test_index_register_wrong_size(self, n):
        """Test that index registers not of size exactly ceil(log2(M + 1)) raise an error."""
        # M = 8 needs ceil(log2(9)) = 4 wires per register: 2 is too few, 5 too many.
        zeta, t_ell = self._dummy(8, 2)
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        with pytest.raises(ValueError, match="exactly ceil"):
            qp.AliasSamplingTHC(
                8, 2, zeta, t_ell, mu_wires, nu_wires, 2 * n, list(range(2 * n + 1, 60)), 3
            )

    def test_not_enough_work_wires(self):
        """Test that too few work wires raise an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="At least"):
            qp.AliasSamplingTHC(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, [5, 6, 7], 3)

    @pytest.mark.parametrize("aleph", [0, -1, 2.0, 3.5, True, "3", None])
    def test_invalid_aleph(self, aleph):
        """Test that a non-integer or non-positive aleph raises an error."""
        zeta, t_ell = self._dummy(2, 2)
        with pytest.raises(ValueError, match="aleph must be a positive integer"):
            qp.AliasSamplingTHC(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), aleph)

    @pytest.mark.parametrize(
        ("zeta", "t_ell", "match"),
        [
            (np.ones((2, 2)), (1.0,), "zeta must be a tuple of tuples"),
            (((1.0, 0.0), [0.0, 1.0]), (1.0,), "zeta must be a tuple of tuples"),
            (((1.0, 0.0), (0.0, 1.0)), np.ones(1), "t_ell must be a tuple of floats"),
            (((1.0, 0.0), (0.0, 1.0)), ((1.0,),), "t_ell must be a tuple of floats"),
        ],
    )
    def test_coefficients_must_be_tuples(self, zeta, t_ell, match):
        """Test that array or nested-list coefficients are rejected as compilable data."""
        with pytest.raises(ValueError, match=match):
            qp.AliasSamplingTHC(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

    @pytest.mark.parametrize(
        ("zeta", "t_ell", "match"),
        [
            (((1.0, 1.0, 1.0),) * 3, (1.0,), r"zeta must be of shape \(2, 2\)"),
            (((1.0, 1.0),), (1.0,), r"zeta must be of shape \(2, 2\)"),
            (((1.0, 0.0), (0.0, 1.0)), (), r"t_ell must be of shape \(1,\)"),
            (((1.0, 0.0), (0.0, 1.0)), (1.0, 0.0), r"t_ell must be of shape \(1,\)"),
        ],
    )
    def test_bad_coefficient_shapes(self, zeta, t_ell, match):
        """Test that coefficients indexing out of bounds raise a ValueError, not IndexError."""
        with pytest.raises(ValueError, match=match):
            qp.AliasSamplingTHC(2, 2, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

    def test_bad_n_over_two(self):
        """Test that a value of N // 2 larger than M + 1 raises an error."""
        zeta, t_ell = _static_coeffs(np.ones((2, 2)), np.ones(4))
        with pytest.raises(ValueError, match="N // 2 must be"):
            qp.AliasSamplingTHC(2, 8, zeta, t_ell, [0, 1], [2, 3], 4, list(range(5, 40)), 3)

    def test_odd_spin_orbitals_allowed(self):
        """Test that an odd N is floor-divided, matching ``SuperpositionTHC``: N = 5, M = 1."""
        # ``N // 2 = 2 <= M + 1 = 2``, so the previous ``N / 2 = 2.5 > 2`` check was wrong.
        zeta, t_ell = _static_coeffs(np.ones((1, 1)), np.ones(5 // 2))
        sizes = qp.alias_sampling_thc_wires(1, 5, aleph=3)
        n = sizes["mu_wires"]
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n + 1, 2 * n + 1 + sizes["work_wires"]))
        # Queued without raising; the wire helper agrees with the template's own checks.
        with qp.queuing.AnnotatedQueue():
            qp.AliasSamplingTHC(1, 5, zeta, t_ell, mu_wires, nu_wires, 2 * n, work_wires, 3)


class TestWiresHelper:
    """Test ``alias_sampling_thc_wires``."""

    def test_reported_sizes_are_accepted(self):
        """Test that the reported register sizes satisfy every check in ``AliasSamplingTHC``."""
        M, N, aleph = 5, 2, 4
        sizes = qp.alias_sampling_thc_wires(M, N, aleph)
        n = sizes["mu_wires"]
        assert n == sizes["nu_wires"] == int(np.ceil(np.log2(M + 1)))
        assert sizes["superposition_work_wires"] == 3 * n + 5
        n_d = qp.math.ceil_log2(N // 2 + M * (M + 1) // 2) + 1
        assert sizes["work_wires"] == n_d + 2 * n + 2 * aleph + 4 + max(aleph, n_d - 2)

        zeta, t_ell = _static_coeffs(np.ones((M, M)), np.ones(N // 2))
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n + 1, 2 * n + 1 + sizes["work_wires"]))
        qp.AliasSamplingTHC(M, N, zeta, t_ell, mu_wires, nu_wires, 2 * n, work_wires, aleph)

    def test_minimum_tops_up_qrom_work_wires(self):
        """Test that ``work_wires`` is large enough to accomodate QROM unary iteration if aleph
        is comparably small, so that the comparator scratch space alone is not enough for
        unary iteration. Also tests the opposite, where aleph is large and the QROM scratch space
        is small.
        """
        M, N, aleph = 4, 2, 2
        n = qp.math.ceil_log2(M + 1)
        n_d = qp.math.ceil_log2(N // 2 + M * (M + 1) // 2) + 1
        assert n_d - 2 > aleph  # Comparably small aleph
        num_work = qp.alias_sampling_thc_wires(M, N, aleph)["work_wires"]
        print(qp.alias_sampling_thc_wires(M, N, aleph))
        num_work_other = n_d + 2 * n + 2 * aleph + 4
        assert num_work - num_work_other == n_d - 2  # Work wires suffice for unary iteration

        # Other way around: aleph is large
        aleph = 6
        assert n_d - 2 < aleph  # Comparably large aleph
        num_work = qp.alias_sampling_thc_wires(M, N, aleph)["work_wires"]
        num_work_other = n_d + 2 * n + 2 * aleph + 4
        assert num_work - num_work_other == aleph  # Work wires suffice for unary iteration

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
        """Test that invalid arguments to alias_sampling_thc_wires raise an error."""
        with pytest.raises(ValueError, match=match):
            qp.alias_sampling_thc_wires(M, N, aleph)

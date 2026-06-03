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
Tests for the SuperpositionTHC template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates.superposition_thc import SuperpositionTHC
from pennylane.ops.functions.assert_valid import assert_valid


def _wire_layout(n, work_offset=None):
    """Build disjoint mu / nu / work registers for ``n`` index wires.

    The minimum number of work wires required by ``SuperpositionTHC`` is
    ``3 * n + 5``.
    """
    mu_wires = list(range(0, n))
    nu_wires = list(range(n, 2 * n))
    start = 2 * n if work_offset is None else work_offset
    work_wires = list(range(start, start + 3 * n + 5))
    return mu_wires, nu_wires, work_wires


def _valid_pairs(M, N):
    r"""The valid THC index pairs :math:`\mathcal{S}` flagged by the success qubit.

    Following Lee et al. (2021), Eqs. (27)-(28) of `arXiv:2011.03494
    <https://arxiv.org/abs/2011.03494>`_, the template prepares a *uniform*
    superposition over

    * the upper-triangular two-body pairs ``{(mu, nu): mu <= nu <= M - 1}``
      (0-indexed registers), of size ``M (M + 1) / 2``, and
    * the ``N / 2`` one-body terms ``{(mu, M): mu < N / 2}``, flagged by the
      sentinel column ``nu = M`` (the value ``M + 1`` in the paper's 1-indexed
      convention).

    The total size is the THC coefficient count ``d = N / 2 + M (M + 1) / 2``.

    .. note::
        This assumes the physical THC regime where the rank ``M`` is large
        relative to the number of spatial orbitals ``N / 2`` (specifically
        ``N / 2 <= M + 1``), so that every one-body index ``mu < N / 2`` also
        satisfies the always-enforced ``mu <= nu = M`` constraint.
    """
    two_body = {(mu, nu) for nu in range(M) for mu in range(nu + 1)}
    one_body = {(mu, M) for mu in range(N // 2)}
    return two_body | one_body


def _d(M, N):
    """Number of THC coefficients (Eq. (28) of arXiv:2011.03494)."""
    return N // 2 + M * (M + 1) // 2


def _full_state(M, N, n):
    """Return the full state vector after applying SuperpositionTHC."""
    mu_wires, nu_wires, work_wires = _wire_layout(n)
    dev = qp.device("default.qubit", wires=2 * n + len(work_wires))

    @qp.qnode(dev)
    def circuit():
        SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
        return qp.state()

    return np.asarray(circuit())


@pytest.mark.parametrize(
    ("M", "N", "n"),
    [
        (2, 4, 3),
        (1, 2, 2),
        (3, 4, 4),
    ],
)
def test_standard_validity(M, N, n):
    """Check the operation using the assert_valid function.

    The full check (including the graph-based decomposition path) is run, which
    also exercises ``_superposition_thc_resources``: the resource function must
    report exactly the gate counts produced by the decomposition.
    """
    mu_wires, nu_wires, work_wires = _wire_layout(n)

    gate = SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
    assert_valid(gate)

    assert gate.hyperparameters["M"] == M
    assert gate.hyperparameters["N"] == N
    assert gate.hyperparameters["mu_wires"] == qp.wires.Wires(mu_wires)
    assert gate.hyperparameters["nu_wires"] == qp.wires.Wires(nu_wires)
    assert gate.hyperparameters["work_wires"] == qp.wires.Wires(work_wires)


class TestSuperpositionTHC:
    """Test the SuperpositionTHC template."""

    @pytest.mark.parametrize(
        ("M", "N", "n"),
        [
            (1, 2, 2),
            (2, 4, 3),
            (3, 4, 3),
            (3, 6, 3),
            (4, 4, 3),
            (4, 8, 3),
        ],
    )
    def test_operation_result(self, M, N, n):
        """The template prepares a uniform superposition over the valid index set
        ``S = {(mu, nu): mu <= nu <= M - 1} U {(mu, M): mu < N / 2}`` conditioned on
        the output success flag (``work_wires[6] == 1``).

        We let ``qml.probs`` marginalize out every other wire for us: it returns
        ``P(mu, nu, flag)`` over the ``2 ** (2n + 1)`` basis states of the
        ``mu_wires + nu_wires + [success_flag]`` register, ordered big-endian
        (mu is the most significant block, the flag is the least significant bit).
        """
        mu_wires, nu_wires, work_wires = _wire_layout(n)
        success_flag = work_wires[6]
        dev = qp.device("default.qubit", wires=2 * n + len(work_wires))

        @qp.qnode(dev)
        def circuit():
            SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
            return qp.probs(wires=mu_wires + nu_wires + [success_flag])

        probs = circuit()

        # Keep only the "success" half (flag == 1, the odd-indexed entries) and
        # decode the remaining index into (mu, nu).
        support = {}
        for index, prob in enumerate(probs):
            if (index & 1) and prob > 1e-9:
                mu, nu = divmod(index >> 1, 2**n)
                support[(mu, nu)] = float(prob)

        assert set(support) == _valid_pairs(M, N)
        assert len(support) == _d(M, N)

        # The flagged amplitudes must be uniform.
        weights = np.array(list(support.values()))
        assert np.allclose(weights, weights[0])

    @pytest.mark.parametrize(
        ("M", "N", "n"),
        [
            (4, 4, 3),
            (4, 8, 3),
            (3, 6, 3),
        ],
    )
    def test_one_body_block(self, M, N, n):
        """The one-body sector is flagged by the sentinel column ``nu = M`` on the
        ``work_wires[3]`` flag and contains exactly ``N / 2`` terms ``{(mu, M):
        mu < N / 2}``.

        This is the part of the valid set that depends on ``N`` (Eq. (28) of
        arXiv:2011.03494). We read ``P(mu, nu, w3, w6)`` and keep the success
        subspace (``w6 == 1``).
        """
        mu_wires, nu_wires, work_wires = _wire_layout(n)
        w3, w6 = work_wires[3], work_wires[6]
        dev = qp.device("default.qubit", wires=2 * n + len(work_wires))

        @qp.qnode(dev)
        def circuit():
            SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
            return qp.probs(wires=mu_wires + nu_wires + [w3, w6])

        probs = circuit()

        one_body = set()
        two_body = set()
        for index, prob in enumerate(probs):
            if prob <= 1e-9:
                continue
            w6_bit = index & 1
            w3_bit = (index >> 1) & 1
            mu, nu = divmod(index >> 2, 2**n)
            if w6_bit:  # success subspace only
                (one_body if w3_bit else two_body).add((mu, nu))

        # The one-body block carries exactly N/2 terms, all in the sentinel column.
        assert one_body == {(mu, M) for mu in range(N // 2)}
        assert len(one_body) == N // 2
        assert all(nu == M for _, nu in one_body)

        # The two-body block is the N-independent upper triangle mu <= nu <= M-1.
        assert two_body == {(mu, nu) for nu in range(M) for mu in range(nu + 1)}
        assert len(two_body) == M * (M + 1) // 2

    @pytest.mark.parametrize(("M", "n"), [(4, 3), (3, 3)])
    def test_n_dependence(self, M, n):
        """The size of the prepared superposition grows with ``N / 2``: the valid
        set is ``d = N / 2 + M (M + 1) / 2``. This guards against the ``mu > N / 2``
        flag becoming dead code (it must select the one-body terms)."""
        mu_wires, nu_wires, work_wires = _wire_layout(n)
        success_flag = work_wires[6]
        dev = qp.device("default.qubit", wires=2 * n + len(work_wires))

        def support_size(N):
            @qp.qnode(dev)
            def circuit():
                SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
                return qp.probs(wires=mu_wires + nu_wires + [success_flag])

            probs = circuit()
            return sum(1 for i, p in enumerate(probs) if (i & 1) and p > 1e-9)

        # N//2 must stay < 2**n for the classical comparator constant.
        sizes = {N: support_size(N) for N in (2, 4, 6)}
        for N, size in sizes.items():
            assert size == _d(M, N)
        # Strictly increasing with N (one extra one-body term per 2 spin orbitals).
        assert sizes[2] < sizes[4] < sizes[6]

    @pytest.mark.parametrize(
        ("M", "N", "n"),
        [
            (1, 2, 2),
            (2, 4, 3),
            (4, 8, 3),
        ],
    )
    def test_work_wires_clean(self, M, N, n):
        """All work wires are returned to |0> except the output flag wires
        ``work_wires[0]`` (the amplitude-amplification ancilla), ``work_wires[3]``
        (the one-body sentinel) and ``work_wires[6]`` (success), which carry the
        prepared flags.

        For each work wire we ask ``qml.probs`` for its single-qubit marginal
        ``[P(0), P(1)]`` and check whether it still has any weight on |1>.
        """
        mu_wires, nu_wires, work_wires = _wire_layout(n)
        dev = qp.device("default.qubit", wires=2 * n + len(work_wires))

        @qp.qnode(dev)
        def circuit():
            SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
            return [qp.probs(wires=[w]) for w in work_wires]

        marginals = circuit()

        flag_indices = {0, 3, 6}
        for fw, marginal in enumerate(marginals):
            prob_one = float(marginal[1])
            if fw in flag_indices:
                continue  # may or may not carry weight depending on (M, N)
            assert np.isclose(prob_one, 0.0), f"work_wires[{fw}] not returned to zero"

    @pytest.mark.parametrize(
        ("M", "N", "n"),
        [
            (1, 2, 2),
            (2, 4, 3),
            (4, 8, 3),
        ],
    )
    def test_no_phase_errors(self, M, N, n):
        """The template introduces no complex phases: the full prepared state
        vector is real-valued."""
        state = _full_state(M, N, n)
        assert np.allclose(state.imag, 0.0), "Phase error: imaginary components detected"

    @pytest.mark.parametrize(
        ("M", "N", "mu_wires", "nu_wires", "work_wires", "msg_match"),
        [
            (
                2,
                4,
                [0, 1, 2],
                [3, 4],
                list(range(5, 5 + 14)),
                "mu_wires and nu_wires must contain the same number of wires",
            ),
            (
                2,
                4,
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8, 9, 10],
                r"At least 14 work_wires \(3 \* len\(mu_wires\) \+ 5\) should be provided",
            ),
            (
                2,
                4,
                [0, 1, 2],
                [0, 4, 5],
                list(range(6, 6 + 14)),
                r"mu_wires and nu_wires must be disjoint, but share: \[0\]",
            ),
            (
                2,
                4,
                [0, 1, 2],
                [3, 4, 5],
                [0] + list(range(6, 6 + 13)),
                r"work_wires and mu_wires must be disjoint, but share: \[0\]",
            ),
            (
                2,
                4,
                [0, 1, 2],
                [3, 4, 5],
                [3] + list(range(6, 6 + 13)),
                r"work_wires and nu_wires must be disjoint, but share: \[3\]",
            ),
        ],
    )
    def test_wires_error(self, M, N, mu_wires, nu_wires, work_wires, msg_match):
        """An error is raised when the registers do not meet the requirements."""
        with pytest.raises(ValueError, match=msg_match):
            SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)

    def test_flatten_unflatten(self):
        """The operator round-trips through _flatten / _unflatten."""
        M, N, n = 2, 4, 3
        mu_wires, nu_wires, work_wires = _wire_layout(n)
        op = SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)

        data, metadata = op._flatten()
        new_op = type(op)._unflatten(data, metadata)

        assert op.hyperparameters == new_op.hyperparameters
        assert op.wires == new_op.wires
        assert qp.equal(op, new_op)

    def test_map_wires(self):
        """map_wires relabels every register consistently."""
        M, N, n = 2, 4, 3
        mu_wires, nu_wires, work_wires = _wire_layout(n)
        op = SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)

        wire_map = {w: w + 100 for w in op.wires}
        mapped = op.map_wires(wire_map)

        assert mapped.hyperparameters["mu_wires"] == qp.wires.Wires(
            [w + 100 for w in mu_wires]
        )
        assert mapped.hyperparameters["nu_wires"] == qp.wires.Wires(
            [w + 100 for w in nu_wires]
        )
        assert mapped.hyperparameters["work_wires"] == qp.wires.Wires(
            [w + 100 for w in work_wires]
        )
        assert mapped.hyperparameters["M"] == M
        assert mapped.hyperparameters["N"] == N

    def test_decomposition_queue(self):
        """compute_decomposition returns a non-empty list of operators."""
        M, N, n = 2, 4, 3
        mu_wires, nu_wires, work_wires = _wire_layout(n)
        op = SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)

        decomp = op.decomposition()
        assert isinstance(decomp, list)
        assert len(decomp) > 0
        assert all(isinstance(o, qp.operation.Operator) for o in decomp)

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

    Following Lee et al. (2021) `arXiv:2011.03494
    <https://arxiv.org/abs/2011.03494>`_, the template prepares a *uniform*
    superposition over

    * the upper-triangular two-body pairs ``{(mu, nu): mu <= nu < M}``
      (0-indexed registers), of size ``M (M + 1) / 2``, and
    * the ``N / 2`` one-body terms ``{(mu, M): mu < N / 2}``, flagged by the
      sentinel column ``nu = M`` (the value ``M`` in the paper's 1-indexed
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
        (7, 3, 3),
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

    @pytest.mark.external
    @pytest.mark.parametrize("qjit", [False, True])
    @pytest.mark.parametrize(
        ("M", "N", "n"),
        [
            (1, 2, 2),
            (2, 4, 3),
            (3, 4, 3),
            (3, 6, 3),
            (4, 4, 3),
            (4, 8, 3),
            (7, 4, 3),
        ],
    )
    def test_operation_result(self, M, N, n, qjit):
        """The template prepares a uniform superposition over the valid index set
        ``S = {(mu, nu): mu <= nu < M} U {(mu, M): mu < N / 2}`` conditioned on
        the output success flag (``work_wires[6] == 1``).
        """

        mu_wires, nu_wires, work_wires = _wire_layout(n)
        success_flag = work_wires[6]
        dev = qp.device("lightning.qubit", wires=2 * n + len(work_wires))

        @qp.qnode(dev)
        def circuit():
            SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
            return qp.probs(wires=mu_wires + nu_wires + [success_flag])

        if qjit:
            circuit = qp.qjit(circuit)

        # probs[mu, nu, flag]: flag == 1 is the "success" subspace.
        probs = np.asarray(circuit()).reshape((2**n, 2**n, 2))
        success = probs[:, :, 1]

        support = {
            (mu, nu): float(success[mu, nu])
            for mu in range(2**n)
            for nu in range(2**n)
            if success[mu, nu] > 1e-9
        }

        assert set(support) == _valid_pairs(M, N)

        d = N // 2 + M * (M + 1) // 2
        assert len(support) == d

        # The flagged amplitudes must be uniform.
        weights = np.array(list(support.values()))
        assert np.allclose(weights, weights[0])

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
        ``work_wires[0]`` (the amplitude-amplification auxiliary wire), ``work_wires[3]``
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

        probs = circuit()

        flag_indices = {0, 3, 6}
        for i, prob in enumerate(probs):
            prob_one = float(prob[1])
            if i in flag_indices:
                continue
            assert np.isclose(prob_one, 0.0), f"work_wires[{i}] not returned to zero"

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
        assert np.allclose(state, np.abs(state)), "Phase error: negative components detected"

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
            (
                8,
                4,
                [0, 1, 2],
                [3, 4, 5],
                list(range(6, 6 + 14)),
                r"each need at least ceil\(log2\(M \+ 1\)\) wires",
            ),
        ],
    )
    def test_wires_error(
        self, M, N, mu_wires, nu_wires, work_wires, msg_match
    ):  # pylint:disable=too-many-arguments
        """An error is raised when the registers do not meet the requirements."""
        with pytest.raises(ValueError, match=msg_match):
            SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_max_rank_boundary(self, n):
        """``M = 2 ** n - 1`` is the largest rank the ``n``-wire registers can
        hold; it must construct successfully and prepare exactly the valid set.
        """
        M = 2**n - 1
        N = 2
        mu_wires, nu_wires, work_wires = _wire_layout(n)
        success_flag = work_wires[6]
        dev = qp.device("lightning.qubit", wires=2 * n + len(work_wires))

        @qp.qnode(dev)
        def circuit():
            SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
            return qp.probs(wires=mu_wires + nu_wires + [success_flag])

        probs = np.asarray(circuit()).reshape((2**n, 2**n, 2))
        success = probs[:, :, 1]
        support = {(mu, nu) for mu in range(2**n) for nu in range(2**n) if success[mu, nu] > 1e-9}
        assert support == _valid_pairs(M, N)

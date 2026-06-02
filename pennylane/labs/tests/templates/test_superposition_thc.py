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


def _success_pairs(M):
    """The valid (mu, nu) index pairs flagged by the success qubit.

    Conditioned on the output success flag (``work_wires[6] == 1``), the template
    prepares a *uniform* superposition over the upper-triangular index pairs
    ``{(mu, nu) : mu <= nu <= M}``, whose size is ``(M + 1)(M + 2) / 2``.
    """
    return {(mu, nu) for nu in range(M + 1) for mu in range(nu + 1)}


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

    The full check (including the new graph-based decomposition path) is run, which
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
        ("qjit"),
        [True, False],
    )
    @pytest.mark.parametrize(
        ("M", "N", "n"),
        [
            (0, 2, 2),
            (1, 2, 2),
            (2, 4, 3),
            (3, 4, 3),
            (2, 8, 3),
        ],
    )
    def test_operation_result(self, qjit, M, N, n):
        """The template prepares a uniform superposition over {(mu, nu): mu <= nu <= M}
        conditioned on the output success flag (work_wires[6] == 1).

        We let ``qml.probs`` marginalize out every other wire for us: it returns
        ``P(mu, nu, flag)`` over the ``2 ** (2n + 1)`` basis states of the
        ``mu_wires + nu_wires + [success_flag]`` register, ordered big-endian
        (mu is the most significant block, the flag is the least significant bit).
        """
        mu_wires, nu_wires, work_wires = _wire_layout(n)
        success_flag = work_wires[6]
        dev = qp.device("lightning.qubit", wires=2 * n + len(work_wires))

        @qp.qnode(dev)
        def circuit(M, N, mu_wires, nu_wires, work_wires):
            SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
            return qp.probs(wires=mu_wires + nu_wires + [success_flag])

        if qjit:
            probs = qp.qjit(circuit)(M, N, mu_wires, nu_wires, work_wires)
        else:
            probs = circuit(M, N, mu_wires, nu_wires, work_wires )

        # Keep only the "success" half (flag == 1, the odd-indexed entries) and
        # decode the remaining index into (mu, nu).
        support = {}
        for index, prob in enumerate(probs):
            if (index & 1) and prob > 1e-9:
                mu, nu = divmod(index >> 1, 2**n)
                support[(mu, nu)] = float(prob)

        assert set(support) == _success_pairs(M)
        assert len(support) == (M + 1) * (M + 2) // 2

        # The flagged amplitudes must be uniform.
        weights = np.array(list(support.values()))
        assert np.allclose(weights, weights[0])

    @pytest.mark.parametrize(
        ("M", "N", "n"),
        [
            (1, 2, 2),
            (2, 4, 3),
        ],
    )
    def test_work_wires_clean(self, M, N, n):
        """All work wires are returned to |0> except the two flag wires
        (work_wires[3] and work_wires[6]), which carry the prepared flags.

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

        flag_indices = {3, 6}
        for fw, marginal in enumerate(marginals):
            prob_one = float(marginal[1])
            if fw in flag_indices:
                assert prob_one > 1e-9, f"flag work_wires[{fw}] unexpectedly clean"
            else:
                assert np.isclose(prob_one, 0.0), f"work_wires[{fw}] not returned to zero"

    @pytest.mark.parametrize(
        ("M", "N", "n"),
        [
            (1, 2, 2),
            (2, 4, 3),
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

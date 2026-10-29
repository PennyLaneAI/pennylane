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
"""Tests for LeftClassicalComparator."""

from collections import Counter

import numpy as np
import pytest

import pennylane as qp
from pennylane.core import abstractify
from pennylane.templates.subroutines.arithmetic.left_classical_comparator import (
    _left_classical_comparator,
)
from pennylane.testing import assert_valid, decomp_rule_to_tape
from pennylane.typing import Int, Wire


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_assert_valid():
    """Test that LeftClassicalComparator passes the standard Operator2 checks,
    with capture enabled and disabled."""
    op = qp.LeftClassicalComparator(
        x_wires=[0, 1, 2], L=2, target_wire=3, work_wires=[4, 5], comparator=">="
    )
    assert_valid(op, skip_differentiation=True)

    assert op.arguments["x_wires"] == qp.wires.Wires([0, 1, 2])
    assert op.arguments["L"] == 2
    assert op.arguments["target_wire"] == qp.wires.Wires(3)
    assert op.arguments["work_wires"] == qp.wires.Wires([4, 5])
    assert op.arguments["comparator"] == ">="


class TestLeftClassicalComparator:
    """Test LeftClassicalComparator template."""

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        "qjit",
        [
            pytest.param(
                True,
                marks=pytest.mark.xfail(reason="Requires Catalyst companion PR [sc-134032]"),
            ),
            False,
        ],
    )
    @pytest.mark.parametrize("comparator", ["<", "<=", ">", ">="])
    @pytest.mark.parametrize(
        ("x_wires", "L", "target_wire", "work_wires", "x"),
        [
            ([0, 3, 6, 9], 1, 11, [2, 5, 8], 1),
            ([0, 3, 6, 9], 1, 11, [2, 5, 8], 2),
            ([0, 3, 6, 9], 2, 11, [2, 5, 8], 1),
            ([0, 3, 6], 5, 11, [2, 5], 2),
        ],
    )
    def test_operation_result(
        self, x_wires, L, target_wire, work_wires, x, comparator, qjit
    ):  # pylint: disable=too-many-arguments
        """Test that the LeftClassicalComparator template produces the correct output."""

        @qp.qnode(qp.device("lightning.qubit", wires=range(13)), shots=1)
        def circuit(x_wires, L):
            qp.BasisState(qp.math.int_to_binary(x, len(x_wires)), wires=x_wires)
            qp.LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator)
            qp.CNOT([11, 12])
            qp.adjoint(
                lambda: qp.LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator)
            )()
            qp.BasisState(qp.math.int_to_binary(x, len(x_wires)), wires=x_wires)
            return qp.sample(wires=[12]), qp.sample(wires=work_wires), qp.sample(wires=x_wires)

        if qjit:
            circuit = qp.qjit(circuit)
        output = circuit(x_wires, L)
        expected = {"<": x < L, "<=": x <= L, ">": x > L, ">=": x >= L}[comparator]
        assert bool(output[0]) == expected
        assert np.isclose(sum(*output[1]), 0)  # work wires are clean
        assert np.isclose(sum(*output[2]), 0)  # x_wires are not modified

    @pytest.mark.parametrize(
        ("target_wire", "x_wires", "L", "work_wires", "comparator", "msg_match"),
        [
            (
                8,
                [0, 1, 2],
                1,
                [1],
                "<",
                "At least 2 work_wires are required, but only 1 were provided",
            ),
            (
                6,
                [0, 1, 2],
                1,
                [6, 7],
                "<=",
                r"work_wires and target_wire must be disjoint, but share: \[6\]",
            ),
            (
                8,
                [0, 1, 2],
                1,
                [1, 6],
                ">=",
                r"work_wires and x_wires must be disjoint, but share: \[1\]",
            ),
            (
                1,
                [0, 1, 2],
                1,
                [6, 7],
                "<=",
                r"x_wires and target_wire must be disjoint, but share: \[1\]",
            ),
            (
                8,
                [0, 1, 2],
                10,
                [6, 7],
                "<",
                "L must be less than",
            ),
        ],
    )
    def test_wires_error(
        self, target_wire, x_wires, L, work_wires, comparator, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test that an error is raised when some work_wires don't meet the requirements."""
        with pytest.raises(ValueError, match=msg_match):
            qp.LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator=comparator)

    @pytest.mark.parametrize("comparator", ["<", "<=", ">", ">="])
    @pytest.mark.parametrize(
        ("x_wires", "L", "target_wire", "work_wires"),
        [
            ([0, 3, 6, 9], 1, 11, [2, 5, 8]),
            ([0, 3, 6, 9], 2, 11, [2, 5, 8]),
            ([0, 3, 6], 5, 11, [2, 5]),
        ],
    )
    @pytest.mark.parametrize("seed", [42, 123])
    def test_no_phase_errors(  # pylint: disable=too-many-arguments
        self, x_wires, L, target_wire, work_wires, comparator, seed
    ):
        """Test that the comparator introduces no complex phases.
        A correct classical reversible circuit is a real permutation matrix,
        so a real positive input must produce a real positive output."""

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(x_state):
            qp.StatePrep(x_state, x_wires)
            qp.LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator)
            return qp.state()

        num_x = 2 ** len(x_wires)
        rng = np.random.default_rng(seed)

        # Real positive superposition: all components strictly > 0
        x_state = rng.random(num_x)
        x_state /= np.linalg.norm(x_state)

        state = np.asarray(circuit(x_state))

        # All amplitudes must be real (no imaginary component)
        assert np.allclose(state.imag, 0.0), "Phase error: imaginary components detected"
        # All non-zero amplitudes must be non-negative
        assert np.all(state.real >= -1e-10), "Phase error: negative amplitudes detected"

    @pytest.mark.parametrize("comparator", ["<", "<=", ">", ">="])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_max_bound_all_inputs(self, comparator, n):
        """Test that the comparator is correct for the largest allowed bound ``L = 2 ** n - 1``."""

        x_wires = list(range(n))
        target_wire = n
        work_wires = list(range(n + 1, n + 1 + max(1, n - 1)))
        total = n + 1 + max(1, n - 1)
        L = 2**n - 1

        dev = qp.device("lightning.qubit", wires=total)

        @qp.qnode(dev)
        def circuit(x):
            qp.BasisState(qp.math.int_to_binary(x, len(x_wires)), wires=x_wires)
            qp.LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator)
            return qp.probs(wires=[target_wire])

        expected_fn = {
            "<": lambda x: x < L,
            "<=": lambda x: x <= L,
            ">": lambda x: x > L,
            ">=": lambda x: x >= L,
        }[comparator]

        for x in range(2**n):
            probs = np.asarray(circuit(x))
            got = np.isclose(probs[1], 1.0)
            assert got == expected_fn(x), (
                f"comparator={comparator} L={L} x={x}: got {got}, "
                f"expected {bool(expected_fn(x))}"
            )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("comparator", ["<", "<=", ">", ">="])
def test_resource_upper_bound(n, comparator):
    """Worst-case estimates must cover each gate count for every valid threshold, including endpoints."""
    rule = _left_classical_comparator
    abstract_op = qp.LeftClassicalComparator(Wire[n], Int, Wire[1], Wire[n - 1], comparator)
    estimates = rule.compute_resources(**abstract_op.arguments).gate_counts
    assert not rule.exact_resources
    max_L = 2 ** n if comparator in ["<", ">="] else 2**n - 1
    for threshold in range(max_L + 1):
        op = qp.LeftClassicalComparator(
            list(range(n)), threshold, n, list(range(n + 1, 2 * n)), comparator
        )
        tape = decomp_rule_to_tape(op, rule)
        counts = Counter(abstractify(gate) for gate in tape.operations)
        assert all(count <= estimates.get(gate, 0) for gate, count in counts.items())


@pytest.mark.catalyst
@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("comparator", ["<", "<=", ">", ">="])
def test_capture_dynamic_threshold(comparator):
    """A captured circuit returns the correct comparison result as the runtime threshold changes."""
    target = qp.gate_sets.CLIFFORD_T_PLUS_RZ

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=target)
    @qp.qnode(qp.device("lightning.qubit", wires=4))
    def circuit(threshold):
        qp.X(1)  # x = 1
        qp.LeftClassicalComparator([0, 1], threshold, 2, [3], comparator)
        return qp.probs(wires=[2])

    max_L = 4 if comparator in ["<", ">="] else 3
    for threshold in range(max_L + 1):
        expected_bit = {
            "<": 1 < threshold,
            "<=": 1 <= threshold,
            ">": 1 > threshold,
            ">=": 1 >= threshold,
        }[comparator]
        expected = np.zeros(2)
        expected[int(expected_bit)] = 1
        assert qp.math.allclose(circuit(threshold), expected, atol=1e-8)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("threshold", [[1, 2], np.array([1]), Int[2]])
def test_non_scalar_threshold_rejected(threshold):
    """Reject concrete and abstract non-scalar thresholds with capture enabled or disabled."""
    with pytest.raises(ValueError, match="L must be a scalar integer"):
        qp.LeftClassicalComparator([0, 1], threshold, 2, [3], "<")

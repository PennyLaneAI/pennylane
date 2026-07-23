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
Tests for the LeftClassicalComparator template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates.left_classical_comparator import LeftClassicalComparator
from pennylane.ops.functions.assert_valid import assert_valid


def test_standard_validity_left_comparator():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2]
    L = 2
    work_wires = [6, 7]
    target_wire = 8
    comparator = ">="

    gate = LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator=comparator)
    assert_valid(gate)

    assert gate.hyperparameters["target_wire"] == qp.wires.Wires(8)
    assert gate.hyperparameters["x_wires"] == qp.wires.Wires([0, 1, 2])
    assert gate.hyperparameters["L"] == L
    assert gate.hyperparameters["work_wires"] == qp.wires.Wires([6, 7])
    assert gate.hyperparameters["comparator"] == ">="


class TestLeftClassicalComparator:
    """Test LeftClassicalComparator template."""

    @pytest.mark.external
    @pytest.mark.parametrize("qjit", [True, False])
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
        """Test the correctness of the LeftClassicalComparator template output."""

        @qp.qnode(qp.device("lightning.qubit", wires=range(13)), shots=1)
        def circuit(x_wires, L):
            qp.BasisState(x, wires=x_wires)
            LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator)
            qp.CNOT([11, 12])
            qp.adjoint(
                lambda: LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator)
            )()
            qp.BasisState(x, wires=x_wires)
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
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qp.labs.templates.LeftClassicalComparator(
                x_wires, L, target_wire, work_wires, comparator=comparator
            )

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
        """Verify the comparator introduces no complex phases.
        A correct classical reversible circuit is a real permutation matrix,
        so a real positive input must produce a real positive output."""

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(x_state):
            qp.StatePrep(x_state, x_wires)
            LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator)
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
        """Test for the largest allowed bound ``L = 2 ** n - 1``."""

        x_wires = list(range(n))
        target_wire = n
        work_wires = list(range(n + 1, n + 1 + max(1, n - 1)))
        total = n + 1 + max(1, n - 1)
        L = 2**n - 1

        dev = qp.device("lightning.qubit", wires=total)

        @qp.qnode(dev)
        def circuit(x):
            qp.BasisState(x, wires=x_wires)
            LeftClassicalComparator(x_wires, L, target_wire, work_wires, comparator)
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

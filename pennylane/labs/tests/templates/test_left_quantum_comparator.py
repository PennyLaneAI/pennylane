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
Tests for the LeftQuantumComparator template.
"""

import pytest

import pennylane as qp
from pennylane.labs.templates.left_quantum_comparator import LeftQuantumComparator
from pennylane.ops.functions.assert_valid import assert_valid


def test_standard_validity_left_comparator():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2]
    y_wires = [3, 4, 5]
    work_wires = [6, 7]
    target_wire = 8
    comparator = ">="

    gate = LeftQuantumComparator(x_wires, y_wires, target_wire, work_wires, comparator=comparator)
    assert_valid(gate)

    assert gate.hyperparameters["target_wire"] == qp.wires.Wires(8)
    assert gate.hyperparameters["x_wires"] == qp.wires.Wires([0, 1, 2])
    assert gate.hyperparameters["y_wires"] == qp.wires.Wires([3, 4, 5])
    assert gate.hyperparameters["work_wires"] == qp.wires.Wires([6, 7])
    assert gate.hyperparameters["comparator"] == ">="


class TestLeftQuantumComparator:
    """Test LeftQuantumComparator template."""

    @pytest.mark.catalyst
    @pytest.mark.external
    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "target_wire", "work_wires", "x", "y", "comparator", "expected"),
        [
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 1, 1, "<", False),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 1, 1, "<=", True),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 1, 1, ">=", True),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 1, 1, ">", False),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 2, 1, "<", False),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 2, 1, "<=", False),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 2, 1, ">=", True),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 2, 1, ">", True),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 1, 2, "<", True),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 1, 2, "<=", True),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 1, 2, ">=", False),
            ([0, 3, 6, 9], [1, 4, 7, 10], 11, [2, 5, 8], 1, 2, ">", False),
            ([0, 3, 6], [1, 4, 7], 11, [2, 5], 2, 5, "<", True),
            ([0, 3, 6], [1, 4, 7], 11, [2, 5], 2, 5, "<=", True),
            ([0, 3, 6], [1, 4, 7], 11, [2, 5], 2, 5, ">=", False),
            ([0, 3, 6], [1, 4, 7], 11, [2, 5], 2, 5, ">", False),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, target_wire, work_wires, x, y, comparator, expected
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the LeftComparator template output."""

        pytest.importorskip("catalyst")

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=range(13)), shots=1)
        def circuit():
            qp.BasisState(x, wires=x_wires)
            qp.BasisState(y, wires=y_wires)
            LeftQuantumComparator(x_wires, y_wires, target_wire, work_wires, comparator)
            qp.CNOT([11, 12])
            qp.adjoint(
                lambda: LeftQuantumComparator(x_wires, y_wires, target_wire, work_wires, comparator)
            )()
            return qp.sample(wires=[12])

        assert bool(circuit()) == expected

    @pytest.mark.parametrize(
        ("target_wire", "x_wires", "y_wires", "work_wires", "comparator", "msg_match"),
        [
            (
                8,
                [0, 1, 2],
                [3, 4, 5],
                [1],
                "<",
                "At least 2 work_wires should be provided.",
            ),
            (
                6,
                [0, 1, 2],
                [3, 4, 5],
                [6, 7],
                "<=",
                "None of the wires in work_wires should be the target wire.",
            ),
            (
                8,
                [0, 1, 2],
                [3, 4, 5],
                [1, 6],
                ">=",
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                8,
                [0, 1, 2],
                [3, 4, 5],
                [3, 6],
                ">",
                "None of the wires in work_wires should be included in y_wires.",
            ),
            (
                1,
                [0, 1, 2],
                [3, 4, 5],
                [6, 7],
                "<=",
                "None of the wires in x_wires should be the target wire.",
            ),
            (
                9,
                [0, 1, 2],
                [2, 3, 4],
                [6, 7, 8],
                ">=",
                "None of the wires in y_wires should be included in x_wires.",
            ),
            (
                5,
                [0, 1, 2],
                [3, 4, 5],
                [6, 7],
                "<=",
                "None of the wires in y_wires should be the target wire.",
            ),
            (
                8,
                [0, 1, 2],
                [3, 4, 5],
                [6, 7],
                "=",
                "Allowed values for 'comparator' are:",
            ),
            (
                8,
                [0, 2],
                [3, 4, 5],
                [6, 7],
                "<",
                "The number of y_wires should be equal to the number of x_wires",
            ),
        ],
    )
    def test_wires_error(
        self, target_wire, x_wires, y_wires, work_wires, comparator, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qp.labs.templates.LeftQuantumComparator(
                x_wires, y_wires, target_wire, work_wires, comparator=comparator
            )

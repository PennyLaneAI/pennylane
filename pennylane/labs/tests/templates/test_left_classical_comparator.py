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

    @pytest.mark.catalyst
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

        pytest.importorskip("catalyst")

        @qp.qnode(qp.device("lightning.qubit", wires=range(13)), shots=1)
        def circuit():
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
        output = circuit()
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
                "At least 2 work_wires should be provided.",
            ),
            (
                6,
                [0, 1, 2],
                1,
                [6, 7],
                "<=",
                "None of the wires in work_wires should be the target wire.",
            ),
            (
                8,
                [0, 1, 2],
                1,
                [1, 6],
                ">=",
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                1,
                [0, 1, 2],
                1,
                [6, 7],
                "<=",
                "None of the wires in x_wires should be the target wire.",
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

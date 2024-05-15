# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the QROM template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np


def test_standard_checks():
    """Run standard validity tests."""
    bitstrings = ["000", "001", "111", "011", "000", "101", "110", "111"]

    op = qml.QROM(bitstrings, control_wires=[0, 1, 2], target_wires=[3, 4, 5], work_wires=[6, 7, 8])
    qml.ops.functions.assert_valid(op)


class TestQROM:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("b", "target_wires", "control_wires", "work_wires", "clean"),
        [
            (
                ["11", "01", "00", "10"],
                [0, 1],
                [2, 3],
                [4, 5],
                True,
            ),
            (
                ["01", "01", "00", "00"],
                ["a", "b"],
                [2, 3],
                [4, 5, 6],
                False,
            ),
            (
                ["111", "001", "000", "100"],
                [0, 1, "b"],
                [2, 3],
                ["a", 5, 6],
                False,
            ),
            (
                ["1111", "0101", "0100", "1010"],
                [0, 1, "b", "d"],
                [2, 3],
                ["a", 5, 6, 7],
                True,
            ),
        ],
    )
    def test_operation_result(
        self, b, target_wires, control_wires, work_wires, clean
    ):  # pylint: disable=arguments-differ
        """Test the correctness of the Select template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(j):
            qml.BasisEmbedding(j, wires=control_wires)

            qml.QROM(b, target_wires, control_wires, work_wires, clean)
            return qml.sample(wires=target_wires)

        @qml.qnode(dev)
        def circuit_test(j):
            for ind, bit in enumerate(b[j]):
                if bit == "1":
                    qml.PauliX(wires=target_wires[ind])
            return qml.sample(wires=target_wires)

        for j in range(2 ** len(control_wires)):
            assert np.allclose(circuit(j), circuit_test(j))

    @pytest.mark.parametrize(
        ("b", "target_wires", "control_wires", "work_wires"),
        [
            (
                ["11", "01", "00", "10"],
                [0, 1],
                [2, 3],
                [4, 5],
            ),
            (
                ["01", "01", "00", "00"],
                ["a", "b"],
                [2, 3],
                [4, 5, 6],
            ),
            (
                ["111", "001", "000", "100"],
                [0, 1, "b"],
                [2, 3],
                ["a", 5, 6],
            ),
            (
                ["1111", "0101", "0100", "1010"],
                [0, 1, "b", "d"],
                [2, 3],
                None,
            ),
        ],
    )
    def test_work_wires_output(self, b, target_wires, control_wires, work_wires):
        """Tests that the ``clean = True`` version don't modify the initial state in work_wires."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit():

            # Initialize the work wires to a non-zero state
            for ind, wire in enumerate(work_wires):
                qml.RX(ind, wires=wire)

            for wire in control_wires:
                qml.Hadamard(wires=wire)

            qml.QROM(b, target_wires, control_wires, work_wires)

            for ind, wire in enumerate(work_wires):
                qml.RX(-ind, wires=wire)

            return qml.probs(wires=work_wires)

        assert np.isclose(circuit()[0], 1.0)

    def test_decomposition(self):
        """Unit test checking that compute_decomposition and decomposition work as expected."""
        qrom_decomposition = qml.QROM(
            ["1", "0", "0", "1"], control_wires=[0, 1], target_wires=[2], work_wires=[3], clean=True
        ).decomposition()

        expected_gates = [
            qml.Hadamard(wires=[2]),
            qml.CSWAP(wires=[1, 2, 3]),
            qml.Select(
                ops=(
                    qml.BasisEmbedding(1, wires=[2]) @ qml.BasisEmbedding(0, wires=[3]),
                    qml.BasisEmbedding(0, wires=[2]) @ qml.BasisEmbedding(1, wires=[3]),
                ),
                control=[0],
            ),
            qml.CSWAP(wires=[1, 2, 3]),
            qml.Hadamard(wires=[2]),
            qml.CSWAP(wires=[1, 2, 3]),
            qml.Select(
                ops=(
                    qml.BasisEmbedding(1, wires=[2]) @ qml.BasisEmbedding(0, wires=[3]),
                    qml.BasisEmbedding(0, wires=[2]) @ qml.BasisEmbedding(1, wires=[3]),
                ),
                control=0,
            ),
            qml.CSWAP(wires=[1, 2, 3]),
        ]

        assert all(qml.equal(op1, op2) for op1, op2 in zip(qrom_decomposition, expected_gates))


@pytest.mark.parametrize(
    ("control_wires", "target_wires", "work_wires", "msg_match"),
    [
        (
            [0, 1, 2],
            [0, 3],
            [4, 5],
            "Target wires should be different from control wires.",
        ),
        (
            [0, 1, 2],
            [4],
            [2, 5],
            "Control wires should be different from work wires.",
        ),
        (
            [0, 1, 2],
            [4],
            [4],
            "Target wires should be different from work wires.",
        ),
    ],
)
def test_wires_error(control_wires, target_wires, work_wires, msg_match):
    """Test an error is raised when a control wire is in one of the ops"""
    with pytest.raises(ValueError, match=msg_match):
        qml.QROM(["1"] * 8, target_wires, control_wires, work_wires)


def test_repr():
    """Test that the __repr__ method works as expected."""
    op = str(
        qml.QROM(
            ["1", "0", "0", "1"], control_wires=[0, 1], target_wires=[2], work_wires=[3], clean=True
        )
    )
    res = op.__repr__()
    expected = "QROM(target_wires=<Wires = [2]>, control_wires=<Wires = [0, 1]>,  work_wires=<Wires = [3]>)"
    assert res == expected

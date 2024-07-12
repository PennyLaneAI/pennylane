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


def test_assert_valid_qrom():
    """Run standard validity tests."""
    bitstrings = ["000", "001", "111", "011", "000", "101", "110", "111"]

    op = qml.QROM(bitstrings, control_wires=[0, 1, 2], target_wires=[3, 4, 5], work_wires=[6, 7, 8])
    qml.ops.functions.assert_valid(op)


class TestQROM:
    """Test the qml.QROM template."""

    @pytest.mark.parametrize(
        ("bitstrings", "target_wires", "control_wires", "work_wires", "clean"),
        [
            (
                ["11", "01", "00", "10"],
                [0, 1],
                [2, 3],
                [4, 5],
                True,
            ),
            (
                ["11", "01", "00", "10"],
                [0, 1],
                [2, 3],
                [4, 5, 6, 7, 8, 9],
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
                None,
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
        self, bitstrings, target_wires, control_wires, work_wires, clean
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the QROM template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(j):
            qml.BasisEmbedding(j, wires=control_wires)

            qml.QROM(bitstrings, control_wires, target_wires, work_wires, clean)
            return qml.sample(wires=target_wires)

        for j in range(2 ** len(control_wires)):
            assert np.allclose(circuit(j), [int(bit) for bit in bitstrings[j]])

    @pytest.mark.parametrize(
        ("bitstrings", "target_wires", "control_wires", "work_wires"),
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
                ["a", 5, 6, 7],
            ),
        ],
    )
    def test_work_wires_output(self, bitstrings, target_wires, control_wires, work_wires):
        """Tests that the ``clean = True`` version don't modify the initial state in work_wires."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit():

            # Initialize the work wires to a non-zero state
            for ind, wire in enumerate(work_wires):
                qml.RX(ind, wires=wire)

            for wire in control_wires:
                qml.Hadamard(wires=wire)

            qml.QROM(bitstrings, control_wires, target_wires, work_wires)

            for ind, wire in enumerate(work_wires):
                qml.RX(-ind, wires=wire)

            return qml.probs(wires=work_wires)

        assert np.isclose(circuit()[0], 1.0)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
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

        for op1, op2 in zip(qrom_decomposition, expected_gates):
            qml.assert_equal(op1, op2)

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", wires=4)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.QROM(["1", "0", "0", "1"], control_wires=[0, 1], target_wires=[2], work_wires=[3])
            return qml.probs(wires=3)

        assert jax.numpy.allclose(circuit(), jax.numpy.array([1.0, 0.0]))


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
        qml.QROM(["1"] * 8, control_wires, target_wires, work_wires)


def test_repr():
    """Test that the __repr__ method works as expected."""

    op = qml.QROM(
        ["1", "0", "0", "1"], control_wires=[0, 1], target_wires=[2], work_wires=[3], clean=True
    )
    res = op.__repr__()
    expected = "QROM(control_wires=Wires([0, 1]), target_wires=Wires([2]),  work_wires=Wires([3]), clean=True)"
    assert res == expected


@pytest.mark.parametrize(
    ("bitstrings", "control_wires", "target_wires", "msg_match"),
    [
        (
            ["1", "0", "0", "1"],
            [0],
            [2],
            r"Not enough control wires \(1\) for the desired number of bitstrings \(4\). At least 2 control wires are required.",
        ),
        (
            ["1", "0", "0", "1"],
            [0, 1],
            [2, 3],
            r"Bitstring length must match the number of target wires.",
        ),
    ],
)
def test_wrong_wires_error(bitstrings, control_wires, target_wires, msg_match):
    """Test that error is raised if more ops are requested than can fit in control wires"""
    with pytest.raises(ValueError, match=msg_match):
        qml.QROM(bitstrings, control_wires, target_wires, work_wires=None)

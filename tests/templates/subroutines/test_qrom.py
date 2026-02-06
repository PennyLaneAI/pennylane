# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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

import pennylane as qp
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_assert_valid_qrom():
    """Run standard validity tests."""
    bitstrings = ["000", "001", "111", "011", "000", "101", "110", "111"]

    op = qp.QROM(bitstrings, control_wires=[0, 1, 2], target_wires=[3, 4, 5], work_wires=[6, 7, 8])
    qp.ops.functions.assert_valid(op)


@pytest.mark.jax
def test_falsy_zero_as_work_wire():
    """Test that work wire is not treated as a falsy zero."""
    op = qp.QROM(["1", "0", "0", "1"], control_wires=[1, 2], target_wires=[3], work_wires=0)
    qp.ops.functions.assert_valid(op)


class TestQROM:
    """Test the qp.QROM template."""

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize(
        ("bitstrings", "target_wires", "control_wires", "work_wires", "clean"),
        [
            (
                ["111", "101", "100", "110"],
                [0, 1, 2],
                [3, 4],
                None,
                False,
            ),
            (
                ["111", "101", "100", "110"],
                [0, 1, 2],
                [3, 4],
                None,
                True,
            ),
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
        dev = qp.device("default.qubit")

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit(j):
            qp.BasisEmbedding(j, wires=control_wires)
            qp.QROM(bitstrings, control_wires, target_wires, work_wires, clean)
            return qp.sample(wires=target_wires)

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
        dev = qp.device("default.qubit")

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit():

            # Initialize the work wires to a non-zero state
            for ind, wire in enumerate(work_wires):
                qp.RX(ind, wires=wire)

            for wire in control_wires:
                qp.Hadamard(wires=wire)

            qp.QROM(bitstrings, control_wires, target_wires, work_wires)

            for ind, wire in enumerate(work_wires):
                qp.RX(-ind, wires=wire)

            return qp.probs(wires=work_wires)

        assert np.isclose(circuit()[0], 1.0)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        qrom_decomposition = qp.QROM(
            ["1", "0", "0", "1"], control_wires=[0, 1], target_wires=[2], work_wires=[3], clean=True
        ).decomposition()

        expected_gates = [
            qp.Hadamard(wires=[2]),
            qp.CSWAP(wires=[1, 2, 3]),
            qp.Select(
                ops=(
                    qp.BasisEmbedding(1, wires=[2]) @ qp.BasisEmbedding(0, wires=[3]),
                    qp.BasisEmbedding(0, wires=[2]) @ qp.BasisEmbedding(1, wires=[3]),
                ),
                control=[0],
            ),
            qp.CSWAP(wires=[1, 2, 3]),
            qp.Hadamard(wires=[2]),
            qp.CSWAP(wires=[1, 2, 3]),
            qp.Select(
                ops=(
                    qp.BasisEmbedding(1, wires=[2]) @ qp.BasisEmbedding(0, wires=[3]),
                    qp.BasisEmbedding(0, wires=[2]) @ qp.BasisEmbedding(1, wires=[3]),
                ),
                control=0,
            ),
            qp.CSWAP(wires=[1, 2, 3]),
        ]

        for op1, op2 in zip(qrom_decomposition, expected_gates):
            qp.assert_equal(op1, op2)

    @pytest.mark.parametrize(
        ("bitstrings", "control_wires", "target_wires", "work_wires", "clean"),
        [
            (["11", "01", "00", "10"], [2, 3], [0, 1], [4, 5, 6, 7, 8, 9], True),
            (["1", "0", "0", "1"], [0, 1], [2], [3], True),
            (["1"], [], [0], [1], True),
            (["10", "00", "00", "01", "01", "00", "00", "01"], [0, 1, 2], [3, 4], [5], False),
            (["01", "00", "00", "10", "10", "00", "00", "01"], [0, 1, 2], [3, 4], [5], True),
            (["1", "0", "0", "1"], [0, 1], [2], [], False),
            (["1", "0", "0", "1"], [0, 1], [2], [3, 4], False),
        ],  # pylint: disable=too-many-arguments
    )
    def test_decomposition_new(
        self, bitstrings, control_wires, target_wires, work_wires, clean
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = qp.QROM(
            bitstrings,
            control_wires=control_wires,
            target_wires=target_wires,
            work_wires=work_wires,
            clean=clean,
        )
        for rule in qp.list_decomps(qp.QROM):
            _test_decomposition_rule(op, rule)

    def test_zero_control_wires(self):
        """Test that the edge case of zero control wires works"""

        dev = qp.device("default.qubit", wires=2)
        qs = qp.tape.QuantumScript(
            qp.QROM.compute_decomposition(
                ["10"], target_wires=[0, 1], work_wires=None, control_wires=[], clean=False
            ),
            [qp.probs(wires=[0, 1])],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert len(tape[0][0].operations) == 1
        assert qp.equal(tape[0][0][0], qp.BasisEmbedding([1, 0], wires=[0, 1]))
        assert qp.math.allclose(output, [0, 0, 1, 0])

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qp.device("default.qubit", wires=4)

        @jax.jit
        @qp.qnode(dev)
        def circuit():
            qp.QROM(["1", "0", "0", "1"], control_wires=[0, 1], target_wires=[2], work_wires=[3])
            return qp.probs(wires=3)

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
        qp.QROM(["1"] * 8, control_wires, target_wires, work_wires)


def test_repr():
    """Test that the __repr__ method works as expected."""

    op = qp.QROM(
        ["1", "0", "0", "1"], control_wires=[0, 1], target_wires=[2], work_wires=[3], clean=True
    )
    res = repr(op)
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
        qp.QROM(bitstrings, control_wires, target_wires, work_wires=None)


def test_none_work_wires_case():
    """Test that clean version is not applied if work wires are not used"""

    gates_clean = qp.QROM.compute_decomposition(["1", "0", "0", "1"], [0, 1], [2], [], clean=True)
    expected_gates = qp.QROM.compute_decomposition(
        ["1", "0", "0", "1"], [0, 1], [2], [], clean=False
    )

    assert gates_clean == expected_gates


def test_too_many_work_wires_case():
    """Test that QROM works when more work wires are given than necessary"""

    gates_clean = qp.QROM.compute_decomposition(
        ["1", "0", "0", "1"], [0, 1], [2], [3, 4, 5], clean=False
    )
    expected_gates = qp.QROM.compute_decomposition(
        ["1", "0", "0", "1"],
        [0, 1],
        [2],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        clean=False,
    )

    assert gates_clean == expected_gates

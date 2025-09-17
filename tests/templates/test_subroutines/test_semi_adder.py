# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Tests for the SemiAdder template.
"""

from functools import partial

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.semi_adder import (
    _cnot_ladder,
    _fanout_1,
    _fanout_2,
    _semiadder,
    _semiadder_log_depth,
    _toffoli_ladder,
)


def test_standard_validity_SemiAdder():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2]
    y_wires = [4, 5, 6]
    work_wires = [7, 8]
    op = qml.SemiAdder(x_wires, y_wires, work_wires)
    qml.ops.functions.assert_valid(op, heuristic_resources=True)


class TestSemiAdder:
    """Test the qml.SemiAdder template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "work_wires"),
        [
            ([0], [1], []),
            ([0], [1], []),
            ([0, 1], [2], []),
            ([0, 1], [2], []),
            ([0, 1], [2], []),
            ([0, 1], [2, 3], [4]),
            ([0, 1], [2, 3], [4]),
            ([0, 1, 2], [3, 4, 5], [6, 7]),
            ([0, 1, 2], [3, 4, 5], [6, 7]),
            (["a", "b", "d"], ["e", "h", "p"], ["f", "z"]),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the SemiAdder template output."""

        if work_wires:
            total_wires = work_wires + x_wires + y_wires
        else:
            total_wires = x_wires + y_wires

        # We compare matrices when work_wires are initialized to |0>
        matrix1 = qml.matrix(_semiadder_log_depth, wire_order=total_wires)(x_wires, y_wires)[
            : 2 ** (len(x_wires) + len(y_wires) - len(work_wires))
        ]

        matrix2 = qml.matrix(_semiadder, wire_order=total_wires)(x_wires, y_wires, work_wires)[
            : 2 ** (len(x_wires) + len(y_wires) - len(work_wires))
        ]

        assert np.allclose(matrix1, matrix2)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [1],
                "At least 2 work_wires should be provided.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [1, 6],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [3, 6],
                "None of the wires in work_wires should be included in y_wires.",
            ),
            (
                [0, 1, 2],
                [2, 3, 4, 5],
                [6, 7, 8],
                "None of the wires in y_wires should be included in x_wires.",
            ),
        ],
    )
    def test_wires_error(
        self, x_wires, y_wires, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.SemiAdder(x_wires, y_wires, work_wires)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, y_wires, work_wires = (
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13],
        )

        adder_decomposition = qml.SemiAdder(x_wires, y_wires, work_wires).compute_decomposition(
            x_wires, y_wires, work_wires
        )

        names = [op.name for op in adder_decomposition]

        # Example in Fig 1.  https://arxiv.org/pdf/1709.06648
        assert names.count("TemporaryAND") == 4
        assert names.count("Adjoint(TemporaryAND)") == 4
        assert names.count("CNOT") == 21

    @pytest.mark.parametrize(
        ("x_wires"),
        [
            [0, 1, 2],
            [0, 1],
            [0, 1, 2, 3],
        ],
    )
    def test_decomposition_rule(self, x_wires):
        """Tests that SemiAdder is decomposed properly."""

        for rule in qml.list_decomps(qml.SemiAdder):
            _test_decomposition_rule(
                qml.SemiAdder(x_wires, [5, 6], [9, 10, 11]), rule, heuristic_resources=True
            )

    def test_error_length_wires(self):
        """Test that the log-depth decomposition throws an error if len(x_wires) < len(y_wires)."""

        with pytest.raises(AssertionError, match="must be greater or equal"):
            _test_decomposition_rule(
                qml.SemiAdder([0, 1, 2], [5, 6, 7, 8], [9, 10, 11]),
                _semiadder_log_depth,
                heuristic_resources=True,
            )

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x, y = 2, 3

        x_wires = [0, 1, 4]
        y_wires = [2, 3, 5]
        work_wires = [7, 8]
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.SemiAdder(x_wires, y_wires, work_wires)
            return qml.sample(wires=y_wires)

        # pylint: disable=bad-reversed-sequence
        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()[0, :]))),
            (x + y) % 2 ** len(y_wires),
        )


@pytest.mark.parametrize(("wires"), [[0, 1, 2, 3], [2, 3, 4, "a", 6, 7, 8], [0, 4]])
def test_cnot_ladder(wires):
    """Check the auxiliary function _cnot_ladder."""

    def cnot_ladder(wires):
        for i in range(len(wires) - 1):
            qml.CNOT([wires[i], wires[i + 1]])

    target_matrix = qml.matrix(cnot_ladder, wire_order=wires)(wires)
    generated_matrix = qml.matrix(_cnot_ladder, wire_order=wires)(wires)
    assert np.allclose(target_matrix, generated_matrix)


@pytest.mark.parametrize(("wires"), [[0, 1, 2, 3], [2, 3, 4, "a", 6, 7, 8], [0, 4, "c"]])
def test_toffoli_ladder(wires):
    """Check the auxiliary function _toffoli_ladder."""

    def toffoli_ladder(wires):
        for i in range(0, len(wires) - 2, 2):
            qml.Toffoli([wires[i], wires[i + 1], wires[i + 2]])

    target_matrix = qml.matrix(toffoli_ladder, wire_order=wires)(wires)
    generated_matrix = qml.matrix(_toffoli_ladder, wire_order=wires)(wires)
    assert np.allclose(target_matrix, generated_matrix)


@pytest.mark.parametrize(("wires"), [[0, 1, 2, 3], [2, 3, 4, "a", 6, 7, 8], [0, 4]])
def test_fanout1_ladder(wires):
    """Check the auxiliary function _fanout_1."""

    def fanout1(wires):
        for i in range(1, len(wires)):
            qml.CNOT([wires[0], wires[i]])

    target_matrix = qml.matrix(fanout1, wire_order=wires)(wires)
    generated_matrix = qml.matrix(_fanout_1, wire_order=wires)(wires)
    assert np.allclose(target_matrix, generated_matrix)


@pytest.mark.parametrize(("wires"), [[0, 1, 2], [2, 3, 4, "a", 6, 7, 8], [0, 4, "c"]])
def test_fanout_2(wires):
    """Check the auxiliary function _fanout_2."""

    def fanout_2(wires):
        for i in range(1, len(wires) - 1, 2):
            qml.Toffoli([wires[0], wires[i], wires[i + 1]])

    target_matrix = qml.matrix(fanout_2, wire_order=wires)(wires)
    generated_matrix = qml.matrix(_fanout_2, wire_order=wires)(wires[0], wires[1::2], wires[2::2])
    assert np.allclose(target_matrix, generated_matrix)

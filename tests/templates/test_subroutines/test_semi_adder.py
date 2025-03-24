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

import pytest

import pennylane as qml
from pennylane import numpy as np


def test_standard_validity_SemiAdder():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1]
    y_wires = [2, 3, 6]
    mod = 7
    work_wires = [4,5]
    op = qml.SemiAdder(x_wires, y_wires, mod, work_wires)
    qml.ops.functions.assert_valid(op)


class TestSemiAdder:
    """Test the qml.SemiAdder template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "mod", "work_wires", "x", "y"),
        [
            ([0, 1, 2], [3, 4, 5], 3, [6, 7], 1, 2),
            ([0, 1, 2], [3, 4, 5], 7, [6, 7], 5, 6),
            ([0, 1], [2, 3, 4], 5, [5, 6], 3, 2),
            ([0, 1], [2, 3, 4, 5], None, [6, 7], 3, 10),
            ([0, 1, 2], [3, 4, 5], None, None, 7, 7),
            ([0, 1, 2], [3, 4, 5, 6], None, [7, 8], 6, 5),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, mod, work_wires, x, y
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the SemiAdder template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.SemiAdder(x_wires, y_wires, mod, work_wires)
            return qml.sample(wires=y_wires)

        if mod is None:
            mod = 2 ** len(y_wires)

        # pylint: disable=bad-reversed-sequence
        assert np.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x, y)))),
            (x + y) % mod,
        )

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "mod", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                9,
                [6, 7],
                "SemiAdder must have enough wires to represent mod.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                7,
                [1, 6],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                7,
                [3, 6],
                "None of the wires in work_wires should be included in y_wires.",
            ),
            (
                [0, 1, 2],
                [2, 3, 4, 5],
                7,
                [6, 7],
                "None of the wires in y_wires should be included in x_wires.",
            ),
        ],
    )
    def test_wires_error(
        self, x_wires, y_wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.SemiAdder(x_wires, y_wires, mod, work_wires)

    @pytest.mark.parametrize("work_wires", [None, [6], [6, 7, 8]])
    def test_validation_of_num_work_wires(self, work_wires):
        """Test that when mod is not 2**len(y_wires), validation confirms two
        work wires are present, while any work wires are accepted for mod=2**len(y_wires)"""

        # if mod=2**len(output_wires), anything goes
        qml.SemiAdder(
            x_wires=[0, 1, 2],
            y_wires=[3, 4, 5],
            mod=8,
            work_wires=work_wires,
        )

        with pytest.raises(ValueError, match="two work wires should be provided"):
            qml.SemiAdder(
                x_wires=[0, 1, 2],
                y_wires=[3, 4, 5],
                mod=7,
                work_wires=work_wires,
            )

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, y_wires, output_wires, mod, work_wires = (
            [0, 1, 2],
            [3, 4, 5],
            7,
            [6, 7],
        )
        adder_decomposition = qml.SemiAdder(
            x_wires, y_wires, mod, work_wires
        ).compute_decomposition(x_wires, y_wires, mod, work_wires)
        op_list = []
        if mod != 2 ** len(y_wires) and mod is not None:
            qft_wires = work_wires[:1] + output_wires
            work_wire = work_wires[1:]
        else:
            qft_wires = output_wires
            work_wire = None
        op_list.append(qml.QFT(wires=qft_wires))
        op_list.append(
            qml.ControlledSequence(
                qml.PhaseAdder(1, qft_wires, mod, work_wire), control=x_wires
            )
        )
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_wires))

        for op1, op2 in zip(adder_decomposition, op_list):
            qml.assert_equal(op1, op2)

    def test_work_wires_added_correctly(self):
        """Test that no work wires are added if work_wire = None"""
        wires = qml.SemiAdder(x_wires=[1, 2], y_wires=[3, 4]).wires
        assert wires == qml.wires.Wires([1, 2, 3, 4])

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x, y = 2, 3

        mod = 7
        x_wires = [0, 1, 4]
        y_wires = [2, 3, 5]
        work_wires = [7, 8]
        dev = qml.device("default.qubit", shots=1)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.SemiAdder(x_wires, y_wires, mod, work_wires)
            return qml.sample(wires=y_wires)

        # pylint: disable=bad-reversed-sequence
        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()))), (x + y) % mod
        )

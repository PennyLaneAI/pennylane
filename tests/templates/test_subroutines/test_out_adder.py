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
Tests for the OutAdder template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np


def test_standard_validity_OutAdder():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1]
    y_wires = [2, 3, 9]
    output_wires = [4, 5, 8]
    mod = 7
    work_wires = [6, 7]
    op = qml.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
    qml.ops.functions.assert_valid(op)


class TestOutAdder:
    """Test the qml.OutAdder template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            ([0, 1, 2], [3, 4, 5], [9, 10, 11], 7, [7, 8]),
            ([0, 1, 2], [3, 4], [5, 6], 3, [7, 8]),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                None,
                [9, 10],
            ),
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8],
                None,
                None,
            ),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutAdder template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.BasisEmbedding(z, wires=output_wires)
            qml.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        if mod is None:
            max = 2 ** len(output_wires)
        else:
            max = mod
        for x, y, z in zip(range(len(x_wires)), range(len(y_wires)), range(len(output_wires))):
            assert np.allclose(
                sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x, y)))),
                (x + y + z) % max,
            )

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                9,
                [9, 10],
                "OutAdder must have at least enough wires to represent mod.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [1, 10],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [3, 10],
                "None of the wires in work_wires should be included in y_wires.",
            ),
            (
                [0, 1, 2],
                [2, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
                "None of the wires in y_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 7, 5],
                [6, 7, 8],
                7,
                [9, 10],
                "None of the wires in y_wires should be included in output_wires.",
            ),
            (
                [0, 1, 7],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
                "None of the wires in x_wires should be included in output_wires.",
            ),
            ([0, 1, 2], [3, 4], [6, 7, 8], 7, [9, 10], "len"),
        ],
    )
    def test_wires_error(self, x_wires, y_wires, output_wires, mod, work_wires, msg_match):
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, y_wires, output_wires, mod, work_wires = (
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            7,
            [9, 10],
        )
        adder_decomposition = qml.OutAdder(
            x_wires, y_wires, output_wires, mod, work_wires
        ).compute_decomposition(x_wires, y_wires, output_wires, mod, work_wires)
        op_list = []
        if mod != 2 ** len(output_wires) and mod is not None:
            qft_new_output_wires = work_wires[:1] + output_wires
            work_wire = work_wires[1:]
        else:
            qft_new_output_wires = output_wires
            work_wire = None
        op_list.append(qml.QFT(wires=qft_new_output_wires))
        op_list.append(
            qml.ControlledSequence(
                qml.PhaseAdder(1, qft_new_output_wires, mod, work_wire), control=x_wires
            )
        )
        op_list.append(
            qml.ControlledSequence(
                qml.PhaseAdder(1, qft_new_output_wires, mod, work_wire), control=y_wires
            )
        )
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_new_output_wires))

        for op1, op2 in zip(adder_decomposition, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x, y = 2, 3

        # x, y in binary
        x_list = [0, 1, 0]
        y_list = [0, 1, 1]
        mod = 7
        x_wires = [0, 1, 4]
        y_wires = [2, 3, 5]
        output_wires = [6, 7, 8]
        work_wires = [11, 10]
        dev = qml.device("default.qubit", shots=1)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x_list, wires=x_wires)
            qml.BasisEmbedding(y_list, wires=y_wires)
            qml.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()))), (x + y) % mod
        )

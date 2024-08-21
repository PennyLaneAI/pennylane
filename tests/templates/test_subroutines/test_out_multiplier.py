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
Tests for the OutMultiplier template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np


def test_standard_validity_OutMultiplier():
    """Check the operation using the assert_valid function."""
    mod = 12
    x_wires = [0, 1]
    y_wires = [2, 3, 4]
    output_wires = [6, 7, 8, 9]
    work_wires = [5, 10]
    op = qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
    qml.ops.functions.assert_valid(op)


class TestMultiplier:
    """Test the qml.OutMultiplier template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
            ),
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8, 2],
                14,
                [9, 10],
            ),
            (
                [0, 1, 2],
                [3, 4],
                [5, 6, 7, 8],
                8,
                [9, 10],
            ),
            (
                [0, 1, 2, 3],
                [4, 5],
                [6, 7, 8, 9, 10],
                22,
                [11, 12],
            ),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutMultiplier template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        if mod == None:
            max = 2 ** len(output_wires)
        else:
            max = mod
        for x, y in zip(range(len(x_wires)), range(len(y_wires))):
            assert np.allclose(
                sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x, y)))), (x * y) % max
            )

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
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
                6,
                None,
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
    def test_operation_result_args_None(
        self, x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutMultiplier template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        if mod == None:
            max = 2 ** len(output_wires)
        else:
            max = mod
        for x, y in zip(range(len(x_wires)), range(len(y_wires))):
            assert np.allclose(
                sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x, y)))), (x * y) % max
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
                "OutMultiplier must have at least enough wires to represent mod.",
            ),
        ],
    )
    def test_operation_error(self, x_wires, y_wires, output_wires, mod, work_wires, msg_match):
        """Test an error is raised when k or mod don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "msg_match"),
        [
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
        ],
    )
    def test_wires_error(self, x_wires, y_wires, output_wires, mod, work_wires, msg_match):
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                None,
                [9, 10],
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [9, 10, 11],
            ),
            (
                [0, 1, 2],
                [3, 5],
                [6, 8],
                3,
                [9, 10],
            ),
        ],
    )
    def test_decomposition(self, x_wires, y_wires, output_wires, mod, work_wires):
        """Test that compute_decomposition and decomposition work as expected."""
        multiplier_decomposition = OutMultiplier(
            x_wires, y_wires, output_wires, mod, work_wires
        ).compute_decomposition(x_wires, y_wires, output_wires, mod, work_wires)
        op_list = []
        if mod != 2 ** (len(output_wires)):
            qft_output_wires = work_wires[:1] + output_wires
        else:
            qft_output_wires = output_wires
        op_list.append(qml.QFT(wires=qft_output_wires))
        op_list.append(
            qml.ControlledSequence(
                qml.ControlledSequence(
                    PhaseAdder(1, output_wires, mod, work_wires), control=x_wires
                ),
                control=y_wires,
            )
        )
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_output_wires))

        for op1, op2 in zip(multiplier_decomposition, op_list):
            qml.assert_equal(op1, op2)

    # @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x, y = 2, 3

        # x, y in binary
        x_list = [0, 1, 0]
        y_list = [0, 1, 1]
        mod = 12
        x_wires = [0, 1]
        y_wires = [2, 3]
        output_wires = [6, 7, 8, 9]
        work_wires = [5, 10]
        dev = qml.device("default.qubit", shots=1)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()))), (x * y) % mod
        )

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
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.out_multiplier import OutMultiplier


@pytest.mark.jax
def test_standard_validity_OutMultiplier():
    """Check the operation using the assert_valid function."""
    mod = 12
    x_wires = [0, 1]
    y_wires = [2, 3, 4]
    output_wires = [6, 7, 8, 9]
    work_wires = [5, 10]
    op = OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
    qml.ops.functions.assert_valid(op)


class TestOutMultiplier:
    """Test the qml.OutMultiplier template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "x", "y"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
                2,
                3,
            ),
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8, 2],
                14,
                [9, 10],
                1,
                2,
            ),
            (
                [0, 1, 2],
                [3, 4],
                [5, 6, 7, 8],
                8,
                [9, 10],
                3,
                3,
            ),
            (
                [0, 1, 2, 3],
                [4, 5],
                [6, 7, 8, 9, 10],
                22,
                [11, 12],
                0,
                0,
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                None,
                [9, 10],
                1,
                3,
            ),
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8],
                None,
                None,
                3,
                3,
            ),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, output_wires, mod, work_wires, x, y
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutMultiplier template output."""
        dev = qml.device("default.qubit")

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x, y):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        if mod is None:
            mod = 2 ** len(output_wires)

        # pylint: disable=bad-reversed-sequence
        out = circuit(x, y)[0, :]
        assert np.allclose(sum(bit * (2**i) for i, bit in enumerate(reversed(out))), (x * y) % mod)

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
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                9,
                [9, 10],
                "OutMultiplier must have enough wires to represent mod.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                9,
                None,
                "If mod is not",
            ),
        ],
    )
    def test_wires_error(
        self, x_wires, y_wires, output_wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)

    @pytest.mark.parametrize("work_wires", [None, [9], [9, 10, 11]])
    def test_validation_of_num_work_wires(self, work_wires):
        """Test that when mod is not 2**len(output_wires), validation confirms two
        work wires are present, while any work wires are accepted for mod=2**len(output_wires)"""

        # if mod=2**len(output_wires), anything goes
        OutMultiplier(
            x_wires=[0, 1, 2],
            y_wires=[3, 4, 5],
            output_wires=[6, 7, 8],
            mod=8,
            work_wires=work_wires,
        )

        with pytest.raises(ValueError, match="two work wires should be provided"):
            OutMultiplier(
                x_wires=[0, 1, 2],
                y_wires=[3, 4, 5],
                output_wires=[6, 7, 8],
                mod=7,
                work_wires=work_wires,
            )

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, y_wires, output_wires, mod, work_wires = (
            [0, 1, 2],
            [3, 5],
            [6, 8],
            3,
            [9, 10],
        )
        multiplier_decomposition = (
            OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            .compute_decomposition(x_wires, y_wires, output_wires, mod, work_wires)[0]
            .decomposition()
        )

        op_list = []
        if mod != 2 ** len(output_wires):
            qft_output_wires = work_wires[:1] + output_wires
            work_wire = work_wires[1:]
        else:
            qft_output_wires = output_wires
            work_wire = None
        op_list.append(qml.QFT(wires=qft_output_wires))
        op_list.append(
            qml.ControlledSequence(
                qml.ControlledSequence(
                    qml.PhaseAdder(1, qft_output_wires, mod, work_wire), control=x_wires
                ),
                control=y_wires,
            )
        )
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_output_wires))

        for op1, op2 in zip(multiplier_decomposition, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            ([0, 1, 2], [3, 5], [6, 8], 3, [9, 10]),
            ([0, 1, 2], [3, 6], [5, 8], 4, [9, 10]),
        ],
    )
    def test_decomposition_new(
        self, x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
        for rule in qml.list_decomps(qml.OutMultiplier):
            _test_decomposition_rule(op, rule)

    def test_work_wires_added_correctly(self):
        """Test that no work wires are added if work_wire = None"""
        wires = qml.OutMultiplier(x_wires=[1, 2], y_wires=[3, 4], output_wires=[5, 6]).wires
        assert wires == qml.wires.Wires([1, 2, 3, 4, 5, 6])

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x, y = 2, 3
        x_list = [1, 0]
        y_list = [1, 1]
        mod = 12
        x_wires = [0, 1]
        y_wires = [2, 3]
        output_wires = [6, 7, 8, 9]
        work_wires = [5, 10]
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x_list, wires=x_wires)
            qml.BasisEmbedding(y_list, wires=y_wires)
            OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        # pylint: disable=bad-reversed-sequence
        out = circuit()[0, :]
        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(out))), (x * y) % mod
        )

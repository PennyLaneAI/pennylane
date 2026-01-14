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
Tests for the OutSquare template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.out_square import OutSquare


@pytest.mark.parametrize("output_wires_zeroed", [False, True])
@pytest.mark.jax
def test_standard_validity_out_square(output_wires_zeroed):
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2, 3]
    output_wires = [4, 5, 6, 7, 8, 9, 10]
    work_wires = [11, 12, 13, 14, 15]
    op = OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
    qml.ops.functions.assert_valid(op)


class TestOutSquare:
    """Test the OutSquare template."""

    @pytest.mark.parametrize("output_wires_zeroed", [False, True])
    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "work_wires", "x_values"),
        [
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8],
                [0, 1, 2, 3],
            ),
            (
                [0, 1, 2],
                [3, 4, 5, 6, 7],
                [8, 9, 10, 11],
                [0, 1, 3, 4, 5],
            ),
            (
                [0, 1, 2],
                [3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13],
                [0, 1, 3, 6],
            ),
            (
                [0, 1, 2, 3],
                [4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13],
                [0, 2, 5, 9, 13],
            ),
            (
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [9, 10, 11, 12, 13],
                [0, 2, 5, 9, 13],
            ),
        ],
    )
    def test_operation_result(
        self,
        x_wires,
        output_wires,
        work_wires,
        x_values,
        output_wires_zeroed,
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutSquare template output."""
        dev = qml.device("lightning.qubit")

        mod = 2 ** len(output_wires)
        if output_wires_zeroed:
            z = 0
        else:
            z = mod - 2
            print(f"{z=}")
            print(f"{mod=}")

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x, z):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(z, wires=output_wires)
            OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
            return (
                qml.sample(wires=x_wires),
                qml.sample(wires=output_wires),
                qml.sample(wires=work_wires),
            )

        for x in x_values:
            output = circuit(x, z)
            out_ints = [int("".join(map(str, out[0])), 2) for out in output]
            assert np.allclose(out_ints, [x, (z + x**2) % mod, 0])

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
                "OutSquare must have enough wires to represent mod.",
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
            OutSquare(x_wires, y_wires, output_wires, mod, work_wires)

    @pytest.mark.parametrize("work_wires", [None, [9], [9, 10, 11]])
    def test_validation_of_num_work_wires(self, work_wires):
        """Test that when mod is not 2**len(output_wires), validation confirms two
        work wires are present, while any work wires are accepted for mod=2**len(output_wires)"""

        # if mod=2**len(output_wires), anything goes
        OutSquare(
            x_wires=[0, 1, 2],
            y_wires=[3, 4, 5],
            output_wires=[6, 7, 8],
            mod=8,
            work_wires=work_wires,
        )

        with pytest.raises(ValueError, match="two work wires should be provided"):
            OutSquare(
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
            OutSquare(x_wires, y_wires, output_wires, mod, work_wires)
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
            ([0], [3, 6], [5, 8], 4, [9, 10]),
            ([0, 1, 2], [3], [5, 8], None, [9]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], None, [9, 10]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], 16, [9, 10]),
            ([0, 1], [3, 6], [5, 8, 2, 4], 16, [9, 10]),
        ],
    )
    def test_decomposition_new(
        self, x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = OutSquare(x_wires, y_wires, output_wires, mod, work_wires)
        for rule in qml.list_decomps(OutSquare):
            _test_decomposition_rule(op, rule)

    def test_work_wires_added_correctly(self):
        """Test that no work wires are added if work_wire = None"""
        wires = qml.OutSquare(x_wires=[1, 2], y_wires=[3, 4], output_wires=[5, 6]).wires
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
            OutSquare(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        # pylint: disable=bad-reversed-sequence
        out = circuit()[0, :]
        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(out))), (x * y) % mod
        )

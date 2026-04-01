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
Tests for the CAddSub template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import assert_valid


@pytest.mark.jax
def test_standard_validity_c_add_sub():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2]
    y_wires = [3, 4, 5]
    work_wires = [6, 7]
    control_wire = 8
    op = qp.CAddSub(control_wire, x_wires, y_wires, work_wires)
    assert_valid(op)


def validate_c_add_sub(output, x, y, ctrl_state, mod):
    """Validate that a circuit output matches the expected when encoding x, y and ctrl_state
    and applying ``CAddSub``."""
    if not isinstance(output[0], dict):
        # Output from QJIT
        output = tuple(
            {int(key): val for key, val in zip(out[0][np.where(out[1])], out[1][np.where(out[1])])}
            for out in output
        )
    else:
        output = tuple({int(key, 2): val for key, val in out.items()} for out in output)

    assert all(len(counts) == 1 for counts in output)  # Output should be deterministic
    x_val = list(output[0].keys())[0]
    assert x_val == x
    y_val = list(output[1].keys())[0]
    if ctrl_state == 0:
        assert y_val == (y - x) % mod
    else:
        assert y_val == (y + x) % mod

    if len(output) > 2:
        work_val = list(output[2].keys())[0]
        assert work_val == 0


class TestCAddSub:
    """Test the qp.CAddSub template."""

    @pytest.mark.usefixture("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "work_wires", "x", "y"),
        [
            ([0], [1], None, 1, 0),
            ([0], [1], None, 1, 1),
            ([0, 1], [2], None, 0, 1),
            ([0, 1], [2], None, 1, 1),
            ([0, 1], [2], None, 1, 0),
            ([0, 1], [2, 3], [4], 2, 0),
            ([0, 1], [2, 3], [4], 1, 2),
            ([0, 1, 2], [3, 4, 5], [6, 7], 1, 2),
            ([0, 1, 2], [3, 4, 5], [6, 7], 5, 6),
            ([0, 1], [2, 3, 4], [5, 6], 3, 2),
            ([0, 1], [2, 3, 4, 5], [6, 7, 8], 3, 10),
            ([0, 1, 2], [3, 4, 5], [6, 7], 7, 7),
            ([0, 1, 2], [3, 4, 5, 6], [7, 8, 9], 6, 5),
            ([0], [3, 4, 5, 6], [7, 8, 9], 1, 5),
            ([0, 1, 2, 3, 4], [5, 6], [7], 11, 2),
            (["a", "b", "d"], ["e", "h", "p"], ["f", "z"], 4, 2),
            (["a", "b", "d"], ["e", "h", "p"], ["f", "z", "u", "q"], 4, 2),
            (["a", "b", "d"], ["e", "h", "p"], ["f", "z", "u", "q", "v"], 4, 2),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, work_wires, x, y
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the CAddSub template output."""
        dev = qp.device("default.qubit")
        control_wire = "control"

        @qp.set_shots(100)
        @qp.decompose(gate_set={"BasisEmbedding", "C(BasisState)", "SemiAdder"})
        @qp.qnode(dev)
        def circuit(x, y, ctrl_state):
            qp.BasisEmbedding(ctrl_state, wires=control_wire)
            qp.BasisEmbedding(x, wires=x_wires)
            qp.BasisEmbedding(y, wires=y_wires)
            qp.CAddSub(control_wire, x_wires, y_wires, work_wires)
            if not work_wires:
                return qp.counts(wires=x_wires), qp.counts(wires=y_wires)
            return qp.counts(wires=x_wires), qp.counts(wires=y_wires), qp.counts(wires=work_wires)

        mod = 2 ** len(y_wires)
        for ctrl_state in [0, 1]:
            output = circuit(x, y, ctrl_state)
            validate_c_add_sub(output, x, y, ctrl_state, mod)

        gates = qp.specs(circuit, level=1)(x, y, 1)["resources"].gate_types
        assert gates == {"BasisEmbedding": 3, "C(BasisState)": 2, "SemiAdder": 1}

    @pytest.mark.parametrize(
        ("control_wire", "x_wires", "y_wires", "work_wires", "msg_match"),
        [
            (
                8,
                [0, 1, 2],
                [3, 4, 5],
                [1],
                "At least 2 work_wires should be provided.",
            ),
            (
                6,
                [0, 1, 2],
                [3, 4, 5],
                [6, 7],
                "None of the wires in work_wires should be the control wire.",
            ),
            (
                8,
                [0, 1, 2],
                [3, 4, 5],
                [1, 6],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                8,
                [0, 1, 2],
                [3, 4, 5],
                [3, 6],
                "None of the wires in work_wires should be included in y_wires.",
            ),
            (
                1,
                [0, 1, 2],
                [3, 4, 5],
                [6, 7],
                "None of the wires in x_wires should be the control wire.",
            ),
            (
                9,
                [0, 1, 2],
                [2, 3, 4, 5],
                [6, 7, 8],
                "None of the wires in y_wires should be included in x_wires.",
            ),
            (
                5,
                [0, 1, 2],
                [3, 4, 5],
                [6, 7],
                "None of the wires in y_wires should be the control wire.",
            ),
        ],
    )
    def test_wires_error(
        self, control_wire, x_wires, y_wires, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qp.CAddSub(control_wire, x_wires, y_wires, work_wires)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.catalyst
    @pytest.mark.external
    def test_qjit_compatible(self):
        """Test that the template is compatible with the QJIT compiler."""
        x, y = 2, 3

        x_wires = qp.math.array([0, 1, 4], like="jax")
        y_wires = [2, 3, 5]
        work_wires = [7, 8]
        control_wire = 9
        dev = qp.device("lightning.qubit", wires=10)

        @qp.qjit
        @qp.set_shots(100)
        @qp.decompose(max_expansion=3)
        @qp.qnode(dev)
        def circuit(x_wires, y, ctrl_state):
            qp.BasisEmbedding(ctrl_state, wires=control_wire)
            qp.BasisEmbedding(x, wires=x_wires)
            qp.BasisEmbedding(y, wires=y_wires)
            qp.CAddSub(control_wire, x_wires, y_wires, work_wires)
            return qp.counts(wires=x_wires), qp.counts(wires=y_wires), qp.counts(wires=work_wires)

        mod = 2 ** len(y_wires)
        for ctrl_state in [0, 1]:
            output = circuit(x_wires, y, ctrl_state)
            validate_c_add_sub(output, x, y, ctrl_state, mod)

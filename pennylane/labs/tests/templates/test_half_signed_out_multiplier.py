# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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
Tests for the half_signed_out_multiplier function.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates import half_signed_out_multiplier


def bin_to_int(bits):
    """Converts a binary array to an integer."""
    return int("".join(map(str, bits)), 2)


@pytest.mark.parametrize(
    "x_wires, y_wires, output_wires, work_wires, init_state",
    [
        (
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7, 8, 9, 10),
            [1, 1]  # operand one: 3
            + [0, 1]  # operand two: 1
            + [0, 1]  # output register starts in non-zero state!
            + [0, 0, 0, 0, 0],  # work wires are zeroed
        ),
        (
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7, 8, 9, 10),
            [0, 1]  # operand one: 1
            + [1, 1]  # operand two: -1
            + [1, 1]  # output register starts in negative non-zero state!
            + [0, 0, 0, 0, 0],  # work wires are zeroed
        ),
        (
            (0, 1, 2),
            (3, 4),
            (5, 6, 7, 8, 9, 10),
            (11, 12, 13, 14),
            [1, 0, 1]  # operand one: 5
            + [1, 1]  # operand two: -1
            + [0, 0, 0, 0, 0, 0]  # output register starts in |0>
            + [0, 0, 0, 0],  # work wires are zeroed
        ),
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9, 10, 11),
            (12, 13, 14, 15),
            [1, 0, 1]  # operand one: 5
            + [0, 1, 1]  # operand two: 3
            + [0, 0, 0, 0, 0, 0]  # output register starts in |0>
            + [0, 0, 0, 0],  # work wires are zeroed
        ),
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9, 10, 11),
            (12, 13, 14, 15),
            [1, 1, 1]  # operand one: 7
            + [1, 0, 1]  # operand two: -3
            + [0, 0, 0, 0, 0, 0]  # output register starts in |0>
            + [0, 0, 0, 0],  # work wires are zeroed
        ),
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9, 10, 11),
            (12, 13, 14, 15),
            [1, 0, 0]  # operand one: 4
            + [0, 1, 1]  # operand two: 3
            + [1, 0, 1, 0, 1, 1]  # output register starts in non-zero state
            + [0, 0, 0, 0],  # work wires are zeroed
        ),
    ],
)
def test_half_signed_out_multiplier_correct(x_wires, y_wires, output_wires, work_wires, init_state):
    """Tests with a few examples that ``half_signed_out_multiplier`` yields correct results."""

    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner
        dev = qp.device("default.qubit", wires=x_wires + y_wires + output_wires + work_wires)

        @qp.qnode(dev)
        def signed_multiply(init_state, *all_wires):
            sum_wires = sum(all_wires, start=())
            qp.BasisState(init_state, sum_wires)
            half_signed_out_multiplier(*all_wires)
            return qp.state()

        # get the initial state of our inputs
        x_state = [init_state[x] for x in x_wires]
        y_state = [init_state[y] for y in y_wires]
        z_state = [init_state[z] for z in output_wires]

        # get the integer value of the x input
        x = bin_to_int(x_state)

        # get the integer value of the y input (two's complement differs from bin_to_int by
        # _subtracting_ 2**(len(y_state)-1) instead of adding, it, so that we double the term.)
        # This is equivalent to ``y = bin_to_int(y_state[1:]) - 2 ** (len(y_state)-1) * y_state[0]
        y = bin_to_int(y_state) - 2 ** len(y_state) * y_state[0]

        # get the integer value of the z input
        z = bin_to_int(z_state)

        # calculate the expected result
        y_unsigned = y + 2 ** len(y_wires) * int(y < 0)
        z_expected = (x * y + z) % 2 ** len(output_wires)
        # Compute the overall expected position of the 1 in the full output state vector
        total_expected = (
            x * 2 ** (len(y_wires) + len(output_wires) + len(work_wires))
            + y_unsigned * 2 ** (len(output_wires) + len(work_wires))
            + z_expected * 2 ** len(work_wires)
        )

        # execute the quantum signed out multiplier circuit
        result = signed_multiply(init_state, x_wires, y_wires, output_wires, work_wires)

        # isclose will not match entries with wrong phase
        result = np.where(np.isclose(result, 1.0))[0]
        assert np.allclose(result, total_expected)

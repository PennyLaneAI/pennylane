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
Tests for the semi_signed_out_multiplier function.
"""

from functools import reduce

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates import semi_signed_out_multiplier

# from pennylane import SignedOutMultiplier, device, qnode
# from pennylane.decomposition import list_decomps
# from pennylane.measurements import sample, state
# from pennylane.ops import CNOT
# from pennylane.ops.functions.assert_valid import _test_decomposition_rule, assert_valid
# from pennylane.templates.subroutines.arithmetic.signed_out_multiplier import _twos_complement_helper


def bin_to_int(bits):
    """Converts a binary array to an integer."""
    return int("".join(map(str, bits)), 2)


def int_to_bin(integer, pd=""):
    """Converts an integer to a binary array."""
    if integer < 0:
        bin_str = format(integer, f"#0{pd}b")[3:]
    else:
        bin_str = format(integer, f"#0{pd}b")[2:]
    return list(reduce(lambda acc, nxt: acc + [int(nxt)], bin_str, []))


def twos_complement_value(bits):
    """Calculates the value of a number encoded as a twos complement."""
    sum = 0
    for i, bit in enumerate(bits[1:][::-1]):
        sum += (2**i) * bit
    sum -= (2 ** (len(bits) - 1)) * bits[0]
    return sum


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
def test_semi_signed_out_multiplier_correct(x_wires, y_wires, output_wires, work_wires, init_state):
    """Tests with a few examples that ``semi_signed_out_multiplier`` yields correct results."""

    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner
        dev = qp.device("default.qubit", wires=x_wires + y_wires + output_wires + work_wires)

        @qp.qnode(dev)
        def signed_multiply(init_state, *all_wires):
            sum_wires = sum(all_wires, start=())
            qp.BasisState(init_state, sum_wires)
            semi_signed_out_multiplier(*all_wires)
            return qp.state()

        # get the initial state of our inputs
        x_state = [init_state[x] for x in x_wires]
        y_state = [init_state[y] for y in y_wires]
        z_state = [init_state[z] for z in output_wires]

        # get the integer value of the x input
        x = bin_to_int(x_state)

        # get the integer value of the y input
        if y_state[0] == 1:
            # get the value encoded using twos complement if it is negative
            y = twos_complement_value(y_state)
        else:
            # otherwise just convert from binary to int
            y = bin_to_int(y_state)

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

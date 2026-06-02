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
Tests for the SignedOutMultiplier template.
"""

from functools import reduce

import numpy as np
import pytest

from pennylane import SignedOutMultiplier, device, qnode
from pennylane.decomposition import list_decomps
from pennylane.measurements import sample
from pennylane.ops import CNOT
from pennylane.ops.functions.assert_valid import _test_decomposition_rule, assert_valid
from pennylane.templates import BasisEmbedding
from pennylane.templates.subroutines.arithmetic.signed_out_multiplier import _twos_complement_helper

dev = device("default.qubit")


def bin_to_int(bits):
    """Converts a binary array to an integer."""
    return int("".join(map(str, bits)), 2)


def int_to_bin(integer):
    """Converts an integer to a binary array."""
    if integer < 0:
        bin_str = bin(integer)[3:]
    else:
        bin_str = bin(integer)[2:]
    return list(reduce(lambda acc, nxt: acc + [int(nxt)], bin_str, []))


def twos_complement_value(bits):
    """Calculates the value of a number encoded as a twos complement."""
    sum = 0
    for i, bit in enumerate(bits[1:][::-1]):
        sum += (2**i) * bit
    sum -= (2 ** (len(bits) - 1)) * bits[0]
    return sum


@pytest.mark.parametrize(
    "x_wires, y_wires, work_wires, output_wires, zeroed",
    [
        ((0, 1, 2), (3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13, 14, 15), True),
        ((0, 1), (2, 3), (4, 5, 6, 7, 8, 9), (10, 11), False),
    ],
)
def test_assert_valid(x_wires, y_wires, work_wires, output_wires, zeroed):
    op = SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires, zeroed)
    assert_valid(op)


@pytest.mark.parametrize(
    ("x_wires", "y_wires", "output_wires", "work_wires", "msg_match"),
    [
        (
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [1, 10],
            "None of the wires in work_wires should be included in x_wires.",
        ),
        (
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [3, 10],
            "None of the wires in work_wires should be included in y_wires.",
        ),
        (
            [0, 1, 2],
            [2, 4, 5],
            [6, 7, 8],
            [9, 10],
            "None of the wires in y_wires should be included in x_wires.",
        ),
        (
            [0, 1, 2],
            [3, 7, 5],
            [6, 7, 8],
            [9, 10],
            "None of the wires in y_wires should be included in output_wires.",
        ),
        (
            [0, 1, 7],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10],
            "None of the wires in x_wires should be included in output_wires.",
        ),
    ],
)
def test_wires_error(x_wires, y_wires, output_wires, work_wires, msg_match):
    """Test an error is raised when some work_wires don't meet the requirements"""
    with pytest.raises(ValueError, match=msg_match):
        SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires)


@pytest.mark.capture
@pytest.mark.parametrize(
    "x_wires, y_wires, work_wires, output_wires, zeroed",
    [
        ((0, 1, 2), (3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13, 14, 15), True),
        ((0, 1), (2, 3), (4, 5, 6, 7, 8, 9), (10, 11), False),
    ],
)
def test_decomposition(x_wires, y_wires, work_wires, output_wires, zeroed):
    op = SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires, zeroed)

    for rule in list_decomps(SignedOutMultiplier):
        _test_decomposition_rule(op, rule)


@qnode(dev, shots=1)
def signed_multiply(
    x_wires, y_wires, work_wires, output_wires, init_state, zeroed
):  # pylint: disable=too-many-arguments
    BasisEmbedding(
        init_state,
        x_wires + y_wires + work_wires + output_wires,
    )
    SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=zeroed)
    return sample(wires=output_wires)


@pytest.mark.parametrize(
    "x_wires, y_wires, work_wires, output_wires, init_state, zeroed",
    [
        (
            (0, 1),
            (2, 3),
            (4, 5, 6, 7, 8, 9),
            (10, 11),
            [1, 1]  # operand one: -1
            + [0, 1]  # operand two: 1
            + [0, 0, 0, 0, 0, 0]  # work wires are zeroed
            + [0, 1],  # output register starts in non-zero state!
            False,
        ),
        (
            (0, 1),
            (2, 3),
            (4, 5, 6, 7, 8, 9),
            (10, 11),
            [0, 1]  # operand one: 1
            + [0, 1]  # operand two: 1
            + [0, 0, 0, 0, 0, 0]  # work wires are zeroed
            + [1, 1],  # output register starts in negative non-zero state!
            False,
        ),
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9),
            (10, 11, 12, 13, 14, 15),
            [1, 0, 1]  # operand one: -3
            + [0, 1, 1]  # operand two: 3
            + [0, 0, 0, 0]  # work wires are zeroed
            + [0, 0, 0, 0, 0, 0],  # output register starts in |0>
            True,
        ),
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9),
            (10, 11, 12, 13, 14, 15),
            [1, 1, 1]  # operand one: -1
            + [1, 0, 1]  # operand two: -3
            + [0, 0, 0, 0]  # work wires are zeroed
            + [0, 0, 0, 0, 0, 0],  # output register starts in |0>
            True,
        ),
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9),
            (10, 11, 12, 13, 14, 15),
            [1, 0, 0]  # operand one: -4
            + [0, 1, 1]  # operand two: 3
            + [0, 0, 0, 0]  # work wires are zeroed
            + [0, 0, 0, 0, 0, 0],  # output register starts in |0>
            True,
        ),
    ],
)
def test_signed_out_multiplier_correct(
    x_wires, y_wires, work_wires, output_wires, init_state, zeroed
):  # pylint: disable=too-many-arguments
    """Tests with a few examples that the Template yields correct results."""

    # get the initial state of our inputs
    x_state = [init_state[x] for x in x_wires]
    y_state = [init_state[y] for y in y_wires]

    # get the integer value of the x input
    if init_state[0] == 1:
        # get the value encoded using twos complement if it is negative
        x = twos_complement_value(x_state)
    else:
        # otherwise just convert from binary to int
        x = bin_to_int(x_state)

    # get the integer value of the y input
    if init_state[3] == 1:
        # get the value encoded using twos complement if it is negative
        y = twos_complement_value(y_state)
    else:
        # otherwise just convert from binary to int
        y = bin_to_int(y_state)

    # get initial output register value
    if zeroed:
        z = twos_complement_value(init_state[-6:])
    else:
        z = twos_complement_value(init_state[-2:])

    # calculate the expected result
    expected = x * y + z

    # execute the quantum signed out multiplier circuit
    result = signed_multiply(x_wires, y_wires, work_wires, output_wires, init_state, zeroed)[0]

    # get the value encoded as a twos complement if the result is negative
    if result[0] == 1:
        result = twos_complement_value(result)
    else:
        result = bin_to_int(result)

    assert result == expected


@pytest.mark.parametrize(
    "aux, wires, init_state, work_wires, expected",
    [
        (3, [0, 1, 2], [1, 1, 1], [4, 5], [0, 0, 1]),  # -1
        (3, [0, 1, 2], [1, 1, 0], [4, 5], [0, 1, 0]),  # -2
        (3, [0, 1, 2], [1, 0, 1], [4, 5], [0, 1, 1]),  # -3
        (3, [0, 1, 2], [1, 0, 0], [4, 5], [1, 0, 0]),  # -4
    ],
)
def test_twos_complement_helper(aux, wires, init_state, work_wires, expected):
    """Tests that the twos complement helper works correctly."""

    @qnode(dev, shots=1)
    def twos_complement(aux, wires, init_state, work_wires):
        # load value
        BasisEmbedding(init_state, wires)

        # sign extend
        CNOT([wires[0], aux])

        # calculate twos complement
        _twos_complement_helper(wires, aux, work_wires)

        # measure
        return sample(wires=wires)

    expected_calc = -twos_complement_value(init_state)
    assert expected_calc == bin_to_int(expected)

    result = twos_complement(aux, wires, init_state, work_wires)[0]
    assert np.all(result == expected)

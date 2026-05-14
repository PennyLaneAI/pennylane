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
from pennylane.ops import CNOT, PauliX

from pennylane.measurements import probs, sample

from pennylane.templates import BasisEmbedding

from pennylane import device, qnode, SignedOutMultiplier, math, for_loop, Incrementer, draw
from pennylane.templates.subroutines.arithmetic.signed_out_multiplier import _twos_complement_helper

dev = device("default.qubit")


def bin_to_int(bits):
    """Converts a binary array to an integer."""
    return int("".join(map(str, bits)), 2)


def twos_complement_value(bits):
    """Calculates the value of a number encoded as a twos complement."""
    sum = 0
    for i, bit in enumerate(bits[1:][::-1]):
        sum += (2 ** i) * bit
    sum -= (2 ** (len(bits) - 1)) * bits[0]
    return sum


@qnode(dev)
def signed_multiply(x_wires, y_wires, work_wires, output_wires, init_state):
    BasisEmbedding(
        init_state,
        (0, 1, 2) +
        (3, 4, 5) +
        (6, 7, 8, 9) +
        (10, 11, 12, 13, 14, 15),
    )
    SignedOutMultiplier(
        x_wires,
        y_wires,
        output_wires,
        work_wires
    )
    return probs()


@pytest.mark.parametrize(
    "x_wires, y_wires, work_wires, output_wires, init_state",
    [
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9),
            (10, 11, 12, 13, 14, 15),
            [1, 0, 1]  # operand one: -3
            + [0, 1, 1]  # operand two: 3
            + [0, 0, 0, 0]  # work wires are zeroed
            + [0, 0, 0, 0, 0, 0]  # output register starts in |0>
        ),
    ]
)
def test_signed_out_multiplier_correct(x_wires, y_wires, work_wires, output_wires, init_state):
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

    # calculate the expected result
    expected = x * y

    # execute the quantum signed out multiplier circuit
    result = signed_multiply(x_wires, y_wires, work_wires, output_wires, init_state)

    # extract the output from the output histogram
    result = math.ceil_log2(list(math.round(result)).index(1)) % (2 ** len(output_wires))

    # convert the result to binary
    binary_result = reduce(lambda acc, bit: acc + [int(bit)], bin(result)[2:], [])

    # pad the result
    while len(binary_result) < len(output_wires):
        binary_result = [0] + binary_result

    # get the value encoded as a twos complement if the result is negative
    if binary_result[0] == 1:
        result = twos_complement_value(binary_result)

    assert result == expected

@pytest.mark.parametrize(
    "aux, wires, init_state, work_wires",
    [
        (
            3,
            [0, 1, 2],
            [1, 1, 1],  # -1
            [4, 5]
        ),
        (
            3,
            [0, 1, 2],
            [1, 1, 0], # -2
            [4, 5]
        ),
        (
            3,
            [0, 1, 2],
            [1, 0, 1], # -3
            [4, 5]
        )
    ]
)
def test_twos_complement_helper(aux, wires, init_state, work_wires):
    """Tests that the twos complement helper works correctly."""

    @qnode(dev)
    def twos_complement(aux, wires, init_state, work_wires):
        # load value
        BasisEmbedding(init_state, wires)

        # sign extend
        CNOT([wires[0], aux])

        # calculate twos complement
        _twos_complement_helper(wires, aux, work_wires)

        # measure
        return probs(wires=wires)

    expected = -twos_complement_value(init_state)

    result = twos_complement(aux, wires, init_state, work_wires)
    result = math.ceil_log2(list(math.round(result)).index(1))

    assert result == expected


@pytest.mark.parametrize(
    "wires, init_state, work_wires, expected",
    [
        (
            [0, 1, 2],
            [1, 1, 1],  # -1
            [],
            [0, 0, 1]
        ),
        (
            [0, 1, 2],
            [1, 1, 0], # -2
            [3, 4],
            [0, 1, 0]
        ),
        (
            [0, 1, 2],
            [1, 0, 1], # -3
            [3, 4, 5],
            [0, 1, 1]
        )
    ]
)
def test_simple_twos_complement(wires, init_state, work_wires, expected):
    def _twos_complement(input_reg, work_wires):
        # Invert all bits
        @for_loop(len(input_reg))
        def invert(w):
            PauliX(input_reg[w])

        invert()  # pylint: disable=no-value-for-parameter

    @qnode(dev, shots=1)
    def twos_complement(wires, init_state, work_wires):
        # load value
        BasisEmbedding(init_state, wires)

        # calculate twos complement
        _twos_complement(wires, work_wires)

        # Add one
        Incrementer(
            wires=wires,
            work_wires=work_wires,
        )

        # measure
        return sample(wires=wires)

    print(draw(twos_complement)(wires, init_state, work_wires))

    expected_calc = -twos_complement_value(init_state)
    assert expected_calc == bin_to_int(expected)

    result = twos_complement(wires, init_state, work_wires)[0]
    assert np.all(result == expected)

    result = math.ceil_log2(list(math.round(result)).index(1))
    assert result == expected_calc


@pytest.mark.parametrize(
    "wires, init_state, expected",
    [
        (
            [0, 1, 2],
            [1, 1, 1],
            [0, 0, 0]
        ),
        (
            [0, 1, 2],
            [1, 1, 0],
            [0, 0, 1]
        ),
        (
            [0, 1, 2],
            [1, 0, 1],
            [0, 1, 0]
        )
    ]
)
def test_inverter(wires, init_state, expected):

    def _inverter(input_reg):
        @for_loop(len(input_reg))
        def invert(w):
            PauliX(input_reg[w])

        invert()  # pylint: disable=no-value-for-parameter

    @qnode(dev, shots=1)
    def inverter(wires, init_state):
        BasisEmbedding(init_state, wires)
        _inverter(wires)
        return sample(wires=wires)

    result = inverter(wires, init_state)[0]
    assert np.all(result == expected)

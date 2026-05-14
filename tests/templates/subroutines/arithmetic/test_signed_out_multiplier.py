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

import pytest

from pennylane.measurements import probs

from pennylane.templates import BasisEmbedding

from pennylane import device, qnode, SignedOutMultiplier, math

dev = device("default.qubit")


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

    def bin_to_int(bits):
        return int("".join(map(str, bits)), 2)

    def twos_complement(bits):
        sum = 0
        for i, bit in enumerate(bits[1:][::-1]):
            sum += (2 ** i) * bit
        sum -= (2 ** (len(bits) - 1)) * bits[0]
        return sum

    x_state = [init_state[x] for x in x_wires]
    y_state = [init_state[y] for y in y_wires]

    if init_state[0] == 1:
        x = twos_complement(x_state)
    else:
        x = bin_to_int(x_state)
    if init_state[3] == 1:
        y = twos_complement(y_state)
    else:
        y = bin_to_int(y_state)

    expected = x * y

    result = signed_multiply(x_wires, y_wires, output_wires, work_wires, init_state)

    result = math.ceil_log2(list(math.round(result)).index(1)) % (2 ** len(output_wires))
    binary_result = reduce(lambda acc, bit: acc + [int(bit)], bin(result)[2:], [])

    while len(binary_result) < len(output_wires):
        binary_result = [0] + binary_result

    if binary_result[0] == 1:
        result = twos_complement(binary_result)

    assert result == expected

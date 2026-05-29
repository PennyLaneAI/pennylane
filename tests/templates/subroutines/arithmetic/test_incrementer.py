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
Tests for the Incrementer template.
"""

import numpy as np
import pytest

from pennylane import Incrementer, device, qnode
from pennylane.decomposition import list_decomps
from pennylane.measurements import sample, state
from pennylane.ops import Controlled, PauliX
from pennylane.ops.functions.assert_valid import _test_decomposition_rule, assert_valid
from pennylane.templates import BasisEmbedding


@pytest.mark.parametrize(
    "wires, work_wires",
    [
        ((0, 1, 2, 3, 4), (3, 4)),  # enough work wires for work wire decomp
        ((0, 1, 2, 3), (3,)),  # not enough work wires... uses fallback
        ((0, 1, 2), []),  # no work wires
    ],
)
def test_assert_valid(wires, work_wires):
    op = Incrementer(wires, work_wires)
    assert_valid(op)


@pytest.mark.capture
@pytest.mark.parametrize(
    "wires, work_wires",
    [
        ((0, 1, 2), (3, 4)),  # enough work wires for work wire decomp
        ((0, 1, 2), (3,)),  # not enough work wires... uses fallback
        ((0, 1, 2), []),  # no work wires
    ],
)
def test_decomposition_capture(wires, work_wires):
    op = Incrementer(wires, work_wires)

    for rule in list_decomps(Incrementer):
        _test_decomposition_rule(op, rule)


@pytest.mark.parametrize(
    "wires, init_state, expected, work_wires",
    [
        # without work wires
        ([0, 1, 2], [0, 0, 0], [0, 0, 1], []),
        ([0, 1, 2], [0, 0, 1], [0, 1, 0], []),
        ([0, 1, 2], [1, 1, 0], [1, 1, 1], []),
        ([0, 1, 2], [1, 0, 1], [1, 1, 0], []),
        ([0, 1, 2, 3], [1, 0, 1, 1], [1, 1, 0, 0], []),
        ([0, 1, 2, 3], [0, 0, 1, 1], [0, 1, 0, 0], []),
        # with work wires
        ([0, 1, 2], [1, 1, 0], [1, 1, 1], [3, 4]),  # enough work wires for our rule
        ([0, 1, 2], [1, 0, 1], [1, 1, 0], [3, 4, 5]),  # more than enough work wires
        ([0, 1, 2, 3], [1, 0, 1, 1], [1, 1, 0, 0], [4, 5, 6, 7, 8]),  # more than enough work wires
        ([0, 1, 2, 3], [0, 0, 1, 1], [0, 1, 0, 0], [4, 5]),  # some work wires, but not enough
    ],
)
def test_correct(wires, init_state, expected, work_wires):
    """Validates that the incrementer adds one."""
    dev = device("default.qubit", wires=wires + work_wires)

    @qnode(dev)
    def increment(wires, init_state, work_wires=None):
        BasisEmbedding(init_state, wires)
        Incrementer(wires, work_wires)
        return state()

    result = increment(wires, init_state, work_wires)

    expected = np.concatenate([np.array(expected), np.zeros(len(work_wires))])

    value = 0
    for i, bit in enumerate(expected[::-1]):
        if bit == 1:
            value += 2**i

    for j, entry in enumerate(result):
        if j == value:
            assert entry == 1
        else:
            assert entry == 0


@qnode(device("default.qubit"), shots=1)
def controlled_increment(wires, init_state, work_wires=None, control=0):
    BasisEmbedding(init_state, wires)
    if control:
        PauliX(len(wires + work_wires))
    Controlled(Incrementer(wires, work_wires), [len(wires + work_wires)], control_values=[1])
    return sample(wires=wires)


@pytest.mark.parametrize(
    "wires, init_state, expected, work_wires, control",
    [
        # enough work wires for our rule
        ([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [6, 7, 8, 9, 10, 11], 0),
        ([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [6, 7, 8, 9, 10, 11], 1),
        # not enough work wires
        ([0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0], [6, 7], 0),
        ([0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1], [6], 1),
        # no work wires
        ([0, 1, 2, 3, 4, 5], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [], 0),
        ([0, 1, 2, 3, 4, 5], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [], 1),
    ],
)
def test_controlled(wires, init_state, expected, work_wires, control):
    result = controlled_increment(wires, init_state, work_wires, control)
    assert np.all(result == expected)


@pytest.mark.capture
@pytest.mark.parametrize(
    "wires, work_wires",
    [
        ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)),  # enough work wires for work wire decomp
        (
            (0, 1, 2, 3, 4, 5),
            (
                6,
                7,
            ),
        ),  # not enough work wires... uses fallback
        ((0, 1, 2), tuple()),  # no work wires
    ],
)
def test_controlled_decomposition_new(wires, work_wires):
    """Tests the decomposition rule implemented with the new system."""
    op = Controlled(Incrementer(wires, work_wires), [len(wires + work_wires)], control_values=[1])
    for rule in list_decomps("C(Incrementer)"):
        _test_decomposition_rule(op, rule)

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
from pennylane.measurements import sample

from pennylane.templates import BasisEmbedding

from pennylane import device, Incrementer, qnode

dev = device("default.qubit")


@qnode(dev, shots=1)
def increment(wires, init_state, work_wires=None):
    BasisEmbedding(init_state , wires)
    Incrementer(wires, work_wires)
    return sample()


@pytest.mark.parametrize(
    "wires, init_state, expected",
    [
        (
            [0, 1, 2],
            [1, 1, 0],
            [1, 1, 1]
        ),
        (
            [0, 1, 2],
            [1, 0, 1],
            [1, 1, 0]
        ),
        (
            [0, 1, 2, 3],
            [1, 0, 1, 1],
            [1, 1, 0, 0]
        ),
        (
            [0, 1, 2, 3],
            [0, 0, 1, 1],
            [0, 1, 0, 0]
        ),
    ]
)
def test_correct(wires, init_state, expected):
    """Validates that the incrementer adds one."""
    np.all(increment(wires, init_state) == expected)

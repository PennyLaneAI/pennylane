# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the :func:`pennylane.template.subroutines.qram` class.
"""
import numpy as np
import pytest

from pennylane import device, qnode
from pennylane.measurements import probs
from pennylane.templates import BasisEmbedding
from pennylane.templates.subroutines.qram import BBQRAM

dev = device("default.qubit")


@qnode(dev)
def bb_quantum(bitstrings, qram_wires, target_wires, work_wires, address):
    BasisEmbedding(address, wires=qram_wires)

    BBQRAM(
        bitstrings,
        qram_wires=qram_wires,
        target_wires=target_wires,
        work_wires=work_wires,
    )
    return probs(wires=target_wires)


@pytest.mark.parametrize(
    (
        "bitstrings",
        "qram_wires",
        "target_wires",
        "bus",
        "dir_wires",
        "portL_wires",
        "portR_wires",
        "address",
        "probabilities",
    ),
    [
        (
            ["010", "111", "110", "000"],
            [0, 1],
            [2, 3, 4],
            5,
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            2,  # addressed from the left
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # |110>
        ),
(
            ["010", "111", "110", "000"],
            [0, 1],
            [2, 3, 4],
            5,
            [11, 10, 9],
            [6, 7, 8],
            [12, 13, 14],
            1,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # |111>
        ),
        (
            ["010", "111", "110", "000"],
            [0, 1],
            [2, 3, 4],
            5,
            [6, 7, 8],
            [12, 13, 14],
            [9, 10, 11],
            0,
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # |010>
        ),
    ],
)
def test_bb_quantum(
    bitstrings, qram_wires, target_wires, bus, dir_wires, portL_wires, portR_wires, address, probabilities
):  # pylint: disable=too-many-arguments
    assert np.allclose(
        probabilities,
        bb_quantum(bitstrings, qram_wires, target_wires, [bus] + dir_wires + portL_wires + portR_wires, address),
    )

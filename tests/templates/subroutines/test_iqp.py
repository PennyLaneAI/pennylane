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
Unit tests for the :func:`pennylane.template.subroutines.iqp` class.
"""
import re
from itertools import combinations

import numpy as np
import pytest

from pennylane import math, qnode, queuing
from pennylane.decomposition import list_decomps
from pennylane.devices import device
from pennylane.measurements import probs
from pennylane.ops import H, MultiRZ, PauliRot
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.iqp import IQP

dev = device("default.qubit")


def local_gates(n_qubits: int, max_weight=2):
    """
    Generates a gate list for containing all gates whose generators have Pauli weight
    less or equal than max_weight.
    :param n_qubits: The number of qubits in the gate list
    :param max_weight: maximum Pauli weight of gate generators
    :return (list[list[list[int]]]): gate list
    """
    gates = []
    for weight in math.arange(1, max_weight + 1):
        for gate in combinations(math.arange(n_qubits), weight):
            gates.append([list(gate)])
    return gates


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        (
            ([0], 2, [[0, 1], [0]], False),
            ValueError,
            "Number of gates and number of parameters for an Instantaneous Quantum Polynomial circuit must be the same",
        ),
        (
            ([0, 1], 0, [[0, 1], [0]], False),
            ValueError,
            "At least one valid wire",
        ),
    ],
)
def test_raises(params, error, match):
    with pytest.raises(error, match=re.escape(match)):
        IQP(*params)


@pytest.mark.parametrize(
    ("weights", "pattern", "spin_sym", "n_qubits"),
    [
        (
            math.random.uniform(0, 2 * np.pi, 4),
            local_gates(4, 1),
            False,
            4,
        ),
        (
            math.random.uniform(0, 2 * np.pi, 6),
            local_gates(6, 1),
            True,
            6,
        ),
    ],
)
def test_decomposition_new(
    weights, pattern, spin_sym, n_qubits
):  # pylint: disable=too-many-arguments
    op = IQP(weights, n_qubits, pattern, spin_sym)

    for rule in list_decomps(IQP):
        _test_decomposition_rule(op, rule)


@qnode(dev)
def iqp_circuit(weights, pattern, spin_sym, n_qubits):  # pylint: disable=too-many-arguments
    IQP(weights, n_qubits, pattern, spin_sym)
    return probs(wires=list(range(n_qubits)))


@pytest.mark.parametrize(
    ("weights", "pattern", "spin_sym", "n_qubits", "expected_circuit"),
    [
        (
            math.random.uniform(0, 2 * np.pi, 4),
            local_gates(4, 1),
            True,
            4,
            [
                PauliRot,
                H,
                H,
                H,
                H,
                MultiRZ,
                MultiRZ,
                MultiRZ,
                MultiRZ,
                H,
                H,
                H,
                H,
            ],
        ),
    ],
)
def test_decomposition_contents(
    weights, pattern, spin_sym, n_qubits, expected_circuit
):  # pylint: disable=too-many-arguments
    with queuing.AnnotatedQueue() as q:
        iqp_circuit(weights, pattern, spin_sym, n_qubits)

    for op, expected in zip(q.queue, expected_circuit):
        assert isinstance(op, expected)

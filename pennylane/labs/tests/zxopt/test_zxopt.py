# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/labs/dla/zxopt"""
import pytest

import pennylane as qml
from pennylane.labs.zxopt import basic_optimization, full_optimize, full_reduce, todd

circ1 = qml.tape.QuantumScript(
    [
        qml.CNOT((0, 1)),
        qml.T(0),
        qml.CNOT((3, 2)),
        qml.T(1),
        qml.CNOT((1, 2)),
        qml.T(2),
        qml.RZ(0.5, 1),
        qml.CNOT((1, 2)),
        qml.T(1),
        qml.CNOT((3, 2)),
        qml.T(0),
        qml.CNOT((0, 1)),
    ],
    [],
)

circ1_clifford_T = qml.tape.QuantumScript(
    [
        qml.CNOT((0, 1)),
        qml.T(0),
        qml.CNOT((3, 2)),
        qml.T(1),
        qml.CNOT((1, 2)),
        qml.T(2),
        qml.CNOT((1, 2)),
        qml.T(1),
        qml.CNOT((3, 2)),
        qml.T(0),
        qml.CNOT((0, 1)),
    ],
    [],
)


@pytest.mark.parametrize("circ", [circ1, circ1_clifford_T])
def test_full_reduce(circ):
    """Test full_reduce"""
    _ = full_reduce(circ)


@pytest.mark.parametrize("circ", [circ1_clifford_T])
def test_full_optimize(circ):
    """Test full_optimize"""
    _ = full_optimize(circ)


@pytest.mark.parametrize("circ", [circ1, circ1_clifford_T])
def test_basic_optimization(circ):
    """Test basic_optimization"""
    _ = basic_optimization(circ)


@pytest.mark.parametrize("circ", [circ1_clifford_T])
def test_todd(circ):
    """Test TODD"""
    _ = todd(circ)

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
from typing import Callable, List

import pytest
import pyzx as zx

import pennylane as qml
from pennylane.labs.zxopt import basic_optimization, full_optimize, full_reduce, todd
from pennylane.labs.zxopt.util import _tape2pyzx
from pennylane.tape import QuantumScript

# arbitrary circuit
circ1 = qml.tape.QuantumScript(
    [
        qml.CNOT((0, 1)),
        qml.T(0),
        qml.S(0),
        qml.Hadamard(0),
        qml.RZ(0.5, 0),
        qml.RX(0.5, 0),
        qml.RY(0.5, 0),
    ],
    [],
)

# arbitrary phase polynomial circuit
phase_poly1 = qml.tape.QuantumScript(
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

# clifford + T circuit
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
circ2_clifford_T = qml.tape.QuantumScript(
    [
        qml.CNOT((0, 1)),
        qml.T(0),
        qml.S(0),
        qml.Hadamard(0),
    ],
    [],
)


@pytest.mark.parametrize("circ", [circ1, phase_poly1, circ1_clifford_T, circ2_clifford_T])
def test_full_reduce(circ):
    """Test full_reduce"""
    batch, func = full_reduce(circ)

    assert isinstance(batch, List)
    assert isinstance(batch[0], QuantumScript)
    assert isinstance(func, Callable)


@pytest.mark.parametrize("circ", [circ1_clifford_T, circ2_clifford_T])
def test_full_optimize(circ):
    """Test full_optimize"""
    batch, func = full_optimize(circ)

    assert isinstance(batch, List)
    assert isinstance(batch[0], QuantumScript)
    assert isinstance(func, Callable)


@pytest.mark.parametrize("circ", [phase_poly1, circ1_clifford_T, circ2_clifford_T])
def test_basic_optimization(circ):
    """Test basic_optimization"""
    batch, func = basic_optimization(circ)

    assert isinstance(batch, List)
    assert isinstance(batch[0], QuantumScript)
    assert isinstance(func, Callable)


@pytest.mark.parametrize("circ", [circ1_clifford_T, circ2_clifford_T])
def test_todd(circ):
    """Test TODD"""
    batch, func = todd(circ)

    assert isinstance(batch, List)
    assert isinstance(batch[0], QuantumScript)
    assert isinstance(func, Callable)


@pytest.mark.parametrize("circ", [phase_poly1, circ1_clifford_T, circ2_clifford_T])
def test_tape2pyzx(circ):
    """Test that PL circuits are translated to pyzx circuits"""
    zx_circuit = _tape2pyzx(circ)
    assert isinstance(zx_circuit, zx.Circuit)


def test_full_optimize_warns_clifford_t():
    """Test that a warning is raised when attempting to full_optimize a circuit with rotation gates"""
    circ = qml.tape.QuantumScript(
        [
            qml.CNOT((0, 1)),
            qml.T(0),
            qml.RZ(0.5, 0),
            qml.Hadamard(0),
        ],
        [],
    )

    with pytest.warns(UserWarning, match="Input circuit is not in the"):
        (new_circ,), _ = full_optimize(circ)

    assert not any(isinstance(op, qml.RZ) for op in new_circ.operations)
    assert any(isinstance(op, (qml.S, qml.T, qml.Hadamard)) for op in new_circ.operations)


def test_full_optimize_doesnt_warn_clifford_t():
    """Test that no warning is raised when attempting to full_optimize a circuit with rotation gates but setting clifford_t_args"""
    circ = qml.tape.QuantumScript(
        [
            qml.CNOT((0, 1)),
            qml.T(0),
            qml.RZ(0.5, 0),
            qml.Hadamard(0),
        ],
        [],
    )

    (new_circ,), _ = full_optimize(circ, clifford_t_args={"epsilon": 0.1})

    assert not any(isinstance(op, qml.RZ) for op in new_circ.operations)
    assert any(isinstance(op, (qml.S, qml.T, qml.Hadamard)) for op in new_circ.operations)

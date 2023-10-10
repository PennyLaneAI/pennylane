# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the available built-in parametric qubit operations."""

import pytest
import pennylane as qml
from pennylane.transforms.decompositions.clifford_plus_t import (
    check_clifford_op,
    clifford_t_decomposition,
)

# Single qubits Clifford+T gates in PL
_CLIFFORD_T_ONE_GATES = [
    qml.Identity,
    qml.PauliX,
    qml.PauliY,
    qml.PauliZ,
    qml.Hadamard,
    qml.S,
    qml.SX,
    qml.T,
    qml.GlobalPhase
]

# Two qubits Clifford+T gates in PL
_CLIFFORD_T_TWO_GATES = [
    qml.CNOT,
    qml.CY,
    qml.CZ,
    qml.SWAP,
    qml.ISWAP,
]

_CLIFFORD_PHASE_GATES = _CLIFFORD_T_ONE_GATES + _CLIFFORD_T_TWO_GATES

def circuit_1():
    """Circuit 1 with quantum chemistry gates"""
    qml.PhaseShift(1, wires=[1])
    qml.SingleExcitation(2, wires=[1, 2])
    qml.DoubleExcitation(2, wires=[1, 2, 3, 4])
    return qml.expval(qml.PauliZ(0))


def circuit_2():
    """Circuit 2 without chemistry gates"""
    qml.CRX(1, wires=[0, 1])
    qml.IsingXY(2, wires=[1, 2])
    qml.ISWAP(wires=[0, 1])
    qml.Toffoli(wires=[1, 2, 3])
    return qml.expval(qml.PauliZ(0))


def circuit_3():
    """Circuit 3 with Clifford gates"""
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=[1])
    qml.ISWAP(wires=[0, 1])
    qml.Hadamard(wires=[0])
    return qml.expval(qml.PauliZ(0))

def circuit_4():
    """Circuit 4 with a Template"""
    qml.RandomLayers(weights=qml.math.array([[0.1, -2.1, 1.4]]), wires=range(2))
    return qml.expval(qml.PauliZ(0))

class TestCliffordCompile:
    """Unit tests for clifford compilation function."""

    @pytest.mark.parametrize(
        "op, res", [(qml.DoubleExcitation(2, wires=[0, 1, 2, 3]), False), (qml.PauliX(wires=[0]), True), (qml.CH(wires=[0, 1]), False)]
    )
    def test_clifford_checker(self, op, res):
        """Test Clifford checker operation for gate"""
        assert check_clifford_op(op) == res

    @pytest.mark.parametrize(("circuit"), [circuit_1, circuit_2, circuit_3, circuit_4])
    def test_decomposition(self, circuit):
        """Test decomposition for the Clifford transform."""

        dev = qml.device("default.qubit")
        circ = qml.QNode(circuit, dev)

        basic_tape = qml.tape.make_qscript(circ.func)()
        program, _ = dev.preprocess()

        circuit_transform = qml.transforms.core.TransformProgram([clifford_t_decomposition])
        program = circuit_transform + program
        [tape], _ = program([basic_tape])

        assert all(
            any(
                (
                    isinstance(op, gate) or isinstance(getattr(op, "base", None), gate)
                    for gate in _CLIFFORD_PHASE_GATES
                )
            )
            for op in tape.operations
        )

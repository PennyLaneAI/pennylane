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
"""
Tests for the QSVT template and qsvt wrapper function.
"""
import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines.qsvt import *


class TestQSVT:
    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "wires", "expected_output"),
        [
            (
                qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                [qml.PCPhase(0.5, dim=2, wires=[0, 1]), qml.PCPhase(0.5, dim=2, wires=[0, 1])],
                [0, 1],
                0.06587008470151662,
            ),
            (
                qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                [qml.PCPhase(0.5, dim=2, wires=[0, 1]), qml.PCPhase(0.3, dim=2, wires=[0, 1])],
                [0, 1],
                0.24662762785797077,
            ),
        ],
    )
    def test_output(self, U_A, lst_projectors, wires, expected_output):
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.QSVT(U_A, lst_projectors, wires)
            return qml.expval(qml.PauliY(wires=0))

        assert np.isclose(circuit(), expected_output)

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "wires", "results"),
        [
            (
                qml.BlockEncode(0.1, wires=0),
                [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)],
                0,
                [
                    qml.BlockEncode(np.array([[0.1]]), wires=[0]),
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                    qml.adjoint(qml.BlockEncode(np.array([[0.1]]), wires=[0])),
                    qml.PCPhase(0.3, dim=2, wires=[0]),
                ],
            ),
            (
                qml.PauliZ(wires=0),
                [qml.RZ(0.1, wires=0), qml.RY(0.2, wires=0), qml.RZ(0.3, wires=1)],
                [0, 1],
                [
                    qml.PauliZ(wires=[0]),
                    qml.RZ(0.1, wires=[0]),
                    qml.adjoint(qml.PauliZ(wires=[0])),
                    qml.RY(0.2, wires=[0]),
                    qml.PauliZ(wires=[0]),
                    qml.RZ(0.3, wires=[1]),
                ],
            ),
        ],
    )
    def test_queuing_ops(self, U_A, lst_projectors, wires, results):
        with qml.tape.QuantumTape() as tape:
            qml.QSVT(U_A, lst_projectors, wires)

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    def test_queuing_ops_defined_in_circuit(self):
        lst_projectors = [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)]
        wires = [0, 1]
        results = [
            qml.PauliX(wires=[0]),
            qml.PCPhase(0.2, dim=1, wires=[0]),
            qml.adjoint(qml.PauliX(wires=[0])),
            qml.PCPhase(0.3, dim=1, wires=[0]),
        ]

        with qml.tape.QuantumTape() as tape:
            qml.QSVT(qml.PauliX(wires=0), lst_projectors, wires)

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    def test_queuing_callables(self):
        def my_qfunc(A):
            return qml.PauliX(wires=0)

        def lst_phis(phis):
            return [qml.PCPhase(i, 2, wires=[0, 1]) for i in phis]

        A = np.array([[0.1, 0.2], [0.3, 0.4]])
        phis = np.array([0.2, 0.3])

        with qml.tape.QuantumTape() as tape:
            qml.QSVT(my_qfunc(A), lst_phis(phis), wires=[0, 1])

        results = [
            qml.PauliX(wires=[0]),
            qml.PCPhase(0.2, dim=2, wires=[0]),
            qml.adjoint(qml.PauliX(wires=[0])),
            qml.PCPhase(0.3, dim=2, wires=[0]),
        ]

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    def test_queuing_callables2(self):
        def my_qfunc2(A):
            return qml.prod(qml.PauliX(wires=0), qml.RZ(A[0][0], wires=0))

        def lst_phis(phis):
            return [qml.PCPhase(i, 2, wires=[0, 1]) for i in phis]

        A = np.array([[0.1, 0.2], [0.3, 0.4]])
        phis = np.array([0.1, 0.2])

        with qml.tape.QuantumTape() as tape:
            qml.QSVT(my_qfunc2(A), lst_phis(phis), wires=[0, 1])

        results = [
            qml.prod(qml.PauliX(wires=0), qml.RZ(A[0][0], wires=0)),
            qml.PCPhase(0.1, dim=2, wires=[0]),
            qml.adjoint(qml.prod(qml.PauliX(wires=0), qml.RZ(A[0][0], wires=0))),
            qml.PCPhase(0.2, dim=2, wires=[0]),
        ]

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters


class Testqsvt:
    @pytest.mark.parametrize(
        ("A", "phis", "wires", "expected_output"),
        [
            ([[0.1, 0.2], [0.3, 0.4]], [0.2, 0.3], [0, 1], 0.9943867483200935),
            ([[0.3, 0.1], [0.2, 0.9]], [0.1, 0.2, 0.3], [0, 1], -0.7494860171717178),
        ],
    )
    def test_output(self, A, phis, wires, expected_output):
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.qsvt(A, phis, wires)
            return qml.expval(qml.PauliZ(wires=0))

        assert np.isclose(circuit(), expected_output)

    def test_circuit_drawing(self):
        return None

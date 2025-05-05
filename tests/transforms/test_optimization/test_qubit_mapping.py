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
"""
Unit tests for the optimization transform ``qubit_mapping``.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.optimization import qubit_mapping


class TestQubitMapping:
    """Test that circuit is mapped to a correct architecture."""

    def test_all_conected(self):
        """Test that an all-conected architecture does not modify the circuit."""

        def qfunc():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.CZ(wires=[0, 3])
            qml.CZ(wires=[0, 1])
            qml.SWAP(wires=[0, 2])

        graph = {
            "a": ["b", "c", "d"],
            "b": ["a", "c", "d"],
            "c": ["a", "b", "d"],
            "d": ["a", "b", "c"],
        }

        initial_map = {0: "a", 1: "b", 2: "c", 3: "d"}
        transformed_qfunc = qubit_mapping(qfunc, graph, initial_map)

        new_tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(new_tape.operations) == 5

    @pytest.mark.parametrize(
        ("ops", "meas"),
        [
            (
                [
                    qml.RY(2, wires=0),
                    qml.CNOT(wires=[0, 1]),
                    qml.CNOT(wires=[0, 2]),
                    qml.RX(3, wires=1),
                    qml.CZ(wires=[0, 3]),
                    qml.CZ(wires=[0, 1]),
                    qml.RY(4, wires=2),
                    qml.SWAP(wires=[0, 2]),
                ],
                [qml.expval(qml.X(0))],
            ),
            (
                [
                    qml.Hadamard(0),
                    qml.Hadamard(2),
                    qml.CRY(2, wires=[0, 3]),
                    qml.CNOT(wires=[2, 0]),
                    qml.SWAP(wires=[0, 2]),
                    qml.RX(3, wires=3),
                    qml.CZ(wires=[0, 3]),
                    qml.CZ(wires=[0, 2]),
                    qml.RY(4, wires=2),
                    qml.SWAP(wires=[0, 2]),
                ],
                [qml.expval(qml.Y(3))],
            ),
        ],
    )
    def test_correctness_solution(self, ops, meas):
        """Test that the output is not modified by the transform"""

        qs = qml.tape.QuantumScript(
            ops,
            meas,
        )

        graph = {"a": ["b", "d"], "b": ["a", "c"], "c": ["b"], "d": ["a"]}

        initial_map = {0: "a", 1: "b", 2: "c", 3: "d"}
        transformed_qs = qubit_mapping(qs, graph, initial_map)

        dev = qml.device("default.qubit")

        program, _ = dev.preprocess()
        tape = program([qs])
        initial_output = dev.execute(tape[0])

        program, _ = dev.preprocess()
        tape = program([transformed_qs[0][0]])
        new_output = dev.execute(tape[0])

        assert np.isclose(initial_output, new_output)

    @pytest.mark.parametrize(
        ("ops", "meas"),
        [
            (
                [
                    qml.RY(2, wires=0),
                    qml.CNOT(wires=[0, 1]),
                    qml.CNOT(wires=[0, 2]),
                    qml.RX(3, wires=1),
                    qml.CZ(wires=[0, 3]),
                    qml.CZ(wires=[0, 1]),
                    qml.RY(4, wires=2),
                    qml.SWAP(wires=[0, 2]),
                ],
                [qml.expval(qml.X(0))],
            ),
            (
                [
                    qml.Hadamard(0),
                    qml.Hadamard(2),
                    qml.CRY(2, wires=[0, 3]),
                    qml.CNOT(wires=[2, 0]),
                    qml.SWAP(wires=[0, 2]),
                    qml.RX(3, wires=3),
                    qml.CZ(wires=[0, 3]),
                    qml.CZ(wires=[0, 2]),
                    qml.RY(4, wires=2),
                    qml.SWAP(wires=[0, 2]),
                ],
                [qml.expval(qml.Y(3))],
            ),
        ],
    )
    def test_correctness_connectivity(self, ops, meas):
        """Test that after routing only allowed physical connections appear."""

        qs = qml.tape.QuantumScript(
            ops,
            meas,
        )
        graph = {
            "a": ["b", "d"],
            "b": ["a", "c"],
            "c": ["b"],
            "d": ["a"],
        }

        initial_map = {0: "a", 1: "b", 2: "c", 3: "d"}

        transformed_qs = qubit_mapping(qs, graph, initial_map)
        new_tape = transformed_qs[0][0]

        for op in new_tape.operations:
            wires = op.wires
            if len(wires) == 2:
                w0, w1 = wires
                assert w1 in graph.get(w0, []) or w0 in graph.get(w1, [])

    def test_wires_is_mapped(self):
        """Test that the output tape has been labelled with the new qubits"""

        graph = {
            "physical 0": ["physical 1", "physical 2"],
            "physical 1": ["physical 0"],
            "physical 2": ["physical 0"],
        }

        initial_map = {f"logical {i}": f"physical {i}" for i in range(3)}

        qs = qml.tape.QuantumScript(
            [qml.Hadamard("logical 0"), qml.CZ(["logical 1", "logical 2"])],
            [qml.expval(qml.Z("logical 0"))],
        )

        transformed_qs = qubit_mapping(qs, graph, initial_map)

        for op in transformed_qs[0][0].operations:
            for wire in op.wires:
                assert "physical" in wire

    def test_more_physical_wires(self):
        """Test that the target architecture could have more wires than the initial circuit"""

        graph = {
            "physical 0": ["physical 1", "physical 2"],
            "physical 1": ["physical 0"],
            "physical 2": ["physical 0"],
        }

        initial_map = {f"logical {i}": f"physical {i}" for i in range(1, 3)}

        qs = qml.tape.QuantumScript(
            [qml.Hadamard("logical 1"), qml.CZ(["logical 1", "logical 2"])],
            [qml.expval(qml.Z("logical 2"))],
        )

        transformed_qs = qubit_mapping(qs, graph, initial_map)

        for op in transformed_qs[0][0].operations:
            for wire in op.wires:
                assert "physical" in wire

    def test_error_too_many_qubits(self):
        """Test that an error appears if there is an operator that acts in more than 2 qubits"""

        def qfunc():
            qml.GroverOperator(wires=[0, 1, 2])
            qml.CNOT(wires=[0, 2])
            qml.CZ(wires=[0, 3])

        graph = {
            "a": ["b", "c", "d"],
            "b": ["a", "c", "d"],
            "c": ["a", "b", "d"],
            "d": ["a", "b", "c"],
        }

        initial_map = {0: "a", 1: "b", 2: "c", 3: "d"}

        with pytest.raises(ValueError, match="All operations should act in less than 3 wires."):
            transformed_qfunc = qubit_mapping(qfunc, graph, initial_map)
            _ = qml.tape.make_qscript(transformed_qfunc)()

    def test_error_not_enough_qubits(self):
        """Test that an error appears if the number of logical qubits is not greater than the physical qubits"""

        def qfunc():
            qml.CNOT(wires=[2, 1])
            qml.CZ(wires=[0, 3])

        graph = {
            "a": ["b", "c"],
            "b": ["a", "c"],
            "c": ["a", "b"],
        }

        with pytest.raises(ValueError, match="Insufficient physical qubits"):
            transformed_qfunc = qubit_mapping(qfunc, graph)
            _ = qml.tape.make_qscript(transformed_qfunc)()

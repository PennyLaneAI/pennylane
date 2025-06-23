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
                    qml.Hadamard(0),
                    qml.CNOT(wires=[0, 1]),
                    qml.RY(0.5, wires=2),
                    qml.CZ(wires=[1, 3]),
                    qml.SWAP(wires=[3, 4]),
                    qml.RX(1.0, wires=5),
                    qml.CNOT(wires=[4, 6]),
                    qml.RZ(0.3, wires=7),
                    qml.CRY(0.9, wires=[6, 8]),
                ],
                [qml.expval(qml.Z(9))],
            ),
            (
                [
                    qml.PauliX(1),
                    qml.RY(1.4, wires=3),
                    qml.CNOT(wires=[3, 5]),
                    qml.CRY(1.2, wires=[5, 2]),
                    qml.SWAP(wires=[2, 6]),
                    qml.RZ(2.1, wires=4),
                    qml.CNOT(wires=[4, 7]),
                    qml.Hadamard(8),
                    qml.RX(1.8, wires=9),
                ],
                [qml.expval(qml.X(0))],
            ),
            (
                [
                    qml.Hadamard(5),
                    qml.CNOT(wires=[5, 1]),
                    qml.RY(0.9, wires=1),
                    qml.RZ(1.0, wires=2),
                    qml.CZ(wires=[2, 3]),
                    qml.CRY(1.7, wires=[3, 6]),
                    qml.SWAP(wires=[6, 0]),
                    qml.RX(2.2, wires=9),
                ],
                [qml.expval(qml.Y(4))],
            ),
            (
                [
                    qml.RX(1.1, wires=6),
                    qml.CNOT(wires=[6, 0]),
                    qml.Hadamard(0),
                    qml.RZ(2.3, wires=8),
                    qml.CRY(0.6, wires=[8, 7]),
                    qml.SWAP(wires=[7, 3]),
                    qml.CZ(wires=[3, 5]),
                    qml.RY(1.9, wires=4),
                ],
                [qml.expval(qml.Z(2))],
            ),
            (
                [
                    qml.PauliZ(9),
                    qml.CNOT(wires=[9, 8]),
                    qml.CRY(1.3, wires=[8, 7]),
                    qml.SWAP(wires=[7, 6]),
                    qml.RX(0.7, wires=5),
                    qml.CZ(wires=[5, 4]),
                    qml.RY(0.4, wires=3),
                    qml.RZ(1.5, wires=2),
                ],
                [qml.expval(qml.Y(1))],
            ),
            (
                [
                    qml.Hadamard(2),
                    qml.CRY(0.8, wires=[2, 6]),
                    qml.CNOT(wires=[6, 4]),
                    qml.SWAP(wires=[4, 1]),
                    qml.RZ(1.2, wires=1),
                    qml.RY(0.9, wires=5),
                    qml.CZ(wires=[5, 9]),
                ],
                [qml.expval(qml.X(3))],
            ),
            (
                [
                    qml.RY(1.1, wires=7),
                    qml.CNOT(wires=[7, 6]),
                    qml.PauliY(6),
                    qml.CRY(1.5, wires=[6, 3]),
                    qml.SWAP(wires=[3, 2]),
                    qml.RX(0.2, wires=2),
                    qml.Hadamard(0),
                ],
                [qml.expval(qml.Z(4))],
            ),
            (
                [
                    qml.PauliX(5),
                    qml.CNOT(wires=[5, 7]),
                    qml.RZ(2.0, wires=7),
                    qml.CZ(wires=[7, 9]),
                    qml.RX(1.6, wires=9),
                    qml.CRY(1.9, wires=[9, 1]),
                    qml.RY(1.3, wires=0),
                ],
                [qml.expval(qml.Y(3))],
            ),
            (
                [
                    qml.Hadamard(4),
                    qml.CNOT(wires=[4, 8]),
                    qml.RX(1.0, wires=8),
                    qml.CRY(0.7, wires=[8, 6]),
                    qml.SWAP(wires=[6, 0]),
                    qml.RY(2.4, wires=1),
                    qml.RZ(0.5, wires=2),
                ],
                [qml.expval(qml.Z(3))],
            ),
            (
                [
                    qml.PauliY(2),
                    qml.CZ(wires=[2, 5]),
                    qml.RY(1.0, wires=5),
                    qml.SWAP(wires=[5, 7]),
                    qml.CNOT(wires=[7, 9]),
                    qml.RX(0.9, wires=3),
                    qml.RZ(1.6, wires=6),
                ],
                [qml.expval(qml.X(0))],
            ),
            (
                [
                    qml.RX(0.3, wires=1),
                    qml.CNOT(wires=[1, 3]),
                    qml.CRY(2.1, wires=[3, 2]),
                    qml.SWAP(wires=[2, 4]),
                    qml.Hadamard(4),
                    qml.RY(1.4, wires=6),
                    qml.CZ(wires=[6, 8]),
                ],
                [qml.expval(qml.Z(9))],
            ),
            (
                [
                    qml.RY(1.5, wires=0),
                    qml.CNOT(wires=[0, 2]),
                    qml.CZ(wires=[2, 3]),
                    qml.PauliX(3),
                    qml.SWAP(wires=[3, 5]),
                    qml.CRY(0.6, wires=[5, 7]),
                    qml.RZ(2.2, wires=9),
                ],
                [qml.expval(qml.Y(8))],
            ),
            (
                [
                    qml.Hadamard(6),
                    qml.RX(1.9, wires=6),
                    qml.CNOT(wires=[6, 0]),
                    qml.SWAP(wires=[0, 1]),
                    qml.RZ(1.0, wires=1),
                    qml.CRY(1.1, wires=[1, 4]),
                    qml.RY(0.8, wires=3),
                ],
                [qml.expval(qml.X(2))],
            ),
            (
                [
                    qml.RZ(1.3, wires=8),
                    qml.CRY(1.8, wires=[8, 7]),
                    qml.CNOT(wires=[7, 6]),
                    qml.SWAP(wires=[6, 5]),
                    qml.PauliZ(5),
                    qml.RX(1.6, wires=4),
                    qml.Hadamard(3),
                ],
                [qml.expval(qml.Z(2))],
            ),
            (
                [
                    qml.RY(2.0, wires=9),
                    qml.CNOT(wires=[9, 8]),
                    qml.RZ(1.7, wires=8),
                    qml.CRY(0.4, wires=[8, 6]),
                    qml.SWAP(wires=[6, 2]),
                    qml.Hadamard(0),
                    qml.RX(1.5, wires=1),
                ],
                [qml.expval(qml.X(3))],
            ),
        ],
    )
    def test_correctness_solution(self, ops, meas):
        """Test that the output is not modified by the transform"""

        qs = qml.tape.QuantumScript(
            ops,
            meas,
        )

        graph = {
            "a": ["b", "d", "e", "f"],
            "b": ["a", "c"],
            "c": ["b"],
            "d": ["a"],
            "e": ["a"],
            "f": ["a"],
            "g": ["a"],
            "h": ["a"],
            "i": ["a"],
            "j": ["a"],
        }

        initial_map = {
            0: "a",
            1: "b",
            2: "c",
            3: "d",
            4: "e",
            5: "f",
            6: "g",
            7: "h",
            8: "i",
            9: "j",
        }
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
        graph = {"a": ["b", "d", "e"], "b": ["a", "c"], "c": ["b"], "d": ["a"], "e": ["a"]}

        transformed_qs = qubit_mapping(qs, graph)
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

    def test_mapping_logical_physical_same(self):
        """Test that if the name of all the logical and physical wires match, the initial mapping respects the positions"""

        def qfunc():
            qml.CNOT(wires=["d", "b"])
            qml.CNOT(wires=["a", "c"])
            qml.CZ(wires=["b", "d"])

        graph = {
            "a": ["b", "c", "d"],
            "b": ["a", "c", "d"],
            "c": ["a", "b", "d"],
            "d": ["a", "b", "c"],
        }

        transformed_qfunc = qubit_mapping(qfunc, graph)

        new_tape = qml.tape.make_qscript(transformed_qfunc)()

        assert new_tape.operations == [
            qml.CNOT(wires=["d", "b"]),
            qml.CNOT(wires=["a", "c"]),
            qml.CZ(wires=["b", "d"]),
        ]

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

"""Unit tests for the GateSet class."""

# pylint: disable=protected-access

import pytest

import pennylane as qml
from pennylane.decomposition.gate_set import GateSet


class TestGateSet:
    """Unit tests for the GateSet class."""

    def test_initialization_from_set(self):
        """Tests creating a GateSet from a set."""

        gate_set = GateSet({qml.RX, qml.RY, qml.CNOT})
        assert gate_set._gate_set == {"RX": 1, "RY": 1, "CNOT": 1}
        assert "RX" in gate_set
        assert qml.RX in gate_set
        assert gate_set[qml.RX] == 1

    def test_initialization_from_dict(self):
        """Tests creating a GateSet from a dictionary."""

        gate_set = GateSet({qml.RX: 1, qml.RY: 1, qml.CNOT: 2})
        assert gate_set._gate_set == {"RX": 1, "RY": 1, "CNOT": 2}
        assert "RX" in gate_set
        assert qml.RX in gate_set
        assert gate_set[qml.RX] == 1
        assert gate_set["CNOT"] == 2

    def test_gate_set_eq(self):
        """Tests comparing gate sets."""

        gate_set = GateSet({qml.RX: 1, qml.RY: 1, qml.CNOT: 1})
        gate_set_two = GateSet({"RX", "RY", "CNOT"})
        assert gate_set == gate_set_two
        assert gate_set != qml.X

    def test_gate_set_immutable(self):
        """Tests that GateSet cannot be mutated."""

        gate_set = GateSet({qml.RX, qml.RY, qml.CNOT})
        with pytest.raises(TypeError, match="immutable"):
            gate_set[qml.RZ] = 1

    def test_gate_set_negative_weights(self):
        """Tests that an error is raised when weights are negative."""

        with pytest.raises(ValueError, match="Negative weights"):
            GateSet({qml.RX: 1, qml.RY: 1, qml.CNOT: -2})

    def test_gate_set_join(self):
        """Tests joining two gate sets."""

        gate_set_one = GateSet({qml.RX, qml.RY})
        gate_set_two = GateSet({qml.CNOT})
        gate_set = gate_set_one | gate_set_two
        assert gate_set._gate_set == {"RX": 1, "RY": 1, "CNOT": 1}

    def test_gate_set_iter(self):
        """Tests that iterating over a GateSet is supported."""

        gate_set = GateSet({qml.RX: 1, qml.RY: 1, qml.CNOT: 2})
        gates = list(gate_set)
        assert gates == ["RX", "RY", "CNOT"]

    def test_gate_set_repr(self):
        """Tests the __repr__ of gate sets."""

        gate_set = GateSet({qml.RX: 1, qml.RY: 1, qml.CNOT: 2})
        assert repr(gate_set) == "{RX, RY, CNOT}"

        gate_set.name = "ROTATIONS_PLUS_CNOT"
        assert repr(gate_set) == "ROTATIONS_PLUS_CNOT"

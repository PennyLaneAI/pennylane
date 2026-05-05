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

from collections.abc import Mapping

import pytest

import pennylane as qp
from pennylane.decomposition.gate_set import GateSet


class TestGateSet:
    """Unit tests for the GateSet class."""

    def test_initialization_from_set(self):
        """Tests creating a GateSet from a set."""

        gate_set = GateSet({qp.RX, qp.RY, qp.CNOT})
        assert gate_set._gate_set == {"RX": 1, "RY": 1, "CNOT": 1}
        assert "RX" in gate_set
        assert qp.RX in gate_set
        assert gate_set[qp.RX] == 1

    def test_initialization_from_dict(self):
        """Tests creating a GateSet from a dictionary."""

        gate_set = GateSet({qp.RX: 1, qp.RY: 1, qp.CNOT: 2})
        assert gate_set._gate_set == {"RX": 1, "RY": 1, "CNOT": 2}
        assert "RX" in gate_set
        assert qp.RX in gate_set
        assert gate_set[qp.RX] == 1
        assert gate_set["CNOT"] == 2

    def test_sort_on_initialization(self):
        """Test that gates are sorted by weight and then alphabetically."""
        gateset = GateSet({"RX": 1, "Adjoint(RX)": 1, qp.CNOT: 3.0, qp.CZ: 3.0, "I": 0.0})
        assert list(gateset) == ["Identity", "Adjoint(RX)", "RX", "CNOT", "CZ"]

    def test_gate_set_eq(self):
        """Tests comparing gate sets."""

        gate_set = GateSet({qp.RX: 1, qp.RY: 1, qp.CNOT: 1})
        gate_set_two = GateSet({"RX", "RY", "CNOT"})
        assert gate_set == gate_set_two
        assert gate_set != qp.X

    def test_gate_set_is_mapping(self):
        """Tests that the GateSet is a mapping."""

        gate_set = GateSet({qp.RX: 1, qp.RY: 1, qp.CNOT: 2})
        assert isinstance(gate_set, Mapping)
        assert len(gate_set) == 3
        assert gate_set.get(qp.RX) == 1
        assert list(gate_set.keys()) == ["RX", "RY", "CNOT"]
        assert list(gate_set.values()) == [1, 1, 2]
        assert list(gate_set.items()) == [("RX", 1), ("RY", 1), ("CNOT", 2)]

    def test_gate_set_immutable(self):
        """Tests that GateSet cannot be mutated."""

        gate_set = GateSet({qp.RX, qp.RY, qp.CNOT})
        with pytest.raises(TypeError, match="immutable"):
            gate_set[qp.RZ] = 1

    def test_gate_set_negative_weights(self):
        """Tests that an error is raised when weights are negative."""

        with pytest.raises(ValueError, match="Negative weights"):
            GateSet({qp.RX: 1, qp.RY: 1, qp.CNOT: -2})

    def test_gate_set_join(self):
        """Tests joining two gate sets."""

        gate_set_one = GateSet({qp.RX, qp.RY})
        gate_set_two = GateSet({qp.CNOT})
        gate_set = gate_set_one | gate_set_two
        assert gate_set._gate_set == {"RX": 1, "RY": 1, "CNOT": 1}

        gate_set = gate_set | {qp.RZ}
        assert gate_set._gate_set == {"RX": 1, "RY": 1, "CNOT": 1, "RZ": 1}

    def test_gate_set_subtract(self):
        """Tests subtracting a set of gates from a gate set."""

        gate_set = GateSet({qp.RX, qp.RZ, qp.RY, qp.CNOT, qp.H})
        gate_set_two = gate_set - {qp.RX}
        expected = {"RZ": 1, "RY": 1, "CNOT": 1, "Hadamard": 1}
        assert gate_set_two._gate_set == expected
        assert (gate_set - {"RX"})._gate_set == expected
        assert (gate_set - qp.RX)._gate_set == expected
        assert (gate_set - GateSet({qp.RX}))._gate_set == expected

        gate_set_three = gate_set - {qp.RY, qp.S}
        assert gate_set_three._gate_set == {"RZ": 1, "RX": 1, "CNOT": 1, "Hadamard": 1}

        gs5 = gate_set - qp.S
        expected = {"RX": 1, "RY": 1, "RZ": 1, "CNOT": 1, "Hadamard": 1}
        assert gs5._gate_set == expected
        assert (gate_set - "S")._gate_set == expected

    def test_gate_set_unsupported_arithmetic(self):
        """Tests that a TypeError is raised."""

        gate_set = GateSet({qp.RZ, qp.RY})

        with pytest.raises(TypeError):
            _ = gate_set - 1

        with pytest.raises(TypeError):
            _ = gate_set | 1

    def test_gate_set_iter(self):
        """Tests that iterating over a GateSet is supported."""

        gate_set = GateSet({qp.RX: 1, qp.RY: 1, qp.CNOT: 2})
        gates = list(gate_set)
        assert gates == ["RX", "RY", "CNOT"]

    def test_gate_set_str(self):
        """Tests the __str__ of gate sets."""

        gate_set = GateSet({qp.RX: 1, qp.RY: 1, qp.CNOT: 2})
        assert str(gate_set) == "{RX, RY, CNOT=2}"

        gate_set.name = "ROTATIONS_PLUS_CNOT"
        assert str(gate_set) == "ROTATIONS_PLUS_CNOT"

    def test_gate_set_repr(self):
        """Tests the __repr__ of gate sets."""

        gate_set = GateSet({qp.RX: 1, qp.RY: 1, qp.CNOT: 2})
        assert repr(gate_set) == "GateSet({RX, RY, CNOT=2})"

        gate_set.name = "ROTATIONS_PLUS_CNOT"
        assert repr(gate_set) == "GateSet({RX, RY, CNOT=2}, name='ROTATIONS_PLUS_CNOT')"

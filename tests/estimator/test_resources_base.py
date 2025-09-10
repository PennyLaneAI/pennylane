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
This module contains tests for the Resources container class.
"""
from collections import defaultdict
from dataclasses import dataclass

import pytest

from pennylane.estimator.resources_base import Resources
from pennylane.estimator.wires_manager import WireResourceManager

# pylint: disable= no-self-use,too-few-public-methods,comparison-with-itself


@dataclass(frozen=True)
class DummyResOp:
    """A dummy class to populate the gate types dictionary for testing."""

    name: str


h = DummyResOp("Hadamard")
x = DummyResOp("X")
y = DummyResOp("Y")
z = DummyResOp("Z")
cnot = DummyResOp("CNOT")
phase_shift = DummyResOp("PhaseShift")

gate_types_data = (
    defaultdict(
        int,
        {h: 2, x: 1, z: 1},
    ),
    defaultdict(
        int,
        {h: 467, cnot: 791, phase_shift: 2},
    ),
    defaultdict(
        int,
        {x: 100, y: 120, z: 1000, cnot: 4523},
    ),
)

wire_manager1 = WireResourceManager(zeroed=5)
wire_manager2 = WireResourceManager(zeroed=8753, any_state=2347, algo=22)
wire_manager3 = WireResourceManager(zeroed=400, any_state=222, algo=108)

wire_manager_data = (wire_manager1, wire_manager2, wire_manager3)


class TestResources:
    """Test the Resources class"""

    @pytest.mark.parametrize("gate_types", gate_types_data + (None,))
    @pytest.mark.parametrize("wire_manager", wire_manager_data)
    def test_init(self, wire_manager, gate_types):
        """Test that the class is correctly initialized"""
        resources = Resources(wire_manager=wire_manager, gate_types=gate_types)

        expected_wire_manager = wire_manager
        expected_gate_types = defaultdict(int, {}) if gate_types is None else gate_types

        assert resources.wire_manager == expected_wire_manager
        assert resources.gate_types == expected_gate_types

    str_data = (
        (
            "--- Resources: ---\n"
            + " Total wires: 5\n"
            + "    algorithmic wires: 0\n"
            + "    allocated wires: 5\n"
            + "\t zero state: 5\n"
            + "\t any state: 0\n"
            + " Total gates : 4\n"
            + "  'X': 1,\n"
            + "  'Z': 1,\n"
            + "  'Hadamard': 2"
        ),
        (
            "--- Resources: ---\n"
            + " Total wires: 1.112E+4\n"
            + "    algorithmic wires: 22\n"
            + "    allocated wires: 11100\n"
            + "\t zero state: 8753\n"
            + "\t any state: 2347\n"
            + " Total gates : 1.260E+3\n"
            + "  'PhaseShift': 2,\n"
            + "  'CNOT': 791,\n"
            + "  'Hadamard': 467"
        ),
        (
            "--- Resources: ---\n"
            + " Total wires: 730\n"
            + "    algorithmic wires: 108\n"
            + "    allocated wires: 622\n"
            + "\t zero state: 400\n"
            + "\t any state: 222\n"
            + " Total gates : 5.743E+3\n"
            + "  'CNOT': 4.523E+3,\n"
            + "  'X': 100,\n"
            + "  'Y': 120,\n"
            + "  'Z': 1.000E+3"
        ),
    )

    @pytest.mark.parametrize(
        "resources, expected_str",
        zip(
            tuple(
                Resources(wire_manager, gate_types)
                for wire_manager, gate_types in zip(wire_manager_data, gate_types_data)
            ),
            str_data,
        ),
    )
    def test_str_method(self, resources, expected_str):
        """Test that the str method correctly displays the information."""
        assert str(resources) == expected_str

    @pytest.mark.parametrize("gate_types", gate_types_data + (None,))
    @pytest.mark.parametrize("wire_manager", wire_manager_data)
    def test_repr_method(self, gate_types, wire_manager):
        """Test that the repr method correctly represents the class."""
        resources = Resources(wire_manager=wire_manager, gate_types=gate_types)

        expected_wire_manager = wire_manager
        expected_gate_types = defaultdict(int, {}) if gate_types is None else gate_types
        assert (
            repr(resources)
            == f"Resources(wire_manager={expected_wire_manager}, gate_types={expected_gate_types})"
        )

    def test_gate_counts(self):
        """Test that this function correctly simplifies the gate types
        dictionary by grouping together gates with the same name."""

        class DummyResOp2:
            """A dummy class to populate the gate types dictionary for testing."""

            def __init__(self, name, parameter=None):
                """Initialize dummy class."""
                self.name = name
                self.parameter = parameter

            def __hash__(self):
                """Custom hash which only depends on instance name."""
                return hash((self.name, self.parameter))

        rx1 = DummyResOp2("RX", parameter=3.14)
        rx2 = DummyResOp2("RX", parameter=3.14 / 2)
        cnots = DummyResOp2("CNOT")
        ry1 = DummyResOp2("RY", parameter=3.14)
        ry2 = DummyResOp2("RY", parameter=3.14 / 4)

        gate_types = {rx1: 1, ry1: 2, cnots: 3, rx2: 4, ry2: 5}
        res = Resources(wire_manager=wire_manager1, gate_types=gate_types)

        expected_gate_counts = {"RX": 5, "RY": 7, "CNOT": 3}
        assert res.gate_counts == expected_gate_counts

    def test_equality(self):
        """Test that the equality method works as expected."""
        gate_types1, gate_types2 = (gate_types_data[0], gate_types_data[1])

        res1 = Resources(wire_manager=wire_manager1, gate_types=gate_types1)
        res1_copy = Resources(wire_manager=wire_manager1, gate_types=gate_types1)
        res2 = Resources(wire_manager=wire_manager2, gate_types=gate_types2)

        assert res1 == res1
        assert res1 == res1_copy
        assert res1 != res2

    def test_arithmetic_raises_error(self):
        """Test that an assertion error is raised when arithmetic methods are used"""
        res = Resources(wire_manager=wire_manager1, gate_types=gate_types_data[0])

        with pytest.raises(AssertionError):
            res.add_series(2)  # Can only add two Resources instances

        with pytest.raises(AssertionError):
            res.add_parallel(2)  # Can only add two Resources instances

        with pytest.raises(AssertionError):
            res.multiply_series(res)  # Can only multiply a Resources instance with an int

        with pytest.raises(AssertionError):
            res.multiply_parallel(res)  # Can only multiply a Resources instance with an int

    def test_add_in_series(self):
        """Test that we can add two resources assuming the gates occur in series"""
        res1 = Resources(wire_manager=wire_manager3, gate_types=gate_types_data[2])
        res2 = Resources(wire_manager=wire_manager2, gate_types=gate_types_data[1])

        expected_wire_manager_add = WireResourceManager(
            zeroed=max(wire_manager3.zeroed, wire_manager2.zeroed),
            any_state=wire_manager2.any_state + wire_manager3.any_state,
            algo=max(wire_manager3.algo_wires, wire_manager2.algo_wires),
        )
        expected_gate_types_add = defaultdict(
            int,
            {
                key: gate_types_data[2].get(key, 0) + gate_types_data[1].get(key, 0)
                for key in gate_types_data[2].keys() | gate_types_data[1].keys()
            },
        )

        expected_add = Resources(expected_wire_manager_add, expected_gate_types_add)
        assert res1.add_series(res2) == expected_add

    def test_add_in_parallel(self):
        """Test that we can add two resources assuming the gates occur in parallel"""
        res1 = Resources(wire_manager=wire_manager3, gate_types=gate_types_data[2])
        res2 = Resources(wire_manager=wire_manager2, gate_types=gate_types_data[1])

        expected_wire_manager_and = WireResourceManager(
            zeroed=max(wire_manager3.zeroed, wire_manager2.zeroed),
            any_state=wire_manager3.any_state + wire_manager2.any_state,
            algo=wire_manager3.algo_wires + wire_manager2.algo_wires,
        )
        expected_gate_types_and = defaultdict(
            int,
            {
                key: gate_types_data[2].get(key, 0) + gate_types_data[1].get(key, 0)
                for key in gate_types_data[2].keys() | gate_types_data[1].keys()
            },
        )

        expected_and = Resources(expected_wire_manager_and, expected_gate_types_and)
        assert (res1.add_parallel(res2)) == expected_and

    def test_mul_in_series(self):
        """Test that we can scale resources by an integer assuming the gates occur in series"""
        k = 3
        res = Resources(wire_manager=wire_manager3, gate_types=gate_types_data[2])

        expected_wire_manager_mul = WireResourceManager(
            zeroed=wire_manager3.zeroed,
            any_state=wire_manager3.any_state * k,
            algo=wire_manager3.algo_wires,
        )
        expected_gate_types_mul = defaultdict(
            int, {key: value * k for key, value in gate_types_data[2].items()}
        )

        expected_mul = Resources(expected_wire_manager_mul, expected_gate_types_mul)

        assert (res.multiply_series(k)) == expected_mul

    def test_mul_in_parallel(self):
        """Test that we can scale resources by an integer assuming the gates occur in parallel"""
        k = 3
        res = Resources(wire_manager=wire_manager3, gate_types=gate_types_data[2])

        expected_wire_manager_matmul = WireResourceManager(
            zeroed=wire_manager3.zeroed,
            any_state=wire_manager3.any_state * k,
            algo=wire_manager3.algo_wires * k,
        )
        expected_gate_types_matmul = defaultdict(
            int, {key: value * k for key, value in gate_types_data[2].items()}
        )

        expected_matmul = Resources(expected_wire_manager_matmul, expected_gate_types_matmul)

        assert (res.multiply_parallel(k)) == expected_matmul

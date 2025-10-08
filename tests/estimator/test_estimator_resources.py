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
from dataclasses import dataclass, field

import pytest

from pennylane.estimator.resources_base import Resources

# pylint: disable= no-self-use,too-few-public-methods,comparison-with-itself


@dataclass(frozen=True)
class DummyResOp:
    """A dummy class to populate the gate types dictionary for testing."""

    name: str
    params: dict = field(default_factory=lambda: defaultdict(int), compare=False)


h = DummyResOp("Hadamard")
x = DummyResOp("X")
y = DummyResOp("Y")
z = DummyResOp("Z")
cnot = DummyResOp("CNOT")
phase_shift = DummyResOp("PhaseShift", {"precision": 1.57})
rx = DummyResOp("RX", {"precision": 1e-8})
rx_2 = DummyResOp("RX", {"precision": 1e-6})

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
        {x: 100, y: 120, z: 1000, cnot: 4523, rx: 1, rx_2: 3},
    ),
)

wire_info1 = {"zeroed": 5}
wire_info2 = {"zeroed": 8753, "any_state": 2347, "algo": 22}
wire_info3 = {"zeroed": 400, "any_state": 222, "algo": 108}

wire_data = (wire_info1, wire_info2, wire_info3)


class TestResources:
    """Test the Resources class"""

    @pytest.mark.parametrize("gate_types", gate_types_data + (None,))
    @pytest.mark.parametrize("wire_info", wire_data)
    def test_init(self, wire_info, gate_types):
        """Test that the class is correctly initialized"""
        zeroed = wire_info.get("zeroed", 0)
        any_state = wire_info.get("any_state", 0)
        algo = wire_info.get("algo", 0)
        resources = Resources(zeroed, any_state, algo, gate_types=gate_types)

        expected_gate_types = defaultdict(int, {}) if gate_types is None else gate_types

        assert resources.zeroed_wires == zeroed
        assert resources.any_state_wires == any_state
        assert resources.algo_wires == algo
        assert resources.gate_types == expected_gate_types

    str_data = (
        (
            "--- Resources: ---\n"
            + " Total wires: 5\n"
            + "   algorithmic wires: 0\n"
            + "   allocated wires: 5\n"
            + "     zero state: 5\n"
            + "     any state: 0\n"
            + " Total gates : 4\n"
            + "   'X': 1,\n"
            + "   'Z': 1,\n"
            + "   'Hadamard': 2"
        ),
        (
            "--- Resources: ---\n"
            + " Total wires: 1.112E+4\n"
            + "   algorithmic wires: 22\n"
            + "   allocated wires: 11100\n"
            + "     zero state: 8753\n"
            + "     any state: 2347\n"
            + " Total gates : 1.260E+3\n"
            + "   'PhaseShift': 2,\n"
            + "   'CNOT': 791,\n"
            + "   'Hadamard': 467"
        ),
        (
            "--- Resources: ---\n"
            + " Total wires: 730\n"
            + "   algorithmic wires: 108\n"
            + "   allocated wires: 622\n"
            + "     zero state: 400\n"
            + "     any state: 222\n"
            + " Total gates : 5.746E+3\n"
            + "   'RX': 3,\n"
            + "   'CNOT': 4.523E+3,\n"
            + "   'X': 100,\n"
            + "   'Y': 120,\n"
            + "   'Z': 1.000E+3"
        ),
    )

    @pytest.mark.parametrize(
        "resources, expected_str",
        zip(
            tuple(
                Resources(
                    wire_info.get("zeroed", 0),
                    wire_info.get("any_state", 0),
                    wire_info.get("algo", 0),
                    gate_types,
                )
                for wire_info, gate_types in zip(wire_data, gate_types_data)
            ),
            str_data,
        ),
    )
    def test_str_method(self, resources, expected_str):
        """Test that the str method correctly displays the information."""
        assert str(resources) == expected_str

    test_data = (
        (gate_types_data[0], None, "X total: 1\n" + "Z total: 1\n" + "Hadamard total: 2"),
        (
            gate_types_data[1],
            None,  # Test the default behavior
            "PhaseShift total: 2\n"
            + "    PhaseShift {'precision': 1.57}: 2\n"
            + "CNOT total: 791\n"
            + "Hadamard total: 467",
        ),
        (
            gate_types_data[2],
            ["X", "Y", "RX", "Toffoli"],  # Test with a custom gate set
            "X total: 100\n"
            + "Y total: 120\n"
            + "RX total: 3\n"
            + "    RX {'precision': 1e-08}: 3",
        ),
    )

    @pytest.mark.parametrize("gate_types, gate_set, expected_str", test_data)
    def test_gate_breakdown(self, gate_types, gate_set, expected_str):
        """Test that the gate_breakdown method correctly displays the information."""
        resources = Resources(zeroed_wires=4, gate_types=gate_types)
        assert resources.gate_breakdown(gate_set=gate_set) == expected_str

    @pytest.mark.parametrize("gate_types", gate_types_data + (None,))
    @pytest.mark.parametrize("wire_info", wire_data)
    def test_repr_method(self, gate_types, wire_info):
        """Test that the repr method correctly represents the class."""
        zeroed = wire_info.get("zeroed", 0)
        any_state = wire_info.get("any_state", 0)
        algo = wire_info.get("algo", 0)
        resources = Resources(zeroed, any_state, algo, gate_types=gate_types)

        expected_gate_types = defaultdict(int, {}) if gate_types is None else gate_types
        assert (
            repr(resources)
            == f"Resources(zeroed={zeroed}, any_state={any_state}, algo_wires={algo}, gate_types={expected_gate_types})"
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
        res = Resources(5, 0, 0, gate_types=gate_types)

        expected_gate_counts = {"RX": 5, "RY": 7, "CNOT": 3}
        assert res.gate_counts == expected_gate_counts

    def test_equality(self):
        """Test that the equality method works as expected."""
        gate_types1, gate_types2 = (gate_types_data[0], gate_types_data[1])

        zeroed = wire_info1.get("zeroed", 0)
        any_state = wire_info1.get("any_state", 0)
        algo = wire_info1.get("algo", 0)
        res1 = Resources(zeroed, any_state, algo, gate_types=gate_types1)
        res1_copy = Resources(zeroed, any_state, algo, gate_types=gate_types1)
        res2 = Resources(5, 4, 3, gate_types=gate_types2)

        assert res1 == res1
        assert res1 == res1_copy
        assert res1 != res2

    def test_equality_error(self):
        """Test that the equality method raises an error."""
        res1 = Resources(3, 0, 2, gate_types=gate_types_data[0])
        with pytest.raises(
            TypeError,
            match="Cannot compare Resources with object of type <class 'collections.defaultdict'>.",
        ):
            assert res1 == gate_types_data[0]

    def test_arithmetic_raises_error(self):
        """Test that an assertion error is raised when arithmetic methods are used"""
        res = Resources(4, 0, 0, gate_types=gate_types_data[0])

        with pytest.raises(TypeError, match="Cannot add Resources object to <class 'int'>."):
            res.add_series(2)  # Can only add two Resources instances

        with pytest.raises(TypeError, match="Cannot add Resources object to <class 'int'>."):
            res.add_parallel(2)  # Can only add two Resources instances

        with pytest.raises(
            TypeError,
            match="Cannot multiply Resources object with <class 'pennylane.estimator.resources_base.Resources'>.",
        ):
            res.multiply_series(res)  # Can only multiply a Resources instance with an int

        with pytest.raises(
            TypeError,
            match="Cannot multiply Resources object with <class 'pennylane.estimator.resources_base.Resources'>.",
        ):
            res.multiply_parallel(res)  # Can only multiply a Resources instance with an int

    def test_add_in_series(self):
        """Test that we can add two resources assuming the gates occur in series"""
        zeroed1 = wire_info3.get("zeroed", 0)
        any_state1 = wire_info3.get("any_state", 0)
        algo1 = wire_info3.get("algo", 0)

        zeroed2 = wire_info2.get("zeroed", 0)
        any_state2 = wire_info2.get("any_state", 0)
        algo2 = wire_info2.get("algo", 0)

        res1 = Resources(zeroed1, any_state1, algo1, gate_types=gate_types_data[2])
        res2 = Resources(zeroed2, any_state2, algo2, gate_types=gate_types_data[1])

        zeroed = max(zeroed1, zeroed2)
        any_state = any_state1 + any_state2
        algo = max(algo1, algo2)
        expected_gate_types_add = defaultdict(
            int,
            {
                key: gate_types_data[2].get(key, 0) + gate_types_data[1].get(key, 0)
                for key in gate_types_data[2].keys() | gate_types_data[1].keys()
            },
        )

        expected_add = Resources(zeroed, any_state, algo, expected_gate_types_add)
        assert res1.add_series(res2) == expected_add

    def test_add_in_parallel(self):
        """Test that we can add two resources assuming the gates occur in parallel"""
        zeroed1 = wire_info3.get("zeroed", 0)
        any_state1 = wire_info3.get("any_state", 0)
        algo1 = wire_info3.get("algo", 0)

        zeroed2 = wire_info2.get("zeroed", 0)
        any_state2 = wire_info2.get("any_state", 0)
        algo2 = wire_info2.get("algo", 0)

        res1 = Resources(zeroed1, any_state1, algo1, gate_types=gate_types_data[2])
        res2 = Resources(zeroed2, any_state2, algo2, gate_types=gate_types_data[1])

        zeroed = max(zeroed1, zeroed2)
        any_state = any_state1 + any_state2
        algo = algo1 + algo2

        expected_gate_types_and = defaultdict(
            int,
            {
                key: gate_types_data[2].get(key, 0) + gate_types_data[1].get(key, 0)
                for key in gate_types_data[2].keys() | gate_types_data[1].keys()
            },
        )

        expected_and = Resources(zeroed, any_state, algo, expected_gate_types_and)
        assert (res1.add_parallel(res2)) == expected_and

    def test_mul_in_series(self):
        """Test that we can scale resources by an integer assuming the gates occur in series"""
        k = 3
        zeroed = wire_info3.get("zeroed", 0)
        any_state = wire_info3.get("any_state", 0)
        algo = wire_info3.get("algo", 0)
        res = Resources(zeroed, any_state, algo, gate_types=gate_types_data[2])

        expected_gate_types_mul = defaultdict(
            int, {key: value * k for key, value in gate_types_data[2].items()}
        )

        expected_mul = Resources(zeroed, any_state * k, algo, expected_gate_types_mul)

        assert (res.multiply_series(k)) == expected_mul

    def test_mul_in_parallel(self):
        """Test that we can scale resources by an integer assuming the gates occur in parallel"""
        k = 3
        zeroed = wire_info3.get("zeroed", 0)
        any_state = wire_info3.get("any_state", 0)
        algo = wire_info3.get("algo", 0)
        res = Resources(zeroed, any_state, algo, gate_types=gate_types_data[2])

        expected_gate_types_matmul = defaultdict(
            int, {key: value * k for key, value in gate_types_data[2].items()}
        )

        expected_matmul = Resources(zeroed, any_state * k, algo * k, expected_gate_types_matmul)

        assert (res.multiply_parallel(k)) == expected_matmul

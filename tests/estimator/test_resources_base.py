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

from pennylane.estimator.resources_base import (
    Resources,
    add_in_parallel,
    add_in_series,
    mul_in_parallel,
    mul_in_series,
)
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

wm1 = WireResourceManager(clean=5)
wm2 = WireResourceManager(clean=8753, dirty=2347, algo=22)
wm3 = WireResourceManager(clean=400, dirty=222, algo=108)

wire_manager_data = (wm1, wm2, wm3)


class TestResources:
    """Test the Resources class"""

    @pytest.mark.parametrize("gate_types", gate_types_data + (None,))
    @pytest.mark.parametrize("wm", wire_manager_data)
    def test_init(self, wm, gate_types):
        """Test that the class is correctly initialized"""
        resources = Resources(wire_manager=wm, gate_types=gate_types)

        expected_wm = wm
        expected_gate_types = defaultdict(int, {}) if gate_types is None else gate_types

        assert resources.wire_manager == expected_wm
        assert resources.gate_types == expected_gate_types

    str_data = (
        (
            "--- Resources: ---\n"
            + " Total wires: 5\n"
            + "    algorithmic wires: 0\n"
            + "    allocated wires: 5\n"
            + "\t clean wires: 5\n"
            + "\t dirty wires: 0\n"
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
            + "\t clean wires: 8753\n"
            + "\t dirty wires: 2347\n"
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
            + "\t clean wires: 400\n"
            + "\t dirty wires: 222\n"
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
                Resources(wm, gate_types)
                for wm, gate_types in zip(wire_manager_data, gate_types_data)
            ),
            str_data,
        ),
    )
    def test_str_method(self, resources, expected_str):
        """Test that the str method correctly displays the information."""
        assert str(resources) == expected_str

    @pytest.mark.parametrize("gate_types", gate_types_data + (None,))
    @pytest.mark.parametrize("wm", wire_manager_data)
    def test_repr_method(self, gate_types, wm):
        """Test that the repr method correctly represents the class."""
        resources = Resources(wire_manager=wm, gate_types=gate_types)

        expected_wm = wm
        expected_gate_types = defaultdict(int, {}) if gate_types is None else gate_types
        assert (
            repr(resources)
            == f"Resources(wire_manager={expected_wm}, gate_types={expected_gate_types})"
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
        res = Resources(wire_manager=wm1, gate_types=gate_types)

        expected_gate_counts = {"RX": 5, "RY": 7, "CNOT": 3}
        assert res.gate_counts == expected_gate_counts

    def test_equality(self):
        """Test that the equality method works as expected."""
        gate_types1, gate_types2 = (gate_types_data[0], gate_types_data[1])

        res1 = Resources(wire_manager=wm1, gate_types=gate_types1)
        res1_copy = Resources(wire_manager=wm1, gate_types=gate_types1)
        res2 = Resources(wire_manager=wm2, gate_types=gate_types2)

        assert res1 == res1
        assert res1 == res1_copy
        assert res1 != res2

    def test_arithmetic_raises_error(self):
        """Test that an assertion error is raised when arithmetic methods are used"""
        res = Resources(wire_manager=wm1, gate_types=gate_types_data[0])

        with pytest.raises(AssertionError):
            _ = res + 2  # Can only add two Resources instances

        with pytest.raises(AssertionError):
            _ = res & 2  # Can only add two Resources instances

        with pytest.raises(AssertionError):
            _ = res * res  # Can only multiply a Resources instance with an int

        with pytest.raises(AssertionError):
            _ = res @ res  # Can only multiply a Resources instance with an int

    def test_add_in_series(self):
        """Test that we can add two resources assuming the gates occur in series"""
        res1 = Resources(wire_manager=wm3, gate_types=gate_types_data[2])
        res2 = Resources(wire_manager=wm2, gate_types=gate_types_data[1])

        expected_wm_add = WireResourceManager(
            clean=8753,  # max(clean1, clean2)
            dirty=2569,  # dirty1 + dirty2
        )
        expected_wm_add.algo_wires = 108  # max(algo1, algo2)
        expected_gate_types_add = defaultdict(
            int,
            {h: 467, x: 100, y: 120, z: 1000, cnot: 5314, phase_shift: 2},  # add gate counts
        )

        expected_add = Resources(expected_wm_add, expected_gate_types_add)
        assert (res1 + res2) == expected_add
        assert add_in_series(res1, res2) == expected_add

    def test_add_in_parallel(self):
        """Test that we can add two resources assuming the gates occur in parallel"""
        res1 = Resources(wire_manager=wm3, gate_types=gate_types_data[2])
        res2 = Resources(wire_manager=wm2, gate_types=gate_types_data[1])

        expected_wm_and = WireResourceManager(
            clean=8753,  # max(clean1, clean2)
            dirty=2569,  # dirty1 + dirty2
        )
        expected_wm_and.algo_wires = 130  # algo1 + algo2
        expected_gate_types_and = defaultdict(
            int,
            {h: 467, x: 100, y: 120, z: 1000, cnot: 5314, phase_shift: 2},  # add gate counts
        )

        expected_and = Resources(expected_wm_and, expected_gate_types_and)
        assert (res1 & res2) == expected_and
        assert add_in_parallel(res1, res2) == expected_and

    def test_mul_in_series(self):
        """Test that we can scale resources by an integer assuming the gates occur in series"""
        k = 3
        res = Resources(wire_manager=wm3, gate_types=gate_types_data[2])

        expected_wm_mul = WireResourceManager(
            clean=400,  # clean
            dirty=222 * k,  # k * dirty1
        )
        expected_wm_mul.algo_wires = 108  # algo
        expected_gate_types_mul = defaultdict(
            int,
            {x: 100 * k, y: 120 * k, z: 1000 * k, cnot: 4523 * k},  # multiply gate counts
        )

        expected_mul = Resources(expected_wm_mul, expected_gate_types_mul)

        assert (k * res) == expected_mul
        assert (res * k) == expected_mul
        assert mul_in_series(res, k) == expected_mul

    def test_mul_in_parallel(self):
        """Test that we can scale resources by an integer assuming the gates occur in parallel"""
        k = 3
        res = Resources(wire_manager=wm3, gate_types=gate_types_data[2])

        expected_wm_matmul = WireResourceManager(
            clean=400,  # clean
            dirty=222 * k,  # k * dirty1
        )
        expected_wm_matmul.algo_wires = 108 * k  # k * algo
        expected_gate_types_matmul = defaultdict(
            int,
            {x: 100 * k, y: 120 * k, z: 1000 * k, cnot: 4523 * k},  # multiply gate counts
        )

        expected_matmul = Resources(expected_wm_matmul, expected_gate_types_matmul)
        assert (k @ res) == expected_matmul
        assert (res @ k) == expected_matmul
        assert mul_in_parallel(res, k) == expected_matmul

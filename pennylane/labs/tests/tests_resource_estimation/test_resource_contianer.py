# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Test base Resource class and its associated methods
"""

import pytest
from collections import defaultdict

from pennylane.labs.resource_estimation import Resources, OpTypeWithParams


class TestResources:
    """Test the methods and attributes of the Resource class"""

    resource_quantities = (
        Resources(),
        Resources(5, 0, defaultdict(int)),
        Resources(
            1, 3, defaultdict(
                int, 
                {OpTypeWithParams("Hadamard", (("num_wires", 1),)): 1, OpTypeWithParams("PauliZ", (("num_wires", 1),)): 2},
                )
            ),
        Resources(
            4, 2, defaultdict(
                int, 
                {OpTypeWithParams("Hadamard", (("num_wires", 1),)): 1, OpTypeWithParams("CNOT", (("num_wires", 2),)): 1},
                )
            ),
    )

    resource_parameters = (
        (0, 0, defaultdict(int)),
        (5, 0, defaultdict(int)),
        (1, 3, defaultdict(int, {OpTypeWithParams("Hadamard", (("num_wires", 1),)): 1, OpTypeWithParams("PauliZ", (("num_wires", 1),)): 2})),
        (4, 2, defaultdict(int, {OpTypeWithParams("Hadamard", (("num_wires", 1),)): 1, OpTypeWithParams("CNOT", (("num_wires", 2),)): 1})),
    )

    @pytest.mark.parametrize("r, attribute_tup", zip(resource_quantities, resource_parameters))
    def test_init(self, r, attribute_tup):
        """Test that the Resource class is instantiated as expected."""
        num_wires, num_gates, gate_types = attribute_tup

        assert r.num_wires == num_wires
        assert r.num_gates == num_gates

        for key, value in gate_types.items():
            assert r.gate_types[key] == value
        # assert r.gate_types == gate_types

    test_str_data = (
        (
            "wires: 0\n"
            + "gates: 0\n"
            + "gate_types:\n"
            + "{}"
        ),
        (
            "wires: 5\n"
            + "gates: 0\n"
            + "gate_types:\n"
            + "{}"
        ),
        (
            "wires: 1\n"
            + "gates: 3\n"
            + "gate_types:\n"
            + "{'Hadamard(num_wires=1)': 1, 'PauliZ(num_wires=1)': 2}"
        ),
        (
            "wires: 4\n"
            + "gates: 2\n"
            + "gate_types:\n"
            + "{'Hadamard(num_wires=1)': 1, 'CNOT(num_wires=2)': 1}"
        ),
    )

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_str_data))
    def test_str(self, r, rep):
        """Test the string representation of a Resources instance."""
        assert str(r) == rep

    test_rep_data = (
        "Resources(num_wires=0, num_gates=0, gate_types=defaultdict(<class 'int'>, {}))",
        "Resources(num_wires=5, num_gates=0, gate_types=defaultdict(<class 'int'>, {}))",
        "Resources(num_wires=1, num_gates=3, gate_types=defaultdict(<class 'int'>, {'Hadamard(num_wires=1)': 1, 'PauliZ(num_wires=1)': 2}))",
        "Resources(num_wires=4, num_gates=2, gate_types=defaultdict(<class 'int'>, {'Hadamard(num_wires=1)': 1, 'CNOT(num_wires=2)': 1}))",
    )

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_rep_data))
    def test_repr(self, r, rep):
        """Test the repr method of a Resources instance looks as expected."""
        assert repr(r) == rep

    def test_eq(self):
        """Test that the equality dunder method is correct for Resources."""
        r1 = Resources(4, 2, defaultdict(int, {OpTypeWithParams("Hadamard", (("num_wires", 1),)): 1, OpTypeWithParams("CNOT", (("num_wires", 2),)): 1}))
        r2 = Resources(4, 2, defaultdict(int, {OpTypeWithParams("CNOT", (("num_wires", 2),)): 1, OpTypeWithParams("Hadamard", (("num_wires", 1),)): 1}))  # all equal

        r3 = Resources(1, 2, defaultdict(int, {OpTypeWithParams("Hadamard", (("num_wires", 1),)): 1, OpTypeWithParams("CNOT", (("num_wires", 2),)): 1}))  # diff wires
        r4 = Resources(4, 1, defaultdict(int, {OpTypeWithParams("Hadamard", (("num_wires", 1),)): 1, OpTypeWithParams("CNOT", (("num_wires", 2),)): 1}))  # diff num_gates
        r5 = Resources(4, 2, defaultdict(int, {OpTypeWithParams("CNOT", (("num_wires", 2),)): 1}))  # diff gate_types


        assert r1.__eq__(r1)
        assert r1.__eq__(r2)

        assert not r1.__eq__(r3)
        assert not r1.__eq__(r4)
        assert not r1.__eq__(r5)

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_str_data))
    def test_ipython_display(self, r, rep, capsys):
        """Test that the ipython display prints the string representation of a Resources instance."""
        r._ipython_display_()  # pylint: disable=protected-access
        captured = capsys.readouterr()
        assert rep in captured.out
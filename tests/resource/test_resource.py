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
Test base Resource class and its associated methods
"""

import pytest
from pennylane.resource import Resources


class TestResources:
    """Test the methods and attributes of the Resource class"""

    resource_quantities = (
        Resources(),
        Resources(5, 0, {}, 0, 0),
        Resources(1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
        Resources(4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100),
    )

    resource_parameters = (
        (0, 0, {}, 0, 0),
        (5, 0, {}, 0, 0),
        (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
        (4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100),
    )

    @pytest.mark.parametrize("r, attribute_tup", zip(resource_quantities, resource_parameters))
    def test_init(self, r, attribute_tup):
        """Test that the Resource class is instantiated as expected."""
        num_wires, num_gates, gate_types, depth, shots = attribute_tup

        assert r.num_wires == num_wires
        assert r.num_gates == num_gates
        assert r.depth == depth
        assert r.shots == shots
        assert r.gate_types == gate_types

    type_error = (
        (0, 1, "wrong_type", 2, 3),
        (0.0, 1.1, {"Hadamard": 1}, 2.2, 3.3),
        ("0", [1], {"Identity": 1}, True, False),
    )

    @pytest.mark.parametrize("params", type_error)
    def test_init_type_error(self, params):
        """Test that a type error is raised if an attribute is initialized with the wrong type."""
        with pytest.raises(TypeError, match="Incorrect type of input,"):
            Resources(*params)

    value_error = ((0, -1, {"Hadamard": 1}, 2, -3),)

    @pytest.mark.parametrize("params", value_error)
    def test_init_value_error(self, params):
        """Test that a value error is raised if an attribute is initialized with the wrong type."""
        with pytest.raises(ValueError, match="Incorrect value of input,"):
            Resources(*params)

    def test_set_attributes_error(self):
        """Test that an error is raised if we try to set any attribute."""
        r = Resources()
        attr_lst = ["num_wires", "num_gates", "depth", "shots", "gate_types"]
        message = (
            "<class 'pennylane.resource.resource.Resources'> object does not support assignment"
        )

        for attr_name in attr_lst:
            with pytest.raises(AttributeError, match=message):
                setattr(r, attr_name, 1)

    test_str_data = (
        ("wires: 0\n" + "gates: 0\n" + "depth: 0\n" + "shots: 0\n" + "gate_types: \n" + "{}"),
        ("wires: 5\n" + "gates: 0\n" + "depth: 0\n" + "shots: 0\n" + "gate_types: \n" + "{}"),
        (
            "wires: 1\n"
            + "gates: 3\n"
            + "depth: 3\n"
            + "shots: 10\n"
            + "gate_types: \n"
            + "{'Hadamard': 1, 'PauliZ': 2}"
        ),
        (
            "wires: 4\n"
            + "gates: 2\n"
            + "depth: 2\n"
            + "shots: 100\n"
            + "gate_types: \n"
            + "{'Hadamard': 1, 'CNOT': 1}"
        ),
    )

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_str_data))
    def test_to_str(self, r, rep):
        """Test the string representation of a Resources instance."""
        assert str(r) == rep

    test_rep_data = (
        "<Resource: wires=0, gates=0, depth=0, shots=0, gate_types={}>",
        "<Resource: wires=5, gates=0, depth=0, shots=0, gate_types={}>",
        "<Resource: wires=1, gates=3, depth=3, shots=10, "
        "gate_types={'Hadamard': 1, 'PauliZ': 2}>",
        "<Resource: wires=4, gates=2, depth=2, shots=100, "
        "gate_types={'Hadamard': 1, 'CNOT': 1}>",
    )

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_rep_data))
    def test_repr(self, r, rep):
        """Test the repr method of a Resources instance looks as expected."""
        assert repr(r) == rep

    def test_eq(self):
        """Test that the equality dunder method is correct for Resources."""
        r1 = Resources(4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100)
        r2 = Resources(4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100)
        r3 = Resources(4, 2, {"CNOT": 1, "Hadamard": 1}, 2, 100)  # all equal

        r4 = Resources(1, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100)  # diff wires
        r5 = Resources(4, 1, {"Hadamard": 1, "CNOT": 1}, 2, 100)  # diff num_gates
        r6 = Resources(4, 2, {"CNOT": 1}, 2, 100)  # diff gate_types
        r7 = Resources(4, 2, {"Hadamard": 1, "CNOT": 1}, 1, 100)  # diff depth
        r8 = Resources(4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 1)  # diff shots

        assert r1.__eq__(r1)
        assert r1.__eq__(r2)
        assert r1.__eq__(r3)

        assert not r1.__eq__(r4)
        assert not r1.__eq__(r5)
        assert not r1.__eq__(r6)
        assert not r1.__eq__(r7)
        assert not r1.__eq__(r8)

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_str_data))
    def test_ipython_display(self, r, rep, capsys):
        """Test that the ipython display prints the string representation of a Resources instance."""
        r._ipython_display_()  # pylint: disable=protected-access
        captured = capsys.readouterr()
        assert rep in captured.out

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
from dataclasses import FrozenInstanceError
from collections import defaultdict
import pytest

import pennylane as qml
from pennylane.operation import Operation
from pennylane.tape import QuantumScript
from pennylane.resource.resource import Resources, ResourcesOperation, _count_resources
from pennylane.measurements import Shots


class TestResources:
    """Test the methods and attributes of the Resource class"""

    resource_quantities = (
        Resources(),
        Resources(5, 0, {}, {}, 0),
        Resources(
            1,
            3,
            defaultdict(int, {"Hadamard": 1, "PauliZ": 2}),
            defaultdict(int, {1: 3}),
            3,
            Shots((10, (50, 2))),
        ),
        Resources(4, 2, {"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, 2, Shots(100)),
    )

    resource_parameters = (
        (0, 0, {}, {}, 0, Shots(None)),
        (5, 0, {}, {}, 0, Shots(None)),
        (
            1,
            3,
            defaultdict(int, {"Hadamard": 1, "PauliZ": 2}),
            defaultdict(int, {1: 3}),
            3,
            Shots((10, (50, 2))),
        ),
        (4, 2, {"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, 2, Shots(100)),
    )

    @pytest.mark.parametrize("r, attribute_tup", zip(resource_quantities, resource_parameters))
    def test_init(self, r, attribute_tup):
        """Test that the Resource class is instantiated as expected."""
        num_wires, num_gates, gate_types, gate_sizes, depth, shots = attribute_tup

        assert r.num_wires == num_wires
        assert r.num_gates == num_gates
        assert r.depth == depth
        assert r.shots == shots
        assert r.gate_types == gate_types
        assert r.gate_sizes == gate_sizes

    def test_set_attributes_error(self):
        """Test that an error is raised if we try to set any attribute."""
        r = Resources()
        attr_lst = ["num_wires", "num_gates", "depth", "shots", "gate_types"]

        for attr_name in attr_lst:
            with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
                setattr(r, attr_name, 1)

    test_str_data = (
        (
            "wires: 0\n"
            + "gates: 0\n"
            + "depth: 0\n"
            + "shots: Shots(total=None)\n"
            + "gate_types:\n"
            + "{}\n"
            + "gate_sizes:\n"
            + "{}"
        ),
        (
            "wires: 5\n"
            + "gates: 0\n"
            + "depth: 0\n"
            + "shots: Shots(total=None)\n"
            + "gate_types:\n"
            + "{}\n"
            + "gate_sizes:\n"
            + "{}"
        ),
        (
            "wires: 1\n"
            + "gates: 3\n"
            + "depth: 3\n"
            + "shots: Shots(total=110, vector=[10 shots, 50 shots x 2])\n"
            + "gate_types:\n"
            + "{'Hadamard': 1, 'PauliZ': 2}\n"
            + "gate_sizes:\n"
            + "{1: 3}"
        ),
        (
            "wires: 4\n"
            + "gates: 2\n"
            + "depth: 2\n"
            + "shots: Shots(total=100)\n"
            + "gate_types:\n"
            + "{'Hadamard': 1, 'CNOT': 1}\n"
            + "gate_sizes:\n"
            + "{1: 1, 2: 1}"
        ),
    )

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_str_data))
    def test_str(self, r, rep):
        """Test the string representation of a Resources instance."""
        assert str(r) == rep

    test_rep_data = (
        "Resources(num_wires=0, num_gates=0, gate_types={}, gate_sizes={}, depth=0, "
        "shots=Shots(total_shots=None, shot_vector=()))",
        "Resources(num_wires=5, num_gates=0, gate_types={}, gate_sizes={}, depth=0, "
        "shots=Shots(total_shots=None, shot_vector=()))",
        "Resources(num_wires=1, num_gates=3, gate_types=defaultdict(<class 'int'>, {'Hadamard': 1, 'PauliZ': 2}), "
        "gate_sizes=defaultdict(<class 'int'>, {1: 3}), depth=3, "
        "shots=Shots(total_shots=110, shot_vector=(ShotCopies(10 shots x 1), ShotCopies(50 shots x 2))))",
        "Resources(num_wires=4, num_gates=2, gate_types={'Hadamard': 1, 'CNOT': 1}, "
        "gate_sizes={1: 1, 2: 1}, depth=2, shots=Shots(total_shots=100, shot_vector=(ShotCopies(100 shots x 1),)))",
    )

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_rep_data))
    def test_repr(self, r, rep):
        """Test the repr method of a Resources instance looks as expected."""
        assert repr(r) == rep

    def test_eq(self):
        """Test that the equality dunder method is correct for Resources."""
        r1 = Resources(4, 2, {"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, 2, Shots(100))
        r2 = Resources(4, 2, {"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, 2, Shots(100))
        r3 = Resources(4, 2, {"CNOT": 1, "Hadamard": 1}, {2: 1, 1: 1}, 2, Shots(100))  # all equal

        r4 = Resources(1, 2, {"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, 2, Shots(100))  # diff wires
        r5 = Resources(
            4, 1, {"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, 2, Shots(100)
        )  # diff num_gates
        r6 = Resources(4, 2, {"CNOT": 1}, {1: 1, 2: 1}, 2, Shots(100))  # diff gate_types
        r7 = Resources(
            4, 2, {"Hadamard": 1, "CNOT": 1}, {1: 3, 2: 2}, 2, Shots(100)
        )  # diff gate_sizes
        r8 = Resources(4, 2, {"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, 1, Shots(100))  # diff depth
        r9 = Resources(
            4, 2, {"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, 2, Shots((10, 10))
        )  # diff shots

        assert r1.__eq__(r1)
        assert r1.__eq__(r2)
        assert r1.__eq__(r3)

        assert not r1.__eq__(r4)
        assert not r1.__eq__(r5)
        assert not r1.__eq__(r6)
        assert not r1.__eq__(r7)
        assert not r1.__eq__(r8)
        assert not r1.__eq__(r9)

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_str_data))
    def test_ipython_display(self, r, rep, capsys):
        """Test that the ipython display prints the string representation of a Resources instance."""
        r._ipython_display_()  # pylint: disable=protected-access
        captured = capsys.readouterr()
        assert rep in captured.out


class TestResourcesOperation:  # pylint: disable=too-few-public-methods
    """Test that the ResourcesOperation class is constructed correctly"""

    def test_raise_not_implemented_error(self):
        """Test that a not type error is raised if the class is
        initialized without a `resources` method."""

        class CustomOpNoResource(ResourcesOperation):  # pylint: disable=too-few-public-methods
            num_wires = 2

        class CustomOPWithResources(ResourcesOperation):  # pylint: disable=too-few-public-methods
            num_wires = 2

            def resources(self):
                return Resources(num_wires=self.num_wires)

        with pytest.raises(TypeError, match="Can't instantiate"):
            _ = CustomOpNoResource(wires=[0, 1])  # pylint:disable=abstract-class-instantiated

        assert CustomOPWithResources(wires=[0, 1])  # shouldn't raise an error


class _CustomOpWithResource(ResourcesOperation):  # pylint: disable=too-few-public-methods
    num_wires = 2
    name = "CustomOp1"

    def resources(self):
        return Resources(
            num_wires=self.num_wires,
            num_gates=3,
            gate_types={"Identity": 1, "PauliZ": 2},
            gate_sizes={1: 3},
            depth=3,
        )


class _CustomOpWithoutResource(Operation):  # pylint: disable=too-few-public-methods
    num_wires = 2
    name = "CustomOp2"


lst_ops_and_shots = (
    ([], Shots(None)),
    ([qml.Hadamard(0), qml.CNOT([0, 1])], Shots(None)),
    ([qml.PauliZ(0), qml.CNOT([0, 1]), qml.RX(1.23, 2)], Shots(10)),
    (
        [
            qml.Hadamard(0),
            qml.RX(1.23, 1),
            qml.CNOT([0, 1]),
            qml.RX(4.56, 1),
            qml.Hadamard(0),
            qml.Hadamard(1),
        ],
        Shots(100),
    ),
    ([qml.Hadamard(0), qml.CNOT([0, 1]), _CustomOpWithResource(wires=[1, 0])], Shots(None)),
    (
        [
            qml.PauliZ(0),
            qml.CNOT([0, 1]),
            qml.RX(1.23, 2),
            _CustomOpWithResource(wires=[0, 2]),
            _CustomOpWithoutResource(wires=[0, 1]),
        ],
        Shots((10, (50, 2))),
    ),
    (
        [
            qml.Hadamard(0),
            qml.RX(1.23, 1),
            qml.CNOT([0, 1]),
            qml.RX(4.56, 1),
            qml.Hadamard(0),
            qml.Hadamard(1),
            _CustomOpWithoutResource(wires=[0, 1]),
        ],
        Shots(100),
    ),
)

resources_data = (
    Resources(),
    Resources(2, 2, {"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, 2),
    Resources(3, 3, {"PauliZ": 1, "CNOT": 1, "RX": 1}, {1: 2, 2: 1}, 2, Shots(10)),
    Resources(2, 6, {"Hadamard": 3, "RX": 2, "CNOT": 1}, {1: 5, 2: 1}, 4, Shots(100)),
    Resources(2, 5, {"Hadamard": 1, "CNOT": 1, "Identity": 1, "PauliZ": 2}, {1: 4, 2: 1}, 5),
    Resources(
        3,
        7,
        {"PauliZ": 3, "CNOT": 1, "RX": 1, "Identity": 1, "CustomOp2": 1},
        {1: 5, 2: 2},
        6,
        Shots((10, (50, 2))),
    ),
    Resources(
        2, 7, {"Hadamard": 3, "RX": 2, "CNOT": 1, "CustomOp2": 1}, {1: 5, 2: 2}, 5, Shots(100)
    ),
)  # Resources(wires, gates, gate_types, gate_sizes, depth, shots)


@pytest.mark.parametrize(
    "ops_and_shots, expected_resources", zip(lst_ops_and_shots, resources_data)
)
def test_count_resources(ops_and_shots, expected_resources):
    """Test the count resources method."""
    ops, shots = ops_and_shots
    computed_resources = _count_resources(QuantumScript(ops=ops, shots=shots))
    assert computed_resources == expected_resources

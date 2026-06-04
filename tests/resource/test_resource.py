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

# pylint: disable=unnecessary-dunder-call,protected-access
from collections import defaultdict
from dataclasses import FrozenInstanceError

import pytest

import pennylane as qp
from pennylane.measurements import Shots
from pennylane.operation import Operation
from pennylane.resource.expression import Expression
from pennylane.resource.resource import (
    CircuitSpecs,
    Resources,
    ResourcesOperation,
    SpecsResources,
    SymbolicSpecsResources,
    _combine_dict,
    _count_resources,
    _count_to_str,
    _scale_dict,
    add_in_parallel,
    add_in_series,
    mul_in_parallel,
    mul_in_series,
    num_to_letters,
    resources_from_tape,
    substitute,
)
from pennylane.tape import QuantumScript


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
            "num_wires: 0\n"
            + "num_gates: 0\n"
            + "depth: 0\n"
            + "shots: Shots(total=None)\n"
            + "gate_types:\n"
            + "{}\n"
            + "gate_sizes:\n"
            + "{}"
        ),
        (
            "num_wires: 5\n"
            + "num_gates: 0\n"
            + "depth: 0\n"
            + "shots: Shots(total=None)\n"
            + "gate_types:\n"
            + "{}\n"
            + "gate_sizes:\n"
            + "{}"
        ),
        (
            "num_wires: 1\n"
            + "num_gates: 3\n"
            + "depth: 3\n"
            + "shots: Shots(total=110, vector=[10 shots, 50 shots x 2])\n"
            + "gate_types:\n"
            + "{'Hadamard': 1, 'PauliZ': 2}\n"
            + "gate_sizes:\n"
            + "{1: 3}"
        ),
        (
            "num_wires: 4\n"
            + "num_gates: 2\n"
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

    expected_results_add_series = (
        Resources(
            2,
            6,
            defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            defaultdict(int, {1: 5, 2: 1}),
            3,
            Shots(10),
        ),
        Resources(
            5,
            6,
            defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            defaultdict(int, {1: 5, 2: 1}),
            3,
            Shots(10),
        ),
        Resources(
            2,
            9,
            defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 2, "PauliZ": 2}),
            defaultdict(int, {1: 8, 2: 1}),
            6,
            Shots((10, (50, 2), 10)),
        ),
        Resources(
            4,
            8,
            defaultdict(int, {"RZ": 2, "CNOT": 2, "RY": 2, "Hadamard": 2}),
            defaultdict(int, {1: 6, 2: 2}),
            5,
            Shots((100, 10)),
        ),
    )

    @pytest.mark.parametrize(
        "resource_obj, expected_res_obj", zip(resource_quantities, expected_results_add_series)
    )
    def test_add_in_series(self, resource_obj, expected_res_obj):
        """Test the add_in_series function works with Resoruces"""
        other_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=3,
            shots=Shots(10),
        )

        resultant_obj = add_in_series(resource_obj, other_obj)
        assert resultant_obj == expected_res_obj

    @pytest.mark.parametrize(
        "resource_obj, expected_res_obj", zip(resource_quantities, expected_results_add_series)
    )
    def test_dunder_add(self, resource_obj, expected_res_obj):
        """Test the __add__ function"""
        other_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=3,
            shots=Shots(10),
        )

        resultant_obj = resource_obj + other_obj
        assert resultant_obj == expected_res_obj

    expected_results_add_parallel = (
        Resources(
            2,
            6,
            defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            defaultdict(int, {1: 5, 2: 1}),
            3,
            Shots(10),
        ),
        Resources(
            7,
            6,
            defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            defaultdict(int, {1: 5, 2: 1}),
            3,
            Shots(10),
        ),
        Resources(
            3,
            9,
            defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 2, "PauliZ": 2}),
            defaultdict(int, {1: 8, 2: 1}),
            3,
            Shots((10, (50, 2), 10)),
        ),
        Resources(
            6,
            8,
            defaultdict(int, {"RZ": 2, "CNOT": 2, "RY": 2, "Hadamard": 2}),
            defaultdict(int, {1: 6, 2: 2}),
            3,
            Shots((100, 10)),
        ),
    )

    @pytest.mark.parametrize(
        "resource_obj, expected_res_obj", zip(resource_quantities, expected_results_add_parallel)
    )
    def test_add_in_parallel(self, resource_obj, expected_res_obj):
        """Test the add_in_parallel function works with Resoruces"""
        other_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=3,
            shots=Shots(10),
        )

        resultant_obj = add_in_parallel(resource_obj, other_obj)
        assert resultant_obj == expected_res_obj

    expected_results_mul_series = (
        Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=3,
            shots=Shots(10),
        ),
        Resources(
            num_wires=2,
            num_gates=12,
            gate_types=defaultdict(int, {"RZ": 4, "CNOT": 2, "RY": 4, "Hadamard": 2}),
            gate_sizes=defaultdict(int, {1: 10, 2: 2}),
            depth=6,
            shots=Shots(20),
        ),
        Resources(
            num_wires=2,
            num_gates=18,
            gate_types=defaultdict(int, {"RZ": 6, "CNOT": 3, "RY": 6, "Hadamard": 3}),
            gate_sizes=defaultdict(int, {1: 15, 2: 3}),
            depth=9,
            shots=Shots(30),
        ),
        Resources(
            num_wires=2,
            num_gates=24,
            gate_types=defaultdict(int, {"RZ": 8, "CNOT": 4, "RY": 8, "Hadamard": 4}),
            gate_sizes=defaultdict(int, {1: 20, 2: 4}),
            depth=12,
            shots=Shots(40),
        ),
    )

    @pytest.mark.parametrize(
        "scalar, expected_res_obj", zip((1, 2, 3, 4), expected_results_mul_series)
    )
    def test_mul_in_series(self, scalar, expected_res_obj):
        """Test the mul_in_series function works with Resoruces"""
        resource_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=3,
            shots=Shots(10),
        )

        resultant_obj = mul_in_series(resource_obj, scalar)
        assert resultant_obj == expected_res_obj

    @pytest.mark.parametrize(
        "scalar, expected_res_obj", zip((1, 2, 3, 4), expected_results_mul_series)
    )
    def test_dunder_mul(self, scalar, expected_res_obj):
        """Test the __mul__ function"""
        resource_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=3,
            shots=Shots(10),
        )

        resultant_obj = resource_obj * scalar
        assert resultant_obj == expected_res_obj

    expected_results_mul_parallel = (
        Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=3,
            shots=Shots(10),
        ),
        Resources(
            num_wires=4,
            num_gates=12,
            gate_types=defaultdict(int, {"RZ": 4, "CNOT": 2, "RY": 4, "Hadamard": 2}),
            gate_sizes=defaultdict(int, {1: 10, 2: 2}),
            depth=3,
            shots=Shots(20),
        ),
        Resources(
            num_wires=6,
            num_gates=18,
            gate_types=defaultdict(int, {"RZ": 6, "CNOT": 3, "RY": 6, "Hadamard": 3}),
            gate_sizes=defaultdict(int, {1: 15, 2: 3}),
            depth=3,
            shots=Shots(30),
        ),
        Resources(
            num_wires=8,
            num_gates=24,
            gate_types=defaultdict(int, {"RZ": 8, "CNOT": 4, "RY": 8, "Hadamard": 4}),
            gate_sizes=defaultdict(int, {1: 20, 2: 4}),
            depth=3,
            shots=Shots(40),
        ),
    )

    @pytest.mark.parametrize(
        "scalar, expected_res_obj", zip((1, 2, 3, 4), expected_results_mul_parallel)
    )
    def test_mul_in_parallel(self, scalar, expected_res_obj):
        """Test the mul_in_parallel function works with Resoruces"""
        resource_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=3,
            shots=Shots(10),
        )

        resultant_obj = mul_in_parallel(resource_obj, scalar)
        assert resultant_obj == expected_res_obj

    gate_info = (("RX", 1), ("RZ", 1), ("RZ", 1))

    sub_obj = (
        Resources(
            num_wires=1,
            num_gates=5,
            gate_types=defaultdict(int, {"RX": 5}),
            gate_sizes=defaultdict(int, {1: 5}),
            depth=1,
            shots=Shots(shots=None),
        ),
        Resources(
            num_wires=1,
            num_gates=5,
            gate_types=defaultdict(int, {"RX": 5}),
            gate_sizes=defaultdict(int, {1: 5}),
            depth=1,
            shots=Shots(shots=None),
        ),
        Resources(
            num_wires=5,
            num_gates=5,
            gate_types=defaultdict(int, {"RX": 5}),
            gate_sizes=defaultdict(int, {1: 5}),
            depth=1,
            shots=Shots(shots=None),
        ),
    )

    expected_results_sub = (
        Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=2,
            shots=Shots(10),
        ),
        Resources(
            num_wires=2,
            num_gates=14,
            gate_types=defaultdict(int, {"RX": 10, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 13, 2: 1}),
            depth=3,
            shots=Shots(10),
        ),
        Resources(
            num_wires=6,
            num_gates=14,
            gate_types=defaultdict(int, {"RX": 10, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 13, 2: 1}),
            depth=3,
            shots=Shots(10),
        ),
    )

    @pytest.mark.parametrize(
        "gate_info, sub_obj, expected_res_obj", zip(gate_info, sub_obj, expected_results_sub)
    )
    def test_substitute(self, gate_info, sub_obj, expected_res_obj):
        """Test the substitute function"""
        resource_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=2,
            shots=Shots(10),
        )

        resultant_obj = substitute(resource_obj, gate_info, sub_obj)
        assert resultant_obj == expected_res_obj

    def test_substitute_wire_error(self):
        """Test that substitute raises an exception when the wire count does not exist in gate_sizes"""

        resource_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=2,
            shots=Shots(10),
        )

        sub_obj = Resources(
            num_wires=1,
            num_gates=5,
            gate_types=defaultdict(int, {"RX": 5}),
            gate_sizes=defaultdict(int, {1: 5}),
            depth=1,
            shots=Shots(shots=None),
        )

        gate_info = ("RZ", 100)

        with pytest.raises(
            ValueError, match="initial_resources does not contain a gate acting on 100 wires."
        ):
            substitute(resource_obj, gate_info, sub_obj)

    def test_substitute_gate_count_error(self):
        """Test that substitute raises an exception when the substitution would result in a negative value in gate_sizes"""

        resource_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
            gate_sizes=defaultdict(int, {1: 5, 2: 1}),
            depth=2,
            shots=Shots(10),
        )

        sub_obj = Resources(
            num_wires=1,
            num_gates=5,
            gate_types=defaultdict(int, {"RX": 5}),
            gate_sizes=defaultdict(int, {1: 5}),
            depth=1,
            shots=Shots(shots=None),
        )

        gate_info = ("RZ", 2)
        with pytest.raises(
            ValueError,
            match="Found 2 gates of type RZ, but only 1 gates act on 2 wires in initial_resources.",
        ):
            substitute(resource_obj, gate_info, sub_obj)


class TestResourcesOperation:  # pylint: disable=too-few-public-methods
    """Test that the ResourcesOperation class is constructed correctly"""

    def test_raise_not_implemented_error(self):
        """Test that a not type error is raised if the class is
        initialized without a `resources` method."""

        class CustomOpNoResource(ResourcesOperation):  # pylint: disable=too-few-public-methods
            """A custom operation that does not implement the resources method."""

            num_wires = 2

        class CustomOPWithResources(ResourcesOperation):  # pylint: disable=too-few-public-methods
            """A custom operation that implements the resources method."""

            num_wires = 2

            def resources(self):
                return Resources(num_wires=self.num_wires)

        with pytest.raises(TypeError, match="Can't instantiate"):
            _ = CustomOpNoResource(wires=[0, 1])  # pylint:disable=abstract-class-instantiated

        assert CustomOPWithResources(wires=[0, 1])  # shouldn't raise an error


def test_combine_dict():
    """Test that we can combine dictionaries as expected."""
    d1 = defaultdict(int, {"a": 2, "b": 4, "c": 6})
    d2 = defaultdict(int, {"a": 1, "b": 2, "d": 3})

    result = _combine_dict(d1, d2)
    expected = defaultdict(int, {"a": 3, "b": 6, "c": 6, "d": 3})

    assert result == expected


@pytest.mark.parametrize("scalar", (1, 2, 3))
def test_scale_dict(scalar):
    """Test that we can scale the values of a dictionary as expected."""
    d1 = defaultdict(int, {"a": 2, "b": 4, "c": 6})

    expected = defaultdict(int, {k: scalar * v for k, v in d1.items()})
    result = _scale_dict(d1, scalar)

    assert result == expected


@pytest.mark.parametrize("compute_depth", (True, False))
def test_specs_compute_depth(compute_depth):
    """Test that depth is skipped with `resources_from_tape`."""

    ops = [
        qp.RX(0.432, wires=0),
        qp.Rot(0.543, 0, 0.23, wires=0),
        qp.CNOT(wires=[0, "a"]),
        qp.RX(0.133, wires=4),
    ]
    obs = [qp.expval(qp.PauliX(wires="a")), qp.probs(wires=[0, "a"])]

    tape = QuantumScript(ops=ops, measurements=obs)
    resources = resources_from_tape(tape, compute_depth=compute_depth)

    assert resources.depth == (3 if compute_depth else None)


###########################################################################
##  Tests for specs dataclasses
###########################################################################


class TestSpecsResources:
    """Test the methods and attributes of the SpecsResource class"""

    @pytest.fixture
    def example_specs_resource(self):
        """Generate an example SpecsResources instance."""
        return SpecsResources(
            gate_types={"Hadamard": 2, "CNOT": 1},
            gate_sizes={1: 2, 2: 1},
            measurements={"expval(PauliZ)": 1},
            num_allocs=2,
            depth=2,
        )

    def test_depth_autoassign(self):
        """Test that the SpecsResources class auto-assigns depth as None if not provided."""

        s = SpecsResources(
            gate_types={"Hadamard": 2, "CNOT": 1},
            gate_sizes={1: 2, 2: 1},
            measurements={"expval(PauliZ)": 1},
            num_allocs=2,
        )

        assert s.depth is None

    def test_num_gates(self, example_specs_resource):
        """Test that the SpecsResources class handles `num_gates` as expected."""

        with pytest.raises(
            ValueError,
            match="Inconsistent gate counts: `gate_types` and `gate_sizes` describe different amounts of gates.",
        ):
            # Gate counts don't match
            _ = SpecsResources(
                gate_types={"Hadamard": 1}, gate_sizes={1: 2}, measurements={}, num_allocs=0
            )

        s = example_specs_resource

        assert s.num_gates == 3

    def test_immutable(self, example_specs_resource):
        """Test that SpecsResources is immutable."""

        s = example_specs_resource

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            s.gate_types = {}

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            s.gate_sizes = {}

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            s.measurements = {}

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            s.num_allocs = 1

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            s.depth = 0

    def test_getitem(self, example_specs_resource):
        """Test that SpecsResources supports indexing via __getitem__."""

        s = example_specs_resource

        assert s["gate_types"] == s.gate_types
        assert s["gate_counts"] == s.gate_types
        assert s["gate_sizes"] == s.gate_sizes
        assert s["measurements"] == s.measurements
        assert s["num_allocs"] == s.num_allocs
        assert s["depth"] == s.depth

        assert s["num_gates"] == s.num_gates

        # Check removed keys
        with pytest.raises(
            KeyError,
            match="shots is no longer included within specs's resources, check the top-level object instead.",
        ):
            _ = s["shots"]
        with pytest.raises(
            KeyError,
            match="num_wires has been renamed to num_allocs to more accurate describe what it measures.",
        ):
            _ = s["num_wires"]

        # Try nonexistent key
        with pytest.raises(
            KeyError,
            match="key 'potato' not available. Options are ",
        ):
            _ = s["potato"]

    def test_str(self, example_specs_resource):
        """Test the string representation of a SpecsResources instance."""

        s = example_specs_resource

        expected = "Wire allocations: 2\n"
        expected += "Total gates: 3\n"
        expected += "Gate counts:\n"
        expected += "- Hadamard: 2\n"
        expected += "- CNOT: 1\n"
        expected += "Measurements:\n"
        expected += "- expval(PauliZ): 1\n"
        expected += "Depth: 2"

        expected_indented = ("    " + expected.replace("\n", "\n    ")).replace("\n    \n", "\n\n")

        assert str(s) == expected
        assert s.to_pretty_str() == expected
        assert s.to_pretty_str(preindent=4) == expected_indented

        # Check with no depth, gates, or measurements

        s = SpecsResources(gate_types={}, gate_sizes={}, measurements={}, num_allocs=0)

        expected = "Wire allocations: 0\n"
        expected += "Total gates: 0\n"
        expected += "Gate counts:\n"
        expected += "- No gates.\n"
        expected += "Measurements:\n"
        expected += "- No measurements.\n"
        expected += "Depth: Not computed"

        expected_indented = ("    " + expected.replace("\n", "\n    ")).replace("\n    \n", "\n\n")

        assert str(s) == expected
        assert s.to_pretty_str() == expected
        assert s.to_pretty_str(preindent=4) == expected_indented

    def test_to_dict(self, example_specs_resource):
        """Test the to_dict method of SpecsResources."""

        s = example_specs_resource

        expected = {
            "gate_types": {"Hadamard": 2, "CNOT": 1},
            "gate_sizes": {1: 2, 2: 1},
            "measurements": {"expval(PauliZ)": 1},
            "num_allocs": 2,
            "depth": 2,
            "num_gates": 3,
        }

        assert s.to_dict() == expected


class TestSymbolicSpecsResources:
    @pytest.fixture
    def example_resource(self) -> SymbolicSpecsResources:
        """
        Generate an example SymbolicSpecsResources instance.
        The resources roughly correspond to the following circuit:

        .. code-block:: python

            def circ():
                qp.Hadamard(0)
                qp.PauliX(0)
                for i in range(x):
                    qp.PauliX(i)
                    for _ in range(z):
                        qp.CNOT(wires=[0, 1])
                for j in range(2 * z):
                    qp.PauliZ(j)
                return expval(qp.PauliZ(0))
        """
        return SymbolicSpecsResources(
            gate_types={
                "Hadamard": Expression({(): 1}),
                "PauliX": Expression({("x"): 1, (): 1}),
                "CNOT": Expression({("x", "z"): 1}),
                "PauliZ": Expression({("z",): 2}),
            },
            gate_sizes={1: Expression({("z"): 2, "x": 1, (): 2}), 2: Expression({("x", "z"): 1})},
            measurements={"expval(PauliZ)": 1},
            # The values for allocs and depth are a bit off, but are helpful for testing substitutions
            num_allocs=Expression({("x",): 1, ("z",): 2, (): 1}),
            depth=Expression({("x", "z"): 1, ("z",): 2, ("x",): 1, (): 2}),
        )

    @pytest.fixture
    def example_resource_concrete(self) -> SymbolicSpecsResources:
        """
        Generate an example SymbolicSpecsResources instance for a non-dynamic circuit.

        Specifically, returns the resources for a simple Bell state circuit with a measurement.
        """
        return SymbolicSpecsResources(
            gate_types={"Hadamard": 1, "CNOT": 1},
            gate_sizes={1: 1, 2: 1},
            measurements={"expval(PauliZ)": 1},
            num_allocs=1,
            depth=1,
        )

    def test_init_converts_to_expression(self):
        """Test that SymbolicSpecsResources can be instantiated with ints and correctly converts them."""
        s = SymbolicSpecsResources(
            gate_types={"Hadamard": 1, "CNOT": 1},
            gate_sizes={1: 1, 2: 1},
            measurements={"expval(PauliZ)": 1},
            num_allocs=1,
            depth=1,
        )

        assert isinstance(s.gate_types, dict)
        assert all(isinstance(v, Expression) for v in s.gate_types.values())
        assert isinstance(s.gate_sizes, dict)
        assert all(isinstance(v, Expression) for v in s.gate_sizes.values())
        assert isinstance(s.measurements, dict)
        assert all(isinstance(v, Expression) for v in s.measurements.values())
        assert isinstance(s.num_allocs, Expression)
        assert isinstance(s.depth, Expression)

    def test_blank_subs(self, example_resource):
        s = example_resource
        assert s.subs() == s

    def test_blank_subs_concrete(self, example_resource_concrete):
        s = example_resource_concrete

        concretized = s.subs()
        assert isinstance(concretized, SpecsResources)
        assert not isinstance(concretized, SymbolicSpecsResources)
        assert concretized == SpecsResources(
            gate_types={"Hadamard": 1, "CNOT": 1},
            gate_sizes={1: 1, 2: 1},
            measurements={"expval(PauliZ)": 1},
            num_allocs=1,
            depth=1,
        )

    def test_partial_subs(self, example_resource):
        s = example_resource

        # Substitute x=2, leaving z symbolic
        partially_substituted = s.subs({"x": 2})

        expected = SymbolicSpecsResources(
            gate_types={
                "Hadamard": Expression({(): 1}),
                "PauliX": Expression({(): 3}),
                "CNOT": Expression({("z",): 2}),
                "PauliZ": Expression({("z",): 2}),
            },
            gate_sizes={
                1: Expression({("z",): 2, (): 4}),
                2: Expression({("z",): 2}),
            },
            measurements={"expval(PauliZ)": 1},
            num_allocs=Expression({("z",): 2, (): 3}),
            depth=Expression({("z",): 4, (): 4}),
        )

        assert partially_substituted == expected

    def test_full_subs(self, example_resource):
        s = example_resource

        # Substitute x=2 and z=3
        fully_substituted = s.subs({"x": 2, "z": 3})

        expected = SpecsResources(
            gate_types={"Hadamard": 1, "PauliX": 3, "CNOT": 6, "PauliZ": 6},
            gate_sizes={1: 10, 2: 6},
            measurements={"expval(PauliZ)": 1},
            num_allocs=9,
            depth=16,
        )

        assert fully_substituted == expected
        assert not isinstance(fully_substituted, SymbolicSpecsResources)

    def test_subs_kwargs(self, example_resource):
        assert example_resource.subs(x=2, z=3) == example_resource.subs({"x": 2, "z": 3})

    def test_invalid_subs(self, example_resource):
        """Test that the subs method raises a TypeError for invalid substitutions."""
        with pytest.raises(TypeError):
            example_resource.subs({"x": "not an int"})
        with pytest.raises(ValueError):
            example_resource.subs({"not a var": 3})

    def test_call(self, example_resource):
        assert example_resource(x=2, z=3) == example_resource.subs(x=2, z=3)

    def test_eq(self):
        s1 = SymbolicSpecsResources(
            gate_types={"Hadamard": Expression({("x,"): 1})},
            gate_sizes={1: Expression({("x,"): 1})},
            measurements={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            depth=Expression(1),
        )
        s2 = SymbolicSpecsResources(
            gate_types={"Hadamard": Expression({("x,"): 1})},
            gate_sizes={1: Expression({("x,"): 1})},
            measurements={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            depth=Expression(1),
        )
        s3 = SymbolicSpecsResources(
            gate_types={"Hadamard": Expression({("z,"): 1})},
            gate_sizes={1: Expression({("z,"): 1})},
            measurements={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            depth=Expression(1),
        )

        assert s1 == s2
        assert s1 != s3
        assert s2 != s3
        assert s1 != SpecsResources(
            gate_types={"Hadamard": 1},
            gate_sizes={1: 1},
            measurements={"expval(PauliZ)": 1},
            num_allocs=1,
            depth=1,
        )

    def test_eq_no_var(self):
        s1 = SymbolicSpecsResources(
            gate_types={"Hadamard": Expression(1)},
            gate_sizes={1: Expression(1)},
            measurements={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            depth=Expression(1),
        )

        s2 = SymbolicSpecsResources(
            gate_types={"Hadamard": Expression(1)},
            gate_sizes={1: Expression(1)},
            measurements={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            depth=Expression(1),
        )

        s3 = SymbolicSpecsResources(
            gate_types={"Hadamard": Expression(2)},  # different value here
            gate_sizes={1: Expression(1)},
            measurements={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            depth=Expression(1),
        )

        assert s1 == s2
        assert s1 != s3
        assert s2 != s3

        assert s1 == SpecsResources(
            gate_types={"Hadamard": 1},
            gate_sizes={1: 1},
            measurements={"expval(PauliZ)": 1},
            num_allocs=1,
            depth=1,
        )

    def test_eq_invalid(self, example_resource):
        assert example_resource != "not a SpecsResource"

    def test_str(self, example_resource):
        s = example_resource

        expected = "Symbolic Variables: x, z\n"
        expected += "Wire allocations: 2*z + x + 1\n"
        expected += "Total gates: x*z + 2*z + x + 2\n"
        expected += "Gate counts:\n"
        expected += "- Hadamard: 1\n"
        expected += "- PauliX: x + 1\n"
        expected += "- CNOT: x*z\n"
        expected += "- PauliZ: 2*z\n"
        expected += "Measurements:\n"
        expected += "- expval(PauliZ): 1\n"
        expected += "Depth: x*z + 2*z + x + 2"

        assert str(s) == expected


class TestCircuitSpecs:
    @pytest.fixture
    def example_specs_result(self):
        """Generate an example CircuitSpecs instance."""
        return CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level=2,
            resources=SpecsResources(
                gate_types={"Hadamard": 2, "CNOT": 1},
                gate_sizes={1: 2, 2: 1},
                measurements={"expval(PauliZ)": 1},
                num_allocs=2,
                depth=2,
            ),
        )

    @pytest.fixture
    def example_specs_result_multi(self):
        """Generate an example CircuitSpecs instance with multiple levels and batches."""
        return CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level={1: "l1", 2: "l2"},
            resources={
                1: SpecsResources(
                    gate_types={"Hadamard": 4, "CNOT": 2},
                    gate_sizes={1: 4, 2: 2},
                    measurements={"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    num_allocs=2,
                    depth=2,
                ),
                2: [
                    SpecsResources(
                        gate_types={"CNOT": 1},
                        gate_sizes={2: 1},
                        measurements={"expval(PauliX)": 1},
                        num_allocs=2,
                        depth=1,
                    ),
                    SpecsResources(
                        gate_types={"CNOT": 1},
                        gate_sizes={2: 1},
                        measurements={"expval(PauliZ)": 1},
                        num_allocs=2,
                        depth=1,
                    ),
                ],
            },
        )

    @pytest.fixture
    def example_specs_result_multi_symbolic(self):
        """Generate an example CircuitSpecs instance with multiple levels and batches, as well as symbolic resources."""
        return CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level={1: "l1", 2: "l2"},
            resources={
                1: SymbolicSpecsResources(
                    gate_types={
                        "Hadamard": Expression({("x",): 2, (): 2}),
                        "CNOT": Expression({("x",): 2}),
                    },
                    gate_sizes={1: Expression({("x",): 2, (): 2}), 2: Expression({("x",): 2})},
                    measurements={"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    num_allocs=2,
                    depth=2,
                ),
                2: [
                    SymbolicSpecsResources(
                        gate_types={"CNOT": Expression({("x",): 1})},
                        gate_sizes={2: Expression({("x",): 1})},
                        measurements={"expval(PauliX)": 1},
                        num_allocs=2,
                        depth=1,
                    ),
                    SymbolicSpecsResources(
                        gate_types={"CNOT": Expression({("x",): 1})},
                        gate_sizes={2: Expression({("x",): 1})},
                        measurements={"expval(PauliZ)": 1},
                        num_allocs=2,
                        depth=1,
                    ),
                ],
            },
        )

    def test_blank_init(self):
        """Test that CircuitSpecss can be instantiated with no arguments."""
        r = CircuitSpecs()  # should not raise any errors

        assert r.device_name is None
        assert r.num_device_wires is None
        assert r.shots is None
        assert r.level is None
        assert r.resources is None

    def test_getitem(self, example_specs_result):
        """Test that CircuitSpecs supports indexing via __getitem__."""

        r = example_specs_result

        assert r["device_name"] == r.device_name
        assert r["num_device_wires"] == r.num_device_wires
        assert r["shots"] == r.shots
        assert r["level"] == r.level
        assert r["resources"] == r.resources

    def test_getitem_removed_keys(self, example_specs_result):
        """Test that CircuitSpecs raises more descriptive KeyErrors for removed keys."""

        r = example_specs_result

        with pytest.raises(
            KeyError,
            match="num_observables is no longer in top-level specs and has instead been absorbed into the 'measurements' attribute of the specs's resources.",
        ):
            _ = r["num_observables"]

        for key in ("interface", "diff_method", "errors", "num_tape_wires"):
            with pytest.raises(
                KeyError,
                match=f"key '{key}' is no longer included in specs.",
            ):
                _ = r[key]

        for key in (
            "gradient_fn",
            "gradient_options",
            "num_gradient_executions",
            "num_trainable_params",
        ):
            with pytest.raises(
                KeyError,
                match=f"key '{key}' is no longer included in specs, as specs no longer gathers gradient information.",
            ):
                _ = r[key]

        # Check nonexistent key
        with pytest.raises(
            KeyError,
            match="key 'potato' not available. Options are ",
        ):
            _ = r["potato"]

    def test_to_dict(
        self, example_specs_result, example_specs_result_multi, example_specs_result_multi_symbolic
    ):
        """Test the to_dict method of CircuitSpecs."""

        r = example_specs_result

        expected = {
            "device_name": "default.qubit",
            "num_device_wires": 5,
            "shots": Shots(1000),
            "level": 2,
            "resources": {
                "gate_types": {"Hadamard": 2, "CNOT": 1},
                "gate_sizes": {1: 2, 2: 1},
                "measurements": {"expval(PauliZ)": 1},
                "num_allocs": 2,
                "depth": 2,
                "num_gates": 3,
            },
        }

        assert r.to_dict() == expected

        r = example_specs_result_multi

        expected = {
            "device_name": "default.qubit",
            "num_device_wires": 5,
            "shots": Shots(1000),
            "level": {1: "l1", 2: "l2"},
            "resources": {
                1: {
                    "gate_types": {"Hadamard": 4, "CNOT": 2},
                    "gate_sizes": {1: 4, 2: 2},
                    "measurements": {"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    "num_allocs": 2,
                    "depth": 2,
                    "num_gates": 6,
                },
                2: [
                    {
                        "gate_types": {"CNOT": 1},
                        "gate_sizes": {2: 1},
                        "measurements": {"expval(PauliX)": 1},
                        "num_allocs": 2,
                        "depth": 1,
                        "num_gates": 1,
                    },
                    {
                        "gate_types": {"CNOT": 1},
                        "gate_sizes": {2: 1},
                        "measurements": {"expval(PauliZ)": 1},
                        "num_allocs": 2,
                        "depth": 1,
                        "num_gates": 1,
                    },
                ],
            },
        }

        assert r.to_dict() == expected

        r = example_specs_result_multi_symbolic

        expected = {
            "device_name": "default.qubit",
            "num_device_wires": 5,
            "shots": Shots(1000),
            "level": {1: "l1", 2: "l2"},
            "resources": {
                1: {
                    "gate_types": {
                        "Hadamard": Expression({("x",): 2, (): 2}),
                        "CNOT": Expression({("x",): 2}),
                    },
                    "gate_sizes": {1: Expression({("x",): 2, (): 2}), 2: Expression({("x",): 2})},
                    "measurements": {"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    "num_allocs": 2,
                    "depth": 2,
                    "num_gates": Expression({("x",): 4, (): 2}),
                    "vars": ["x"],
                },
                2: [
                    {
                        "gate_types": {"CNOT": Expression({("x",): 1})},
                        "gate_sizes": {2: Expression({("x",): 1})},
                        "measurements": {"expval(PauliX)": 1},
                        "num_allocs": 2,
                        "depth": 1,
                        "num_gates": Expression({("x",): 1}),
                        "vars": ["x"],
                    },
                    {
                        "gate_types": {"CNOT": Expression({("x",): 1})},
                        "gate_sizes": {2: Expression({("x",): 1})},
                        "measurements": {"expval(PauliZ)": 1},
                        "num_allocs": 2,
                        "depth": 1,
                        "num_gates": Expression({("x",): 1}),
                        "vars": ["x"],
                    },
                ],
            },
        }

        assert r.to_dict() == expected

    def test_str(self, example_specs_result):
        """Test the string representation of a CircuitSpecs instance."""

        r = example_specs_result

        expected = "Device: default.qubit\n"
        expected += "Device wires: 5\n"
        expected += "Shots: Shots(total=1000)\n"
        expected += "Level: 2\n"
        expected += "\n"
        expected += r.resources.to_pretty_str()

        assert str(r) == expected

    def test_str_multi_tabular(self, example_specs_result_multi):
        """Test the tabular string representation of a CircuitSpecs instance."""

        r = example_specs_result_multi
        assert [x.strip() for x in str(r).split()] == [x.strip() for x in """Device: default.qubit
Device wires: 5
Shots: Shots(total=1000)
Levels:
- 1: l1
- 2: l2

↓Metric   Level→ |    1 |  2-a |  2-b
-------------------------------------
Wire allocations |    2 |    2 |    2
Total gates      |    6 |    1 |    1
Gate counts:     |
- Hadamard       |    4 |    0 |    0
- CNOT           |    2 |    1 |    1
Measurements:    |
- expval(PauliX) |    1 |    1 |    0
- expval(PauliZ) |    1 |    0 |    1""".split()]

    def test_str_multi_tabular_symbolic(self, example_specs_result_multi_symbolic):
        """Test the tabular string representation of a CircuitSpecs instance with symbolic resources."""

        r = example_specs_result_multi_symbolic
        assert [x.strip() for x in str(r).split()] == [x.strip() for x in """Device: default.qubit
Device wires: 5
Shots: Shots(total=1000)
Levels:
- 1: l1
- 2: l2

↓Metric   Level→ |     1 |   2-a |   2-b
----------------------------------------
Wire allocations |     2 |     2 |     2
Total gates      | 4*x+2 |     x |     x
Gate counts:     |
- Hadamard       | 2*x+2 |     0 |     0
- CNOT           |   2*x |     x |     x
Measurements:    |
- expval(PauliX) |     1 |     1 |     0
- expval(PauliZ) |     1 |     0 |     1""".split()]

    def test_str_multi_non_tabular(self, example_specs_result_multi):
        """Test the non-tabular string representation of a CircuitSpecs instance."""
        r = example_specs_result_multi

        expected = "Device: default.qubit\n"
        expected += "Device wires: 5\n"
        expected += "Shots: Shots(total=1000)\n"
        expected += "Levels:\n"
        expected += "- 1: l1\n"
        expected += "- 2: l2\n"
        expected += "\n\n"

        expected += "Level = 1:\n"
        expected += r.resources[1].to_pretty_str(preindent=4)

        expected += "\n\n" + "-" * 60 + "\n\n"

        expected += "Level = 2:\n"
        expected += "    Batched tape a:\n"
        expected += r.resources[2][0].to_pretty_str(preindent=8)
        expected += "\n\n    Batched tape b:\n"
        expected += r.resources[2][1].to_pretty_str(preindent=8)

        assert r.to_pretty_str(tabular=False) == expected


class TestIPythonDisplays:
    """
    Test the IPython display methods for all applicable resource classes.

    Note that we don't test the `_ipython_display_` methods directly since this method does not
    return a value (it uses side-effects only). Instead, we check the output of the _repr_markdown_
    or other related methods.

    See also: https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
    """

    @pytest.fixture
    def example_specs_resource(self) -> SpecsResources:
        return SpecsResources(
            # Pick a number that forces scientific notation
            gate_types={"Hadamard": 1, "CNOT": 100_001},
            gate_sizes={1: 1, 2: 100_001},
            measurements={"expval(PauliZ)": 1},
            num_allocs=2,
            depth=2,
        )

    @pytest.fixture
    def example_symbolic_specs_resource(self) -> SymbolicSpecsResources:
        return SymbolicSpecsResources(
            gate_types={
                "Hadamard": Expression({("a", "a", "b"): 1, ("a", "a"): 1, ("a",): 1}),
                "CNOT": 1,
            },
            gate_sizes={1: Expression({("a", "a", "b"): 1, ("a", "a"): 1, ("a",): 1}), 2: 1},
            measurements={"expval(PauliZ)": 1},
            num_allocs=2,
            depth=2,
        )

    def test_specs_resources_ipython_display(self, example_specs_resource):
        """Test the IPython display of a SpecsResources instance."""
        expected = """\
| **Metric** | **Value** |
| :--- | ---: |
| **Wire allocations** | 2 |
| **Total gates** | 1.000E+5 |
| **Gate counts:** | |
| Hadamard | 1 |
| CNOT | 1.000E+5 |
| **Measurements:** | |
| expval(PauliZ) | 1 |
| **Depth** | 2 |
"""
        actual = example_specs_resource._repr_markdown_()

        assert actual.strip() == expected.strip()

    def test_symbolic_specs_resources_ipython_display(self, example_symbolic_specs_resource):
        """Test the IPython display of a SymbolicSpecsResources instance."""
        expected = """\
| **Metric** | **Value** |
| :--- | ---: |
| **Wire allocations** | 2 |
| **Total gates** | a\\*a\\*b + a\\*a + a + 1 |
| **Gate counts:** | |
| Hadamard | a\\*a\\*b + a\\*a + a |
| CNOT | 1 |
| **Measurements:** | |
| expval(PauliZ) | 1 |
| **Depth** | 2 |
"""
        actual = example_symbolic_specs_resource._repr_markdown_()

        assert actual.strip() == expected.strip()

    def test_single_level_circuit_specs_ipython_display(self, example_specs_resource):
        """Test the IPython display of a single-level CircuitSpecs instance."""
        s = CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level=2,
            resources=example_specs_resource,
        )
        actual = s._repr_markdown_()
        expected = """\
**Circuit Specs:**

| Metric | Value |
| :--- | ---: |
| **Device** | default.qubit |
| **Device wires** | 5 |
| **Shots** | Shots(total=1000) |
| **Level** | 2 |

**Resources:**

| **Metric** | **Value** |
| :--- | ---: |
| **Wire allocations** | 2 |
| **Total gates** | 1.000E+5 |
| **Gate counts:** | |
| Hadamard | 1 |
| CNOT | 1.000E+5 |
| **Measurements:** | |
| expval(PauliZ) | 1 |
| **Depth** | 2 |
"""

        assert actual.strip() == expected.strip()

    def test_batched_circuit_specs_ipython_display(self, example_specs_resource):
        """Test the IPython display of a single-level CircuitSpecs instance."""
        s = CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level=2,
            resources=[example_specs_resource, example_specs_resource],
        )
        actual = s._repr_markdown_()
        expected = """\
**Circuit Specs:**

| Metric | Value |
| :--- | ---: |
| **Device** | default.qubit |
| **Device wires** | 5 |
| **Shots** | Shots(total=1000) |
| **Level** | 2 |

**Resources:**

**Batched tape a:**

| **Metric** | **Value** |
| :--- | ---: |
| **Wire allocations** | 2 |
| **Total gates** | 1.000E+5 |
| **Gate counts:** | |
| Hadamard | 1 |
| CNOT | 1.000E+5 |
| **Measurements:** | |
| expval(PauliZ) | 1 |
| **Depth** | 2 |

**Batched tape b:**

| **Metric** | **Value** |
| :--- | ---: |
| **Wire allocations** | 2 |
| **Total gates** | 1.000E+5 |
| **Gate counts:** | |
| Hadamard | 1 |
| CNOT | 1.000E+5 |
| **Measurements:** | |
| expval(PauliZ) | 1 |
| **Depth** | 2 |
"""

        assert actual.strip() == expected.strip()

    def test_multi_level_circuit_specs_ipython_display(
        self, example_symbolic_specs_resource, example_specs_resource
    ):
        """Test the IPython display of a single-level CircuitSpecs instance."""
        s = CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level={0: "l1", 1: "l2"},
            resources={
                0: example_symbolic_specs_resource,
                1: [example_specs_resource, example_specs_resource],
            },
        )
        actual = s._repr_markdown_()
        expected = """\
**Circuit Specs:**

| Metric | Value |
| :--- | ---: |
| **Device** | default.qubit |
| **Device wires** | 5 |
| **Shots** | Shots(total=1000) |
| **Levels** | |
| 0 | l1 |
| 1 | l2 |

**Resources:**

| ↓Metric / Level→ | 0 | 1-a | 1-b |
| :--- | ---: | ---: | ---: |
| **Wire allocations** | 2 | 2 | 2 |
| **Total gates** | a\\*a\\*b + a\\*a + a + 1 | 1.000E+5 | 1.000E+5 |
| **Gate counts** |  |  |  |
| Hadamard | a\\*a\\*b + a\\*a + a | 1 | 1 |
| CNOT | 1 | 1.000E+5 | 1.000E+5 |
| **Measurements** |  |  |  |
| expval(PauliZ) | 1 | 1 | 1 |
"""

        assert actual.strip() == expected.strip()


class TestCountResources:
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

    scripts = (
        QuantumScript(ops=[], measurements=[]),
        QuantumScript(
            ops=[qp.Hadamard(0), qp.CNOT([0, 1])], measurements=[qp.expval(qp.PauliZ(0))]
        ),
        QuantumScript(
            ops=[qp.PauliZ(0), qp.CNOT([0, 1]), qp.RX(1.23, 2)],
            measurements=[qp.expval(qp.exp(qp.PauliZ(0)))],
            shots=Shots(10),
        ),
        QuantumScript(
            ops=[
                qp.PauliZ(0),
                qp.CNOT([0, 1]),
                qp.RX(1.23, 2),
                _CustomOpWithResource(wires=[0, 2]),
                _CustomOpWithoutResource(wires=[0, 1]),
            ],
            measurements=[qp.probs()],
        ),
        QuantumScript(
            ops=[
                qp.ctrl(op=qp.IsingXX(0.5, wires=[10, 11]), control=range(10)),
                qp.ctrl(op=qp.IsingXX(0.5, wires=[10, 11]), control=range(5)),
                qp.ctrl(op=qp.IsingXX(0.5, wires=[10, 11]), control=[0]),
                qp.CNOT([0, 1]),
                qp.Toffoli([0, 1, 2]),
                qp.ctrl(op=qp.PauliX(10), control=[0]),
                qp.ctrl(op=qp.PauliX(10), control=[0, 1]),
            ],
            measurements=[qp.probs()],
        ),
    )

    expected_resources = (
        SpecsResources({}, {}, {}, 0, 0),
        SpecsResources({"Hadamard": 1, "CNOT": 1}, {1: 1, 2: 1}, {"expval(PauliZ)": 1}, 2, 2),
        SpecsResources(
            {"PauliZ": 1, "CNOT": 1, "RX": 1}, {1: 2, 2: 1}, {"expval(Exp(PauliZ))": 1}, 3, 2
        ),
        SpecsResources(
            {"PauliZ": 3, "CNOT": 1, "RX": 1, "Identity": 1, "CustomOp2": 1},
            {1: 5, 2: 2},
            {"probs(all wires)": 1},
            3,
            6,
        ),
        SpecsResources(
            {"10C(IsingXX)": 1, "5C(IsingXX)": 1, "C(IsingXX)": 1, "CNOT": 2, "Toffoli": 2},
            {12: 1, 7: 1, 3: 3, 2: 2},
            {"probs(all wires)": 1},
            12,
            7,
        ),
    )  # SpecsResources(gate_types, gate_sizes, measurements, num_allocs, depth)

    @pytest.mark.parametrize("script, expected_resources", zip(scripts, expected_resources))
    def test_count_resources(self, script, expected_resources):
        """Test the count resources method."""
        computed_resources = _count_resources(script)
        assert computed_resources == expected_resources

    @pytest.mark.parametrize("script, expected_resources", zip(scripts, expected_resources))
    def test_count_resources_no_depth(self, script, expected_resources):
        """Test the count resources method with depth disabled."""

        computed_resources = _count_resources(script, compute_depth=False)
        expected_resources = SpecsResources(
            gate_types=expected_resources.gate_types,
            gate_sizes=expected_resources.gate_sizes,
            measurements=expected_resources.measurements,
            num_allocs=expected_resources.num_allocs,
        )

        assert computed_resources == expected_resources


def test_count_to_str():
    """Test the _count_to_str helper function."""
    assert _count_to_str(0) == "0"
    assert _count_to_str(999) == "999"
    assert _count_to_str(1_000) == "1,000"
    assert _count_to_str(10_000) == "10,000"
    assert _count_to_str(100_000) == "1.000E+5"
    assert _count_to_str(12_345_678) == "1.235E+7"
    assert _count_to_str(Expression(0)) == "0"
    assert _count_to_str(Expression(15)) == "15"
    assert _count_to_str(Expression(100_000)) == "1.000E+5"
    assert _count_to_str(Expression({("y",): 3, ("x",): 2})) == "3*y + 2*x"


def test_num_to_letters():
    """Test the num_to_letters helper function."""
    assert num_to_letters(0) == "a"
    assert num_to_letters(1) == "b"
    assert num_to_letters(25) == "z"
    assert num_to_letters(26) == "aa"
    assert num_to_letters(27) == "ab"
    assert num_to_letters(51) == "az"
    assert num_to_letters(52) == "ba"

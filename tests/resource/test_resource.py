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
# pylint: disable=unnecessary-dunder-call
from collections import defaultdict
from dataclasses import FrozenInstanceError

import pytest

import pennylane as qml
from pennylane import estimator, resource
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.measurements import Shots
from pennylane.operation import Operation
from pennylane.resource.resource import (
    CircuitSpecs,
    Resources,
    ResourcesOperation,
    SpecsResources,
    _combine_dict,
    _count_resources,
    _scale_dict,
    add_in_parallel,
    add_in_series,
    mul_in_parallel,
    mul_in_series,
    resources_from_tape,
    substitute,
)
from pennylane.tape import QuantumScript

_DEPRECATED_CLASSES_FUNCTIONS = (
    "estimate_shots",
    "estimate_error",
    "FirstQuantization",
    "DoubleFactorization",
)


@pytest.mark.parametrize("item", _DEPRECATED_CLASSES_FUNCTIONS)
def test_deprecated_items(item):
    """Test that accessing the items from the resource module raises
    a deprecation error."""
    with pytest.warns(
        PennyLaneDeprecationWarning,
        match=f"pennylane.resource.{item} is no longer accessible from the resource module",
    ):
        cls_or_fn = getattr(resource, item)

    assert cls_or_fn is getattr(estimator, item)


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
        qml.RX(0.432, wires=0),
        qml.Rot(0.543, 0, 0.23, wires=0),
        qml.CNOT(wires=[0, "a"]),
        qml.RX(0.133, wires=4),
    ]
    obs = [qml.expval(qml.PauliX(wires="a")), qml.probs(wires=[0, "a"])]

    tape = QuantumScript(ops=ops, measurements=obs)
    resources = resources_from_tape(tape, compute_depth=compute_depth)

    assert resources.depth == (3 if compute_depth else None)


###########################################################################
##  Tests for specs dataclasses
###########################################################################


class TestSpecsResources:
    """Test the methods and attributes of the SpecsResource class"""

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

    def test_num_gates(self):
        """Test that the SpecsResources class handles `num_gates` as expected."""

        with pytest.raises(
            ValueError,
            match="Inconsistent gate counts: `gate_types` and `gate_sizes` describe different amounts of gates.",
        ):
            # Gate counts don't match
            _ = SpecsResources(
                gate_types={"Hadamard": 1}, gate_sizes={1: 2}, measurements={}, num_allocs=0
            )

        s = self.example_specs_resource()

        assert s.num_gates == 3

    def test_immutable(self):
        """Test that SpecsResources is immutable."""

        s = self.example_specs_resource()

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

    def test_getitem(self):
        """Test that SpecsResources supports indexing via __getitem__."""

        s = self.example_specs_resource()

        assert s["gate_types"] == s.gate_types
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

    def test_str(self):
        """Test the string representation of a SpecsResources instance."""

        s = self.example_specs_resource()

        expected = "Total wire allocations: 2\n"
        expected += "Total gates: 3\n"
        expected += "Circuit depth: 2\n"
        expected += "\n"
        expected += "Gate types:\n"
        expected += "  Hadamard: 2\n"
        expected += "  CNOT: 1\n"
        expected += "\n"
        expected += "Measurements:\n"
        expected += "  expval(PauliZ): 1"

        expected_indented = ("    " + expected.replace("\n", "\n    ")).replace("\n    \n", "\n\n")

        assert str(s) == expected
        assert s.to_pretty_str() == expected
        assert s.to_pretty_str(preindent=4) == expected_indented

        # Check with no depth, gates, or measurements

        s = SpecsResources(gate_types={}, gate_sizes={}, measurements={}, num_allocs=0)

        expected = "Total wire allocations: 0\n"
        expected += "Total gates: 0\n"
        expected += "Circuit depth: Not computed\n"
        expected += "\n"
        expected += "Gate types:\n"
        expected += "  No gates.\n"
        expected += "\n"
        expected += "Measurements:\n"
        expected += "  No measurements."

        expected_indented = ("    " + expected.replace("\n", "\n    ")).replace("\n    \n", "\n\n")

        assert str(s) == expected
        assert s.to_pretty_str() == expected
        assert s.to_pretty_str(preindent=4) == expected_indented

    def test_to_dict(self):
        """Test the to_dict method of SpecsResources."""

        s = self.example_specs_resource()

        expected = {
            "gate_types": {"Hadamard": 2, "CNOT": 1},
            "gate_sizes": {1: 2, 2: 1},
            "measurements": {"expval(PauliZ)": 1},
            "num_allocs": 2,
            "depth": 2,
            "num_gates": 3,
        }

        assert s.to_dict() == expected

    def test_addition(self):
        """Test the addition of SpecsResources."""
        s0 = self.example_specs_resource()
        s_add = s0 + s0
        s_add_expected = SpecsResources(
            gate_types={"Hadamard": 4, "CNOT": 2},
            gate_sizes={1: 4, 2: 2},
            measurements={"expval(PauliZ)": 2},
            num_allocs=2,
            depth=4,
        )
        assert s_add == s_add_expected

        s_more_alloc = SpecsResources(
            gate_types={"Hadamard": 7, "CNOT": 2},
            gate_sizes={1: 7, 2: 2},
            measurements={"expval(PauliZ)": 2},
            num_allocs=3,
            depth=2,
        )
        s_add = s0 + s_more_alloc
        s_add_expected = SpecsResources(
            gate_types={"Hadamard": 9, "CNOT": 3},
            gate_sizes={1: 9, 2: 3},
            measurements={"expval(PauliZ)": 3},
            num_allocs=3,
            depth=4,
        )
        assert s_add == s_add_expected

        s_no_depth = SpecsResources(
            gate_types={"Hadamard": 7, "CNOT": 2},
            gate_sizes={1: 7, 2: 2},
            measurements={"expval(PauliZ)": 2},
            num_allocs=3,
        )
        s_add = s0 + s_no_depth
        s_add_expected = SpecsResources(
            gate_types={"Hadamard": 9, "CNOT": 3},
            gate_sizes={1: 9, 2: 3},
            measurements={"expval(PauliZ)": 3},
            num_allocs=3,
        )
        assert s_add == s_add_expected


class TestCircuitSpecs:

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

    def example_specs_result_multi(self):
        """Generate an example CircuitSpecs instance with multiple levels and batches."""
        return CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level=[1, 2],
            resources={
                1: SpecsResources(
                    gate_types={"Hadamard": 4, "CNOT": 2},
                    gate_sizes={1: 4, 2: 2},
                    measurements={"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    num_allocs=2,
                    depth=2,
                ),
                3: [
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

    def test_blank_init(self):
        """Test that CircuitSpecss can be instantiated with no arguments."""
        r = CircuitSpecs()  # should not raise any errors

        assert r.device_name is None
        assert r.num_device_wires is None
        assert r.shots is None
        assert r.level is None
        assert r.resources is None

    def test_getitem(self):
        """Test that CircuitSpecs supports indexing via __getitem__."""

        r = self.example_specs_result()

        assert r["device_name"] == r.device_name
        assert r["num_device_wires"] == r.num_device_wires
        assert r["shots"] == r.shots
        assert r["level"] == r.level
        assert r["resources"] == r.resources

    def test_getitem_removed_keys(self):
        """Test that CircuitSpecs raises more descriptive KeyErrors for removed keys."""

        r = self.example_specs_result()

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

    def test_to_dict(self):
        """Test the to_dict method of CircuitSpecs."""

        r = self.example_specs_result()

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

        r = self.example_specs_result_multi()

        expected = {
            "device_name": "default.qubit",
            "num_device_wires": 5,
            "shots": Shots(1000),
            "level": [1, 2],
            "resources": {
                1: {
                    "gate_types": {"Hadamard": 4, "CNOT": 2},
                    "gate_sizes": {1: 4, 2: 2},
                    "measurements": {"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    "num_allocs": 2,
                    "depth": 2,
                    "num_gates": 6,
                },
                3: [
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

    def test_str(self):
        """Test the string representation of a CircuitSpecs instance."""

        r = self.example_specs_result()

        expected = "Device: default.qubit\n"
        expected += "Device wires: 5\n"
        expected += "Shots: Shots(total=1000)\n"
        expected += "Level: 2\n"
        expected += "\n"
        expected += "Resource specifications:\n"
        expected += r.resources.to_pretty_str(preindent=2)

        assert str(r) == expected

    def test_str_multi(self):
        """Test the string representation of a CircuitSpecs instance."""

        r = self.example_specs_result_multi()

        expected = "Device: default.qubit\n"
        expected += "Device wires: 5\n"
        expected += "Shots: Shots(total=1000)\n"
        expected += "Level: [1, 2]\n"
        expected += "\n"
        expected += "Resource specifications:\n"

        expected += "Level = 1:\n"
        expected += r.resources[1].to_pretty_str(preindent=2)

        expected += "\n\n" + "-" * 60 + "\n\n"

        expected += "Level = 3:\n"
        expected += "  Batched tape 0:\n"
        expected += r.resources[3][0].to_pretty_str(preindent=4)
        expected += "\n\n  Batched tape 1:\n"
        expected += r.resources[3][1].to_pretty_str(preindent=4)

        assert str(r) == expected


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
            ops=[qml.Hadamard(0), qml.CNOT([0, 1])], measurements=[qml.expval(qml.PauliZ(0))]
        ),
        QuantumScript(
            ops=[qml.PauliZ(0), qml.CNOT([0, 1]), qml.RX(1.23, 2)],
            measurements=[qml.expval(qml.exp(qml.PauliZ(0)))],
            shots=Shots(10),
        ),
        QuantumScript(
            ops=[
                qml.PauliZ(0),
                qml.CNOT([0, 1]),
                qml.RX(1.23, 2),
                _CustomOpWithResource(wires=[0, 2]),
                _CustomOpWithoutResource(wires=[0, 1]),
            ],
            measurements=[qml.probs()],
        ),
        QuantumScript(
            ops=[
                qml.ctrl(op=qml.IsingXX(0.5, wires=[10, 11]), control=range(10)),
                qml.ctrl(op=qml.IsingXX(0.5, wires=[10, 11]), control=range(5)),
                qml.ctrl(op=qml.IsingXX(0.5, wires=[10, 11]), control=[0]),
                qml.CNOT([0, 1]),
                qml.Toffoli([0, 1, 2]),
                qml.ctrl(op=qml.PauliX(10), control=[0]),
                qml.ctrl(op=qml.PauliX(10), control=[0, 1]),
            ],
            measurements=[qml.probs()],
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

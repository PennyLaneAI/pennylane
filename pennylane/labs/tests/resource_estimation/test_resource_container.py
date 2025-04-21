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
# pylint:disable=protected-access, no-self-use, too-few-public-methods
import copy
from collections import defaultdict

import pytest

import pennylane as qml
from pennylane.labs.resource_estimation.resource_container import (
    CompressedResourceOp,
    Resources,
    _combine_dict,
    _scale_dict,
    add_in_parallel,
    add_in_series,
    mul_in_parallel,
    mul_in_series,
    substitute,
)
from pennylane.labs.resource_estimation.resource_operator import ResourceOperator
from pennylane.operation import Operator


class ResourceDummyX(Operator, ResourceOperator):
    """Dummy testing class representing X gate"""


class ResourceDummyQFT(Operator, ResourceOperator):
    """Dummy testing class representing QFT gate"""


class ResourceDummyQSVT(Operator, ResourceOperator):
    """Dummy testing class representing QSVT gate"""


class ResourceDummyTrotterProduct(Operator, ResourceOperator):
    """Dummy testing class representing TrotterProduct gate"""


class ResourceDummyAdjoint(Operator, ResourceOperator):
    """Dummy testing class representing the Adjoint symbolic operator"""


class TestCompressedResourceOp:
    """Testing the methods and attributes of the CompressedResourceOp class"""

    test_hamiltonian = qml.dot([1, -1, 0.5], [qml.X(0), qml.Y(1), qml.Z(0) @ qml.Z(1)])
    compressed_ops_and_params_lst = (
        ("DummyX", ResourceDummyX, {"num_wires": 1}, None),
        ("DummyQFT", ResourceDummyQFT, {"num_wires": 5}, None),
        ("DummyQSVT", ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5}, None),
        (
            "DummyTrotterProduct",
            ResourceDummyTrotterProduct,
            {"Hamiltonian": test_hamiltonian, "num_steps": 5, "order": 2},
            None,
        ),
        ("X", ResourceDummyX, {"num_wires": 1}, "X"),
    )

    compressed_op_reprs = (
        "DummyX",
        "DummyQFT",
        "DummyQSVT",
        "DummyTrotterProduct",
        "X",
    )

    @pytest.mark.parametrize("name, op_type, parameters, name_param", compressed_ops_and_params_lst)
    def test_init(self, name, op_type, parameters, name_param):
        """Test that we can correctly instantiate CompressedResourceOp"""
        cr_op = CompressedResourceOp(op_type, parameters, name=name_param)

        assert cr_op._name == name
        assert cr_op.op_type is op_type
        assert cr_op.params == parameters
        assert cr_op._hashable_params == tuple(sorted((str(k), v) for k, v in parameters.items()))

    def test_hash(self):
        """Test that the hash method behaves as expected"""
        CmprssedQSVT1 = CompressedResourceOp(ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5})
        CmprssedQSVT2 = CompressedResourceOp(ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5})
        Other = CompressedResourceOp(ResourceDummyQFT, {"num_wires": 3})

        assert hash(CmprssedQSVT1) == hash(CmprssedQSVT1)  # compare same object
        assert hash(CmprssedQSVT1) == hash(CmprssedQSVT2)  # compare identical instance
        assert hash(CmprssedQSVT1) != hash(Other)

        # test dictionary as parameter
        CmprssedAdjoint1 = CompressedResourceOp(
            ResourceDummyAdjoint, {"base_class": ResourceDummyQFT, "base_params": {"num_wires": 1}}
        )
        CmprssedAdjoint2 = CompressedResourceOp(
            ResourceDummyAdjoint, {"base_class": ResourceDummyQFT, "base_params": {"num_wires": 1}}
        )
        Other = CompressedResourceOp(
            ResourceDummyAdjoint, {"base_class": ResourceDummyQFT, "base_params": {"num_wires": 2}}
        )

        assert hash(CmprssedAdjoint1) == hash(CmprssedAdjoint1)
        assert hash(CmprssedAdjoint1) == hash(CmprssedAdjoint2)
        assert hash(CmprssedAdjoint1) != hash(Other)

    def test_equality(self):
        """Test that the equality methods behaves as expected"""
        CmprssedQSVT1 = CompressedResourceOp(ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5})
        CmprssedQSVT2 = CompressedResourceOp(ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5})
        CmprssedQSVT3 = CompressedResourceOp(ResourceDummyQSVT, {"num_angles": 5, "num_wires": 3})
        Other = CompressedResourceOp(ResourceDummyQFT, {"num_wires": 3})

        assert CmprssedQSVT1 == CmprssedQSVT2  # compare identical instance
        assert CmprssedQSVT1 == CmprssedQSVT3  # compare swapped parameters
        assert CmprssedQSVT1 != Other

    @pytest.mark.parametrize("args, repr", zip(compressed_ops_and_params_lst, compressed_op_reprs))
    def test_repr(self, args, repr):
        """Test that the repr method behaves as expected."""
        _, op_type, parameters, name_param = args
        cr_op = CompressedResourceOp(op_type, parameters, name=name_param)

        assert str(cr_op) == repr


class TestResources:
    """Test the methods and attributes of the Resource class"""

    resource_quantities = (
        Resources(),
        Resources(5, 0, defaultdict(int, {})),
        Resources(1, 3, defaultdict(int, {"Hadamard": 1, "PauliZ": 2})),
        Resources(4, 2, defaultdict(int, {"Hadamard": 1, "CNOT": 1})),
    )

    resource_parameters = (
        (0, 0, defaultdict(int, {})),
        (5, 0, defaultdict(int, {})),
        (1, 3, defaultdict(int, {"Hadamard": 1, "PauliZ": 2})),
        (4, 2, defaultdict(int, {"Hadamard": 1, "CNOT": 1})),
    )

    @pytest.mark.parametrize("r, attribute_tup", zip(resource_quantities, resource_parameters))
    def test_init(self, r, attribute_tup):
        """Test that the Resource class is instantiated as expected."""
        num_wires, num_gates, gate_types = attribute_tup

        assert r.num_wires == num_wires
        assert r.num_gates == num_gates
        assert r.gate_types == gate_types

    expected_results_add_series = (
        Resources(2, 6, defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1})),
        Resources(5, 6, defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1})),
        Resources(
            2, 9, defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 2, "PauliZ": 2})
        ),
        Resources(4, 8, defaultdict(int, {"RZ": 2, "CNOT": 2, "RY": 2, "Hadamard": 2})),
    )

    @pytest.mark.parametrize("in_place", (False, True))
    @pytest.mark.parametrize(
        "resource_obj, expected_res_obj", zip(resource_quantities, expected_results_add_series)
    )
    def test_add_in_series(self, resource_obj, expected_res_obj, in_place):
        """Test the add_in_series function works with Resoruces"""
        resource_obj = copy.deepcopy(resource_obj)
        other_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
        )

        resultant_obj = add_in_series(resource_obj, other_obj, in_place=in_place)
        assert resultant_obj == expected_res_obj

        if in_place:
            assert resultant_obj is resource_obj

    expected_results_add_parallel = (
        Resources(2, 6, defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1})),
        Resources(7, 6, defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1})),
        Resources(
            3, 9, defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 2, "PauliZ": 2})
        ),
        Resources(6, 8, defaultdict(int, {"RZ": 2, "CNOT": 2, "RY": 2, "Hadamard": 2})),
    )

    @pytest.mark.parametrize("in_place", (False, True))
    @pytest.mark.parametrize(
        "resource_obj, expected_res_obj", zip(resource_quantities, expected_results_add_parallel)
    )
    def test_add_in_parallel(self, resource_obj, expected_res_obj, in_place):
        """Test the add_in_parallel function works with Resoruces"""
        resource_obj = copy.deepcopy(resource_obj)
        other_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
        )

        resultant_obj = add_in_parallel(resource_obj, other_obj, in_place=in_place)
        assert resultant_obj == expected_res_obj

        if in_place:
            assert resultant_obj is resource_obj

    expected_results_mul_series = (
        Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
        ),
        Resources(
            num_wires=2,
            num_gates=12,
            gate_types=defaultdict(int, {"RZ": 4, "CNOT": 2, "RY": 4, "Hadamard": 2}),
        ),
        Resources(
            num_wires=2,
            num_gates=18,
            gate_types=defaultdict(int, {"RZ": 6, "CNOT": 3, "RY": 6, "Hadamard": 3}),
        ),
        Resources(
            num_wires=2,
            num_gates=30,
            gate_types=defaultdict(int, {"RZ": 10, "CNOT": 5, "RY": 10, "Hadamard": 5}),
        ),
    )

    @pytest.mark.parametrize("in_place", (False, True))
    @pytest.mark.parametrize(
        "scalar, expected_res_obj", zip((1, 2, 3, 5), expected_results_mul_series)
    )
    def test_mul_in_series(self, scalar, expected_res_obj, in_place):
        """Test the mul_in_series function works with Resoruces"""
        resource_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
        )

        resultant_obj = mul_in_series(resource_obj, scalar, in_place=in_place)
        assert resultant_obj == expected_res_obj

        if in_place:
            assert resultant_obj is resource_obj
        assert True

    expected_results_mul_parallel = (
        Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
        ),
        Resources(
            num_wires=4,
            num_gates=12,
            gate_types=defaultdict(int, {"RZ": 4, "CNOT": 2, "RY": 4, "Hadamard": 2}),
        ),
        Resources(
            num_wires=6,
            num_gates=18,
            gate_types=defaultdict(int, {"RZ": 6, "CNOT": 3, "RY": 6, "Hadamard": 3}),
        ),
        Resources(
            num_wires=10,
            num_gates=30,
            gate_types=defaultdict(int, {"RZ": 10, "CNOT": 5, "RY": 10, "Hadamard": 5}),
        ),
    )

    @pytest.mark.parametrize("in_place", (False, True))
    @pytest.mark.parametrize(
        "scalar, expected_res_obj", zip((1, 2, 3, 5), expected_results_mul_parallel)
    )
    def test_mul_in_parallel(self, scalar, expected_res_obj, in_place):
        """Test the mul_in_parallel function works with Resoruces"""
        resource_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
        )

        resultant_obj = mul_in_parallel(resource_obj, scalar, in_place=in_place)
        assert resultant_obj == expected_res_obj

        if in_place:
            assert resultant_obj is resource_obj
        assert True

    test_str_data = (
        ("wires: 0\n" + "gates: 0\n" + "gate_types:\n" + "{}"),
        ("wires: 5\n" + "gates: 0\n" + "gate_types:\n" + "{}"),
        ("wires: 1\n" + "gates: 3\n" + "gate_types:\n" + "{'Hadamard': 1, 'PauliZ': 2}"),
        ("wires: 4\n" + "gates: 2\n" + "gate_types:\n" + "{'Hadamard': 1, 'CNOT': 1}"),
    )

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_str_data))
    def test_str(self, r, rep):
        """Test the string representation of a Resources instance."""
        assert str(r) == rep

    @pytest.mark.parametrize("r, rep", zip(resource_quantities, test_str_data))
    def test_ipython_display(self, r, rep, capsys):
        """Test that the ipython display prints the string representation of a Resources instance."""
        r._ipython_display_()  # pylint: disable=protected-access
        captured = capsys.readouterr()
        assert rep in captured.out

    gate_names = ("RX", "RZ")
    expected_results_sub = (
        Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
        ),
        Resources(
            num_wires=2,
            num_gates=14,
            gate_types=defaultdict(int, {"RX": 10, "CNOT": 1, "RY": 2, "Hadamard": 1}),
        ),
    )

    @pytest.mark.parametrize("gate_name, expected_res_obj", zip(gate_names, expected_results_sub))
    def test_substitute(self, gate_name, expected_res_obj):
        """Test the substitute function"""
        resource_obj = Resources(
            num_wires=2,
            num_gates=6,
            gate_types=defaultdict(int, {"RZ": 2, "CNOT": 1, "RY": 2, "Hadamard": 1}),
        )

        sub_obj = Resources(
            num_wires=1,
            num_gates=5,
            gate_types=defaultdict(int, {"RX": 5}),
        )

        resultant_obj1 = substitute(resource_obj, gate_name, sub_obj, in_place=False)
        assert resultant_obj1 == expected_res_obj

        resultant_obj2 = substitute(resource_obj, gate_name, sub_obj, in_place=True)
        assert resultant_obj2 == expected_res_obj
        assert resultant_obj2 is resource_obj


@pytest.mark.parametrize("in_place", [False, True])
def test_combine_dict(in_place):
    """Test that we can combine dictionaries as expected."""
    d1 = defaultdict(int, {"a": 2, "b": 4, "c": 6})
    d2 = defaultdict(int, {"a": 1, "b": 2, "d": 3})

    result = _combine_dict(d1, d2, in_place=in_place)
    expected = defaultdict(int, {"a": 3, "b": 6, "c": 6, "d": 3})

    assert result == expected

    if in_place:
        assert result is d1
    else:
        assert result is not d1


@pytest.mark.parametrize("scalar", (1, 2, 3))
@pytest.mark.parametrize("in_place", (False, True))
def test_scale_dict(scalar, in_place):
    """Test that we can scale the values of a dictionary as expected."""
    d1 = defaultdict(int, {"a": 2, "b": 4, "c": 6})

    expected = defaultdict(int, {k: scalar * v for k, v in d1.items()})
    result = _scale_dict(d1, scalar, in_place=in_place)

    assert result == expected

    if in_place:
        assert result is d1
    else:
        assert result is not d1

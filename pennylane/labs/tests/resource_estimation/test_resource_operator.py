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
Test the base and abstract Resource class
"""
from typing import Hashable

import numpy as np
import pytest

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator
from pennylane.labs.resource_estimation.resource_operator import _make_hashable

# pylint: disable=protected-access, too-few-public-methods, no-self-use


class ResourceDummyX(ResourceOperator):
    """Dummy testing class representing X gate"""


class ResourceDummyQFT(ResourceOperator):
    """Dummy testing class representing QFT gate"""


class ResourceDummyQSVT(ResourceOperator):
    """Dummy testing class representing QSVT gate"""


class ResourceDummyTrotterProduct(ResourceOperator):
    """Dummy testing class representing TrotterProduct gate"""


class ResourceDummyAdjoint(ResourceOperator):
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

    compressed_op_names = (
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
        assert sorted(cr_op._hashable_params) == sorted(tuple(parameters.items()))

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

    @pytest.mark.parametrize("args, name", zip(compressed_ops_and_params_lst, compressed_op_names))
    def test_name(self, args, name):
        """Test that the name method behaves as expected."""
        _, op_type, parameters, name_param = args
        cr_op = CompressedResourceOp(op_type, parameters, name=name_param)

        assert cr_op.name == name

    @pytest.mark.parametrize("args", compressed_ops_and_params_lst)
    def test_repr(self, args):
        """Test that the name method behaves as expected."""
        _, op_type, parameters, name_param = args

        cr_op = CompressedResourceOp(op_type, parameters, name=name_param)

        op_name = op_type.__name__
        expected_params_str_parts = [f"{k!r}:{v!r}" for k, v in sorted(parameters.items())]
        expected_params_str = ", ".join(expected_params_str_parts)

        expected_repr_string = f"CompressedResourceOp({op_name}, params={{{expected_params_str}}})"
        assert str(cr_op) == expected_repr_string

    def test_type_error(self):
        """Test that an error is raised if wrong type is provided for op_type."""
        with pytest.raises(TypeError, match="op_type must be a subclass of ResourceOperator."):
            CompressedResourceOp(type(1))


@pytest.mark.parametrize(
    "input_obj, expected_hashable",
    [
        (123, 123),
        ("hello", "hello"),
        (3.14, 3.14),
        (None, None),
        ((1, 2, 3), (1, 2, 3)),
        ([], ()),
        ([1, 2, 3], (1, 2, 3)),
        ([[1, 2], [3, 4]], ((1, 2), (3, 4))),
        (set(), ()),
        ({3, 1, 2}, (1, 2, 3)),
        ({}, ()),
        ({"b": 2, "a": 1}, (("a", 1), ("b", 2))),
        ({"key": [1, 2]}, (("key", (1, 2)),)),
        ({"nested": {"x": 1, "y": [2, 3]}}, (("nested", (("x", 1), ("y", (2, 3)))),)),
        (np.array([1, 2, 3]), (1, 2, 3)),
        (np.array([["a", "b"], ["c", "d"]]), (("a", "b"), ("c", "d"))),
        ([{"a": 1}, {2, 3}], ((("a", 1),), (2, 3))),
        (
            {"list_key": [1, {"nested_dict": "val"}]},
            (("list_key", (1, (("nested_dict", "val"),))),),
        ),
    ],
)
def test_make_hashable(input_obj, expected_hashable):
    """Test that _make_hashable function works as expected"""
    result = _make_hashable(input_obj)
    assert result == expected_hashable
    assert isinstance(result, Hashable)
    assert hash(result) is not None

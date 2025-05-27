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
import pytest

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator
from pennylane.operation import Operator

# pylint: disable=protected-access


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

    @pytest.mark.parametrize("args, repr", zip(compressed_ops_and_params_lst, compressed_op_reprs))
    def test_repr(self, args, repr):
        """Test that the repr method behaves as expected."""
        _, op_type, parameters, name_param = args
        cr_op = CompressedResourceOp(op_type, parameters, name=name_param)

        assert str(cr_op) == repr

    def test_type_error(self):
        """Test that an error is raised if wrong type is provided for op_type."""
        with pytest.raises(TypeError, match="op_type must be a subclass of ResourceOperator."):
            CompressedResourceOp(type(1))

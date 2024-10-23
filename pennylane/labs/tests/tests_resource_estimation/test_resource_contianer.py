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

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp


class TestCompressedResourceOp:

    hamiltonian_arg = qml.dot([1, -1, 0.5], [qml.X(0), qml.Y(1), qml.Z(0)@qml.Z(1)])
    compressed_op_args_lst = (
        ("X", qml.X, tuple({"num_wires": 1})),
        ("QFT", qml.QFT, tuple({"num_wires": 5})),
        ("QSVT", qml.QSVT, tuple({"num_wires": 3, "num_angles":5})),
        (qml.TrotterProduct, tuple({"Hamiltonian": hamiltonian_arg, "num_steps": 5, "order": 2}))
    )

    @pytest.mark.parametrize("name, op_type, parameters", compressed_op_args_lst)
    def test_init(self, name, op_type, parameters):
        """Test that we can correctly instantiate CompressedResourceOp"""
        cr_op = CompressedResourceOp(op_type, parameters)

        assert cr_op._name == name
        assert cr_op.op_type is op_type
        assert cr_op.params == parameters

    def test_hash(self):
        """Test that the hash method behaves as expected"""
        CmprssedQSVT1 = CompressedResourceOp(qml.QSVT, tuple({"num_wires": 3, "num_angles":5}))
        CmprssedQSVT2 = CompressedResourceOp(qml.QSVT, tuple({"num_wires": 3, "num_angles":5}))
        Other = CompressedResourceOp(qml.QFT, tuple({"num_wires": 3}))

        assert hash(CmprssedQSVT1) == hash(CmprssedQSVT1)
        assert hash(CmprssedQSVT1) == hash(CmprssedQSVT2)
        assert hash(CmprssedQSVT1) != hash(Other)
    
    def test_equality(self):
        """Test that the equality methods behaves as expected"""
        
        assert True

    def test_repr(self):
        """Test that the repr method behaves as expected."""
        assert True
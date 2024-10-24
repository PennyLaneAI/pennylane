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

from collections import defaultdict

import pytest

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, Resources


class TestCompressedResourceOp:

    hamiltonian_arg = qml.dot([1, -1, 0.5], [qml.X(0), qml.Y(1), qml.Z(0) @ qml.Z(1)])
    compressed_op_args_lst = (
        ("PauliX", qml.X, {"num_wires": 1}),
        ("QFT", qml.QFT, {"num_wires": 5}),
        ("QSVT", qml.QSVT, {"num_wires": 3, "num_angles": 5}),
        (
            "TrotterProduct",
            qml.TrotterProduct,
            {"Hamiltonian": hamiltonian_arg, "num_steps": 5, "order": 2},
        ),
    )

    compressed_op_reprs = (
        "PauliX(num_wires=1)",
        "QFT(num_wires=5)",
        "QSVT(num_wires=3, num_angles=5)",
        "TrotterProduct(Hamiltonian=X(0) + -1 * Y(1) + 0.5 * (Z(0) @ Z(1)), num_steps=5, order=2)",
    )

    @pytest.mark.parametrize("name, op_type, parameters", compressed_op_args_lst)
    def test_init(self, name, op_type, parameters):
        """Test that we can correctly instantiate CompressedResourceOp"""
        cr_op = CompressedResourceOp(op_type, parameters)

        assert cr_op._name == name
        assert cr_op.op_type is op_type
        assert cr_op.params == parameters
        assert cr_op._hashable_params == tuple(parameters.items())

    def test_hash(self):
        """Test that the hash method behaves as expected"""
        CmprssedQSVT1 = CompressedResourceOp(qml.QSVT, {"num_wires": 3, "num_angles": 5})
        CmprssedQSVT2 = CompressedResourceOp(qml.QSVT, {"num_wires": 3, "num_angles": 5})
        Other = CompressedResourceOp(qml.QFT, {"num_wires": 3})

        assert hash(CmprssedQSVT1) == hash(CmprssedQSVT1)  # compare same object
        assert hash(CmprssedQSVT1) == hash(CmprssedQSVT2)  # compare identical instance
        assert hash(CmprssedQSVT1) != hash(Other)

    def test_equality(self):
        """Test that the equality methods behaves as expected"""
        CmprssedQSVT1 = CompressedResourceOp(qml.QSVT, {"num_wires": 3, "num_angles": 5})
        CmprssedQSVT2 = CompressedResourceOp(qml.QSVT, {"num_wires": 3, "num_angles": 5})
        CmprssedQSVT3 = CompressedResourceOp(qml.QSVT, {"num_angles": 5, "num_wires": 3})
        Other = CompressedResourceOp(qml.QFT, {"num_wires": 3})

        assert CmprssedQSVT1 == CmprssedQSVT1  # compare same object
        assert CmprssedQSVT1 == CmprssedQSVT2  # compare identical instance
        assert CmprssedQSVT1 == CmprssedQSVT3  # compare swapped parameters
        assert CmprssedQSVT1 != Other

    @pytest.mark.parametrize("args, repr", zip(compressed_op_args_lst, compressed_op_reprs))
    def test_repr(self, args, repr):
        """Test that the repr method behaves as expected."""
        _, op_type, parameters = args
        cr_op = CompressedResourceOp(op_type, parameters)

        assert str(cr_op) == repr


class TestResources:
    """Test the methods and attributes of the Resource class"""

    resource_quantities = (
        Resources(),
        Resources(5, 0, {}),
        Resources(
            1,
            3,
            defaultdict(int, {"Hadamard": 1, "PauliZ": 2}),
        ),
        Resources(4, 2, {"Hadamard": 1, "CNOT": 1}),
    )

    resource_parameters = (
        (0, 0, {}),
        (5, 0, {}),
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

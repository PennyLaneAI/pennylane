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
Unit tests for the comm function
"""
# pylint: disable=too-many-arguments
import pytest

import pennylane as qml
from pennylane.ops import SProd, Sum
from pennylane.operation import Operator
from pennylane.pauli import PauliWord, PauliSentence

X, Y, Z, Id = qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity

X0 = PauliWord({0: "X"})
Y0 = PauliWord({0: "Y"})
Z0 = PauliWord({0: "Z"})
pw_id = PauliWord({})


def _pauli_to_op(p):
    """convert PauliWord or PauliSentence to Operator"""
    return p.operation()


def _pw_to_ps(p):
    """convert PauliWord to PauliSentence"""
    return PauliSentence({p: 1.0})


def _id(p):
    """Leave operator as is"""
    # this is used for parametrization of tests
    return p


class TestLegacySupport:
    """Test support for legacy operator classes like Tensor and Hamiltonian"""

    def test_Hamiltonian_single(self):
        """Test that Hamiltonians get transformed to new operator classes and return the correct result"""
        H1 = qml.Hamiltonian([1.0], [qml.PauliX(0)])
        H2 = qml.Hamiltonian([1.0], [qml.PauliY(0)])
        res = qml.comm(H1, H2)
        true_res = qml.s_prod(2j, qml.PauliZ(0))
        assert isinstance(res, SProd)
        assert true_res == res

    @pytest.mark.xfail
    def test_Hamiltonian_sum(self):
        """Test that Hamiltonians with Tensors and sums get transformed to new operator classes and return the correct result"""
        H1 = qml.Hamiltonian([1.0], [qml.PauliX(0) @ qml.PauliX(1)])
        H2 = qml.Hamiltonian([1.0], [qml.PauliY(0) + qml.PauliY(1)])
        true_res = qml.sum(qml.s_prod(2j, qml.PauliZ(0) @ qml.PauliX(1)), qml.s_prod(2j, qml.PauliX(0) @ qml.PauliZ(1)))
        res = qml.comm(H1, H2).simplify()
        assert isinstance(res, Sum)
        assert qml.equal(true_res, res) # issue https://github.com/PennyLaneAI/pennylane/issues/5060 as well as potential fix https://github.com/PennyLaneAI/pennylane/pull/5037


class TestcommPauli:
    """Test qml.comm for pauli=True"""

    data_pauli_relations = (
        # word and word
        (X0, X0, PauliSentence({pw_id: 0})),
        (Y0, Y0, PauliSentence({pw_id: 0})),
        (Z0, Z0, PauliSentence({pw_id: 0})),
        (X0, Y0, PauliSentence({Z0: 2j})),
        (Y0, Z0, PauliSentence({X0: 2j})),
        (Z0, X0, PauliSentence({Y0: 2j})),
        (Y0, X0, PauliSentence({Z0: -2j})),
        (Z0, Y0, PauliSentence({X0: -2j})),
        (X0, Z0, PauliSentence({Y0: -2j})),
    )

    @pytest.mark.parametrize("transform_type1", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("transform_type2", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations)
    def test_basic_comm_relations(self, op1, op2, true_res, transform_type1, transform_type2):
        """Test basic comm relations between Paulis for PauliWord, PauliSentence and Operator instances"""
        res = qml.comm(transform_type1(op1), transform_type2(op2), pauli=True)
        assert res == true_res
        assert isinstance(res, PauliSentence)

    data_more_comm_relations = (
        (
            PauliWord({0: "X", 1: "X"}),
            PauliWord({0: "Y", 1: "Y"}),
            PauliSentence({PauliWord({0: "Z", 1: "Z"}): 0.0}),
        ),
        (
            PauliWord({0: "X", 1: "X"}),
            PauliWord({"a": "X", "b": "Y"}),
            PauliSentence({PauliWord({0: "X", 1: "X", "a": "X", "b": "Y"}): 0.0}),
        ),
    )

    @pytest.mark.parametrize("transform_type1", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("transform_type2", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_more_comm_relations)
    def test_comm_relations_pauli_words(
        self, op1, op2, true_res, transform_type1, transform_type2
    ):
        """Test more comm relations between Paulis"""
        res = qml.comm(transform_type1(op1), transform_type2(op2), pauli=True)
        assert res == true_res
        assert isinstance(res, PauliSentence)


class TestcommPauliFalseSimplify:
    """Test qml.comm for pauli=False (default behavior)"""

    data_pauli_relations_ops = (
        # word and word
        (X0, X0, qml.s_prod(0.0, Id(0))),
        (Y0, Y0, qml.s_prod(0.0, Id(0))),
        (Z0, Z0, qml.s_prod(0.0, Id(0))),
        (X0, Y0, qml.s_prod(2j, Z(0))),
        (Y0, Z0, qml.s_prod(2j, X(0))),
        (Z0, X0, qml.s_prod(2j, Y(0))),
        (Y0, X0, qml.s_prod(-2j, Z(0))),
        (Z0, Y0, qml.s_prod(-2j, X(0))),
        (X0, Z0, qml.s_prod(-2j, Y(0))),
    )

    @pytest.mark.parametrize("transform_type1", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("transform_type2", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations_ops)
    def test_basic_comm_relations(self, op1, op2, true_res, transform_type1, transform_type2):
        """Test basic comm relations between Paulis for PauliWord, PauliSentence and Operator instances"""
        res = qml.comm(transform_type1(op1), transform_type2(op2), pauli=False)
        assert res == true_res
        assert isinstance(res, Operator)
        assert isinstance(res, SProd)

    data_more_comm_relations_op = (
        (
            PauliWord({0: "X", 1: "X"}),
            PauliWord({0: "Y", 1: "Y"}),
            qml.s_prod(0.0, Id([0, 1])),
        ),
        (
            PauliWord({0: "X", 1: "X"}),
            PauliWord({"a": "X", "b": "Y"}),
            qml.s_prod(0.0, Id([0, 1, "a", "b"])),
        ),
    )

    @pytest.mark.parametrize("transform_type1", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("transform_type2", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_more_comm_relations_op)
    def test_comm_relations_pauli_words(
        self, op1, op2, true_res, transform_type1, transform_type2
    ):
        """Test more comm relations between Paulis"""
        res = qml.comm(transform_type1(op1), transform_type2(op2), pauli=False)
        assert res == true_res
        assert isinstance(res, Operator)
        assert isinstance(res, SProd)

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
Unit tests for the dot function
"""
import pytest

import pennylane as qml
from pennylane.ops import SProd
from pennylane.operation import Operator
from pennylane.pauli import PauliWord, PauliSentence

X, Y, Z, Id = qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity

X0 = PauliWord({0: "X"})
Y0 = PauliWord({0: "Y"})
Z0 = PauliWord({0: "Z"})
pw_id = PauliWord({})


def _pauli_to_op(p):
    return p.operation()


def _pw_to_ps(p):
    return PauliSentence({p: 1.0})


def _id(p):
    return p


class TestCommutatorPauli:
    """Test qml.commutator for pauli=True"""

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
    def test_basic_commutator_relations(self, op1, op2, true_res, transform_type1, transform_type2):
        """Test basic commutator relations between Paulis for PauliWord, PauliSentence and Operator instances"""
        res = qml.commutator(transform_type1(op1), transform_type2(op2), pauli=True)
        assert res == true_res
        assert isinstance(res, PauliSentence)

    data_more_commutator_relations = (
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
    @pytest.mark.parametrize("op1, op2, true_res", data_more_commutator_relations)
    def test_commutator_relations_pauli_words(
        self, op1, op2, true_res, transform_type1, transform_type2
    ):
        """Test more commutator relations between Paulis"""
        res = qml.commutator(transform_type1(op1), transform_type2(op2), pauli=True)
        assert res == true_res
        assert isinstance(res, PauliSentence)


class TestCommutatorPauliFalseSimplify:
    """Test qml.commutator for pauli=False (default behavior)"""

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
    def test_basic_commutator_relations(self, op1, op2, true_res, transform_type1, transform_type2):
        """Test basic commutator relations between Paulis for PauliWord, PauliSentence and Operator instances"""
        res = qml.commutator(transform_type1(op1), transform_type2(op2), pauli=False)
        assert res == true_res
        assert isinstance(res, Operator)
        assert isinstance(res, SProd)

    data_more_commutator_relations_op = (
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
    @pytest.mark.parametrize("op1, op2, true_res", data_more_commutator_relations_op)
    def test_commutator_relations_pauli_words(
        self, op1, op2, true_res, transform_type1, transform_type2
    ):
        """Test more commutator relations between Paulis"""
        res = qml.commutator(transform_type1(op1), transform_type2(op2), pauli=False)
        assert res == true_res
        assert isinstance(res, Operator)
        assert isinstance(res, SProd)

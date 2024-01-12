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
from pennylane.pauli import PauliWord, PauliSentence

X0 = PauliWord({0: "X"})
Y0 = PauliWord({0: "Y"})
Z0 = PauliWord({0: "Z"})

pw1 = PauliWord({0: "I", 1: "X", 2: "Y"})
pw2 = PauliWord({"a": "X", "b": "X", "c": "Z"})
pw3 = PauliWord({0: "Z", "b": "Z", "c": "Z"})
pw4 = PauliWord({})
pw_id = pw4  # Identity PauliWord

words = [pw1, pw2, pw3, pw4]

words = [pw1, pw2, pw3, pw4]

ps1 = PauliSentence({pw1: 1.23, pw2: 4j, pw3: -0.5})
ps2 = PauliSentence({pw1: -1.23, pw2: -4j, pw3: 0.5})
ps1_hamiltonian = PauliSentence({pw1: 1.23, pw2: 4, pw3: -0.5})
ps2_hamiltonian = PauliSentence({pw1: -1.23, pw2: -4, pw3: 0.5})
ps3 = PauliSentence({pw3: -0.5, pw4: 1})
ps4 = PauliSentence({pw4: 1})
ps5 = PauliSentence({})


def _pauli_to_op(p):
    return p.operation()


def _pw_to_ps(p):
    return PauliSentence({p: 1.0})


def _id(p):
    return p


sentences = [ps1, ps2, ps3, ps4, ps5, ps1_hamiltonian, ps2_hamiltonian]


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
    """Test qml.commutator for pauli=False and simplify=True (default behavior)"""


class TestCommutatorPauliFalseSimplifyFalse:
    """Test qml.commutator for pauli=False and simplify=False"""

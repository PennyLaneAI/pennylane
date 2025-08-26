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
# pylint: disable=too-many-arguments, unused-variable
import pytest

import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops import SProd
from pennylane.pauli import PauliSentence, PauliWord

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


def test_alias():
    """Test that the alias qml.comm() works as expected"""
    res1 = qml.comm(X0, Y0)
    res1_true = qml.commutator(X0, Y0)
    assert res1 == res1_true
    res2 = qml.comm(X0, Y0, pauli=True)
    res2_true = qml.commutator(X0, Y0, pauli=True)
    assert res2 == res2_true


def test_no_recording_in_context():
    """Test that commutator is not recorded"""
    with qml.queuing.AnnotatedQueue() as q1:
        a = qml.PauliX(0)  # gets recorded
        b = qml.PauliY(0)  # gets recorded
        comm = qml.commutator(a, b)

    with qml.queuing.AnnotatedQueue() as q2:
        qml.PauliX(0)
        qml.PauliY(0)

    expected = [qml.X(0), qml.Y(0)]
    for op1, op2, exp_op in zip(q1.queue, q2.queue, expected, strict=True):
        qml.assert_equal(op1, exp_op)
        qml.assert_equal(op2, exp_op)


def test_no_recording_in_context_with_pauli():
    """Test that commutator is not recorded while one of the ops is a Pauli"""
    with qml.queuing.AnnotatedQueue() as q1:
        a = qml.PauliX(0)  # gets recorded
        b = PauliWord({0: "Y"})  # does not get recorded
        comm = qml.commutator(a, b)

    with qml.queuing.AnnotatedQueue() as q2:
        qml.PauliX(0)

    expected = [qml.X(0)]
    for op1, op2, exp_op in zip(q1.queue, q2.queue, expected, strict=True):
        qml.assert_equal(op1, exp_op)
        qml.assert_equal(op2, exp_op)


def test_recording_wanted():
    """Test that commutator can be correctly recorded with qml.apply still"""
    with qml.queuing.AnnotatedQueue() as q1:
        a = qml.PauliX(0)
        b = qml.PauliY(0)
        comm = qml.commutator(a, b)
        qml.apply(comm)

    with qml.queuing.AnnotatedQueue() as q2:
        qml.PauliX(0)
        qml.PauliY(0)
        qml.s_prod(2j, qml.PauliZ(0))

    expected = [qml.X(0), qml.Y(0), 2j * qml.Z(0)]
    for op1, op2, exp_op in zip(q1.queue, q2.queue, expected, strict=True):
        qml.assert_equal(op1, exp_op)
        qml.assert_equal(op2, exp_op)


class TestcommPauli:
    """Test qml.comm for pauli=True"""

    data_pauli_relations = (
        # word and word
        (X0, X0, PauliSentence({})),
        (Y0, Y0, PauliSentence({})),
        (Z0, Z0, PauliSentence({})),
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
        res = qml.commutator(transform_type1(op1), transform_type2(op2), pauli=True)
        assert res == true_res
        assert isinstance(res, PauliSentence)

    data_more_comm_relations_commutes = (
        (
            PauliWord({0: "X", 1: "X"}),
            PauliWord({0: "Y", 1: "Y"}),
            PauliSentence({}),
        ),
        (
            PauliWord({0: "X", 1: "X"}),
            PauliWord({"a": "X", "b": "Y"}),
            PauliSentence({}),
        ),
    )

    @pytest.mark.parametrize("transform_type1", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("transform_type2", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_more_comm_relations_commutes)
    def test_comm_relations_pauli_words_that_commute(
        self, op1, op2, true_res, transform_type1, transform_type2
    ):
        """Test more comm relations between Paulis"""
        res = qml.commutator(transform_type1(op1), transform_type2(op2), pauli=True)
        assert res == true_res
        assert isinstance(res, PauliSentence)

    def test_consistency_with_native_pauli_comms(self):
        """Test consistent behavior between native comms in PauliWord and PauliSentence and qml.commutor"""
        op1 = qml.PauliX(0) @ qml.PauliX(1)
        op2 = qml.PauliY(0) + qml.PauliY(1)
        res1 = qml.commutator(op1, op2, pauli=True)
        res2 = PauliWord({0: "X", 1: "X"}).commutator(PauliWord({0: "Y"}) + PauliWord({1: "Y"}))
        assert isinstance(res1, PauliSentence)
        assert isinstance(res2, PauliSentence)
        assert res1 == res2


class TestcommPauliFalse:
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
        res = qml.commutator(transform_type1(op1), transform_type2(op2), pauli=False)
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
    def test_comm_relations_pauli_words(self, op1, op2, true_res, transform_type1, transform_type2):
        """Test more comm relations between Paulis"""
        res = qml.commutator(transform_type1(op1), transform_type2(op2), pauli=False)
        assert res == true_res
        assert isinstance(res, Operator)
        assert isinstance(res, SProd)

    def test_paulis_used_when_ever_possible(self, mocker):
        """Test that pauli_rep is used whenever possible even when ``pauli=False``"""
        spy = mocker.spy(PauliSentence, "operation")
        op1 = qml.PauliX(0) @ qml.PauliX(1)
        op2 = qml.PauliY(0) + qml.PauliY(1)
        res = qml.commutator(op1, op2, pauli=False)
        spy.assert_called()

        assert isinstance(res, Operator)

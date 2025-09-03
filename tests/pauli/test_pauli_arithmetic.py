# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit Tests for the PauliWord and PauliSentence classes"""
# pylint: disable=too-many-public-methods,too-many-arguments,protected-access

import pickle
from copy import copy, deepcopy

import pytest
from scipy import sparse

import pennylane as qml
from pennylane import numpy as np
from pennylane.pauli.pauli_arithmetic import I, PauliSentence, PauliWord, X, Y, Z

matI = np.eye(2)
matX = np.array([[0, 1], [1, 0]])
matY = np.array([[0, -1j], [1j, 0]])
matZ = np.array([[1, 0], [0, -1]])

sparse_matI = sparse.eye(2, format="csr")
sparse_matX = sparse.csr_matrix([[0, 1], [1, 0]])
sparse_matY = sparse.csr_matrix([[0, -1j], [1j, 0]])
sparse_matZ = sparse.csr_matrix([[1, 0], [0, -1]])

pw1 = PauliWord({0: I, 1: X, 2: Y})
pw2 = PauliWord({"a": X, "b": X, "c": Z})
pw3 = PauliWord({0: Z, "b": Z, "c": Z})
pw4 = PauliWord({})
pw_id = pw4  # Identity PauliWord

words = [pw1, pw2, pw3, pw4]

ps1 = PauliSentence({pw1: 1.23, pw2: 4j, pw3: -0.5})
ps2 = PauliSentence({pw1: -1.23, pw2: -4j, pw3: 0.5})
ps1_hamiltonian = PauliSentence({pw1: 1.23, pw2: 4, pw3: -0.5})
ps2_hamiltonian = PauliSentence({pw1: -1.23, pw2: -4, pw3: 0.5})
ps3 = PauliSentence({pw3: -0.5, pw4: 1})
ps4 = PauliSentence({pw4: 1})
ps5 = PauliSentence({})

sentences = [ps1, ps2, ps3, ps4, ps5, ps1_hamiltonian, ps2_hamiltonian]

X0 = PauliWord({0: "X"})
Y0 = PauliWord({0: "Y"})
Z0 = PauliWord({0: "Z"})


def test_pw_pw_multiplication_non_commutativity():
    """Test that pauli word matrix multiplication is non-commutative and returns correct result"""

    res1 = X0 @ Y0
    res2 = Y0 @ X0
    assert res1 == 1j * Z0
    assert res2 == -1j * Z0


def test_ps_ps_multiplication_non_commutativity():
    """Test that pauli sentence matrix multiplication is non-commutative and returns correct result"""

    pauliX = PauliSentence({PauliWord({0: "X"}): 1.0})
    pauliY = PauliSentence({PauliWord({0: "Y"}): 1.0})
    pauliZ = PauliSentence({PauliWord({0: "Z"}): 1j})

    res1 = pauliX @ pauliY
    res2 = pauliY @ pauliX
    assert res1 == pauliZ
    assert res2 == -1 * pauliZ


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


class TestPauliWord:
    def test_identity_removed_on_init(self):
        """Test that identities are removed on init."""
        pw = PauliWord({0: I, 1: X, 2: Y})
        assert 0 not in pw.keys()  # identity ops are removed from pw

    def test_missing(self):
        """Test the result when a missing key is indexed"""
        pw = PauliWord({0: I, 1: X, 2: Y})
        assert 3 not in pw.keys()
        assert pw[3] == I

    @pytest.mark.parametrize("pw", words)
    def test_trivial_pauli_rep(self, pw):
        """Test the pauli_rep property of PauliWord instances"""
        assert pw.pauli_rep is not None
        assert pw.pauli_rep == PauliSentence({pw: 1})

    def test_set_items(self):
        """Test that setting items raises an error"""
        pw = PauliWord({0: I, 1: X, 2: Y})
        with pytest.raises(TypeError, match="PauliWord object does not support assignment"):
            pw[3] = Z  # trying to add to a pw after instantiation is prohibited

    def test_update_items(self):
        """Test that updating items raises an error"""
        pw = PauliWord({0: I, 1: X, 2: Y})
        with pytest.raises(TypeError, match="PauliWord object does not support assignment"):
            pw.update({3: Z})  # trying to add to a pw after instantiation is prohibited

    # pylint: disable=unnecessary-dunder-call
    def test_hash(self):
        """Test that a unique hash exists for different PauliWords."""
        pw_1 = PauliWord({0: I, 1: X, 2: Y})
        pw_2 = PauliWord({0: I, 1: X, 2: Y})  # same as 1
        pw_3 = PauliWord({1: X, 2: Y, 0: I})  # same as 1 but reordered
        pw_4 = PauliWord({1: Z, 2: Z})  # distinct from above

        assert pw_1.__hash__() == pw_2.__hash__()
        assert pw_1.__hash__() == pw_3.__hash__()
        assert pw_1.__hash__() != pw_4.__hash__()

    @pytest.mark.parametrize("pw", (pw1, pw2, pw3, pw4))
    def test_copy(self, pw):
        """Test that the copy is identical to the original."""
        copy_pw = copy(pw)
        deep_copy_pw = deepcopy(pw)

        assert copy_pw == pw
        assert deep_copy_pw == pw
        assert copy_pw is not pw
        assert deep_copy_pw is not pw

    tup_pws_wires = ((pw1, {1, 2}), (pw2, {"a", "b", "c"}), (pw3, {0, "b", "c"}), (pw4, set()))

    @pytest.mark.parametrize("pw, wires", tup_pws_wires)
    def test_wires(self, pw, wires):
        """Test that the wires are tracked correctly."""
        assert set(pw.wires) == wires

    tup_pw_str = (
        (pw1, "X(1) @ Y(2)"),
        (pw2, "X(a) @ X(b) @ Z(c)"),
        (pw3, "Z(0) @ Z(b) @ Z(c)"),
        (pw4, "I"),
    )

    @pytest.mark.parametrize("pw, str_rep", tup_pw_str)
    def test_str(self, pw, str_rep):
        assert str(pw) == str_rep
        assert repr(pw) == str_rep

    def test_add_pw_pw_different(self):
        """Test adding two pauli words that are distinct"""
        res1 = pw1 + pw2
        true_res1 = PauliSentence({pw1: 1.0, pw2: 1.0})
        assert res1 == true_res1

    def test_add_pw_pw_same(self):
        """Test adding two pauli words that are the same"""
        res1 = pw1 + pw1
        true_res1 = PauliSentence({pw1: 2.0})
        assert res1 == true_res1

    def test_add_pw_sclar_different(self):
        """Test adding pauli word and scalar that are distinct (i.e. non-identity)"""
        res1 = pw1 + 1.5
        true_res1 = PauliSentence({pw1: 1.0, pw_id: 1.5})
        assert res1 == true_res1

    def test_add_pw_scalar_same(self):
        """Test adding pauli word and scalar that are the same (i.e. both identity)"""
        res1 = pw_id + 1.5
        true_res1 = PauliSentence({pw_id: 2.5})
        assert res1 == true_res1

    def test_iadd_pw_pw_different(self):
        """Test inplace-adding two pauli words that are distinct"""
        res1 = copy(pw1)
        res2 = copy(pw1)
        res1 += pw2
        true_res1 = PauliSentence({pw1: 1.0, pw2: 1.0})
        assert res1 == true_res1
        assert res2 == pw1  # original pw is unaltered after copy

    def test_iadd_pw_pw_same(self):
        """Test inplace-adding two pauli words that are the same"""
        res1 = copy(pw1)
        res2 = copy(pw1)
        res1 += pw1
        true_res1 = PauliSentence({pw1: 2.0})
        assert res1 == true_res1
        assert res2 == pw1  # original pw is unaltered after copy

    def test_iadd_pw_scalar_different(self):
        """Test inplace-adding pauli word and scalar that are distinct (i.e. non-identity)"""
        res1 = copy(pw1)
        res2 = copy(pw1)
        res1 += 1.5
        true_res1 = PauliSentence({pw1: 1.0, pw_id: 1.5})
        assert res1 == true_res1
        assert res2 == pw1  # original pw is unaltered after copy

    def test_iadd_pw_scalar_same(self):
        """Test inplace-adding two pauli words that are the same (i.e. both identity)"""
        res1 = copy(pw_id)
        res2 = copy(pw_id)
        res1 += 1.5
        true_res1 = PauliSentence({pw_id: 2.5})
        assert res1 == true_res1
        assert res2 == pw_id  # original pw is unaltered after copy

    sub_pw_pw = (
        (
            pw1,
            pw1,
            PauliSentence({pw1: 0.0}),
            PauliSentence({pw1: 0.0}),
        ),
        (
            pw1,
            pw2,
            PauliSentence({pw1: 1.0, pw2: -1.0}),
            PauliSentence({pw1: -1.0, pw2: 1.0}),
        ),
    )

    @pytest.mark.parametrize("pauli1, pauli2, true_res1, true_res2", sub_pw_pw)
    def test_sub_PW_and_PW(self, pauli1, pauli2, true_res1, true_res2):
        """Test subtracting PauliWord from PauliWord"""
        res1 = pauli1 - pauli2
        res2 = pauli2 - pauli1
        assert res1 == true_res1
        assert res2 == true_res2

    # psx does not contain identity, so simply add it to the definition with pw_id
    sub_pw_scalar = (
        (
            pw1,
            1.0,
            PauliSentence({pw1: 1.0, pw_id: -1}),
            PauliSentence({pw1: -1, pw_id: 1}),
        ),
        (
            pw_id,
            0.5,
            PauliSentence({pw_id: 0.5}),
            PauliSentence({pw_id: -0.5}),
        ),
    )

    @pytest.mark.parametrize("pw, scalar, true_res1, true_res2", sub_pw_scalar)
    def test_sub_PW_and_scalar(self, pw, scalar, true_res1, true_res2):
        """Test subtracting scalar from PauliWord"""
        res1 = pw - scalar
        res2 = scalar - pw
        assert res1 == true_res1
        assert res2 == true_res2

    tup_pws_matmult = (
        (pw1, pw1, PauliWord({}), 1.0),  # identities are automatically removed !
        (pw1, pw3, PauliWord({0: Z, 1: X, 2: Y, "b": Z, "c": Z}), 1.0),
        (pw2, pw3, PauliWord({"a": X, "b": Y, 0: Z}), -1.0j),
        (pw3, pw4, pw3, 1.0),
    )

    @pytest.mark.parametrize("word1, word2, result_pw, coeff", tup_pws_matmult)
    def test_matmul(self, word1, word2, result_pw, coeff):
        """Test the user facing matrix multiplication between two pauli words"""
        copy_pw1 = copy(word1)
        copy_pw2 = copy(word2)

        assert word1 @ word2 == PauliSentence({result_pw: coeff})
        assert copy_pw1 == word1  # check for mutation of the pw themselves
        assert copy_pw2 == word2

    @pytest.mark.parametrize("word1, word2, result_pw, coeff", tup_pws_matmult)
    def test_private_private_matmul(self, word1, word2, result_pw, coeff):
        """Test the private matrix multiplication that returns a tuple (new_word, coeff)"""
        copy_pw1 = copy(word1)
        copy_pw2 = copy(word2)

        assert word1._matmul(word2) == (result_pw, coeff)
        assert copy_pw1 == word1  # check for mutation of the pw themselves
        assert copy_pw2 == word2

    @pytest.mark.parametrize("pw", words)
    @pytest.mark.parametrize("scalar", [0.0, 0.5, 1, 1j, 0.5j + 1.0, np.int64(1), np.float32(0.5)])
    def test_mul(self, pw, scalar):
        """Test scalar multiplication"""
        res1 = scalar * pw
        res2 = pw * scalar
        assert isinstance(res1, PauliSentence)
        assert list(res1.values()) == [scalar]
        assert isinstance(res2, PauliSentence)
        assert list(res2.values()) == [scalar]

    @pytest.mark.parametrize("pw", words)
    @pytest.mark.parametrize("scalar", [0.5, 1, 1j, 0.5j + 1.0])
    def test_truediv(self, pw, scalar):
        """Test scalar multiplication"""
        res1 = pw / scalar
        assert isinstance(res1, PauliSentence)
        assert list(res1.values()) == [1 / scalar]

    @pytest.mark.parametrize("pw", words)
    def test_raise_error_for_non_scalar(self, pw):
        """Test that the correct error is raised when attempting to multiply a PauliWord by a sclar"""
        with pytest.raises(ValueError, match="Attempting to multiply"):
            _ = [0.5] * pw

    def test_mul_raise_not_implemented_non_numerical_data_recursive(self):
        """Test that TypeError is raised when trying to multiply by non-numerical data"""
        with pytest.raises(TypeError, match="PauliWord can only"):
            _ = "0.5" * pw1

    def test_mul_raise_not_implemented_non_numerical_data(self):
        """Test that TypeError is raised when trying to multiply by non-numerical data"""
        with pytest.raises(TypeError, match="PauliWord can only"):
            _ = pw1 * "0.5"

    def test_truediv_raise_not_implemented_non_numerical_data(self):
        """Test that TypeError is raised when trying to divide by non-numerical data"""
        with pytest.raises(TypeError, match="PauliWord can only be"):
            _ = pw1 / "0.5"

    tup_pws_mat_wire = (
        (pw1, [2, 0, 1], np.kron(np.kron(matY, matI), matX)),
        (pw1, [2, 1, 0], np.kron(np.kron(matY, matX), matI)),
        (pw2, ["c", "b", "a"], np.kron(np.kron(matZ, matX), matX)),
        (pw3, [0, "b", "c"], np.kron(np.kron(matZ, matZ), matZ)),
    )

    def test_to_mat_empty(self):
        """Test that an empty PauliWord is turned to the trivial 1 matrix"""
        res = pw4.to_mat(wire_order=[])
        assert res == np.ones((1, 1))

    pw_wire_order = ((pw1, [0, 1]), (pw1, [0, 1, 3]), (pw2, [0]))

    @pytest.mark.parametrize("pw, wire_order", pw_wire_order)
    def test_to_mat_error_incomplete(self, pw, wire_order):
        """Test that an appropriate error is raised when the wire order does
        not contain all the PauliWord's wires."""
        match = "Can't get the matrix for the specified wire order"
        with pytest.raises(ValueError, match=match):
            pw.to_mat(wire_order=wire_order)

    def test_to_mat_identity(self):
        """Test that an identity matrix is return if wire_order is provided."""
        assert np.allclose(pw4.to_mat(wire_order=[0, 1]), np.eye(4))
        assert sparse.issparse(pw4.to_mat(wire_order=[0, 1], format="csr"))
        assert not (pw4.to_mat(wire_order=[0, 1], format="csr") != sparse.eye(4)).sum()

    @pytest.mark.parametrize("pw, wire_order, true_matrix", tup_pws_mat_wire)
    def test_to_mat(self, pw, wire_order, true_matrix):
        """Test that the wire_order is correctly incorporated in computing the
        matrix representation."""
        assert np.allclose(pw.to_mat(wire_order=wire_order), true_matrix)

    @pytest.mark.parametrize("pw, wire_order, true_matrix", tup_pws_mat_wire)
    def test_to_mat_format(self, pw, wire_order, true_matrix):
        """Test that the correct type of matrix is returned given the
        format kwarg."""
        sparse_mat = pw.to_mat(wire_order, format="csr")
        assert sparse.issparse(sparse_mat)
        assert np.allclose(sparse_mat.toarray(), true_matrix)

    tup_pw_operation = (
        (PauliWord({0: X}), qml.PauliX(wires=0)),
        (pw1, qml.prod(qml.PauliX(wires=1), qml.PauliY(wires=2))),
        (pw2, qml.prod(qml.PauliX(wires="a"), qml.PauliX(wires="b"), qml.PauliZ(wires="c"))),
        (pw3, qml.prod(qml.PauliZ(wires=0), qml.PauliZ(wires="b"), qml.PauliZ(wires="c"))),
    )

    @pytest.mark.parametrize("pw, op", tup_pw_operation)
    def test_operation(self, pw, op):
        """Test that a PauliWord can be cast to a PL operation."""
        with qml.queuing.AnnotatedQueue() as q:
            pw_op = pw.operation()
        assert len(q.queue) == 0

        if len(pw) > 1:
            for pw_factor, op_factor in zip(pw_op.operands, op.operands):
                assert pw_factor.name == op_factor.name
                assert pw_factor.wires == op_factor.wires
        else:
            assert pw_op.name == op.name
            assert pw_op.wires == op.wires

    def test_operation_empty(self):
        """Test that an empty PauliWord with wire_order returns Identity."""
        with qml.queuing.AnnotatedQueue() as q:
            op = PauliWord({}).operation(wire_order=[0, 1])
        assert len(q.queue) == 0
        id = qml.Identity(wires=[0, 1])
        assert op.name == id.name
        assert op.wires == id.wires

    def test_operation_empty_nowires(self):
        """Test that an empty PauliWord is cast to qml.Identity() operation."""
        with qml.queuing.AnnotatedQueue() as q:
            res = pw4.operation()
        assert len(q.queue) == 0
        assert res == qml.Identity()

    def test_pickling(self):
        """Check that pauliwords can be pickled and unpickled."""
        pw = PauliWord({2: "X", 3: "Y", 4: "Z"})
        serialization = pickle.dumps(pw)
        new_pw = pickle.loads(serialization)
        assert pw == new_pw

    @pytest.mark.parametrize(
        "word,wire_map,expected",
        [
            (PauliWord({0: X, 1: Y}), {0: "a", 1: "b"}, PauliWord({"a": X, "b": Y})),
            (PauliWord({0: X, 1: Y}), {1: "b"}, PauliWord({0: X, "b": Y})),
            (PauliWord({0: X, 1: Y}), {0: 1, 1: 0}, PauliWord({0: Y, 1: X})),
            (PauliWord({"a": X, 0: Y}), {"a": 2, 0: 1, "c": "C"}, PauliWord({2: X, 1: Y})),
        ],
    )
    def test_map_wires(self, word, wire_map, expected):
        """Test the map_wires conversion method."""
        assert word.map_wires(wire_map) == expected

    TEST_TRACE = (
        (PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({}): 3.0}), 3.0),
        (PauliSentence({PauliWord({0: "Y"}): 1.0, PauliWord({1: "X"}): 3.0}), 0.0),
    )

    @pytest.mark.parametrize("op, res", TEST_TRACE)
    def test_trace(self, op, res):
        """Test the trace method of PauliSentence"""
        assert op.trace() == res


class TestPauliSentence:
    def test_missing(self):
        """Test the result when a missing key is indexed"""
        pw = PauliWord({0: X})
        new_pw = PauliWord({"a": Z})
        ps = PauliSentence({pw: 1.0})

        assert new_pw not in ps.keys()
        assert ps[new_pw] == 0.0

    @pytest.mark.parametrize("pw", words)
    def test_wires_not_reordered(self, pw):
        """Test that wires are set correctly and not reshuffled when put in a PS"""
        true_wires = pw.wires
        ps = PauliSentence({pw: 1.0})
        assert ps.wires == true_wires

    @pytest.mark.parametrize("ps", sentences)
    def test_trivial_pauli_rep(self, ps):
        """Test the pauli_rep property of PauliSentence instances"""
        assert ps.pauli_rep is not None
        assert ps.pauli_rep == ps

    def test_set_items(self):
        """Test that we can add to a PauliSentence"""
        pw = PauliWord({0: X})
        ps = PauliSentence({pw: 1.0})

        new_pw = PauliWord({"a": Z})
        assert new_pw not in ps.keys()

        ps[new_pw] = 3.45
        assert new_pw in ps.keys() and ps[new_pw] == 3.45

    def test_pauli_rep(self):
        """Test trivial pauli_rep property"""
        ps = PauliSentence({PauliWord({0: "I", 1: "X", 2: Y}): 1j, X0: 2.0})
        assert ps.pauli_rep == ps

    tup_ps_str = (
        (
            ps1,
            "1.23 * X(1) @ Y(2)\n+ 4j * X(a) @ X(b) @ Z(c)\n+ -0.5 * Z(0) @ Z(b) @ Z(c)",
        ),
        (
            ps2,
            "-1.23 * X(1) @ Y(2)\n+ (-0-4j) * X(a) @ X(b) @ Z(c)\n+ 0.5 * Z(0) @ Z(b) @ Z(c)",
        ),
        (ps3, "-0.5 * Z(0) @ Z(b) @ Z(c)\n+ 1 * I"),
        (ps4, "1 * I"),
        (ps5, "0 * I"),
    )

    @pytest.mark.parametrize("ps, str_rep", tup_ps_str)
    def test_str(self, ps, str_rep):
        """Test the string representation of the PauliSentence."""
        assert str(ps) == str_rep
        assert repr(ps) == str_rep

    tup_ps_wires = (
        (ps1, {0, 1, 2, "a", "b", "c"}),
        (ps2, {0, 1, 2, "a", "b", "c"}),
        (ps3, {0, "b", "c"}),
        (ps4, set()),
    )

    @pytest.mark.parametrize("ps, wires", tup_ps_wires)
    def test_wires(self, ps, wires):
        """Test the correct wires are given for the PauliSentence."""
        assert set(ps.wires) == wires

    @pytest.mark.parametrize("ps", (ps1, ps2, ps3, ps4))
    def test_copy(self, ps):
        """Test that the copy is identical to the original."""
        copy_ps = copy(ps)
        deep_copy_ps = deepcopy(ps)

        assert copy_ps == ps
        assert deep_copy_ps == ps
        assert copy_ps is not ps
        assert deep_copy_ps is not ps

    tup_ps_mult = (  # computed by hand
        (
            ps1,
            ps1,
            PauliSentence(
                {
                    PauliWord({}): -14.2371,
                    PauliWord({1: X, 2: Y, "a": X, "b": X, "c": Z}): 9.84j,
                    PauliWord({0: Z, 1: X, 2: Y, "b": Z, "c": Z}): -1.23,
                }
            ),
        ),
        (
            ps1,
            ps3,
            PauliSentence(
                {
                    PauliWord({0: Z, 1: X, 2: Y, "b": Z, "c": Z}): -0.615,
                    PauliWord({0: Z, "a": X, "b": Y}): -2,
                    PauliWord({}): 0.25,
                    PauliWord({0: I, 1: X, 2: Y}): 1.23,
                    PauliWord({"a": X, "b": X, "c": Z}): 4j,
                    PauliWord({0: Z, "b": Z, "c": Z}): -0.5,
                }
            ),
        ),
        (ps3, ps4, ps3),
        (ps4, ps3, ps3),
        (ps1, ps5, ps5),
        (ps5, ps1, ps5),
        (
            PauliSentence(
                {PauliWord({0: "Z"}): np.array(1.0), PauliWord({0: "Z", 1: "X"}): np.array(1.0)}
            ),
            PauliSentence({PauliWord({1: "Z"}): np.array(1.0), PauliWord({1: "Y"}): np.array(1.0)}),
            PauliSentence(
                {
                    PauliWord({0: "Z", 1: "Z"}): np.array(1.0 + 1.0j),
                    PauliWord({0: "Z", 1: "Y"}): np.array(1.0 - 1.0j),
                }
            ),
        ),
        (
            PauliSentence({PauliWord({0: "X"}): 1.0}),  # ps @ pw disjoint wires
            PauliWord({1: "X"}),
            PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0}),
        ),
        (
            PauliSentence({PauliWord({0: "X"}): 1.0}),  # ps @ pw same wires same op
            PauliWord({0: "X"}),
            PauliSentence({PauliWord({}): 1.0}),
        ),
        (
            PauliSentence({PauliWord({0: "X"}): 1.0}),  # ps @ pw same wires different op
            PauliWord({0: "Y"}),
            PauliSentence({PauliWord({0: "Z"}): 1j}),
        ),
        (
            PauliSentence(
                {PauliWord({0: "Y"}): 1.0}
            ),  # ps @ pw same wires different op check minus sign
            PauliWord({0: "X"}),
            PauliSentence({PauliWord({0: "Z"}): -1j}),
        ),
        (
            PauliWord({1: "X"}),  # pw @ ps disjoint wires
            PauliSentence({PauliWord({0: "X"}): 1.0}),
            PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0}),
        ),
        (
            PauliWord({0: "X"}),  # ps @ pw same wires same op
            PauliSentence({PauliWord({0: "X"}): 1.0}),
            PauliSentence({PauliWord({}): 1.0}),
        ),
        (
            PauliWord({0: "X"}),  # ps @ pw same wires different op
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1j}),
        ),
        (
            PauliWord({0: "Y"}),  # ps @ pw same wires different op check minus sign
            PauliSentence({PauliWord({0: "X"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): -1j}),
        ),
    )

    @pytest.mark.parametrize("pauli1, pauli2, res", tup_ps_mult)
    def test_matmul(self, pauli1, pauli2, res):
        """Test that the correct result of matrix multiplication is produced."""
        copy_ps1 = copy(pauli1)
        copy_ps2 = copy(pauli2)

        simplified_product = pauli1 @ pauli2
        simplified_product.simplify()

        assert simplified_product == res
        assert pauli1 == copy_ps1
        assert pauli2 == copy_ps2

    @pytest.mark.parametrize("ps", sentences)
    @pytest.mark.parametrize("scalar", [0.0, 0.5, 1, 1j, 0.5j + 1.0, np.int64(1), np.float64(1.0)])
    def test_mul(self, ps, scalar):
        """Test scalar multiplication"""
        res1 = scalar * ps
        res2 = ps * scalar
        assert list(res1.values()) == [scalar * coeff for coeff in ps.values()]
        assert list(res2.values()) == [scalar * coeff for coeff in ps.values()]

    def test_mul_raise_not_implemented_non_numerical_data_recursive(self):
        """Test that TypeError is raised when trying to multiply by non-numerical data"""
        with pytest.raises(TypeError, match="PauliSentence can only"):
            _ = "0.5" * ps1

    def test_mul_raise_not_implemented_non_numerical_data(self):
        """Test that TypeError is raised when trying to multiply by non-numerical data"""
        with pytest.raises(TypeError, match="PauliSentence can only"):
            _ = ps1 * "0.5"

    def test_truediv_raise_not_implemented_non_numerical_data(self):
        """Test that TypeError is raised when trying to divide by non-numerical data"""
        with pytest.raises(TypeError, match="PauliSentence can only"):
            _ = ps1 / "0.5"

    @pytest.mark.parametrize("ps", sentences)
    def test_raise_error_for_non_scalar(self, ps):
        """Test that the correct error is raised when attempting to multiply a PauliSentence by a sclar"""
        with pytest.raises(ValueError, match="Attempting to multiply"):
            _ = [0.5] * ps

    def test_add_raises_other_types(self):
        """Test that adding types other than PauliWord, PauliSentence or a scalar raises an error"""
        with pytest.raises(TypeError, match="Cannot add"):
            _ = ps1 + qml.PauliX(0)

        with pytest.raises(TypeError, match="Cannot add"):
            _ = qml.PauliX(0) + ps1

        with pytest.raises(TypeError, match="Cannot add"):
            _ = ps1 + "asd"

        copy_ps = copy(ps1)
        with pytest.raises(TypeError, match="Cannot add"):
            copy_ps += qml.PauliX(0)

        copy_ps = copy(ps1)
        with pytest.raises(TypeError, match="Cannot add"):
            copy_ps += "asd"

    add_ps_ps = (  # computed by hand
        (ps1, ps1, PauliSentence({pw1: 2.46, pw2: 8j, pw3: -1})),
        (ps1, ps2, PauliSentence({})),
        (ps1, ps3, PauliSentence({pw1: 1.23, pw2: 4j, pw3: -1, pw4: 1})),
        (ps2, ps5, ps2),
    )

    @pytest.mark.parametrize("string1, string2, result", add_ps_ps)
    def test_add_PS_and_PS(self, string1, string2, result):
        """Test adding two PauliSentences"""
        copy_ps1 = copy(string1)
        copy_ps2 = copy(string2)

        simplified_product = string1 + string2
        simplified_product.simplify()

        assert simplified_product == result
        assert string1 == copy_ps1
        assert string2 == copy_ps2

    add_ps_pw = (
        (ps1, pw1, PauliSentence({pw1: 2.23, pw2: 4j, pw3: -0.5})),
        (ps1, pw2, PauliSentence({pw1: 1.23, pw2: 1 + 4j, pw3: -0.5})),
        (ps1, pw4, PauliSentence({pw1: 1.23, pw2: 4j, pw3: -0.5, pw4: 1.0})),
        (ps3, pw1, PauliSentence({pw1: 1.0, pw3: -0.5, pw4: 1})),
    )

    @pytest.mark.parametrize("ps, pw, true_res", add_ps_pw)
    def test_add_PS_and_PW(self, ps, pw, true_res):
        """Test adding PauliSentence and PauliWord"""
        res1 = ps + pw
        res2 = pw + ps
        assert res1 == true_res
        assert res2 == true_res

    @pytest.mark.parametrize("scalar", [0.0, 0.5, 0.5j, 0.5 + 0.5j])
    def test_add_PS_and_scalar(self, scalar):
        """Test adding PauliSentence and scalar"""
        res1 = ps1 + scalar
        res2 = scalar + ps1
        assert res1[pw_id] == scalar
        assert res2[pw_id] == scalar

    @pytest.mark.parametrize("scalar", [0.0, 0.5, 0.5j, 0.5 + 0.5j])
    def test_iadd_PS_and_scalar(self, scalar):
        """Test inplace adding PauliSentence and scalar"""
        copy_ps1 = copy(ps1)
        copy_ps2 = copy(ps1)
        copy_ps1 += scalar
        assert copy_ps1[pw_id] == scalar
        assert copy_ps2 == ps1

    @pytest.mark.parametrize("scalar", [0.0, 0.5, 0.5j, 0.5 + 0.5j])
    def test_add_PS_and_scalar_with_1_present(self, scalar):
        """Test adding scalar to a PauliSentence that already contains identity"""
        res1 = ps4 + scalar
        res2 = scalar + ps4
        assert res1[pw_id] == 1 + scalar
        assert res2[pw_id] == 1 + scalar

    @pytest.mark.parametrize("scalar", [0.0, 0.5, 0.5j, 0.5 + 0.5j])
    def test_iadd_PS_and_scalar_1_present(self, scalar):
        """Test inplace adding scalar to PauliSentence that already contains identity"""
        copy_ps1 = copy(ps4)
        copy_ps2 = copy(ps4)
        copy_ps1 += scalar
        assert copy_ps1[pw_id] == 1 + scalar
        assert copy_ps2 == ps4

    psx = PauliSentence(
        {pw1: 1.5, pw2: 4j, pw3: -0.5}
    )  # problems with numerical accuracy for subtracting 1.23 - 1 = 0.2299999998
    sub_ps_pw = (
        (
            psx,
            pw1,
            PauliSentence({pw1: 0.5, pw2: 4j, pw3: -0.5}),
            PauliSentence({pw1: -0.5, pw2: -4j, pw3: +0.5}),
        ),
        (
            psx,
            pw2,
            PauliSentence({pw1: 1.5, pw2: -1.0 + 4j, pw3: -0.5}),
            PauliSentence({pw1: -1.5, pw2: 1.0 - 4j, pw3: +0.5}),
        ),
        (
            psx,
            pw4,
            PauliSentence({pw1: 1.5, pw2: 4j, pw3: -0.5, pw4: -1.0}),
            PauliSentence({pw1: -1.5, pw2: -4j, pw3: +0.5, pw4: 1.0}),
        ),
        (
            ps3,
            pw1,
            PauliSentence({pw1: -1.0, pw3: -0.5, pw4: 1}),
            PauliSentence({pw1: 1.0, pw3: 0.5, pw4: -1}),
        ),
    )

    @pytest.mark.parametrize("ps, pw, true_res1, true_res2", sub_ps_pw)
    def test_sub_PS_and_PW(self, ps, pw, true_res1, true_res2):
        """Test subtracting PauliWord from PauliSentence"""
        res1 = ps - pw
        res2 = pw - ps
        assert res1 == true_res1
        assert res2 == true_res2

    # psx does not contain identity, so simply add it to the definition with pw_id
    sub_ps_scalar = (
        (
            psx,
            1.0,
            PauliSentence({pw1: 1.5, pw2: 4j, pw3: -0.5, pw_id: -1}),
            PauliSentence({pw1: -1.5, pw2: -4j, pw3: 0.5, pw_id: 1}),
        ),
        (
            psx,
            0.5,
            PauliSentence({pw1: 1.5, pw2: 4j, pw3: -0.5, pw_id: -0.5}),
            PauliSentence({pw1: -1.5, pw2: -4j, pw3: 0.5, pw_id: 0.5}),
        ),
        (
            psx,
            0.5j,
            PauliSentence({pw1: 1.5, pw2: 4j, pw3: -0.5, pw_id: -0.5j}),
            PauliSentence({pw1: -1.5, pw2: -4j, pw3: 0.5, pw_id: 0.5j}),
        ),
        (
            psx,
            0.5 + 0.5j,
            PauliSentence({pw1: 1.5, pw2: 4j, pw3: -0.5, pw_id: -0.5 - 0.5j}),
            PauliSentence({pw1: -1.5, pw2: -4j, pw3: 0.5, pw_id: 0.5 + 0.5j}),
        ),
        (
            ps3,
            0.5 + 0.5j,
            PauliSentence({pw3: -0.5, pw4: 0.5 - 0.5j}),
            PauliSentence({pw3: 0.5, pw4: -0.5 + 0.5j}),
        ),
    )

    @pytest.mark.parametrize("ps, scalar, true_res1, true_res2", sub_ps_scalar)
    def test_sub_PS_and_scalar(self, ps, scalar, true_res1, true_res2):
        """Test subtracting scalar from PauliSentence"""
        res1 = ps - scalar
        res2 = scalar - ps
        assert res1 == true_res1
        assert res2 == true_res2

    @pytest.mark.parametrize("string1, string2, result", add_ps_ps)
    def test_iadd_ps_ps(self, string1, string2, result):
        """Test that the correct result of inplace addition with PauliSentence is produced and other object is not changed."""
        copied_string1 = copy(string1)
        copied_string2 = copy(string2)
        copied_string1 += copied_string2
        copied_string1.simplify()

        assert copied_string1 == result  # Check if the modified object matches the expected result
        assert copied_string2 == string2  # Ensure the original object is not modified

    @pytest.mark.parametrize("ps, pw, res", add_ps_pw)
    def test_iadd_ps_pw(self, ps, pw, res):
        """Test that the correct result of inplace addition with PauliWord is produced and other object is not changed."""
        copy_ps1 = copy(ps)
        copy_ps2 = copy(ps)
        copy_ps1 += pw
        assert copy_ps1 == res  # Check if the modified object matches the expected result
        assert copy_ps2 == ps  # Ensure the original object is not modified

    def test_simplify(self):
        """Test that simplify removes terms in the PauliSentence with
        coefficient less than the threshold"""
        un_simplified_ps = PauliSentence({pw1: 0.001, pw2: 0.05, pw3: 1})

        expected_simplified_ps0 = PauliSentence({pw1: 0.001, pw2: 0.05, pw3: 1})
        expected_simplified_ps1 = PauliSentence({pw2: 0.05, pw3: 1})
        expected_simplified_ps2 = PauliSentence({pw3: 1})

        un_simplified_ps.simplify()
        assert un_simplified_ps == expected_simplified_ps0  # default tol = 1e-8
        un_simplified_ps.simplify(tol=1e-2)
        assert un_simplified_ps == expected_simplified_ps1
        un_simplified_ps.simplify(tol=1e-1)
        assert un_simplified_ps == expected_simplified_ps2

    tup_ps_operation = (
        (PauliSentence({PauliWord({0: X}): 1}), qml.s_prod(1, qml.PauliX(wires=0))),
        (
            ps1_hamiltonian,
            qml.sum(
                1.23 * qml.prod(qml.PauliX(wires=1), qml.PauliY(wires=2)),
                4 * qml.prod(qml.PauliX(wires="a"), qml.PauliX(wires="b"), qml.PauliZ(wires="c")),
                -0.5 * qml.prod(qml.PauliZ(wires=0), qml.PauliZ(wires="b"), qml.PauliZ(wires="c")),
            ),
        ),
        (
            ps2_hamiltonian,
            qml.sum(
                -1.23 * qml.prod(qml.PauliX(wires=1), qml.PauliY(wires=2)),
                -4 * qml.prod(qml.PauliX(wires="a"), qml.PauliX(wires="b"), qml.PauliZ(wires="c")),
                0.5 * qml.prod(qml.PauliZ(wires=0), qml.PauliZ(wires="b"), qml.PauliZ(wires="c")),
            ),
        ),
    )

    @pytest.mark.parametrize("ps, op", tup_ps_operation)
    def test_operation(self, ps, op):
        """Test that a PauliSentence can be cast to a PL operation."""

        def _compare_ops(op1, op2):
            assert op1.name == op2.name
            assert op1.wires == op2.wires

        with qml.queuing.AnnotatedQueue() as q:
            ps_op = ps.operation()
        assert len(q.queue) == 0
        if len(ps) > 1:
            for ps_summand, op_summand in zip(ps_op.operands, op.operands):
                assert ps_summand.scalar == op_summand.scalar
                if isinstance(ps_summand.base, qml.ops.Prod):  # pylint: disable=no-member
                    for pw_factor, op_factor in zip(ps_summand.base, op_summand.base):
                        _compare_ops(pw_factor, op_factor)
                else:
                    ps_base, op_base = (ps_summand.base, op_summand.base)
                    _compare_ops(ps_base, op_base)

    def test_operation_with_identity(self):
        """Test that a PauliSentence with an empty PauliWord can be cast to
        operation correctly."""
        with qml.queuing.AnnotatedQueue() as q:
            full_ps_op = ps3.operation()
        assert len(q.queue) == 0
        full_op = qml.sum(
            -0.5 * qml.prod(qml.PauliZ(wires=0), qml.PauliZ(wires="b"), qml.PauliZ(wires="c")),
            qml.s_prod(1, qml.Identity(wires=[0, "b", "c"])),
        )

        ps_op, op = (
            full_ps_op.operands[1],
            full_op.operands[1],
        )  # testing that the identity term is constructed well
        if op.scalar != 1:
            assert ps_op.scalar == op.scalar
            ps_base, op_base = (ps_op.base, op.base)
        else:
            ps_base, op_base = ps_op, op.base

        assert ps_base.name == op_base.name
        assert set(ps_base.wires) == set(op_base.wires)
        # in constructing the identity wires are cast from set -> list and the order is not preserved

    def test_operation_empty(self):
        """Test that an empty PauliSentence with wire_order returns Identity."""
        with qml.queuing.AnnotatedQueue() as q:
            op = ps5.operation(wire_order=[0, 1])
        assert len(q.queue) == 0
        id = qml.s_prod(0.0, qml.Identity(wires=[0, 1]))

        assert op.name == id.name
        assert op.wires == id.wires

    def test_operation_empty_nowires(self):
        """Test that a ValueError is raised if an empty PauliSentence is
        cast to a PL operation."""
        with qml.queuing.AnnotatedQueue() as q:
            res1 = ps4.operation()
        assert len(q.queue) == 0
        assert res1 == qml.Identity()

        with qml.queuing.AnnotatedQueue() as q:
            res2 = ps5.operation()
        assert len(q.queue) == 0
        assert res2 == qml.s_prod(0, qml.Identity())

    def test_operation_wire_order(self):
        """Test that the wire_order parameter is used when the pauli representation is empty"""
        op = ps5.operation(wire_order=["a", "b"])
        id = qml.s_prod(0.0, qml.Identity(wires=["a", "b"]))

        qml.assert_equal(op, id)

    # pylint: disable=W0621
    @pytest.mark.parametrize("coeff0", [qml.math.array([0.6, 0.2, 4.3])])
    @pytest.mark.parametrize("coeff1", [qml.math.array([1.2, -0.9, 2.7])])
    def test_operation_array_input(self, coeff0, coeff1):
        pw0 = qml.pauli.PauliWord({0: "X", "a": "Y"})
        pw1 = qml.pauli.PauliWord({0: "Z", 1: "Y", "b": "Y"})
        ps = qml.pauli.PauliSentence({pw0: coeff0, pw1: coeff1})
        assert ps.operation() is not None

    def test_pickling(self):
        """Check that paulisentences can be pickled and unpickled."""
        word1 = PauliWord({2: "X", 3: "Y", 4: "Z"})
        word2 = PauliWord({2: "Y", 3: "Z"})
        ps = PauliSentence({word1: 1.5, word2: -0.5})

        serialization = pickle.dumps(ps)
        new_ps = pickle.loads(serialization)
        assert ps == new_ps

    def test_dot_wrong_wire_order(self):
        """Check that paulisentences can be dotted with a vector."""
        wire_order = list(range(4))
        word1 = PauliWord({2: "X", 3: "Y", 4: "Z"})
        word2 = PauliWord({2: "Y", 3: "Z"})
        ps = PauliSentence({word1: 1.5, word2: -0.5})
        vector = np.random.rand(2)
        with pytest.raises(
            ValueError,
            match="get the matrix for the specified wire order because it does not contain all the Pauli",
        ):
            ps.dot(vector, wire_order=wire_order)

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
    @pytest.mark.parametrize("wire_order", [None, range(8)])
    def test_dot(self, wire_order, batch_size):
        """Check that paulisentences can be dotted with a vector."""
        word1 = PauliWord({2: "X", 3: "Y", 4: "Z"})
        word2 = PauliWord({2: "Y", 3: "Z"})
        ps = PauliSentence({word1: 1.5, word2: -0.5})
        psmat = ps.to_mat(wire_order=wire_order)
        vector = (
            np.random.rand(psmat.shape[0])
            if batch_size is None
            else np.random.rand(batch_size, psmat.shape[0])
        )
        v0 = ps.dot(vector, wire_order=wire_order)
        v1 = (psmat @ vector.T).T
        assert np.allclose(v0, v1)

    def test_map_wires(self):
        """Test the map_wires conversion method."""
        assert ps1.map_wires({1: "u", 2: "v", "a": 1, "b": 2, "c": 3}) == PauliSentence(
            {
                PauliWord({"u": X, "v": Y}): 1.23,
                PauliWord({1: X, 2: X, 3: Z}): 4j,
                PauliWord({0: Z, 2: Z, 3: Z}): -0.5,
            }
        )

    # pylint: disable=W0621
    @pytest.mark.parametrize("coeff0", [[0.6, 0.2, 4.3], -0.7])
    @pytest.mark.parametrize("coeff1", [[1.2, -0.9, 2.7], -0.7])
    def test_to_mat_with_broadcasting(self, coeff0, coeff1):
        wire_order = [0, 1, "a", "b"]
        pw0 = qml.pauli.PauliWord({0: "X", "a": "Y"})
        pw1 = qml.pauli.PauliWord({0: "Z", 1: "Y", "b": "Y"})
        ps = qml.pauli.PauliSentence({pw0: coeff0, pw1: coeff1})
        mat0 = ps.to_mat(wire_order=wire_order)
        mat1 = qml.matrix(ps.operation(), wire_order=wire_order)
        assert qml.math.allclose(mat0, mat1)


class TestPauliSentenceMatrix:
    """Tests for calculating the matrix of a pauli sentence."""

    PS_EMPTY_CASES = (
        (PauliSentence({}), np.zeros((1, 1))),
        (PauliSentence({PauliWord({}): 1.0}), np.ones((1, 1))),
        (PauliSentence({PauliWord({}): 2.5}), 2.5 * np.ones((1, 1))),
        (
            PauliSentence({PauliWord({}): 2.5, PauliWord({0: "X"}): 1.0}),
            2.5 * np.eye(2) + qml.matrix(qml.PauliX(0)),
        ),
    )

    @pytest.mark.parametrize("ps, true_res", PS_EMPTY_CASES)
    def test_to_mat_empty(self, ps, true_res):
        """Test that empty PauliSentences and PauliSentences with empty PauliWords are handled correctly"""

        res_dense = ps.to_mat()
        assert np.allclose(res_dense, true_res)
        res_sparse = ps.to_mat(format="csr")
        assert sparse.issparse(res_sparse)
        assert qml.math.allclose(res_sparse.todense(), true_res)

    def test_empty_pauli_to_mat_with_wire_order(self):
        """Test the to_mat method with an empty PauliSentence and PauliWord and an external wire order."""
        actual = PauliSentence({PauliWord({}): 1.5}).to_mat([0, 1])
        assert np.allclose(actual, 1.5 * np.eye(4))

    ps_wire_order = ((ps1, []), (ps1, [0, 1, 2, "a", "b"]), (ps3, [0, 1, "c"]))

    @pytest.mark.parametrize("ps, wire_order", ps_wire_order)
    def test_to_mat_error_incomplete(self, ps, wire_order):
        """Test that an appropriate error is raised when the wire order does
        not contain all the PauliSentence's wires."""
        match = "Can't get the matrix for the specified wire order"
        with pytest.raises(ValueError, match=match):
            ps.to_mat(wire_order=wire_order)

    def test_to_mat_empty_sentence_with_wires(self):
        """Test that a zero matrix is returned if wire_order is provided on an empty PauliSentence."""
        true_res = np.zeros((4, 4))
        res_dense = ps5.to_mat(wire_order=[0, 1])
        assert np.allclose(res_dense, true_res)
        res_sparse = ps5.to_mat(wire_order=[0, 1], format="csr")
        assert sparse.issparse(res_sparse)
        assert qml.math.allclose(res_sparse.todense(), true_res)

    tup_ps_mat = (
        (
            ps1,
            [0, 1, 2, "a", "b", "c"],
            1.23 * np.kron(np.kron(matI, np.kron(matX, matY)), np.eye(8))
            + 4j * np.kron(np.eye(8), np.kron(matX, np.kron(matX, matZ)))
            - 0.5 * np.kron(matZ, np.kron(np.eye(8), np.kron(matZ, matZ))),
        ),
        (
            ps2,
            ["a", "b", "c", 0, 1, 2],
            -1.23 * np.kron(np.eye(8), np.kron(matI, np.kron(matX, matY)))
            - 4j * np.kron(np.kron(matX, np.kron(matX, matZ)), np.eye(8))
            + 0.5 * np.kron(np.kron(matI, np.kron(matZ, np.kron(matZ, matZ))), np.eye(4)),
        ),
        (
            ps3,
            [0, "b", "c"],
            -0.5 * np.kron(matZ, np.kron(matZ, matZ)) + 1 * np.eye(8),
        ),
    )

    @pytest.mark.parametrize("ps, wire_order, true_matrix", tup_ps_mat)
    def test_to_mat_wire_order(self, ps, wire_order, true_matrix):
        """Test that the wire_order is correctly incorporated in computing the
        matrix representation."""
        assert np.allclose(ps.to_mat(wire_order), true_matrix)

    @pytest.mark.parametrize("ps, wire_order, true_matrix", tup_ps_mat)
    def test_to_mat_format(self, ps, wire_order, true_matrix):
        """Test that the correct type of matrix is returned given the format kwarg."""
        sparse_mat = ps.to_mat(wire_order, format="csr")
        assert sparse.issparse(sparse_mat)
        assert np.allclose(sparse_mat.toarray(), true_matrix)

    @pytest.mark.parametrize("ps,wire_order,true_matrix", tup_ps_mat)
    def test_to_mat_buffer(self, ps, wire_order, true_matrix):
        """Test that the intermediate matrices are added correctly once the maximum buffer
        size is reached."""
        buffer_size = 2 ** len(wire_order) * 48  # Buffer size for 2 matrices
        sparse_mat = ps.to_mat(wire_order, format="csr", buffer_size=buffer_size)
        assert np.allclose(sparse_mat.toarray(), true_matrix)

    @pytest.mark.tf
    def test_dense_matrix_tf(self):
        """Test calculating the matrix for a pauli sentence is differentaible with tensorflow."""
        import tensorflow as tf

        x = tf.Variable(0.1 + 0j)
        y = tf.Variable(0.2 + 0j)

        with tf.GradientTape() as tape:
            _pw1 = PauliWord({0: "X", 1: "Y"})
            _pw2 = PauliWord({0: "Y", 1: "X"})
            H = x * _pw1 + y * _pw2
            mat = H.to_mat()

        gx, gy = tape.jacobian(mat, [x, y])

        pw1_mat = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]])
        pw2_mat = np.array([[0, 0, 0, -1j], [0, 0, -1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]])

        assert qml.math.allclose(mat, x * pw1_mat + y * pw2_mat)
        assert qml.math.allclose(gx, qml.math.conj(pw1_mat))  # tf complex number convention
        assert qml.math.allclose(gy, qml.math.conj(pw2_mat))

    @pytest.mark.torch
    def test_dense_matrix_torch(self):
        """Test calculating and differentiating the matrix with torch."""

        import torch

        _pw1 = qml.pauli.PauliWord({0: "X", 1: "Z"})
        _pw2 = qml.pauli.PauliWord({0: "X"})

        pw1_mat = torch.tensor([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]])
        pw2_mat = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])

        x = torch.tensor(0.1, requires_grad=True)
        y = torch.tensor(0.2, requires_grad=True)

        def f(x, y):
            H = x * _pw1 + y * _pw2
            return qml.math.real(H.to_mat())

        mat = f(x, y)
        assert qml.math.allclose(mat, x * pw1_mat + y * pw2_mat)

        gx, gy = torch.autograd.functional.jacobian(f, (x, y))
        assert qml.math.allclose(gx, pw1_mat)
        assert qml.math.allclose(gy, pw2_mat)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_dense_matrix_jax(self, use_jit):
        """Test calculating and differentiating the matrix with jax."""

        import jax

        def f(x, y):
            _pw1 = qml.pauli.PauliWord({0: "X", 1: "Y"})
            _pw2 = qml.pauli.PauliWord({0: "Y", 1: "X"})
            H = x * _pw1 + y * _pw2
            return H.to_mat()

        if use_jit:
            f = jax.jit(f)

        x = jax.numpy.array(0.1 + 0j)
        y = jax.numpy.array(0.2 + 0j)

        pw1_mat = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]])
        pw2_mat = np.array([[0, 0, 0, -1j], [0, 0, -1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]])

        mat = f(x, y)
        assert qml.math.allclose(mat, x * pw1_mat + y * pw2_mat)

        gx, gy = jax.jacobian(f, holomorphic=True, argnums=(0, 1))(x, y)
        assert qml.math.allclose(gx, pw1_mat)
        assert qml.math.allclose(gy, pw2_mat)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_tracer_coefficients_jax(self, use_jit):
        """Tests that functions with abstract coefficients can be jitted."""
        import jax

        def f(c1, c2):
            op = c1 * qml.X(0) + c2 * qml.X(0)
            return op.simplify()

        if use_jit:
            f = jax.jit(f)
            # Need for assert_equal to recognize they are the same.
            # Complains about mismatching interfaces (numpy vs jax).
            coeffs = jax.numpy.array(1.0)
        else:
            coeffs = 1.0

        qml.assert_equal(
            f(1.0, 1.0),
            (coeffs * qml.X(0) + coeffs * qml.X(0)).simplify(),
        )


class TestPaulicomms:
    """Test 'native' commutator in PauliWord and PauliSentence"""

    def test_pauli_word_comm_raises_NotImplementedError(self):
        """Test that a NotImplementedError is raised when a PauliWord.commutator() for type that is not PauliWord, PauliSentence or Operator"""
        op1 = PauliWord({0: "X"})
        matrix = np.eye(2)
        with pytest.raises(NotImplementedError, match="Cannot compute natively a commutator"):
            op1.commutator(matrix)

    def test_pauli_raises_NotImplementedError_without_pauli_rep_ps(self):
        """Test that a NotImplementedError is raised in PauliSentence when ``other`` is an operator without a pauli_rep"""
        with pytest.raises(
            NotImplementedError, match="Cannot compute a native commutator of a Pauli word"
        ):
            _ = ps1.commutator(qml.Hadamard(0))

    def test_pauli_raises_NotImplementedError_without_pauli_rep_pw(self):
        """Test that a NotImplementedError is raised in PauliWord when ``other`` is an operator without a pauli_rep"""
        with pytest.raises(
            NotImplementedError, match="Cannot compute a native commutator of a Pauli word"
        ):
            _ = pw1.commutator(qml.Hadamard(0))

    def test_commutators_with_zeros_ps(self):
        """Test that commutators between PauliSentences where one represents the 0 word is treated correctly"""
        op_zero = PauliSentence({})
        op = PauliSentence({PauliWord({0: "X"}): 1.0})

        assert op_zero.commutator(op) == op_zero
        assert op.commutator(op_zero) == op_zero

    data_pauli_relations_commutes = [
        # word and word
        (X0, X0, PauliSentence({})),
        (Y0, Y0, PauliSentence({})),
        (Z0, Z0, PauliSentence({})),
        (PauliWord({"a": X}), PauliWord({"a": X}), PauliSentence({})),
        (PauliWord({"a": X}), PauliWord({"b": Y}), PauliSentence({})),
    ]

    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations_commutes)
    def test_zero_return_pauli_word(self, op1, op2, true_res):
        """Test the return when both operators are zero"""
        res1 = op1.commutator(op2)
        res1m = op2.commutator(op1)
        assert res1 == true_res
        assert res1m == true_res

    @pytest.mark.parametrize("convert1", [_id, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations_commutes)
    def test_zero_return_pauli_word_different_types(self, op1, op2, true_res, convert1):
        """Test the return when both operators are zero and potentially of different type"""
        op1 = convert1(op1)
        res2 = op1.commutator(op2)
        res2m = op2.commutator(op1)
        assert res2 == true_res
        assert res2m == true_res

    @pytest.mark.parametrize("convert1", [_id, _pauli_to_op, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations_commutes)
    def test_zero_return_pauli_word_different_types_with_operator(
        self, op1, op2, true_res, convert1
    ):
        """Test the return when both operators are zero and potentially of different type"""
        op1 = convert1(op1)
        res = op2.commutator(op1)
        assert res == true_res

    data_pauli_relations = [
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
    ]

    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations)
    def test_pauli_word_commutator(self, op1, op2, true_res):
        """Test native comm in PauliWord class"""
        res1 = op1.commutator(op2)
        res1m = op2.commutator(op1)
        assert res1 == true_res
        assert res1m == -1 * true_res

    data_pauli_relations_private_func = (
        # test when using the private _comm function that returns a tuple instead of a PauliSentence
        (X0, X0, pw_id, 0),
        (Y0, Y0, pw_id, 0),
        (Z0, Z0, pw_id, 0),
        (X0, Y0, Z0, 2j),
        (Y0, Z0, X0, 2j),
        (Z0, X0, Y0, 2j),
        (Y0, X0, Z0, -2j),
        (Z0, Y0, X0, -2j),
        (X0, Z0, Y0, -2j),
    )

    @pytest.mark.parametrize("op1, op2, true_word, true_coeff", data_pauli_relations_private_func)
    def test_pauli_word_private_commutator(self, op1, op2, true_word, true_coeff):
        """Test native _comm in PauliWord class that returns tuples"""
        res1 = op1._commutator(op2)
        res1m = op2._commutator(op1)
        assert res1[0] == true_word
        assert res1[1] == true_coeff
        assert res1m[0] == true_word
        assert res1m[1] == -1 * true_coeff

    data_pauli_relations_different_types = [
        (X0, Y0, PauliSentence({Z0: 2j})),
        (Y0, Z0, PauliSentence({X0: 2j})),
        (Z0, X0, PauliSentence({Y0: 2j})),
        (Y0, X0, PauliSentence({Z0: -2j})),
        (Z0, Y0, PauliSentence({X0: -2j})),
        (X0, Z0, PauliSentence({Y0: -2j})),
    ]

    @pytest.mark.parametrize("convert1", [_id, _pw_to_ps])
    @pytest.mark.parametrize("convert2", [_id, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations_different_types)
    def test_pauli_word_comm_different_types(
        self, op1, op2, true_res, convert1, convert2
    ):  # pylint: disable=too-many-positional-arguments
        """Test native comm in between a PauliSentence and either of PauliWord, PauliSentence, Operator"""
        op1 = convert1(op1)
        op2 = convert2(op2)
        res2 = op1.commutator(op2)
        res2m = op2.commutator(op1)
        assert res2 == true_res
        assert res2m == -1 * true_res
        assert all(isinstance(res, PauliSentence) for res in [res2, res2m])

    @pytest.mark.parametrize("convert1", [_id, _pw_to_ps])
    @pytest.mark.parametrize("convert2", [_pauli_to_op])
    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations_different_types)
    def test_pauli_word_comm_different_types_with_ops(
        self, op1, op2, true_res, convert1, convert2
    ):  # pylint: disable=too-many-positional-arguments
        """Test native comm in between a PauliWord, PauliSentence and Operator"""
        op1 = convert1(op1)
        op2 = convert2(op2)
        res2 = op1.commutator(op2)
        assert res2 == true_res
        assert isinstance(res2, PauliSentence)

    data_pauli_relations_commutes = [
        (X0, X0, PauliSentence({})),
        (Y0, Y0, PauliSentence({})),
        (Z0, Z0, PauliSentence({})),
    ]

    @pytest.mark.parametrize("convert1", [_id, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations_different_types)
    def test_pauli_word_comm_commutes(self, op1, op2, true_res, convert1):
        """Test native comm in between a PauliSentence and either of PauliWord, PauliSentence, Operator for the case when the operators commute"""
        op1 = convert1(op1)
        op2 = qml.pauli.pauli_sentence(op2)
        res2 = op1.commutator(op2)
        res2m = op2.commutator(op1)
        assert res2 == true_res
        assert res2m == -1 * true_res
        assert all(isinstance(res, PauliSentence) for res in [res2, res2m])

    @pytest.mark.parametrize("convert1", [_id, _pw_to_ps])
    @pytest.mark.parametrize("op1, op2, true_res", data_pauli_relations_different_types)
    def test_pauli_commutator_with_operator(self, op1, op2, true_res, convert1):
        """Test native comm in between a PauliSentence and either of PauliWord, PauliSentence, Operator for the case when the operators commute"""
        op1 = convert1(op1)
        op2 = _pauli_to_op(op2)
        res2 = op1.commutator(op2)
        assert res2 == true_res
        assert isinstance(res2, PauliSentence)

    data_consistency_paulis = [
        PauliWord({0: "X", 1: "X"}),
        PauliWord({0: "Y"}) + PauliWord({1: "Y"}),
        1.5 * PauliWord({0: "Y"}) + 0.5 * PauliWord({1: "Y"}),
    ]

    @pytest.mark.parametrize("op1", data_consistency_paulis)
    @pytest.mark.parametrize("op2", data_consistency_paulis)
    def test_consistency_pw_ps(self, op1, op2):
        """Test consistent behavior when inputting PauliWord and PauliSentence"""
        op1 = PauliWord({0: "X", 1: "X"})
        op2 = PauliWord({0: "Y"}) + PauliWord({1: "Y"})
        res1 = op1.commutator(op2)
        res2 = qml.commutator(op1, op2, pauli=True)
        assert res1 == res2


class TestPauliArithmeticIntegration:
    def test_pauli_arithmetic_integration(self):
        """Test creating operators from PauliWord, PauliSentence and scalars"""
        res = 1.0 + 3.0 * pw1 + 1j * ps3 - 1.0 * ps1
        true_res = PauliSentence({pw1: -1.23 + 3, pw2: -4j, pw3: 0.5 - 0.5j, pw_id: 1 + 1j})
        assert res == true_res

    def test_construct_XXZ_model(self):
        """Test that constructing the XXZ model results in the correct matrix"""
        n_wires = 4
        J_orthogonal = 1.5
        J_zz = 0.5
        h = 2.0
        # Construct XXZ Hamiltonian using paulis
        paulis = [
            J_orthogonal
            * (
                PauliWord({i: "X", (i + 1) % n_wires: "X"})
                + PauliWord({i: "Y", (i + 1) % n_wires: "Y"})
            )
            for i in range(n_wires)
        ]
        paulis += [J_zz * PauliWord({i: "Z", (i + 1) % n_wires: "Z"}) for i in range(n_wires)]
        paulis += [h * PauliWord({i: "Z"}) for i in range(n_wires)]
        H = sum(paulis) + 10.0

        # Construct XXZ Hamiltonian using PL ops
        ops = [
            J_orthogonal
            * (
                qml.PauliX(i) @ qml.PauliX((i + 1) % n_wires)
                + qml.PauliY(i) @ qml.PauliY((i + 1) % n_wires)
            )
            for i in range(n_wires)
        ]
        ops += [J_zz * qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_wires) for i in range(n_wires)]
        ops += [h * qml.PauliZ(i) for i in range(n_wires)]
        H_true = qml.sum(*ops) + 10.0

        assert qml.math.allclose(H.to_mat(), qml.matrix(H_true))


@pytest.mark.all_interfaces
class TestPauliArithmeticWithADInterfaces:
    """Test pauli arithmetic with different automatic differentiation interfaces"""

    @pytest.mark.torch
    @pytest.mark.parametrize("scalar", [0.0, 0.5, 1, 1j, 0.5j + 1.0])
    def test_torch_initialization(self, scalar):
        """Test initializing PauliSentence from torch tensor"""
        import torch

        tensor = scalar * torch.ones(4)
        res = PauliSentence(dict(zip(words, tensor)))
        assert all(isinstance(val, torch.Tensor) for val in res.values())

    @pytest.mark.autograd
    @pytest.mark.parametrize("scalar", [0.0, 0.5, 1, 1j, 0.5j + 1.0])
    def test_autograd_initialization(self, scalar):
        """Test initializing PauliSentence from autograd array"""

        tensor = scalar * np.ones(4)
        res = PauliSentence(dict(zip(words, tensor)))
        assert all(isinstance(val, np.ndarray) for val in res.values())

    @pytest.mark.jax
    @pytest.mark.parametrize("scalar", [0.0, 0.5, 1, 1j, 0.5j + 1.0])
    def test_jax_initialization(self, scalar):
        """Test initializing PauliSentence from jax array"""
        import jax.numpy as jnp

        tensor = scalar * jnp.ones(4)
        res = PauliSentence(dict(zip(words, tensor)))
        assert all(isinstance(val, jnp.ndarray) for val in res.values())

    @pytest.mark.tf
    @pytest.mark.parametrize("scalar", [0.0, 0.5, 1, 1j, 0.5j + 1.0])
    def test_tf_initialization(self, scalar):
        """Test initializing PauliSentence from tf tensor"""
        import tensorflow as tf

        tensor = scalar * tf.ones(4, dtype=tf.complex64)
        res = PauliSentence(dict(zip(words, tensor)))
        assert all(isinstance(val, tf.Tensor) for val in res.values())

    @pytest.mark.torch
    @pytest.mark.parametrize("ps", sentences)
    @pytest.mark.parametrize("scalar", [0.5, 1, 1j, 0.5j + 1.0])
    def test_torch_scalar_multiplication(self, ps, scalar):
        """Test that multiplying with a torch tensor works and results in the correct types"""
        import torch

        res1 = torch.tensor(scalar) * ps
        res2 = ps * torch.tensor(scalar)
        res3 = ps / torch.tensor(scalar)
        assert isinstance(res1, PauliSentence)
        assert isinstance(res2, PauliSentence)
        assert isinstance(res3, PauliSentence)
        assert list(res1.values()) == [scalar * coeff for coeff in ps.values()]
        assert list(res2.values()) == [scalar * coeff for coeff in ps.values()]
        assert list(res3.values()) == [coeff / scalar for coeff in ps.values()]
        assert all(isinstance(val, torch.Tensor) for val in res1.values())
        assert all(isinstance(val, torch.Tensor) for val in res2.values())
        assert all(isinstance(val, torch.Tensor) for val in res3.values())

    @pytest.mark.autograd
    @pytest.mark.parametrize("ps", sentences)
    @pytest.mark.parametrize("scalar", [0.5, 1, 1j, 0.5j + 1.0])
    def test_autograd_scalar_multiplication(self, ps, scalar):
        """Test that multiplying with an autograd array works and results in the correct types"""

        res1 = np.array(scalar) * ps
        res2 = ps * np.array(scalar)
        res3 = ps / np.array(scalar)
        assert isinstance(res1, PauliSentence)
        assert isinstance(res2, PauliSentence)
        assert isinstance(res3, PauliSentence)
        assert list(res1.values()) == [scalar * coeff for coeff in ps.values()]
        assert list(res2.values()) == [scalar * coeff for coeff in ps.values()]
        assert list(res3.values()) == [coeff / scalar for coeff in ps.values()]
        assert all(isinstance(val, np.ndarray) for val in res1.values())
        assert all(isinstance(val, np.ndarray) for val in res2.values())
        assert all(isinstance(val, np.ndarray) for val in res3.values())

    @pytest.mark.jax
    @pytest.mark.parametrize("ps", sentences)
    @pytest.mark.parametrize("scalar", [0.5, 1, 1j, 0.5j + 1.0])
    def test_jax_scalar_multiplication(self, ps, scalar):
        """Test that multiplying with a jax array works and results in the correct types"""
        import jax.numpy as jnp

        res1 = jnp.array(scalar) * ps
        res2 = ps * jnp.array(scalar)
        res3 = ps / jnp.array(scalar)
        assert isinstance(res1, PauliSentence)
        assert isinstance(res2, PauliSentence)
        assert isinstance(res3, PauliSentence)
        assert list(res1.values()) == [scalar * coeff for coeff in ps.values()]
        assert list(res2.values()) == [scalar * coeff for coeff in ps.values()]
        assert list(res3.values()) == [coeff / scalar for coeff in ps.values()]
        assert all(isinstance(val, jnp.ndarray) for val in res1.values())
        assert all(isinstance(val, jnp.ndarray) for val in res2.values())
        assert all(isinstance(val, jnp.ndarray) for val in res3.values())

    @pytest.mark.tf
    @pytest.mark.parametrize("ps", sentences)
    @pytest.mark.parametrize("scalar", [0.5, 1, 1j, 0.5j + 1.0])
    def test_tf_scalar_multiplication(self, ps, scalar):
        """Test that multiplying with a tf tensor works and results in the correct types"""
        import tensorflow as tf

        res1 = tf.constant(scalar, dtype=tf.complex64) * ps
        res2 = ps * tf.constant(scalar, dtype=tf.complex64)
        res3 = ps / tf.constant(scalar, dtype=tf.complex64)
        assert isinstance(res1, PauliSentence)
        assert isinstance(res2, PauliSentence)
        assert isinstance(res3, PauliSentence)
        assert list(res1.values()) == [scalar * coeff for coeff in ps.values()]
        assert list(res2.values()) == [scalar * coeff for coeff in ps.values()]
        assert list(res3.values()) == [coeff / scalar for coeff in ps.values()]
        assert all(isinstance(val, tf.Tensor) for val in res1.values())
        assert all(isinstance(val, tf.Tensor) for val in res2.values())
        assert all(isinstance(val, tf.Tensor) for val in res3.values())

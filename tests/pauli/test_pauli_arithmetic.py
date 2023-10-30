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
# pylint: disable=too-many-public-methods
import pickle
from copy import copy, deepcopy
import pytest

from scipy import sparse
import pennylane as qml
from pennylane import numpy as np
from pennylane.wires import Wires
from pennylane.pauli.pauli_arithmetic import PauliWord, PauliSentence, I, X, Y, Z


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

ps1 = PauliSentence({pw1: 1.23, pw2: 4j, pw3: -0.5})
ps2 = PauliSentence({pw1: -1.23, pw2: -4j, pw3: 0.5})
ps1_hamiltonian = PauliSentence({pw1: 1.23, pw2: 4, pw3: -0.5})
ps2_hamiltonian = PauliSentence({pw1: -1.23, pw2: -4, pw3: 0.5})
ps3 = PauliSentence({pw3: -0.5, pw4: 1})
ps4 = PauliSentence({pw4: 1})
ps5 = PauliSentence({})


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

    tup_pws_mult = (
        (pw1, pw1, PauliWord({}), 1.0),  # identities are automatically removed !
        (pw1, pw3, PauliWord({0: Z, 1: X, 2: Y, "b": Z, "c": Z}), 1.0),
        (pw2, pw3, PauliWord({"a": X, "b": Y, 0: Z}), -1.0j),
        (pw3, pw4, pw3, 1.0),
    )

    @pytest.mark.parametrize("word1, word2, result_pw, coeff", tup_pws_mult)
    def test_mul(self, word1, word2, result_pw, coeff):
        copy_pw1 = copy(word1)
        copy_pw2 = copy(word2)

        assert word1 * word2 == (result_pw, coeff)
        assert copy_pw1 == word1  # check for mutation of the pw themselves
        assert copy_pw2 == word2

    tup_pws_mat_wire = (
        (pw1, [2, 0, 1], np.kron(np.kron(matY, matI), matX)),
        (pw1, [2, 1, 0], np.kron(np.kron(matY, matX), matI)),
        (pw2, ["c", "b", "a"], np.kron(np.kron(matZ, matX), matX)),
        (pw3, [0, "b", "c"], np.kron(np.kron(matZ, matZ), matZ)),
    )

    def test_to_mat_error(self):
        """Test that an appropriate error is raised when an empty
        PauliWord is cast to matrix."""
        with pytest.raises(ValueError, match="Can't get the matrix of an empty PauliWord."):
            pw4.to_mat(wire_order=[])

        with pytest.raises(ValueError, match="Can't get the matrix of an empty PauliWord."):
            pw4.to_mat(wire_order=Wires([]))

    pw1 = PauliWord({0: I, 1: X, 2: Y})
    pw2 = PauliWord({"a": X, "b": X, "c": Z})
    pw3 = PauliWord({0: Z, "b": Z, "c": Z})
    pw4 = PauliWord({})

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
        pw_op = pw.operation()
        if len(pw) > 1:
            for pw_factor, op_factor in zip(pw_op.operands, op.operands):
                assert pw_factor.name == op_factor.name
                assert pw_factor.wires == op_factor.wires
        else:
            assert pw_op.name == op.name
            assert pw_op.wires == op.wires

        if isinstance(op, qml.ops.Prod):  # pylint: disable=no-member
            pw_tensor_op = pw.operation(get_as_tensor=True)
            expected_tensor_op = qml.operation.Tensor(*op.operands)
            assert qml.equal(pw_tensor_op, expected_tensor_op)

    def test_operation_empty(self):
        """Test that an empty PauliWord with wire_order returns Identity."""
        op = PauliWord({}).operation(wire_order=[0, 1])
        id = qml.Identity(wires=[0, 1])
        assert op.name == id.name
        assert op.wires == id.wires

    def test_operation_empty_error(self):
        """Test that a ValueError is raised if an empty PauliWord is
        cast to a PL operation."""
        with pytest.raises(ValueError, match="Can't get the operation for an empty PauliWord."):
            pw4.operation()

    tup_pw_hamiltonian = (
        (PauliWord({0: X}), 1 * qml.PauliX(wires=0)),
        (pw1, 1 * qml.PauliX(wires=1) @ qml.PauliY(wires=2)),
        (pw2, 1 * qml.PauliX(wires="a") @ qml.PauliX(wires="b") @ qml.PauliZ(wires="c")),
        (pw3, 1 * qml.PauliZ(wires=0) @ qml.PauliZ(wires="b") @ qml.PauliZ(wires="c")),
    )

    @pytest.mark.parametrize("pw, h", tup_pw_hamiltonian)
    def test_hamiltonian(self, pw, h):
        """Test that a PauliWord can be cast to a Hamiltonian."""
        pw_h = pw.hamiltonian()
        assert pw_h.compare(h)

    def test_hamiltonian_empty(self):
        """Test that an empty PauliWord with wire_order returns Identity Hamiltonian."""
        op = PauliWord({}).hamiltonian(wire_order=[0, 1])
        id = 1 * qml.Identity(wires=[0, 1])
        assert op.compare(id)

    def test_hamiltonian_empty_error(self):
        """Test that a ValueError is raised if an empty PauliWord is
        cast to a Hamiltonian."""
        with pytest.raises(ValueError, match="Can't get the Hamiltonian for an empty PauliWord."):
            pw4.hamiltonian()

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


class TestPauliSentence:
    def test_missing(self):
        """Test the result when a missing key is indexed"""
        pw = PauliWord({0: X})
        new_pw = PauliWord({"a": Z})
        ps = PauliSentence({pw: 1.0})

        assert new_pw not in ps.keys()
        assert ps[new_pw] == 0.0

    def test_set_items(self):
        """Test that we can add to a PauliSentence"""
        pw = PauliWord({0: X})
        ps = PauliSentence({pw: 1.0})

        new_pw = PauliWord({"a": Z})
        assert new_pw not in ps.keys()

        ps[new_pw] = 3.45
        assert new_pw in ps.keys() and ps[new_pw] == 3.45

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
        (ps1, ps5, ps1),
        (ps5, ps1, ps1),
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
    )

    @pytest.mark.parametrize("string1, string2, res", tup_ps_mult)
    def test_mul(self, string1, string2, res):
        """Test that the correct result of multiplication is produced."""
        copy_ps1 = copy(string1)
        copy_ps2 = copy(string2)

        simplified_product = string1 * string2
        simplified_product.simplify()

        assert simplified_product == res
        assert string1 == copy_ps1
        assert string2 == copy_ps2

    tup_ps_add = (  # computed by hand
        (ps1, ps1, PauliSentence({pw1: 2.46, pw2: 8j, pw3: -1})),
        (ps1, ps2, PauliSentence({})),
        (ps1, ps3, PauliSentence({pw1: 1.23, pw2: 4j, pw3: -1, pw4: 1})),
        (ps2, ps5, ps2),
    )

    @pytest.mark.parametrize("string1, string2, result", tup_ps_add)
    def test_add(self, string1, string2, result):
        """Test that the correct result of addition is produced."""
        copy_ps1 = copy(string1)
        copy_ps2 = copy(string2)

        simplified_product = string1 + string2
        simplified_product.simplify()

        assert simplified_product == result
        assert string1 == copy_ps1
        assert string2 == copy_ps2

    ps_match = (
        (ps4, "Can't get the matrix of an empty PauliWord."),
        (ps5, "Can't get the matrix of an empty PauliSentence."),
    )

    @pytest.mark.parametrize("string1, string2, result", tup_ps_add)
    def test_iadd(self, string1, string2, result):
        """Test that the correct result of inplace addition is produced and other object is not changed."""
        copied_string1 = copy(string1)
        copied_string2 = copy(string2)
        copied_string1 += copied_string2
        copied_string1.simplify()

        assert copied_string1 == result # Check if the modified object matches the expected result
        assert copied_string2 == string2  # Ensure the original object is not modified
    @pytest.mark.parametrize("ps, match", ps_match)
    def test_to_mat_error_empty(self, ps, match):
        """Test that an appropriate error is raised when an empty
        PauliSentence or PauliWord is cast to matrix."""
        with pytest.raises(ValueError, match=match):
            ps.to_mat(wire_order=[])

        with pytest.raises(ValueError, match=match):
            ps.to_mat(wire_order=Wires([]))

    ps_wire_order = ((ps1, []), (ps1, [0, 1, 2, "a", "b"]), (ps3, [0, 1, "c"]))

    @pytest.mark.parametrize("ps, wire_order", ps_wire_order)
    def test_to_mat_error_incomplete(self, ps, wire_order):
        """Test that an appropriate error is raised when the wire order does
        not contain all the PauliSentence's wires."""
        match = "Can't get the matrix for the specified wire order"
        with pytest.raises(ValueError, match=match):
            ps.to_mat(wire_order=wire_order)

    def test_to_mat_identity(self):
        """Test that an identity matrix is return if wire_order is provided."""
        assert np.allclose(ps5.to_mat(wire_order=[0, 1]), np.eye(4))
        assert sparse.issparse(ps5.to_mat(wire_order=[0, 1], format="csr"))

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

        ps_op = ps.operation()
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
        full_ps_op = ps3.operation()
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
        op = ps5.operation(wire_order=[0, 1])
        id = qml.s_prod(0.0, qml.Identity(wires=[0, 1]))

        assert op.name == id.name
        assert op.wires == id.wires

    def test_operation_empty_error(self):
        """Test that a ValueError is raised if an empty PauliSentence is
        cast to a PL operation."""
        with pytest.raises(ValueError, match="Can't get the operation for an empty PauliWord."):
            ps4.operation()
        with pytest.raises(ValueError, match="Can't get the operation for an empty PauliSentence."):
            ps5.operation()

    def test_operation_wire_order(self):
        """Test that the wire_order parameter is used when the pauli representation is empty"""
        op = ps5.operation(wire_order=["a", "b"])
        id = qml.s_prod(0.0, qml.Identity(wires=["a", "b"]))

        assert qml.equal(op, id)

    tup_ps_hamiltonian = (
        (PauliSentence({PauliWord({0: X}): 1}), 1 * qml.PauliX(wires=0)),
        (
            ps1_hamiltonian,
            +1.23 * qml.PauliX(wires=1) @ qml.PauliY(wires=2)
            + 4 * qml.PauliX(wires="a") @ qml.PauliX(wires="b") @ qml.PauliZ(wires="c")
            - 0.5 * qml.PauliZ(wires=0) @ qml.PauliZ(wires="b") @ qml.PauliZ(wires="c"),
        ),
        (
            ps2_hamiltonian,
            -1.23 * qml.PauliX(wires=1) @ qml.PauliY(wires=2)
            - 4 * qml.PauliX(wires="a") @ qml.PauliX(wires="b") @ qml.PauliZ(wires="c")
            + 0.5 * qml.PauliZ(wires=0) @ qml.PauliZ(wires="b") @ qml.PauliZ(wires="c"),
        ),
        (
            ps3,
            -0.5 * qml.PauliZ(wires=0) @ qml.PauliZ(wires="b") @ qml.PauliZ(wires="c")
            + 1 * qml.Identity(wires=[0, "b", "c"]),
        ),
    )

    @pytest.mark.parametrize("ps, h", tup_ps_hamiltonian)
    def test_hamiltonian(self, ps, h):
        """Test that a PauliSentence can be cast to a Hamiltonian."""
        ps_h = ps.hamiltonian()
        assert ps_h.compare(h)

    def test_hamiltonian_empty(self):
        """Test that an empty PauliSentence with wire_order returns Identity."""
        op = ps5.hamiltonian(wire_order=[0, 1])
        id = qml.Hamiltonian([], [])
        assert op.compare(id)

    def test_hamiltonian_empty_error(self):
        """Test that a ValueError is raised if an empty PauliSentence is
        cast to a Hamiltonian."""
        with pytest.raises(
            ValueError, match="Can't get the Hamiltonian for an empty PauliSentence."
        ):
            ps5.hamiltonian()

    def test_hamiltonian_wire_order(self):
        """Test that the wire_order parameter is used when the pauli representation is empty"""
        op = ps5.hamiltonian(wire_order=["a", "b"])
        id = qml.Hamiltonian([], [])

        assert qml.equal(op, id)

    def test_pickling(self):
        """Check that paulisentences can be pickled and unpickled."""
        word1 = PauliWord({2: "X", 3: "Y", 4: "Z"})
        word2 = PauliWord({2: "Y", 3: "Z"})
        ps = PauliSentence({word1: 1.5, word2: -0.5})

        serialization = pickle.dumps(ps)
        new_ps = pickle.loads(serialization)
        assert ps == new_ps

    def test_map_wires(self):
        """Test the map_wires conversion method."""
        assert ps1.map_wires({1: "u", 2: "v", "a": 1, "b": 2, "c": 3}) == PauliSentence(
            {
                PauliWord({"u": X, "v": Y}): 1.23,
                PauliWord({1: X, 2: X, 3: Z}): 4j,
                PauliWord({0: Z, 2: Z, 3: Z}): -0.5,
            }
        )

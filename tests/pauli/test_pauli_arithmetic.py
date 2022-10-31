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
import pytest
from copy import copy

from scipy import sparse
from pennylane import numpy as np
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
        with pytest.raises(NotImplementedError):
            pw[3] = Z  # trying to add to a pw after instantiation is prohibited

    def test_update_items(self):
        """Test that updating items raises an error"""
        pw = PauliWord({0: I, 1: X, 2: Y})
        with pytest.raises(NotImplementedError):
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
        assert copy_pw == pw
        assert copy_pw is not pw

    tup_pws_wires = ((pw1, {1, 2}), (pw2, {"a", "b", "c"}), (pw3, {0, "b", "c"}), (pw4, set()))

    @pytest.mark.parametrize("pw, wires", tup_pws_wires)
    def test_wires(self, pw, wires):
        """Test that the wires are tracked correctly."""
        assert pw.wires == wires

    tup_pw_str = (
        (pw1, "[X(1)] @ [Y(2)]"),
        (pw2, "[X(a)] @ [X(b)] @ [Z(c)]"),
        (pw3, "[Z(0)] @ [Z(b)] @ [Z(c)]"),
        (pw4, "[()]"),
    )

    @pytest.mark.parametrize("pw, str_rep", tup_pw_str)
    def test_str(self, pw, str_rep):
        assert str(pw) == str_rep

    tup_pws_mult = (
        (pw1, pw1, PauliWord({}), 1.0),  # identities are automatically removed !
        (pw1, pw3, PauliWord({0: Z, 1: X, 2: Y, "b": Z, "c": Z}), 1.0),
        (pw2, pw3, PauliWord({"a": X, "b": Y, 0: Z}), -1.0j),
        (pw3, pw4, pw3, 1.0),
    )

    @pytest.mark.parametrize("pw1, pw2, result_pw, coeff", tup_pws_mult)
    def test_mul(self, pw1, pw2, result_pw, coeff):
        copy_pw1 = copy(pw1)
        copy_pw2 = copy(pw2)

        assert pw1 * pw2 == (result_pw, coeff)
        assert copy_pw1 == pw1  # check for mutation of the pw themselves
        assert copy_pw2 == pw2

    tup_pws_mat_wire = (
        (pw1, [2, 0, 1], np.kron(np.kron(matY, matI), matX)),
        (pw2, ["c", "b", "a"], np.kron(np.kron(matZ, matX), matX)),
        (pw3, [0, "b", "c"], np.kron(np.kron(matZ, matZ), matZ)),
    )

    def test_to_mat_error(self):
        """Test that an appropriate error is raised when an empty
        PauliWord is cast to matrix."""
        with pytest.raises(ValueError, match="Can't get the matrix of an empty PauliWord."):
            pw4.to_mat(wire_order=[])

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
            "(1.23) * [X(1)] @ [Y(2)]\n"
            "+ (4j) * [X(a)] @ [X(b)] @ [Z(c)]\n"
            "+ (-0.5) * [Z(0)] @ [Z(b)] @ [Z(c)]",
        ),
        (
            ps2,
            "(-1.23) * [X(1)] @ [Y(2)]\n"
            "+ ((-0-4j)) * [X(a)] @ [X(b)] @ [Z(c)]\n"
            "+ (0.5) * [Z(0)] @ [Z(b)] @ [Z(c)]",
        ),
        (ps3, "(-0.5) * [Z(0)] @ [Z(b)] @ [Z(c)]\n" "+ (1) * [()]"),
        (ps4, "(1) * [()]"),
    )

    @pytest.mark.parametrize("ps, str_rep", tup_ps_str)
    def test_str(self, ps, str_rep):
        """Test the string representation of the PauliSentence."""
        assert str(ps) == str_rep

    tup_ps_wires = (
        (ps1, {0, 1, 2, "a", "b", "c"}),
        (ps2, {0, 1, 2, "a", "b", "c"}),
        (ps3, {0, "b", "c"}),
        (ps4, set()),
    )

    @pytest.mark.parametrize("ps, wires", tup_ps_wires)
    def test_wires(self, ps, wires):
        """Test the correct wires are given for the PauliSentence."""
        assert ps.wires == wires

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
    )

    @pytest.mark.parametrize("ps1, ps2, res", tup_ps_mult)
    def test_mul(self, ps1, ps2, res):
        """Test that the correct result of multiplication is produced."""
        copy_ps1 = copy(ps1)
        copy_ps2 = copy(ps2)

        simplified_product = ps1 * ps2
        simplified_product.simplify()

        assert simplified_product == res
        assert ps1 == copy_ps1
        assert ps2 == copy_ps2

    tup_ps_add = (  # computed by hand
        (ps1, ps1, PauliSentence({pw1: 2.46, pw2: 8j, pw3: -1})),
        (ps1, ps2, PauliSentence({})),
        (ps1, ps3, PauliSentence({pw1: 1.23, pw2: 4j, pw3: -1, pw4: 1})),
        (ps2, ps5, ps2),
    )

    @pytest.mark.parametrize("ps1, ps2, result", tup_ps_add)
    def test_add(self, ps1, ps2, result):
        """Test that the correct result of addition is produced."""
        copy_ps1 = copy(ps1)
        copy_ps2 = copy(ps2)

        simplified_product = ps1 + ps2
        simplified_product.simplify()

        assert simplified_product == result
        assert ps1 == copy_ps1
        assert ps2 == copy_ps2

    ps_match = (
        (ps3, "Can't get the matrix of an empty PauliWord."),
        (ps4, "Can't get the matrix of an empty PauliWord."),
        (ps5, "Can't get the matrix of an empty PauliSentence."),
    )

    @pytest.mark.parametrize("ps, match", ps_match)
    def test_to_mat_error(self, ps, match):
        """Test that an appropriate error is raised when an empty
        PauliSentence or PauliWord is cast to matrix."""
        with pytest.raises(ValueError, match=match):
            ps.to_mat(wire_order=[])

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

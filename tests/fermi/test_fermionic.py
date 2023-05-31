# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit Tests for the Fermionic representation classes."""
import pickle
from copy import copy, deepcopy

import pytest

from pennylane.fermi.fermionic import FermiWord, FermiSentence

fw1 = FermiWord({(0, 0): "+", (1, 1): "-"})
fw2 = FermiWord({(0, 0): "+", (1, 0): "-"})
fw3 = FermiWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
fw4 = FermiWord({})


class TestFermiWord:
    def test_missing(self):
        """Test that empty string is returned for missing key."""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        assert (2, 3) not in fw.keys()
        assert fw[(2, 3)] == ""

    def test_set_items(self):
        """Test that setting items raises an error"""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        with pytest.raises(TypeError, match="FermiWord object does not support assignment"):
            fw[(2, 2)] = "+"

    def test_update_items(self):
        """Test that updating items raises an error"""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        with pytest.raises(TypeError, match="FermiWord object does not support assignment"):
            fw.update({(2, 2): "+"})

    def test_hash(self):
        """Test that a unique hash exists for different FermiWords."""
        fw_1 = FermiWord({(0, 0): "+", (1, 1): "-"})
        fw_2 = FermiWord({(0, 0): "+", (1, 1): "-"})  # same as 1
        fw_3 = FermiWord({(1, 1): "-", (0, 0): "+"})  # same as 1 but reordered
        fw_4 = FermiWord({(0, 0): "+", (2, 2): "-"})  # distinct from above

        assert fw_1.__hash__() == fw_2.__hash__()
        assert fw_1.__hash__() == fw_3.__hash__()
        assert fw_1.__hash__() != fw_4.__hash__()

    @pytest.mark.parametrize("fw", (fw1, fw2, fw3, fw4))
    def test_copy(self, fw):
        """Test that the copy is identical to the original."""
        copy_fw = copy(fw)
        deep_copy_fw = deepcopy(fw)

        assert copy_fw == fw
        assert deep_copy_fw == fw
        assert copy_fw is not fw
        assert deep_copy_fw is not fw

    tup_fws_wires = ((fw1, [0, 1]), (fw2, [0, 0]), (fw3, [0, 3, 0, 4]), (fw4, []))

    @pytest.mark.parametrize("fw, wires", tup_fws_wires)
    def test_wires(self, fw, wires):
        """Test that the wires are tracked correctly."""
        assert fw.wires == wires

    tup_fw_compact = (
        (fw1, "0+ 1-"),
        (fw2, "0+ 0-"),
        (fw3, "0+ 3- 0+ 4-"),
        (fw4, ""),
    )

    @pytest.mark.parametrize("fw, str_rep", tup_fw_compact)
    def test_compact(self, fw, str_rep):
        assert fw.to_string() == str_rep

    tup_fw_str = (
        (fw1, "<FermiWord = '0+ 1-'>"),
        (fw2, "<FermiWord = '0+ 0-'>"),
        (fw3, "<FermiWord = '0+ 3- 0+ 4-'>"),
        (fw4, "<FermiWord = ''>"),
    )

    @pytest.mark.parametrize("fw, str_rep", tup_fw_str)
    def test_str(self, fw, str_rep):
        assert str(fw) == str_rep
        assert repr(fw) == str_rep

    tup_fws_mult = (
        (fw1, fw1, FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"})),
        (
            fw1,
            fw3,
            FermiWord(
                {(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 3): "-", (4, 0): "+", (5, 4): "-"}
            ),
        ),
        (fw2, fw1, FermiWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-"})),
    )

    @pytest.mark.parametrize("f1, f2, result_fw", tup_fws_mult)
    def test_mul(self, f1, f2, result_fw):
        assert f1 * f2 == result_fw

    tup_fws_pow = (
        (fw1, 0, FermiWord({})),
        (fw1, 1, FermiWord({(0, 0): "+", (1, 1): "-"})),
        (fw1, 2, FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"})),
        (
            fw2,
            3,
            FermiWord(
                {(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 0): "-", (4, 0): "+", (5, 0): "-"}
            ),
        ),
    )

    @pytest.mark.parametrize("f1, pow, result_fw", tup_fws_pow)
    def test_pow(self, f1, pow, result_fw):
        assert f1**pow == result_fw

    def test_pickling(self):
        """Check that FermiWords can be pickled and unpickled."""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        serialization = pickle.dumps(fw)
        new_fw = pickle.loads(serialization)
        assert fw == new_fw


fs1 = FermiSentence({fw1: 1.23, fw2: 4j, fw3: -0.5})
fs2 = FermiSentence({fw1: -1.23, fw2: -4j, fw3: 0.5})
fs1_hamiltonian = FermiSentence({fw1: 1.23, fw2: 4, fw3: -0.5})
fs2_hamiltonian = FermiSentence({fw1: -1.23, fw2: -4, fw3: 0.5})
fs3 = FermiSentence({fw3: -0.5, fw4: 1})
fs4 = FermiSentence({fw4: 1})
fs5 = FermiSentence({})


class TestFermiSentence:
    def test_missing(self):
        """Test the result when a missing key is indexed."""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        new_fw = FermiWord({(0, 2): "+", (1, 3): "-"})
        fs = FermiSentence({fw: 1.0})

        assert new_fw not in fs.keys()
        assert fs[new_fw] == 0.0

    def test_set_items(self):
        """Test that we can add to a FermiSentence."""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        fs = FermiSentence({fw: 1.0})

        new_fw = FermiWord({(0, 2): "+", (1, 3): "-"})
        assert new_fw not in fs.keys()

        fs[new_fw] = 3.45
        assert new_fw in fs.keys() and fs[new_fw] == 3.45

    tup_fs_str = (
        (fs1, "1.23 * '0+ 1-'\n" "+ 4j * '0+ 0-'\n" "+ -0.5 * '0+ 3- 0+ 4-'"),
        (fs2, "-1.23 * '0+ 1-'\n" "+ (-0-4j) * '0+ 0-'\n" "+ 0.5 * '0+ 3- 0+ 4-'"),
        (fs3, "-0.5 * '0+ 3- 0+ 4-'\n" "+ 1 * I"),
        # (fs4, "1 * I"),
        # (fs5, "0 * I"),
    )

    @pytest.mark.parametrize("fs, str_rep", tup_fs_str)
    def test_str(self, fs, str_rep):
        """Test the string representation of the FermiSentence."""
        print(str(fs))
        assert str(fs) == str_rep
        assert repr(fs) == str_rep

    #
    # tup_fs_wires = (
    #     (fs1, {0, 1, 2, "a", "b", "c"}),
    #     (fs2, {0, 1, 2, "a", "b", "c"}),
    #     (fs3, {0, "b", "c"}),
    #     (fs4, set()),
    # )
    #
    # @pytest.mark.parametrize("fs, wires", tup_fs_wires)
    # def test_wires(self, fs, wires):
    #     """Test the correct wires are given for the FermiSentence."""
    #     assert fs.wires == wires
    #
    # @pytest.mark.parametrize("fs", (fs1, fs2, fs3, fs4))
    # def test_copy(self, fs):
    #     """Test that the copy is identical to the original."""
    #     copy_fs = copy(fs)
    #     deep_copy_fs = deepcopy(fs)
    #
    #     assert copy_fs == fs
    #     assert deep_copy_fs == fs
    #     assert copy_fs is not fs
    #     assert deep_copy_fs is not fs
    #
    # tup_fs_mult = (  # computed by hand
    #     (
    #         fs1,
    #         fs1,
    #         FermiSentence(
    #             {
    #                 FermiWord({}): -14.2371,
    #                 FermiWord({1: X, 2: Y, "a": X, "b": X, "c": Z}): 9.84j,
    #                 FermiWord({0: Z, 1: X, 2: Y, "b": Z, "c": Z}): -1.23,
    #             }
    #         ),
    #     ),
    #     (
    #         fs1,
    #         fs3,
    #         FermiSentence(
    #             {
    #                 FermiWord({0: Z, 1: X, 2: Y, "b": Z, "c": Z}): -0.615,
    #                 FermiWord({0: Z, "a": X, "b": Y}): -2,
    #                 FermiWord({}): 0.25,
    #                 FermiWord({0: I, 1: X, 2: Y}): 1.23,
    #                 FermiWord({"a": X, "b": X, "c": Z}): 4j,
    #                 FermiWord({0: Z, "b": Z, "c": Z}): -0.5,
    #             }
    #         ),
    #     ),
    #     (fs3, fs4, fs3),
    #     (fs4, fs3, fs3),
    #     (fs1, fs5, fs1),
    #     (fs5, fs1, fs1),
    #     (
    #         FermiSentence(
    #             {FermiWord({0: "Z"}): np.array(1.0), FermiWord({0: "Z", 1: "X"}): np.array(1.0)}
    #         ),
    #         FermiSentence({FermiWord({1: "Z"}): np.array(1.0), FermiWord({1: "Y"}): np.array(1.0)}),
    #         FermiSentence(
    #             {
    #                 FermiWord({0: "Z", 1: "Z"}): np.array(1.0 + 1.0j),
    #                 FermiWord({0: "Z", 1: "Y"}): np.array(1.0 - 1.0j),
    #             }
    #         ),
    #     ),
    # )
    #
    # @pytest.mark.parametrize("fs1, fs2, res", tup_fs_mult)
    # def test_mul(self, fs1, fs2, res):
    #     """Test that the correct result of multiplication is produced."""
    #     copy_fs1 = copy(fs1)
    #     copy_fs2 = copy(fs2)
    #
    #     simplified_product = fs1 * fs2
    #     simplified_product.simplify()
    #
    #     assert simplified_product == res
    #     assert fs1 == copy_fs1
    #     assert fs2 == copy_fs2
    #
    # tup_fs_add = (  # computed by hand
    #     (fs1, fs1, FermiSentence({fw1: 2.46, fw2: 8j, fw3: -1})),
    #     (fs1, fs2, FermiSentence({})),
    #     (fs1, fs3, FermiSentence({fw1: 1.23, fw2: 4j, fw3: -1, fw4: 1})),
    #     (fs2, fs5, fs2),
    # )
    #
    # @pytest.mark.parametrize("fs1, fs2, result", tup_fs_add)
    # def test_add(self, fs1, fs2, result):
    #     """Test that the correct result of addition is produced."""
    #     copy_fs1 = copy(fs1)
    #     copy_fs2 = copy(fs2)
    #
    #     simplified_product = fs1 + fs2
    #     simplified_product.simplify()
    #
    #     assert simplified_product == result
    #     assert fs1 == copy_fs1
    #     assert fs2 == copy_fs2
    #
    # fs_match = (
    #     (fs4, "Can't get the matrix of an empty FermiWord."),
    #     (fs5, "Can't get the matrix of an empty FermiSentence."),
    # )
    #
    # @pytest.mark.parametrize("fs, match", fs_match)
    # def test_to_mat_error(self, fs, match):
    #     """Test that an appropriate error is raised when an empty
    #     FermiSentence or FermiWord is cast to matrix."""
    #     with pytest.raises(ValueError, match=match):
    #         fs.to_mat(wire_order=None)
    #
    #     with pytest.raises(ValueError, match=match):
    #         fs.to_mat(wire_order=Wires([]))
    #
    # def test_to_mat_identity(self):
    #     """Test that an identity matrix is return if wire_order is provided."""
    #     assert np.allclose(fs5.to_mat(wire_order=[0, 1]), np.eye(4))
    #     assert sparse.issparse(fs5.to_mat(wire_order=[0, 1], format="csr"))
    #
    # tup_fs_mat = (
    #     (
    #         fs1,
    #         [0, 1, 2, "a", "b", "c"],
    #         1.23 * np.kron(np.kron(matI, np.kron(matX, matY)), np.eye(8))
    #         + 4j * np.kron(np.eye(8), np.kron(matX, np.kron(matX, matZ)))
    #         - 0.5 * np.kron(matZ, np.kron(np.eye(8), np.kron(matZ, matZ))),
    #     ),
    #     (
    #         fs2,
    #         ["a", "b", "c", 0, 1, 2],
    #         -1.23 * np.kron(np.eye(8), np.kron(matI, np.kron(matX, matY)))
    #         - 4j * np.kron(np.kron(matX, np.kron(matX, matZ)), np.eye(8))
    #         + 0.5 * np.kron(np.kron(matI, np.kron(matZ, np.kron(matZ, matZ))), np.eye(4)),
    #     ),
    #     (
    #         fs3,
    #         [0, "b", "c"],
    #         -0.5 * np.kron(matZ, np.kron(matZ, matZ)) + 1 * np.eye(8),
    #     ),
    # )
    #
    # @pytest.mark.parametrize("fs, wire_order, true_matrix", tup_fs_mat)
    # def test_to_mat_wire_order(self, fs, wire_order, true_matrix):
    #     """Test that the wire_order is correctly incorporated in computing the
    #     matrix representation."""
    #     assert np.allclose(fs.to_mat(wire_order), true_matrix)
    #
    # @pytest.mark.parametrize("fs, wire_order, true_matrix", tup_fs_mat)
    # def test_to_mat_format(self, fs, wire_order, true_matrix):
    #     """Test that the correct type of matrix is returned given the format kwarg."""
    #     sparse_mat = fs.to_mat(wire_order, format="csr")
    #     assert sparse.issparse(sparse_mat)
    #     assert np.allclose(sparse_mat.toarray(), true_matrix)
    #
    # def test_simplify(self):
    #     """Test that simplify removes terms in the FermiSentence with
    #     coefficient less than the threshold"""
    #     un_simplified_fs = FermiSentence({fw1: 0.001, fw2: 0.05, fw3: 1})
    #
    #     expected_simplified_fs0 = FermiSentence({fw1: 0.001, fw2: 0.05, fw3: 1})
    #     expected_simplified_fs1 = FermiSentence({fw2: 0.05, fw3: 1})
    #     expected_simplified_fs2 = FermiSentence({fw3: 1})
    #
    #     un_simplified_fs.simplify()
    #     assert un_simplified_fs == expected_simplified_fs0  # default tol = 1e-8
    #     un_simplified_fs.simplify(tol=1e-2)
    #     assert un_simplified_fs == expected_simplified_fs1
    #     un_simplified_fs.simplify(tol=1e-1)
    #     assert un_simplified_fs == expected_simplified_fs2
    #
    # tup_fs_operation = (
    #     (FermiSentence({FermiWord({0: X}): 1}), qml.s_prod(1, qml.FermiX(wires=0))),
    #     (
    #         fs1_hamiltonian,
    #         qml.sum(
    #             1.23 * qml.prod(qml.FermiX(wires=1), qml.FermiY(wires=2)),
    #             4 * qml.prod(qml.FermiX(wires="a"), qml.FermiX(wires="b"), qml.FermiZ(wires="c")),
    #             -0.5 * qml.prod(qml.FermiZ(wires=0), qml.FermiZ(wires="b"), qml.FermiZ(wires="c")),
    #         ),
    #     ),
    #     (
    #         fs2_hamiltonian,
    #         qml.sum(
    #             -1.23 * qml.prod(qml.FermiX(wires=1), qml.FermiY(wires=2)),
    #             -4 * qml.prod(qml.FermiX(wires="a"), qml.FermiX(wires="b"), qml.FermiZ(wires="c")),
    #             0.5 * qml.prod(qml.FermiZ(wires=0), qml.FermiZ(wires="b"), qml.FermiZ(wires="c")),
    #         ),
    #     ),
    # )
    #
    # @pytest.mark.parametrize("fs, op", tup_fs_operation)
    # def test_operation(self, fs, op):
    #     """Test that a FermiSentence can be cast to a PL operation."""
    #
    #     def _compare_ofs(op1, op2):
    #         assert op1.name == op2.name
    #         assert op1.wires == op2.wires
    #
    #     fs_op = fs.operation()
    #     if len(fs) > 1:
    #         for fs_summand, op_summand in zip(fs_op.operands, op.operands):
    #             assert fs_summand.scalar == op_summand.scalar
    #             if isinstance(fs_summand.base, qml.ofs.Prod):
    #                 for fw_factor, op_factor in zip(fs_summand.base, op_summand.base):
    #                     _compare_ofs(fw_factor, op_factor)
    #             else:
    #                 fs_base, op_base = (fs_summand.base, op_summand.base)
    #                 _compare_ofs(fs_base, op_base)
    #
    # def test_operation_with_identity(self):
    #     """Test that a FermiSentence with an empty FermiWord can be cast to
    #     operation correctly."""
    #     full_fs_op = fs3.operation()
    #     full_op = qml.sum(
    #         -0.5 * qml.prod(qml.FermiZ(wires=0), qml.FermiZ(wires="b"), qml.FermiZ(wires="c")),
    #         qml.s_prod(1, qml.Identity(wires=[0, "b", "c"])),
    #     )
    #
    #     fs_op, op = (
    #         full_fs_op.operands[1],
    #         full_op.operands[1],
    #     )  # testing that the identity term is constructed well
    #     if op.scalar != 1:
    #         assert fs_op.scalar == op.scalar
    #         fs_base, op_base = (fs_op.base, op.base)
    #     else:
    #         fs_base, op_base = fs_op, op.base
    #
    #     assert fs_base.name == op_base.name
    #     assert set(fs_base.wires) == set(op_base.wires)
    #     # in constructing the identity wires are cast from set -> list and the order is not preserved
    #
    # def test_operation_empty(self):
    #     """Test that an empty FermiSentence with wire_order returns Identity."""
    #     op = fs5.operation(wire_order=[0, 1])
    #     id = qml.s_prod(0.0, qml.Identity(wires=[0, 1]))
    #
    #     assert op.name == id.name
    #     assert op.wires == id.wires
    #
    # def test_operation_empty_error(self):
    #     """Test that a ValueError is raised if an empty FermiSentence is
    #     cast to a PL operation."""
    #     with pytest.raises(ValueError, match="Can't get the operation for an empty FermiWord."):
    #         fs4.operation()
    #     with pytest.raises(ValueError, match="Can't get the operation for an empty FermiSentence."):
    #         fs5.operation()
    #
    # def test_operation_wire_order(self):
    #     """Test that the wire_order parameter is used when the Fermi representation is empty"""
    #     op = fs5.operation(wire_order=["a", "b"])
    #     id = qml.s_prod(0.0, qml.Identity(wires=["a", "b"]))
    #
    #     assert qml.equal(op, id)
    #
    # tup_fs_hamiltonian = (
    #     (FermiSentence({FermiWord({0: X}): 1}), 1 * qml.FermiX(wires=0)),
    #     (
    #         fs1_hamiltonian,
    #         +1.23 * qml.FermiX(wires=1) @ qml.FermiY(wires=2)
    #         + 4 * qml.FermiX(wires="a") @ qml.FermiX(wires="b") @ qml.FermiZ(wires="c")
    #         - 0.5 * qml.FermiZ(wires=0) @ qml.FermiZ(wires="b") @ qml.FermiZ(wires="c"),
    #     ),
    #     (
    #         fs2_hamiltonian,
    #         -1.23 * qml.FermiX(wires=1) @ qml.FermiY(wires=2)
    #         - 4 * qml.FermiX(wires="a") @ qml.FermiX(wires="b") @ qml.FermiZ(wires="c")
    #         + 0.5 * qml.FermiZ(wires=0) @ qml.FermiZ(wires="b") @ qml.FermiZ(wires="c"),
    #     ),
    #     (
    #         fs3,
    #         -0.5 * qml.FermiZ(wires=0) @ qml.FermiZ(wires="b") @ qml.FermiZ(wires="c")
    #         + 1 * qml.Identity(wires=[0, "b", "c"]),
    #     ),
    # )
    #
    # @pytest.mark.parametrize("fs, h", tup_fs_hamiltonian)
    # def test_hamiltonian(self, fs, h):
    #     """Test that a FermiSentence can be cast to a Hamiltonian."""
    #     fs_h = fs.hamiltonian()
    #     assert fs_h.compare(h)
    #
    # def test_hamiltonian_empty(self):
    #     """Test that an empty FermiSentence with wire_order returns Identity."""
    #     op = fs5.hamiltonian(wire_order=[0, 1])
    #     id = qml.Hamiltonian([], [])
    #     assert op.compare(id)
    #
    # def test_hamiltonian_empty_error(self):
    #     """Test that a ValueError is raised if an empty FermiSentence is
    #     cast to a Hamiltonian."""
    #     with pytest.raises(
    #         ValueError, match="Can't get the Hamiltonian for an empty FermiSentence."
    #     ):
    #         fs5.hamiltonian()
    #
    # def test_hamiltonian_wire_order(self):
    #     """Test that the wire_order parameter is used when the Fermi representation is empty"""
    #     op = fs5.hamiltonian(wire_order=["a", "b"])
    #     id = qml.Hamiltonian([], [])
    #
    #     assert qml.equal(op, id)
    #
    # def test_pickling(self):
    #     """Check that Fermisentences can be pickled and unpickled."""
    #     fw1 = FermiWord({2: "X", 3: "Y", 4: "Z"})
    #     fw2 = FermiWord({2: "Y", 3: "Z"})
    #     fs = FermiSentence({fw1: 1.5, fw2: -0.5})
    #
    #     serialization = pickle.dumfs(fs)
    #     new_fs = pickle.loads(serialization)
    #     assert fs == new_fs
    #
    # def test_map_wires(self):
    #     """Test the map_wires conversion method."""
    #     assert fs1.map_wires({1: "u", 2: "v", "a": 1, "b": 2, "c": 3}) == FermiSentence(
    #         {
    #             FermiWord({"u": X, "v": Y}): 1.23,
    #             FermiWord({1: X, 2: X, 3: Z}): 4j,
    #             FermiWord({0: Z, 2: Z, 3: Z}): -0.5,
    #         }
    #     )

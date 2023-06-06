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

import pennylane as qml
from pennylane.fermi.fermionic import FermiSentence, FermiWord
from pennylane.pauli import PauliWord, PauliSentence


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

        with pytest.raises(TypeError, match="FermiWord object does not support assignment"):
            fw[(1, 1)] = "+"

    def test_hash(self):
        """Test that a unique hash exists for different FermiWords."""
        fw_1 = FermiWord({(0, 0): "+", (1, 1): "-"})
        fw_2 = FermiWord({(0, 0): "+", (1, 1): "-"})  # same as 1
        fw_3 = FermiWord({(1, 1): "-", (0, 0): "+"})  # same as 1 but reordered
        fw_4 = FermiWord({(0, 0): "+", (1, 2): "-"})  # distinct from above

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

    tup_fws_wires = ((fw1, {0, 1}), (fw2, {0}), (fw3, {0, 3, 4}), (fw4, set()))

    @pytest.mark.parametrize("fw, wires", tup_fws_wires)
    def test_wires(self, fw, wires):
        """Test that the wires are tracked correctly."""
        assert fw.wires == wires

    tup_fw_compact = (
        (fw1, "0+ 1-"),
        (fw2, "0+ 0-"),
        (fw3, "0+ 3- 0+ 4-"),
        (fw4, "I"),
    )

    @pytest.mark.parametrize("fw, str_rep", tup_fw_compact)
    def test_compact(self, fw, str_rep):
        assert fw.to_string() == str_rep

    tup_fw_str = (
        (fw1, "<FermiWord = '0+ 1-'>"),
        (fw2, "<FermiWord = '0+ 0-'>"),
        (fw3, "<FermiWord = '0+ 3- 0+ 4-'>"),
        (fw4, "<FermiWord = 'I'>"),
    )

    @pytest.mark.parametrize("fw, str_rep", tup_fw_str)
    def test_str(self, fw, str_rep):
        assert str(fw) == str_rep
        assert repr(fw) == str_rep

    tup_fw_mult = (
        (fw1, fw1, FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"})),
        (
            fw1,
            fw3,
            FermiWord(
                {(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 3): "-", (4, 0): "+", (5, 4): "-"}
            ),
        ),
        (fw2, fw1, FermiWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-"})),
        (fw1, fw4, fw1),
        (fw4, fw3, fw3),
        (fw4, fw4, fw4),
    )

    @pytest.mark.parametrize("f1, f2, result_fw", tup_fw_mult)
    def test_mul(self, f1, f2, result_fw):
        assert f1 * f2 == result_fw

    tup_fw_mult_error = (
        (fw1, [1.5]),
        (fw4, "string"),
    )

    @pytest.mark.parametrize("f1, f2", tup_fw_mult_error)
    def test_mul_error(self, f1, f2):
        with pytest.raises(TypeError, match=f"Cannot multiply FermiWord by {type(f2)}."):
            f1 * f2  # pylint: disable=pointless-statement

    tup_fw_pow = (
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

    @pytest.mark.parametrize("f1, pow, result_fw", tup_fw_pow)
    def test_pow(self, f1, pow, result_fw):
        assert f1**pow == result_fw

    tup_fw_pow_error = ((fw1, -1), (fw3, 1.5))

    @pytest.mark.parametrize("f1, pow", tup_fw_pow_error)
    def test_pow_error(self, f1, pow):
        with pytest.raises(ValueError, match="The exponent must be a positive integer."):
            f1**pow  # pylint: disable=pointless-statement

    def test_pickling(self):
        """Check that FermiWords can be pickled and unpickled."""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        serialization = pickle.dumps(fw)
        new_fw = pickle.loads(serialization)
        assert fw == new_fw

    @pytest.mark.parametrize(
        "operator",
        [
            ({(0, 0): "+", (2, 1): "-"}),
            ({(0, 0): "+", (1, 1): "-", (3, 0): "+", (4, 1): "-"}),
            ({(-1, 0): "+", (0, 1): "-", (1, 0): "+", (2, 1): "-"}),
        ],
    )
    def test_init_error(self, operator):
        """Test that an error is raised if the operator orders are not correct."""
        with pytest.raises(ValueError, match="The operator indices must belong to the set"):
            FermiWord(operator)


    FERMI_AND_PAULI_WORDS = (
        (
            FermiWord({(0, 1): "+", (1, 1): "-", (2, 0): "+", (3, 0): "-"}),  # [1, 1, 0, 0],
            # obtained with openfermion using: jordan_wigner(FermionOperator('1^ 1 0^ 0', 1))
            (
                [(0.25 + 0j), (-0.25 + 0j), (0.25 + 0j), (-0.25 + 0j)],
                [
                    PauliWord({0: "I"}),
                    PauliWord({0: "Z"}),
                    PauliWord({0: "Z", 1: "Z"}),
                    PauliWord({1: "Z"}),
                ],
            ),
        ),
        (
            FermiWord({(0, 5): "+", (1, 5): "-", (2, 5): "+", (3, 5): "-"}),  # [5, 5, 5, 5],
            # obtained with openfermion using: jordan_wigner(FermionOperator('5^ 5 5^ 5', 1))
            (
                [(0.5 + 0j), (-0.5 + 0j)],
                [PauliWord({0: "I"}), PauliWord({5: "Z"})],
            ),
        ),
        (
            FermiWord({(0, 3): "+", (1, 3): "-", (2, 3): "+", (3, 1): "-"}),
            # obtained with openfermion using: jordan_wigner(FermionOperator('3^ 3 3^ 1', 1))
            (
                [(0.25 + 0j), (-0.25j), (0.25j), (0.25 + 0j)],
                [
                    PauliWord({1: "X", 2: "Z", 3: "X"}),
                    PauliWord({1: "X", 2: "Z", 3: "Y"}),
                    PauliWord({1: "Y", 2: "Z", 3: "X"}),
                    PauliWord({1: "Y", 2: "Z", 3: "Y"}),
                ],
            ),
        ),
    )

    @pytest.mark.parametrize("operator, pauli_equivalent", FERMI_AND_PAULI_WORDS)
    def test_to_qubit(self, operator, pauli_equivalent):

        ps = operator.to_qubit()
        coeffs, words = pauli_equivalent

        assert ps == PauliSentence(dict(zip(words, coeffs)))


fs1 = FermiSentence({fw1: 1.23, fw2: 4j, fw3: -0.5})
fs2 = FermiSentence({fw1: -1.23, fw2: -4j, fw3: 0.5})
fs1_hamiltonian = FermiSentence({fw1: 1.23, fw2: 4, fw3: -0.5})
fs2_hamiltonian = FermiSentence({fw1: -1.23, fw2: -4, fw3: 0.5})
fs3 = FermiSentence({fw3: -0.5, fw4: 1})
fs4 = FermiSentence({fw4: 1})
fs5 = FermiSentence({})

fs1_x_fs2 = FermiSentence(  # fs1 * fs1, computed by hand
    {
        FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}): 1.5129,
        FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 0): "-"}): 4.92j,
        FermiWord(
            {
                (0, 0): "+",
                (1, 1): "-",
                (2, 0): "+",
                (3, 3): "-",
                (4, 0): "+",
                (5, 4): "-",
            }
        ): -0.615,
        FermiWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-"}): 4.92j,
        FermiWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 0): "-"}): -16,
        FermiWord(
            {
                (0, 0): "+",
                (1, 0): "-",
                (2, 0): "+",
                (3, 3): "-",
                (4, 0): "+",
                (5, 4): "-",
            }
        ): (-0 - 2j),
        FermiWord(
            {
                (0, 0): "+",
                (1, 3): "-",
                (2, 0): "+",
                (3, 4): "-",
                (4, 0): "+",
                (5, 1): "-",
            }
        ): -0.615,
        FermiWord(
            {
                (0, 0): "+",
                (1, 3): "-",
                (2, 0): "+",
                (3, 4): "-",
                (4, 0): "+",
                (5, 0): "-",
            }
        ): (-0 - 2j),
        FermiWord(
            {
                (0, 0): "+",
                (1, 3): "-",
                (2, 0): "+",
                (3, 4): "-",
                (4, 0): "+",
                (5, 3): "-",
                (6, 0): "+",
                (7, 4): "-",
            }
        ): 0.25,
    }
)


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
        (fs1, "1.23 * '0+ 1-'\n" + "+ 4j * '0+ 0-'\n" + "+ -0.5 * '0+ 3- 0+ 4-'"),
        (fs2, "-1.23 * '0+ 1-'\n" + "+ (-0-4j) * '0+ 0-'\n" + "+ 0.5 * '0+ 3- 0+ 4-'"),
        (fs3, "-0.5 * '0+ 3- 0+ 4-'\n" + "+ 1 * 'I'"),
        (fs4, "1 * 'I'"),
        (fs5, "0 * 'I'"),
    )

    @pytest.mark.parametrize("fs, str_rep", tup_fs_str)
    def test_str(self, fs, str_rep):
        """Test the string representation of the FermiSentence."""
        print(str(fs))
        assert str(fs) == str_rep
        assert repr(fs) == str_rep

    tup_fs_wires = (
        (fs1, {0, 1, 3, 4}),
        (fs2, {0, 1, 3, 4}),
        (fs3, {0, 3, 4}),
        (fs4, set()),
    )

    @pytest.mark.parametrize("fs, wires", tup_fs_wires)
    def test_wires(self, fs, wires):
        """Test the correct wires are given for the FermiSentence."""
        assert fs.wires == wires

    @pytest.mark.parametrize("fs", (fs1, fs2, fs3, fs4))
    def test_copy(self, fs):
        """Test that the copy is identical to the original."""
        copy_fs = copy(fs)
        deep_copy_fs = deepcopy(fs)

        assert copy_fs == fs
        assert deep_copy_fs == fs
        assert copy_fs is not fs
        assert deep_copy_fs is not fs

    tup_fs_mult = (  # computed by hand
        (
            fs1,
            fs1,
            fs1_x_fs2,
        ),
        (
            fs3,
            fs4,
            fs3,
        ),
        (
            fs4,
            fs4,
            FermiSentence(
                {
                    FermiWord({}): 1,
                }
            ),
        ),
        (fs5, fs3, fs5),
        (fs3, fs5, fs5),
        (fs4, fs3, fs3),
        (fs3, fs4, fs3),
        (
            FermiSentence({fw2: 1, fw3: 1, fw4: 1}),
            FermiSentence({fw4: 1, fw2: 1}),
            FermiSentence({fw2: 2, fw3: 1, fw4: 1, fw2 * fw2: 1, fw3 * fw2: 1}),
        ),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_fs_mult)
    def test_mul(self, f1, f2, result):
        """Test that the correct result of multiplication is produced."""
        simplified_product = f1 * f2
        simplified_product.simplify()

        assert simplified_product == result

    tup_fs_add = (  # computed by hand
        (fs1, fs1, FermiSentence({fw1: 2.46, fw2: 8j, fw3: -1})),
        (fs1, fs2, FermiSentence({})),
        (fs1, fs3, FermiSentence({fw1: 1.23, fw2: 4j, fw3: -1, fw4: 1})),
        (fs2, fs5, fs2),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_fs_add)
    def test_add(self, f1, f2, result):
        """Test that the correct result of addition is produced."""

        simplified_product = f1 + f2
        simplified_product.simplify()

        assert simplified_product == result

    def test_simplify(self):
        """Test that simplify removes terms in the FermiSentence with coefficient less than the
        threshold."""
        un_simplified_fs = FermiSentence({fw1: 0.001, fw2: 0.05, fw3: 1})

        expected_simplified_fs0 = FermiSentence({fw1: 0.001, fw2: 0.05, fw3: 1})
        expected_simplified_fs1 = FermiSentence({fw2: 0.05, fw3: 1})
        expected_simplified_fs2 = FermiSentence({fw3: 1})

        un_simplified_fs.simplify()
        assert un_simplified_fs == expected_simplified_fs0  # default tol = 1e-8
        un_simplified_fs.simplify(tol=1e-2)
        assert un_simplified_fs == expected_simplified_fs1
        un_simplified_fs.simplify(tol=1e-1)
        assert un_simplified_fs == expected_simplified_fs2

    def test_pickling(self):
        """Check that FermiSentences can be pickled and unpickled."""
        f1 = FermiWord({(0, 0): "+", (1, 1): "-"})
        f2 = FermiWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
        fs = FermiSentence({f1: 1.5, f2: -0.5})

        serialization = pickle.dumps(fs)
        new_fs = pickle.loads(serialization)
        assert fs == new_fs

    tup_fs_pow = (
        (fs1, 0, FermiSentence({FermiWord({}): 1})),
        (fs1, 1, fs1),
        (fs1, 2, fs1_x_fs2),
        (fs3, 0, FermiSentence({FermiWord({}): 1})),
        (fs3, 1, fs3),
        (fs4, 0, FermiSentence({FermiWord({}): 1})),
        (fs4, 3, fs4),
    )

    @pytest.mark.parametrize("f1, pow, result", tup_fs_pow)
    def test_pow(self, f1, pow, result):
        assert f1**pow == result

    tup_fs_pow_error = ((fs1, -1), (fs3, 1.5))

    @pytest.mark.parametrize("f1, pow", tup_fs_pow_error)
    def test_pow_error(self, f1, pow):
        with pytest.raises(ValueError, match="The exponent must be a positive integer."):
            f1**pow  # pylint: disable=pointless-statement


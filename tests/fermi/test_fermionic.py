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

import numpy as np
import pytest

from pennylane import numpy as pnp
from pennylane.fermi.fermionic import FermiSentence, FermiWord, from_string

# pylint: disable=too-many-public-methods

fw1 = FermiWord({(0, 0): "+", (1, 1): "-"})
fw2 = FermiWord({(0, 0): "+", (1, 0): "-"})
fw3 = FermiWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
fw4 = FermiWord({})
fw5 = FermiWord({(0, 10): "+", (1, 30): "-", (2, 0): "+", (3, 400): "-"})
fw6 = FermiWord({(0, 10): "+", (1, 30): "+", (2, 0): "-", (3, 400): "-"})
fw7 = FermiWord({(0, 10): "-", (1, 30): "+", (2, 0): "-", (3, 400): "+"})


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
        (fw1, "a\u207a(0) a(1)"),
        (fw2, "a\u207a(0) a(0)"),
        (fw3, "a\u207a(0) a(3) a\u207a(0) a(4)"),
        (fw4, "I"),
        (fw5, "a\u207a(10) a(30) a\u207a(0) a(400)"),
        (fw6, "a\u207a(10) a\u207a(30) a(0) a(400)"),
        (fw7, "a(10) a\u207a(30) a(0) a\u207a(400)"),
    )

    @pytest.mark.parametrize("fw, str_rep", tup_fw_compact)
    def test_compact(self, fw, str_rep):
        """Test string representation from to_string"""
        assert fw.to_string() == str_rep

    @pytest.mark.parametrize("fw, str_rep", tup_fw_compact)
    def test_str(self, fw, str_rep):
        """Test __str__ and __repr__ methods"""
        assert str(fw) == str_rep
        assert repr(fw) == str_rep

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


class TestFermiWordArithmetic:
    WORDS_MUL = (
        (
            fw1,
            fw1,
            FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
            FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
        ),
        (
            fw1,
            fw1,
            FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
            FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
        ),
        (
            fw1,
            fw3,
            FermiWord(
                {(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 3): "-", (4, 0): "+", (5, 4): "-"}
            ),
            FermiWord(
                {(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-", (4, 0): "+", (5, 1): "-"}
            ),
        ),
        (
            fw2,
            fw1,
            FermiWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-"}),
            FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 0): "-"}),
        ),
        (fw1, fw4, fw1, fw1),
        (fw4, fw3, fw3, fw3),
        (fw4, fw4, fw4, fw4),
    )

    @pytest.mark.parametrize("f1, f2, result_fw_right, result_fw_left", WORDS_MUL)
    def test_mul_fermi_words(self, f1, f2, result_fw_right, result_fw_left):
        """Test that two FermiWords can be multiplied together and return a new
        FermiWord, with operators in the expected order"""
        assert f1 * f2 == result_fw_right
        assert f2 * f1 == result_fw_left

    WORDS_AND_SENTENCES_MUL = (
        (
            fw1,
            FermiSentence({fw3: 1.2}),
            FermiSentence({fw1 * fw3: 1.2}),
        ),
        (
            fw2,
            FermiSentence({fw3: 1.2, fw1: 3.7}),
            FermiSentence({fw2 * fw3: 1.2, fw2 * fw1: 3.7}),
        ),
    )

    @pytest.mark.parametrize("fw, fs, result", WORDS_AND_SENTENCES_MUL)
    def test_mul_fermi_word_and_sentence(self, fw, fs, result):
        """Test that a FermiWord can be multiplied by a FermiSentence
        and return a new FermiSentence"""
        assert fw * fs == result

    WORDS_AND_NUMBERS_MUL = (
        (fw1, 2, FermiSentence({fw1: 2})),  # int
        (fw2, 3.7, FermiSentence({fw2: 3.7})),  # float
        (fw2, 2j, FermiSentence({fw2: 2j})),  # complex
        (fw2, np.array([2]), FermiSentence({fw2: 2})),  # numpy array
        (fw1, pnp.array([2]), FermiSentence({fw1: 2})),  # pennylane numpy array
        (fw1, pnp.array([2, 2])[0], FermiSentence({fw1: 2})),  # pennylane tensor with no length
    )

    @pytest.mark.parametrize("fw, number, result", WORDS_AND_NUMBERS_MUL)
    def test_mul_number(self, fw, number, result):
        """Test that a FermiWord can be multiplied onto a number (from the left)
        and return a FermiSentence"""
        assert fw * number == result

    @pytest.mark.parametrize("fw, number, result", WORDS_AND_NUMBERS_MUL)
    def test_rmul_number(self, fw, number, result):
        """Test that a FermiWord can be multiplied onto a number (from the right)
        and return a FermiSentence"""
        assert number * fw == result

    tup_fw_mult_error = (
        (fw1, [1.5]),
        (fw4, "string"),
    )

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_mul_error(self, fw, bad_type):
        """Test multiply with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply FermiWord by {type(bad_type)}."):
            fw * bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_rmul_error(self, fw, bad_type):
        """Test __rmul__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply FermiWord by {type(bad_type)}."):
            bad_type * fw  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_add_error(self, fw, bad_type):
        """Test __add__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a FermiWord"):
            fw + bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_radd_error(self, fw, bad_type):
        """Test __radd__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add a FermiWord to {type(bad_type)}"):
            bad_type + fw  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_sub_error(self, fw, bad_type):
        """Test __sub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot subtract {type(bad_type)} from a FermiWord"):
            fw - bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_rsub_error(self, fw, bad_type):
        """Test __rsub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot subtract a FermiWord from {type(bad_type)}"):
            bad_type - fw  # pylint: disable=pointless-statement

    WORDS_ADD = [
        (fw1, fw2, FermiSentence({fw1: 1, fw2: 1})),
        (fw3, fw2, FermiSentence({fw2: 1, fw3: 1})),
        (fw2, fw2, FermiSentence({fw2: 2})),
    ]

    @pytest.mark.parametrize("f1, f2, res", WORDS_ADD)
    def test_add_fermi_words(self, f1, f2, res):
        """Test that adding two FermiWords returns the expected FermiSentence"""
        assert f1 + f2 == res
        assert f2 + f1 == res

    WORDS_AND_SENTENCES_ADD = [
        (fw1, FermiSentence({fw1: 1.2, fw3: 3j}), FermiSentence({fw1: 2.2, fw3: 3j})),
        (fw3, FermiSentence({fw1: 1.2, fw3: 3j}), FermiSentence({fw1: 1.2, fw3: (1 + 3j)})),
        (fw1, FermiSentence({fw1: -1.2, fw3: 3j}), FermiSentence({fw1: -0.2, fw3: 3j})),
    ]

    @pytest.mark.parametrize("w, s, res", WORDS_AND_SENTENCES_ADD)
    def test_add_fermi_words_and_sentences(self, w, s, res):
        """Test that adding a FermiSentence to a FermiWord returns the expected FermiSentence"""
        sum = w + s
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    WORDS_AND_CONSTANTS_ADD = [
        (fw1, 5, FermiSentence({fw1: 1, fw4: 5})),  # int
        (fw2, 2.8, FermiSentence({fw2: 1, fw4: 2.8})),  # float
        (fw3, (1 + 3j), FermiSentence({fw3: 1, fw4: (1 + 3j)})),  # complex
        (fw1, np.array([5]), FermiSentence({fw1: 1, fw4: 5})),  # numpy array
        (fw2, pnp.array([2.8]), FermiSentence({fw2: 1, fw4: 2.8})),  # pennylane numpy array
        (
            fw1,
            pnp.array([2, 2])[0],
            FermiSentence({fw1: 1, fw4: 2}),
        ),  # pennylane tensor with no length
        (fw4, 2, FermiSentence({fw4: 3})),  # FermiWord is Identity
    ]

    @pytest.mark.parametrize("w, c, res", WORDS_AND_CONSTANTS_ADD)
    def test_add_fermi_words_and_constants(self, w, c, res):
        """Test that adding a constant (int, float or complex) to a FermiWord
        returns the expected FermiSentence"""
        sum = w + c
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    @pytest.mark.parametrize("w, c, res", WORDS_AND_CONSTANTS_ADD)
    def test_radd_fermi_words_and_constants(self, w, c, res):
        """Test that adding a Fermi word to a constant (int, float or complex)
        returns the expected FermiSentence (__radd__)"""
        sum = c + w
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    WORDS_SUB = [
        (fw1, fw2, FermiSentence({fw1: 1, fw2: -1}), FermiSentence({fw1: -1, fw2: 1})),
        (fw2, fw3, FermiSentence({fw2: 1, fw3: -1}), FermiSentence({fw2: -1, fw3: 1})),
        (fw2, fw2, FermiSentence({fw2: 0}), FermiSentence({fw2: 0})),
    ]

    @pytest.mark.parametrize("f1, f2, res1, res2", WORDS_SUB)
    def test_subtract_fermi_words(self, f1, f2, res1, res2):
        """Test that subtracting one FermiWord from another returns the expected FermiSentence"""
        assert f1 - f2 == res1
        assert f2 - f1 == res2

    WORDS_AND_SENTENCES_SUB = [
        (fw1, FermiSentence({fw1: 1.2, fw3: 3j}), FermiSentence({fw1: -0.2, fw3: -3j})),
        (fw3, FermiSentence({fw1: 1.2, fw3: 3j}), FermiSentence({fw1: -1.2, fw3: (1 - 3j)})),
        (fw1, FermiSentence({fw1: -1.2, fw3: 3j}), FermiSentence({fw1: 2.2, fw3: -3j})),
    ]

    @pytest.mark.parametrize("w, s, res", WORDS_AND_SENTENCES_SUB)
    def test_subtract_fermi_words_and_sentences(self, w, s, res):
        """Test that subtracting a FermiSentence from a FermiWord returns the expected FermiSentence"""
        diff = w - s
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        diff_rounded = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in diff.items()}
        )

        assert diff_rounded == res

    WORDS_AND_CONSTANTS_SUBTRACT = [
        (fw1, 5, FermiSentence({fw1: 1, fw4: -5})),  # int
        (fw2, 2.8, FermiSentence({fw2: 1, fw4: -2.8})),  # float
        (fw3, (1 + 3j), FermiSentence({fw3: 1, fw4: -(1 + 3j)})),  # complex
        (fw1, np.array([5]), FermiSentence({fw1: 1, fw4: -5})),  # numpy array
        (fw2, pnp.array([2.8]), FermiSentence({fw2: 1, fw4: -2.8})),  # pennylane numpy array
        (
            fw1,
            pnp.array([2, 2])[0],
            FermiSentence({fw1: 1, fw4: -2}),
        ),  # pennylane tensor with no length
        (fw4, 2, FermiSentence({fw4: -1})),  # FermiWord is Identity
    ]

    @pytest.mark.parametrize("w, c, res", WORDS_AND_CONSTANTS_SUBTRACT)
    def test_subtract_constant_from_fermi_word(self, w, c, res):
        """Test that subtracting a constant (int, float or complex) from a FermiWord
        returns the expected FermiSentence"""
        diff = w - c
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        diff_rounded = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in diff.items()}
        )
        assert diff_rounded == res

    CONSTANTS_AND_WORDS_SUBTRACT = [
        (fw1, 5, FermiSentence({fw1: -1, fw4: 5})),  # int
        (fw2, 2.8, FermiSentence({fw2: -1, fw4: 2.8})),  # float
        (fw3, (1 + 3j), FermiSentence({fw3: -1, fw4: (1 + 3j)})),  # complex
        (fw1, np.array([5]), FermiSentence({fw1: -1, fw4: 5})),  # numpy array
        (fw2, pnp.array([2.8]), FermiSentence({fw2: -1, fw4: 2.8})),  # pennylane numpy array
        (
            fw1,
            pnp.array([2, 2])[0],
            FermiSentence({fw1: -1, fw4: 2}),
        ),  # pennylane tensor with no length
        (fw4, 2, FermiSentence({fw4: 1})),  # FermiWord is Identity
    ]

    @pytest.mark.parametrize("w, c, res", CONSTANTS_AND_WORDS_SUBTRACT)
    def test_subtract_fermi_words_from_constant(self, w, c, res):
        """Test that subtracting a constant (int, float or complex) from a FermiWord
        returns the expected FermiSentence"""
        diff = c - w
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        diff_rounded = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in diff.items()}
        )
        assert diff_rounded == res

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
        """Test that raising a FermiWord to an integer power returns the expected FermiWord"""
        assert f1**pow == result_fw

    tup_fw_pow_error = ((fw1, -1), (fw3, 1.5))

    @pytest.mark.parametrize("f1, pow", tup_fw_pow_error)
    def test_pow_error(self, f1, pow):
        """Test that invalid values for the exponent raises an error"""
        with pytest.raises(ValueError, match="The exponent must be a positive integer."):
            f1**pow  # pylint: disable=pointless-statement

    @pytest.mark.parametrize(
        "method_name", ("__add__", "__sub__", "__mul__", "__radd__", "__rsub__", "__rmul__")
    )
    def test_array_must_not_exceed_length_1(self, method_name):
        with pytest.raises(
            ValueError, match="Arithmetic Fermi operations can only accept an array of length 1"
        ):
            method_to_test = getattr(fw1, method_name)
            _ = method_to_test(np.array([1, 2]))


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
        """Test that we can add a new key to a FermiSentence."""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        fs = FermiSentence({fw: 1.0})

        new_fw = FermiWord({(0, 2): "+", (1, 3): "-"})
        assert new_fw not in fs.keys()

        fs[new_fw] = 3.45
        assert new_fw in fs.keys() and fs[new_fw] == 3.45

    tup_fs_str = (
        (
            fs1,
            "1.23 * a\u207a(0) a(1)\n"
            + "+ 4j * a\u207a(0) a(0)\n"
            + "+ -0.5 * a\u207a(0) a(3) a\u207a(0) a(4)",
        ),
        (
            fs2,
            "-1.23 * a\u207a(0) a(1)\n"
            + "+ (-0-4j) * a\u207a(0) a(0)\n"
            + "+ 0.5 * a\u207a(0) a(3) a\u207a(0) a(4)",
        ),
        (fs3, "-0.5 * a\u207a(0) a(3) a\u207a(0) a(4)\n" + "+ 1 * I"),
        (fs4, "1 * I"),
        (fs5, "0 * I"),
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


class TestFermiSentenceArithmetic:
    tup_fs_mult = (  # computed by hand
        (
            fs1,
            fs1,
            fs1_x_fs2,
        ),
        (
            fs3,
            fs4,
            FermiSentence(
                {
                    FermiWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"}): -0.5,
                    FermiWord({}): 1,
                }
            ),
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
    def test_mul_fermi_sentences(self, f1, f2, result):
        """Test that the correct result of multiplication between two
        FermiSentences is produced."""

        simplified_product = f1 * f2
        simplified_product.simplify()

        assert simplified_product == result

    SENTENCES_AND_WORDS_MUL = (
        (
            fw1,
            FermiSentence({fw3: 1.2}),
            FermiSentence({fw3 * fw1: 1.2}),
        ),
        (
            fw2,
            FermiSentence({fw3: 1.2, fw1: 3.7}),
            FermiSentence({fw3 * fw2: 1.2, fw1 * fw2: 3.7}),
        ),
    )

    @pytest.mark.parametrize("fw, fs, result", SENTENCES_AND_WORDS_MUL)
    def test_mul_fermi_word_and_sentence(self, fw, fs, result):
        """Test that a FermiWord and a FermiSentence can be multiplied together
        and return a new FermiSentence"""
        assert fs * fw == result

    SENTENCES_AND_NUMBERS_MUL = (
        (fs1, 2, FermiSentence({fw1: 1.23 * 2, fw2: 4j * 2, fw3: -0.5 * 2})),  # int
        (fs2, 3.4, FermiSentence({fw1: -1.23 * 3.4, fw2: -4j * 3.4, fw3: 0.5 * 3.4})),  # float
        (fs1, 3j, FermiSentence({fw1: 3.69j, fw2: -12, fw3: -1.5j})),  # complex
        (fs5, 10, FermiSentence({})),  # null operator times constant
        (
            fs1,
            np.array([2]),
            FermiSentence({fw1: 1.23 * 2, fw2: 4j * 2, fw3: -0.5 * 2}),
        ),  # numpy array
        (
            fs1,
            pnp.array([2]),
            FermiSentence({fw1: 1.23 * 2, fw2: 4j * 2, fw3: -0.5 * 2}),
        ),  # pennylane numpy array
        (
            fs1,
            pnp.array([2, 2])[0],
            FermiSentence({fw1: 1.23 * 2, fw2: 4j * 2, fw3: -0.5 * 2}),
        ),  # pennylane tensor with no length
    )

    @pytest.mark.parametrize("fs, number, result", SENTENCES_AND_NUMBERS_MUL)
    def test_mul_number(self, fs, number, result):
        """Test that a FermiSentence can be multiplied onto a number (from the left)
        and return a FermiSentence"""
        assert fs * number == result

    @pytest.mark.parametrize("fs, number, result", SENTENCES_AND_NUMBERS_MUL)
    def test_rmul_number(self, fs, number, result):
        """Test that a FermiSentence can be multiplied onto a number (from the right)
        and return a FermiSentence"""
        assert number * fs == result

    tup_fs_add = (  # computed by hand
        (fs1, fs1, FermiSentence({fw1: 2.46, fw2: 8j, fw3: -1})),
        (fs1, fs2, FermiSentence({})),
        (fs1, fs3, FermiSentence({fw1: 1.23, fw2: 4j, fw3: -1, fw4: 1})),
        (fs2, fs5, fs2),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_fs_add)
    def test_add_fermi_sentences(self, f1, f2, result):
        """Test that the correct result of addition is produced for two FermiSentences."""

        simplified_product = f1 + f2
        simplified_product.simplify()

        assert simplified_product == result

    SENTENCES_AND_WORDS_ADD = [
        (fw1, FermiSentence({fw1: 1.2, fw3: 3j}), FermiSentence({fw1: 2.2, fw3: 3j})),
        (fw3, FermiSentence({fw1: 1.2, fw3: 3j}), FermiSentence({fw1: 1.2, fw3: (1 + 3j)})),
        (fw1, FermiSentence({fw1: -1.2, fw3: 3j}), FermiSentence({fw1: -0.2, fw3: 3j})),
    ]

    @pytest.mark.parametrize("w, s, res", SENTENCES_AND_WORDS_ADD)
    def test_add_fermi_words_and_sentences(self, w, s, res):
        """Test that adding a FermiWord to a FermiSentence returns the expected FermiSentence"""
        sum = s + w
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    SENTENCES_AND_CONSTANTS_ADD = [
        (FermiSentence({fw1: 1.2, fw3: 3j}), 3, FermiSentence({fw1: 1.2, fw3: 3j, fw4: 3})),  # int
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            1.3,
            FermiSentence({fw1: 1.2, fw3: 3j, fw4: 1.3}),
        ),  # float
        (
            FermiSentence({fw1: -1.2, fw3: 3j}),  # complex
            (1 + 2j),
            FermiSentence({fw1: -1.2, fw3: 3j, fw4: (1 + 2j)}),
        ),
        (FermiSentence({}), 5, FermiSentence({fw4: 5})),  # null sentence
        (FermiSentence({fw4: 3}), 1j, FermiSentence({fw4: 3 + 1j})),  # identity only
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            np.array([3]),
            FermiSentence({fw1: 1.2, fw3: 3j, fw4: 3}),
        ),  # numpy array
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3]),
            FermiSentence({fw1: 1.2, fw3: 3j, fw4: 3}),
        ),  # pennylane numpy array
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3, 0])[0],
            FermiSentence({fw1: 1.2, fw3: 3j, fw4: 3}),
        ),  # pennylane tensor with no length
    ]

    @pytest.mark.parametrize("s, c, res", SENTENCES_AND_CONSTANTS_ADD)
    def test_add_fermi_sentences_and_constants(self, s, c, res):
        """Test that adding a constant to a FermiSentence returns the expected FermiSentence"""
        sum = s + c
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    @pytest.mark.parametrize("s, c, res", SENTENCES_AND_CONSTANTS_ADD)
    def test_radd_fermi_sentences_and_constants(self, s, c, res):
        """Test that adding a FermiSentence to a constant (__radd___) returns the expected FermiSentence"""
        sum = c + s
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    SENTENCE_MINUS_WORD = (  # computed by hand
        (fs1, fw1, FermiSentence({fw1: 0.23, fw2: 4j, fw3: -0.5})),
        (fs2, fw3, FermiSentence({fw1: -1.23, fw2: -4j, fw3: -0.5})),
        (fs3, fw4, FermiSentence({fw3: -0.5})),
        (FermiSentence({fw1: 1.2, fw3: 3j}), fw1, FermiSentence({fw1: 0.2, fw3: 3j})),
        (FermiSentence({fw1: 1.2, fw3: 3j}), fw3, FermiSentence({fw1: 1.2, fw3: (-1 + 3j)})),
        (FermiSentence({fw1: -1.2, fw3: 3j}), fw1, FermiSentence({fw1: -2.2, fw3: 3j})),
    )

    @pytest.mark.parametrize("fs, fw, result", SENTENCE_MINUS_WORD)
    def test_subtract_fermi_word_from_fermi_sentence(self, fs, fw, result):
        """Test that the correct result is produced if a FermiWord is
        subtracted from a FermiSentence"""

        simplified_diff = fs - fw
        simplified_diff.simplify()
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        simplified_diff = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in simplified_diff.items()}
        )

        assert simplified_diff == result

    SENTENCE_MINUS_CONSTANT = (  # computed by hand
        (fs1, 3, FermiSentence({fw1: 1.23, fw2: 4j, fw3: -0.5, fw4: -3})),  # int
        (fs2, -2.7, FermiSentence({fw1: -1.23, fw2: -4j, fw3: 0.5, fw4: 2.7})),  # float
        (fs3, 2j, FermiSentence({fw3: -0.5, fw4: (1 - 2j)})),  # complex
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            -4,
            FermiSentence({fw1: 1.2, fw3: 3j, fw4: 4}),
        ),  # negative int
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            np.array([3]),
            FermiSentence({fw1: 1.2, fw3: 3j, fw4: -3}),
        ),  # numpy array
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3]),
            FermiSentence({fw1: 1.2, fw3: 3j, fw4: -3}),
        ),  # pennylane numpy array
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3, 2])[0],
            FermiSentence({fw1: 1.2, fw3: 3j, fw4: -3}),
        ),  # pennylane tensor with no len
    )

    @pytest.mark.parametrize("fs, c, result", SENTENCE_MINUS_CONSTANT)
    def test_subtract_constant_from_fermi_sentence(self, fs, c, result):
        """Test that the correct result is produced if a FermiWord is
        subtracted from a FermiSentence"""

        simplified_diff = fs - c
        simplified_diff.simplify()
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        simplified_diff = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in simplified_diff.items()}
        )
        assert simplified_diff == result

    CONSTANT_MINUS_SENTENCE = (  # computed by hand
        (fs1, 3, FermiSentence({fw1: -1.23, fw2: -4j, fw3: 0.5, fw4: 3})),
        (fs2, -2.7, FermiSentence({fw1: 1.23, fw2: 4j, fw3: -0.5, fw4: -2.7})),
        (fs3, 2j, FermiSentence({fw3: 0.5, fw4: (-1 + 2j)})),
        (FermiSentence({fw1: 1.2, fw3: 3j}), -4, FermiSentence({fw1: -1.2, fw3: -3j, fw4: -4})),
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            np.array([3]),
            FermiSentence({fw1: -1.2, fw3: -3j, fw4: 3}),
        ),  # numpy array
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3]),
            FermiSentence({fw1: -1.2, fw3: -3j, fw4: 3}),
        ),  # pennylane numpy array
        (
            FermiSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3, 3])[0],
            FermiSentence({fw1: -1.2, fw3: -3j, fw4: 3}),
        ),  # pennylane tensor with to len
    )

    @pytest.mark.parametrize("fs, c, result", CONSTANT_MINUS_SENTENCE)
    def test_subtract_fermi_sentence_from_constant(self, fs, c, result):
        """Test that the correct result is produced if a FermiWord is
        subtracted from a FermiSentence"""

        simplified_diff = c - fs
        simplified_diff.simplify()
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        simplified_diff = FermiSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in simplified_diff.items()}
        )
        assert simplified_diff == result

    tup_fs_subtract = (  # computed by hand
        (fs1, fs1, FermiSentence({})),
        (fs1, fs2, FermiSentence({fw1: 2.46, fw2: 8j, fw3: -1})),
        (fs1, fs3, FermiSentence({fw1: 1.23, fw2: 4j, fw4: -1})),
        (fs2, fs5, fs2),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_fs_subtract)
    def test_subtract_fermi_sentences(self, f1, f2, result):
        """Test that the correct result of subtraction is produced for two FermiSentences."""

        simplified_product = f1 - f2
        simplified_product.simplify()

        assert simplified_product == result

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
        """Test that raising a FermiWord to an integer power returns the expected FermiWord"""
        assert f1**pow == result

    tup_fs_pow_error = ((fs1, -1), (fs3, 1.5))

    @pytest.mark.parametrize("f1, pow", tup_fs_pow_error)
    def test_pow_error(self, f1, pow):
        """Test that invalid values for the exponent raises an error"""
        with pytest.raises(ValueError, match="The exponent must be a positive integer."):
            f1**pow  # pylint: disable=pointless-statement

    TYPE_ERRORS = (
        (fs1, [1.5]),
        (fs4, "string"),
    )

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_add_error(self, fs, bad_type):
        """Test __add__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a FermiSentence."):
            fs + bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_radd_error(self, fs, bad_type):
        """Test __radd__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add a FermiSentence to {type(bad_type)}."):
            bad_type + fs  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_sub_error(self, fs, bad_type):
        """Test __sub__ with unsupported type raises an error"""
        with pytest.raises(
            TypeError, match=f"Cannot subtract {type(bad_type)} from a FermiSentence."
        ):
            fs - bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_rsub_error(self, fs, bad_type):
        """Test __rsub__ with unsupported type raises an error"""
        with pytest.raises(
            TypeError, match=f"Cannot subtract a FermiSentence from {type(bad_type)}."
        ):
            bad_type - fs  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_mul_error(self, fs, bad_type):
        """Test __sub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply FermiSentence by {type(bad_type)}."):
            fs * bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_rmul_error(self, fs, bad_type):
        """Test __rsub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply {type(bad_type)} by FermiSentence."):
            bad_type * fs  # pylint: disable=pointless-statement

    tup_fw_string = (
        ("0+ 1-", fw1),
        ("0+ 0-", fw2),
        ("0+ 3- 0+ 4-", fw3),
        ("0+   3- 0+    4-    ", fw3),
        ("10+ 30- 0+ 400-", fw5),
        ("10+ 30+ 0- 400-", fw6),
        ("0^ 1", fw1),
        ("0^ 0", fw2),
        ("0^    0", fw2),
        ("0^ 3 0^ 4", fw3),
        ("10^ 30 0^ 400", fw5),
        ("10^ 30^ 0 400", fw6),
        ("0+ 1", fw1),
        ("0+  1", fw1),
        ("0+ 0", fw2),
        ("0+ 3 0+ 4", fw3),
        ("10+ 30 0+ 400", fw5),
        ("10+ 30+ 0 400", fw6),
        ("", fw4),
        (" ", fw4),
    )

    @pytest.mark.parametrize("string, result_fw", tup_fw_string)
    def test_from_string(self, string, result_fw):
        assert from_string(string) == result_fw

    tup_fw_string_error = (
        "0+ a-",
        "0+ 1-? 3+ 4-",
    )

    @pytest.mark.parametrize("string", tup_fw_string_error)
    def test_from_string_error(self, string):
        with pytest.raises(ValueError, match="Invalid character encountered in string "):
            from_string(string)  # pylint: disable=pointless-statement

    @pytest.mark.parametrize(
        "method_name", ("__add__", "__sub__", "__mul__", "__radd__", "__rsub__", "__rmul__")
    )
    def test_array_must_not_exceed_length_1(self, method_name):
        with pytest.raises(
            ValueError, match="Arithmetic Fermi operations can only accept an array of length 1"
        ):
            method_to_test = getattr(fs1, method_name)
            _ = method_to_test(np.array([1, 2]))

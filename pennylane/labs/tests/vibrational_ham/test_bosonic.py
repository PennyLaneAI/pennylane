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
"""Unit Tests for the Boseonic representation classes."""
import pickle
from copy import copy, deepcopy

import numpy as np
import pytest
from scipy import sparse

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.labs.vibrational_ham.bosonic import (
    BoseWord,
    BoseSentence,
)

# pylint: disable=too-many-public-methods

fw1 = BoseWord({(0, 0): "+", (1, 1): "-"})
fw1_dag = BoseWord({(0, 1): "+", (1, 0): "-"})

fw2 = BoseWord({(0, 0): "+", (1, 0): "-"})
fw2_dag = BoseWord({(0, 0): "+", (1, 0): "-"})

fw3 = BoseWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
fw3_dag = BoseWord({(0, 4): "+", (1, 0): "-", (2, 3): "+", (3, 0): "-"})

fw4 = BoseWord({})
fw4_dag = BoseWord({})

fw5 = BoseWord({(0, 10): "+", (1, 30): "-", (2, 0): "+", (3, 400): "-"})
fw5_dag = BoseWord({(0, 400): "+", (1, 0): "-", (2, 30): "+", (3, 10): "-"})

fw6 = BoseWord({(0, 10): "+", (1, 30): "+", (2, 0): "-", (3, 400): "-"})
fw6_dag = BoseWord({(0, 400): "+", (1, 0): "+", (2, 30): "-", (3, 10): "-"})

fw7 = BoseWord({(0, 10): "-", (1, 30): "+", (2, 0): "-", (3, 400): "+"})
fw7_dag = BoseWord({(0, 400): "-", (1, 0): "+", (2, 30): "-", (3, 10): "+"})

fw8 = BoseWord({(0, 0): "-", (1, 1): "+"})
fw8c = BoseWord({(0, 1): "+", (1, 0): "-"})
fw8cs = BoseSentence({fw8c: -1})

fw9 = BoseWord({(0, 0): "-", (1, 1): "-"})
fw9c = BoseWord({(0, 1): "-", (1, 0): "-"})
fw9cs = BoseSentence({fw9c: -1})

fw10 = BoseWord({(0, 0): "+", (1, 1): "+"})
fw10c = BoseWord({(0, 1): "+", (1, 0): "+"})
fw10cs = BoseSentence({fw10c: -1})

fw11 = BoseWord({(0, 0): "-", (1, 0): "+"})
fw11c = BoseWord({(0, 0): "+", (1, 0): "-"})
fw11cs = 1 + BoseSentence({fw11c: -1})

fw12 = BoseWord({(0, 0): "+", (1, 0): "+"})
fw12c = BoseWord({(0, 0): "+", (1, 0): "+"})
fw12cs = BoseSentence({fw12c: 1})

fw13 = BoseWord({(0, 0): "-", (1, 0): "-"})
fw13c = BoseWord({(0, 0): "-", (1, 0): "-"})
fw13cs = BoseSentence({fw13c: 1})

fw14 = BoseWord({(0, 0): "+", (1, 0): "-"})
fw14c = BoseWord({(0, 0): "-", (1, 0): "+"})
fw14cs = 1 + BoseSentence({fw14c: -1})

fw15 = BoseWord({(0, 0): "-", (1, 1): "+", (2, 2): "+"})
fw15c = BoseWord({(0, 1): "+", (1, 0): "-", (2, 2): "+"})
fw15cs = BoseSentence({fw15c: -1})

fw16 = BoseWord({(0, 0): "-", (1, 1): "+", (2, 2): "-"})
fw16c = BoseWord({(0, 0): "-", (1, 2): "-", (2, 1): "+"})
fw16cs = BoseSentence({fw16c: -1})

fw17 = BoseWord({(0, 0): "-", (1, 0): "+", (2, 2): "-"})
fw17c1 = BoseWord({(0, 2): "-"})
fw17c2 = BoseWord({(0, 0): "+", (1, 0): "-", (2, 2): "-"})
fw17cs = fw17c1 - fw17c2

fw18 = BoseWord({(0, 0): "+", (1, 1): "+", (2, 2): "-", (3, 3): "-"})
fw18c = BoseWord({(0, 0): "+", (1, 3): "-", (2, 1): "+", (3, 2): "-"})
fw18cs = BoseSentence({fw18c: 1})

fw19 = BoseWord({(0, 0): "+", (1, 1): "+", (2, 2): "-", (3, 2): "+"})
fw19c1 = BoseWord({(0, 0): "+", (1, 1): "+"})
fw19c2 = BoseWord({(0, 2): "+", (1, 0): "+", (2, 1): "+", (3, 2): "-"})
fw19cs = BoseSentence({fw19c1: 1, fw19c2: -1})

fw20 = BoseWord({(0, 0): "-", (1, 0): "+", (2, 1): "-", (3, 0): "-", (4, 0): "+"})
fw20c1 = BoseWord({(0, 0): "-", (1, 0): "+", (2, 1): "-"})
fw20c2 = BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "-"})
fw20c3 = BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-", (4, 0): "-"})
fw20cs = fw20c1 + fw20c2 - fw20c3


class TestBoseWord:
    def test_missing(self):
        """Test that empty string is returned for missing key."""
        fw = BoseWord({(0, 0): "+", (1, 1): "-"})
        assert (2, 3) not in fw.keys()
        assert fw[(2, 3)] == ""

    def test_set_items(self):
        """Test that setting items raises an error"""
        fw = BoseWord({(0, 0): "+", (1, 1): "-"})
        with pytest.raises(TypeError, match="BoseWord object does not support assignment"):
            fw[(2, 2)] = "+"

    def test_update_items(self):
        """Test that updating items raises an error"""
        fw = BoseWord({(0, 0): "+", (1, 1): "-"})
        with pytest.raises(TypeError, match="BoseWord object does not support assignment"):
            fw.update({(2, 2): "+"})

        with pytest.raises(TypeError, match="BoseWord object does not support assignment"):
            fw[(1, 1)] = "+"

    def test_hash(self):
        """Test that a unique hash exists for different BoseWords."""
        fw_1 = BoseWord({(0, 0): "+", (1, 1): "-"})
        fw_2 = BoseWord({(0, 0): "+", (1, 1): "-"})  # same as 1
        fw_3 = BoseWord({(1, 1): "-", (0, 0): "+"})  # same as 1 but reordered
        fw_4 = BoseWord({(0, 0): "+", (1, 2): "-"})  # distinct from above

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
        (fw1, "b\u207a(0) b(1)"),
        (fw2, "b\u207a(0) b(0)"),
        (fw3, "b\u207a(0) b(3) b\u207a(0) b(4)"),
        (fw4, "I"),
        (fw5, "b\u207a(10) b(30) b\u207a(0) b(400)"),
        (fw6, "b\u207a(10) b\u207a(30) b(0) b(400)"),
        (fw7, "b(10) b\u207a(30) b(0) b\u207a(400)"),
    )

    @pytest.mark.parametrize("fw, str_rep", tup_fw_compact)
    def test_compact(self, fw, str_rep):
        """Test string representation from to_string"""
        assert fw.to_string() == str_rep

    @pytest.mark.parametrize("fw, str_rep", tup_fw_compact)
    def test_str(self, fw, str_rep):
        """Test __str__ and __repr__ methods"""
        assert str(fw) == str_rep
        assert repr(fw) == f"BoseWord({fw.sorted_dic})"

    def test_pickling(self):
        """Check that BoseWords can be pickled and unpickled."""
        fw = BoseWord({(0, 0): "+", (1, 1): "-"})
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
            BoseWord(operator)

    tup_fw_shift = (
        (fw8, 0, 1, fw8cs),
        (fw9, 0, 1, fw9cs),
        (fw10, 0, 1, fw10cs),
        (fw11, 0, 1, fw11cs),
        (fw12, 0, 1, fw12cs),
        (fw13, 0, 1, fw13cs),
        (fw14, 0, 1, fw14cs),
        (fw15, 0, 1, fw15cs),
        (fw16, 1, 2, fw16cs),
        (fw17, 0, 1, fw17cs),
        (fw8, 0, 0, BoseSentence({fw8: 1})),
        (fw8, 1, 0, fw8cs),
        (fw11, 1, 0, fw11cs),
        (fw18, 3, 1, fw18cs),
        (fw19, 3, 0, fw19cs),
        (fw20, 4, 0, fw20cs),
    )

    @pytest.mark.parametrize("fw, i, j, fs", tup_fw_shift)
    def test_shift_operator(self, fw, i, j, fs):
        """Test that the shift_operator method correctly applies the anti-commutator relations."""
        assert fw.shift_operator(i, j) == fs

    def test_shift_operator_errors(self):
        """Test that the shift_operator method correctly raises exceptions."""
        with pytest.raises(TypeError, match="Positions must be integers."):
            fw8.shift_operator(0.5, 1)

        with pytest.raises(ValueError, match="Positions must be positive integers."):
            fw8.shift_operator(-1, 0)

        with pytest.raises(ValueError, match="Positions are out of range."):
            fw8.shift_operator(1, 2)

    tup_fw_dag = (
        (fw1, fw1_dag),
        (fw2, fw2_dag),
        (fw3, fw3_dag),
        (fw4, fw4_dag),
        (fw5, fw5_dag),
        (fw6, fw6_dag),
        (fw7, fw7_dag),
    )

    @pytest.mark.parametrize("fw, fw_dag", tup_fw_dag)
    def test_adjoint(self, fw, fw_dag):
        assert fw.adjoint() == fw_dag


class TestBoseWordArithmetic:
    WORDS_MUL = (
        (
            fw1,
            fw1,
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
        ),
        (
            fw1,
            fw1,
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
        ),
        (
            fw1,
            fw3,
            BoseWord(
                {(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 3): "-", (4, 0): "+", (5, 4): "-"}
            ),
            BoseWord(
                {(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-", (4, 0): "+", (5, 1): "-"}
            ),
        ),
        (
            fw2,
            fw1,
            BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-"}),
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 0): "-"}),
        ),
        (fw1, fw4, fw1, fw1),
        (fw4, fw3, fw3, fw3),
        (fw4, fw4, fw4, fw4),
    )

    @pytest.mark.parametrize("f1, f2, result_fw_right, result_fw_left", WORDS_MUL)
    def test_mul_bose_words(self, f1, f2, result_fw_right, result_fw_left):
        """Test that two BoseWords can be multiplied together and return a new
        BoseWord, with operators in the expected order"""
        assert f1 * f2 == result_fw_right
        assert f2 * f1 == result_fw_left

    WORDS_AND_SENTENCES_MUL = (
        (
            fw1,
            BoseSentence({fw3: 1.2}),
            BoseSentence({fw1 * fw3: 1.2}),
        ),
        (
            fw2,
            BoseSentence({fw3: 1.2, fw1: 3.7}),
            BoseSentence({fw2 * fw3: 1.2, fw2 * fw1: 3.7}),
        ),
    )

    @pytest.mark.parametrize("fw, fs, result", WORDS_AND_SENTENCES_MUL)
    def test_mul_bose_word_and_sentence(self, fw, fs, result):
        """Test that a BoseWord can be multiplied by a BoseSentence
        and return a new BoseSentence"""
        assert fw * fs == result

    WORDS_AND_NUMBERS_MUL = (
        (fw1, 2, BoseSentence({fw1: 2})),  # int
        (fw2, 3.7, BoseSentence({fw2: 3.7})),  # float
        (fw2, 2j, BoseSentence({fw2: 2j})),  # complex
        (fw2, np.array([2]), BoseSentence({fw2: 2})),  # numpy array
        (fw1, pnp.array([2]), BoseSentence({fw1: 2})),  # pennylane numpy array
        (fw1, pnp.array([2, 2])[0], BoseSentence({fw1: 2})),  # pennylane tensor with no length
    )

    @pytest.mark.parametrize("fw, number, result", WORDS_AND_NUMBERS_MUL)
    def test_mul_number(self, fw, number, result):
        """Test that a BoseWord can be multiplied onto a number (from the left)
        and return a BoseSentence"""
        assert fw * number == result

    @pytest.mark.parametrize("fw, number, result", WORDS_AND_NUMBERS_MUL)
    def test_rmul_number(self, fw, number, result):
        """Test that a BoseWord can be multiplied onto a number (from the right)
        and return a BoseSentence"""
        assert number * fw == result

    tup_fw_mult_error = ((fw4, "string"),)

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_mul_error(self, fw, bad_type):
        """Test multiply with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply BoseWord by {type(bad_type)}."):
            fw * bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_rmul_error(self, fw, bad_type):
        """Test __rmul__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply BoseWord by {type(bad_type)}."):
            bad_type * fw  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_add_error(self, fw, bad_type):
        """Test __add__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a BoseWord"):
            fw + bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_radd_error(self, fw, bad_type):
        """Test __radd__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a BoseWord"):
            bad_type + fw  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_sub_error(self, fw, bad_type):
        """Test __sub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot subtract {type(bad_type)} from a BoseWord"):
            fw - bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fw, bad_type", tup_fw_mult_error)
    def test_rsub_error(self, fw, bad_type):
        """Test __rsub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot subtract a BoseWord from {type(bad_type)}"):
            bad_type - fw  # pylint: disable=pointless-statement

    WORDS_ADD = [
        (fw1, fw2, BoseSentence({fw1: 1, fw2: 1})),
        (fw3, fw2, BoseSentence({fw2: 1, fw3: 1})),
        (fw2, fw2, BoseSentence({fw2: 2})),
    ]

    @pytest.mark.parametrize("f1, f2, res", WORDS_ADD)
    def test_add_bose_words(self, f1, f2, res):
        """Test that adding two BoseWords returns the expected BoseSentence"""
        assert f1 + f2 == res
        assert f2 + f1 == res

    WORDS_AND_SENTENCES_ADD = [
        (fw1, BoseSentence({fw1: 1.2, fw3: 3j}), BoseSentence({fw1: 2.2, fw3: 3j})),
        (fw3, BoseSentence({fw1: 1.2, fw3: 3j}), BoseSentence({fw1: 1.2, fw3: (1 + 3j)})),
        (fw1, BoseSentence({fw1: -1.2, fw3: 3j}), BoseSentence({fw1: -0.2, fw3: 3j})),
    ]

    @pytest.mark.parametrize("w, s, res", WORDS_AND_SENTENCES_ADD)
    def test_add_bose_words_and_sentences(self, w, s, res):
        """Test that adding a BoseSentence to a BoseWord returns the expected BoseSentence"""
        sum = w + s
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    WORDS_AND_CONSTANTS_ADD = [
        (fw1, 5, BoseSentence({fw1: 1, fw4: 5})),  # int
        (fw2, 2.8, BoseSentence({fw2: 1, fw4: 2.8})),  # float
        (fw3, (1 + 3j), BoseSentence({fw3: 1, fw4: (1 + 3j)})),  # complex
        (fw1, np.array([5]), BoseSentence({fw1: 1, fw4: 5})),  # numpy array
        (fw2, pnp.array([2.8]), BoseSentence({fw2: 1, fw4: 2.8})),  # pennylane numpy array
        (
            fw1,
            pnp.array([2, 2])[0],
            BoseSentence({fw1: 1, fw4: 2}),
        ),  # pennylane tensor with no length
        (fw4, 2, BoseSentence({fw4: 3})),  # BoseWord is Identity
    ]

    @pytest.mark.parametrize("w, c, res", WORDS_AND_CONSTANTS_ADD)
    def test_add_bose_words_and_constants(self, w, c, res):
        """Test that adding a constant (int, float or complex) to a BoseWord
        returns the expected BoseSentence"""
        sum = w + c
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    @pytest.mark.parametrize("w, c, res", WORDS_AND_CONSTANTS_ADD)
    def test_radd_bose_words_and_constants(self, w, c, res):
        """Test that adding a Bose word to a constant (int, float or complex)
        returns the expected BoseSentence (__radd__)"""
        sum = c + w
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    WORDS_SUB = [
        (fw1, fw2, BoseSentence({fw1: 1, fw2: -1}), BoseSentence({fw1: -1, fw2: 1})),
        (fw2, fw3, BoseSentence({fw2: 1, fw3: -1}), BoseSentence({fw2: -1, fw3: 1})),
        (fw2, fw2, BoseSentence({fw2: 0}), BoseSentence({fw2: 0})),
    ]

    @pytest.mark.parametrize("f1, f2, res1, res2", WORDS_SUB)
    def test_subtract_bose_words(self, f1, f2, res1, res2):
        """Test that subtracting one BoseWord from another returns the expected BoseSentence"""
        assert f1 - f2 == res1
        assert f2 - f1 == res2

    WORDS_AND_SENTENCES_SUB = [
        (fw1, BoseSentence({fw1: 1.2, fw3: 3j}), BoseSentence({fw1: -0.2, fw3: -3j})),
        (fw3, BoseSentence({fw1: 1.2, fw3: 3j}), BoseSentence({fw1: -1.2, fw3: (1 - 3j)})),
        (fw1, BoseSentence({fw1: -1.2, fw3: 3j}), BoseSentence({fw1: 2.2, fw3: -3j})),
    ]

    @pytest.mark.parametrize("w, s, res", WORDS_AND_SENTENCES_SUB)
    def test_subtract_bose_words_and_sentences(self, w, s, res):
        """Test that subtracting a BoseSentence from a BoseWord returns the expected BoseSentence"""
        diff = w - s
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        diff_rounded = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in diff.items()}
        )

        assert diff_rounded == res

    WORDS_AND_CONSTANTS_SUBTRACT = [
        (fw1, 5, BoseSentence({fw1: 1, fw4: -5})),  # int
        (fw2, 2.8, BoseSentence({fw2: 1, fw4: -2.8})),  # float
        (fw3, (1 + 3j), BoseSentence({fw3: 1, fw4: -(1 + 3j)})),  # complex
        (fw1, np.array([5]), BoseSentence({fw1: 1, fw4: -5})),  # numpy array
        (fw2, pnp.array([2.8]), BoseSentence({fw2: 1, fw4: -2.8})),  # pennylane numpy array
        (
            fw1,
            pnp.array([2, 2])[0],
            BoseSentence({fw1: 1, fw4: -2}),
        ),  # pennylane tensor with no length
        (fw4, 2, BoseSentence({fw4: -1})),  # BoseWord is Identity
    ]

    @pytest.mark.parametrize("w, c, res", WORDS_AND_CONSTANTS_SUBTRACT)
    def test_subtract_constant_from_bose_word(self, w, c, res):
        """Test that subtracting a constant (int, float or complex) from a BoseWord
        returns the expected BoseSentence"""
        diff = w - c
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        diff_rounded = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in diff.items()}
        )
        assert diff_rounded == res

    CONSTANTS_AND_WORDS_SUBTRACT = [
        (fw1, 5, BoseSentence({fw1: -1, fw4: 5})),  # int
        (fw2, 2.8, BoseSentence({fw2: -1, fw4: 2.8})),  # float
        (fw3, (1 + 3j), BoseSentence({fw3: -1, fw4: (1 + 3j)})),  # complex
        (fw1, np.array([5]), BoseSentence({fw1: -1, fw4: 5})),  # numpy array
        (fw2, pnp.array([2.8]), BoseSentence({fw2: -1, fw4: 2.8})),  # pennylane numpy array
        (
            fw1,
            pnp.array([2, 2])[0],
            BoseSentence({fw1: -1, fw4: 2}),
        ),  # pennylane tensor with no length
        (fw4, 2, BoseSentence({fw4: 1})),  # BoseWord is Identity
    ]

    @pytest.mark.parametrize("w, c, res", CONSTANTS_AND_WORDS_SUBTRACT)
    def test_subtract_bose_words_from_constant(self, w, c, res):
        """Test that subtracting a constant (int, float or complex) from a BoseWord
        returns the expected BoseSentence"""
        diff = c - w
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        diff_rounded = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in diff.items()}
        )
        assert diff_rounded == res

    tup_fw_pow = (
        (fw1, 0, BoseWord({})),
        (fw1, 1, BoseWord({(0, 0): "+", (1, 1): "-"})),
        (fw1, 2, BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"})),
        (
            fw2,
            3,
            BoseWord(
                {(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 0): "-", (4, 0): "+", (5, 0): "-"}
            ),
        ),
    )

    @pytest.mark.parametrize("f1, pow, result_fw", tup_fw_pow)
    def test_pow(self, f1, pow, result_fw):
        """Test that raising a BoseWord to an integer power returns the expected BoseWord"""
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
            ValueError, match="Arithmetic Bose operations can only accept an array of length 1"
        ):
            method_to_test = getattr(fw1, method_name)
            _ = method_to_test(np.array([1, 2]))


fs1 = BoseSentence({fw1: 1.23, fw2: 4j, fw3: -0.5})
fs1_dag = BoseSentence({fw1_dag: 1.23, fw2_dag: -4j, fw3_dag: -0.5})

fs2 = BoseSentence({fw1: -1.23, fw2: -4j, fw3: 0.5})
fs2_dag = BoseSentence({fw1_dag: -1.23, fw2_dag: 4j, fw3_dag: 0.5})

fs1_hamiltonian = BoseSentence({fw1: 1.23, fw2: 4, fw3: -0.5})
fs1_hamiltonian_dag = BoseSentence({fw1_dag: 1.23, fw2_dag: 4, fw3_dag: -0.5})

fs2_hamiltonian = BoseSentence({fw1: -1.23, fw2: -4, fw3: 0.5})
fs2_hamiltonian_dag = BoseSentence({fw1_dag: -1.23, fw2_dag: -4, fw3_dag: 0.5})

fs3 = BoseSentence({fw3: -0.5, fw4: 1})
fs3_dag = BoseSentence({fw3_dag: -0.5, fw4_dag: 1})

fs4 = BoseSentence({fw4: 1})
fs4_dag = BoseSentence({fw4_dag: 1})

fs5 = BoseSentence({})
fs5_dag = BoseSentence({})

fs6 = BoseSentence({fw1: 1.2, fw2: 3.1})
fs6_dag = BoseSentence({fw1_dag: 1.2, fw2_dag: 3.1})

fs7 = BoseSentence(
    {
        BoseWord({(0, 0): "+", (1, 1): "-"}): 1.23,  # b+(0) b(1)
        BoseWord({(0, 0): "+", (1, 0): "-"}): 4.0j,  # b+(0) b(0) = n(0) (number operator)
        BoseWord({(0, 0): "+", (1, 2): "-", (2, 1): "+"}): -0.5,  # b+(0) b(2) b+(1)
    }
)

fs1_x_fs2 = BoseSentence(  # fs1 * fs1, computed by hand
    {
        BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}): 1.5129,
        BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 0): "-"}): 4.92j,
        BoseWord(
            {
                (0, 0): "+",
                (1, 1): "-",
                (2, 0): "+",
                (3, 3): "-",
                (4, 0): "+",
                (5, 4): "-",
            }
        ): -0.615,
        BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-"}): 4.92j,
        BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 0): "-"}): -16,
        BoseWord(
            {
                (0, 0): "+",
                (1, 0): "-",
                (2, 0): "+",
                (3, 3): "-",
                (4, 0): "+",
                (5, 4): "-",
            }
        ): (-0 - 2j),
        BoseWord(
            {
                (0, 0): "+",
                (1, 3): "-",
                (2, 0): "+",
                (3, 4): "-",
                (4, 0): "+",
                (5, 1): "-",
            }
        ): -0.615,
        BoseWord(
            {
                (0, 0): "+",
                (1, 3): "-",
                (2, 0): "+",
                (3, 4): "-",
                (4, 0): "+",
                (5, 0): "-",
            }
        ): (-0 - 2j),
        BoseWord(
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

fs8 = fw8 + fw9
fs8c = fw8 + fw9cs

fs9 = 1.3 * fw8 + (1.4 + 3.8j) * fw9
fs9c = 1.3 * fw8 + (1.4 + 3.8j) * fw9cs

fs10 = -1.3 * fw11 + 2.3 * fw9
fs10c = -1.3 * fw11cs + 2.3 * fw9


class TestBoseSentence:
    def test_missing(self):
        """Test the result when a missing key is indexed."""
        fw = BoseWord({(0, 0): "+", (1, 1): "-"})
        new_fw = BoseWord({(0, 2): "+", (1, 3): "-"})
        fs = BoseSentence({fw: 1.0})

        assert new_fw not in fs.keys()
        assert fs[new_fw] == 0.0

    def test_set_items(self):
        """Test that we can add a new key to a BoseSentence."""
        fw = BoseWord({(0, 0): "+", (1, 1): "-"})
        fs = BoseSentence({fw: 1.0})

        new_fw = BoseWord({(0, 2): "+", (1, 3): "-"})
        assert new_fw not in fs.keys()

        fs[new_fw] = 3.45
        assert new_fw in fs.keys() and fs[new_fw] == 3.45

    tup_fs_str = (
        (
            fs1,
            "1.23 * b\u207a(0) b(1)\n"
            + "+ 4j * b\u207a(0) b(0)\n"
            + "+ -0.5 * b\u207a(0) b(3) a\u207a(0) b(4)",
        ),
        (
            fs2,
            "-1.23 * b\u207a(0) b(1)\n"
            + "+ (-0-4j) * b\u207a(0) b(0)\n"
            + "+ 0.5 * b\u207a(0) b(3) a\u207a(0) b(4)",
        ),
        (fs3, "-0.5 * b\u207a(0) b(3) b\u207a(0) b(4)\n" + "+ 1 * I"),
        (fs4, "1 * I"),
        (fs5, "0 * I"),
    )

    @pytest.mark.parametrize("fs, str_rep", tup_fs_str)
    def test_str(self, fs, str_rep):
        """Test the string representation of the BoseSentence."""
        print(str(fs))
        assert str(fs) == str_rep
        assert repr(fs) == f"BoseSentence({dict(fs)})"

    tup_fs_wires = (
        (fs1, {0, 1, 3, 4}),
        (fs2, {0, 1, 3, 4}),
        (fs3, {0, 3, 4}),
        (fs4, set()),
    )

    @pytest.mark.parametrize("fs, wires", tup_fs_wires)
    def test_wires(self, fs, wires):
        """Test the correct wires are given for the BoseSentence."""
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
        """Test that simplify removes terms in the BoseSentence with coefficient less than the
        threshold."""
        un_simplified_fs = BoseSentence({fw1: 0.001, fw2: 0.05, fw3: 1})

        expected_simplified_fs0 = BoseSentence({fw1: 0.001, fw2: 0.05, fw3: 1})
        expected_simplified_fs1 = BoseSentence({fw2: 0.05, fw3: 1})
        expected_simplified_fs2 = BoseSentence({fw3: 1})

        un_simplified_fs.simplify()
        assert un_simplified_fs == expected_simplified_fs0  # default tol = 1e-8
        un_simplified_fs.simplify(tol=1e-2)
        assert un_simplified_fs == expected_simplified_fs1
        un_simplified_fs.simplify(tol=1e-1)
        assert un_simplified_fs == expected_simplified_fs2

    def test_pickling(self):
        """Check that BoseSentences can be pickled and unpickled."""
        f1 = BoseWord({(0, 0): "+", (1, 1): "-"})
        f2 = BoseWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
        fs = BoseSentence({f1: 1.5, f2: -0.5})

        serialization = pickle.dumps(fs)
        new_fs = pickle.loads(serialization)
        assert fs == new_fs

    fs_dag_tup = (
        (fs1, fs1_dag),
        (fs2, fs2_dag),
        (fs3, fs3_dag),
        (fs4, fs4_dag),
        (fs5, fs5_dag),
        (fs6, fs6_dag),
        (fs1_hamiltonian, fs1_hamiltonian_dag),
        (fs2_hamiltonian, fs2_hamiltonian_dag),
    )

    @pytest.mark.parametrize("fs, fs_dag", fs_dag_tup)
    def test_adjoint(self, fs, fs_dag):
        assert fs.adjoint() == fs_dag


class TestBoseSentenceArithmetic:
    tup_fs_mult = (  # computed by hand
        (
            fs1,
            fs1,
            fs1_x_fs2,
        ),
        (
            fs3,
            fs4,
            BoseSentence(
                {
                    BoseWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"}): -0.5,
                    BoseWord({}): 1,
                }
            ),
        ),
        (
            fs4,
            fs4,
            BoseSentence(
                {
                    BoseWord({}): 1,
                }
            ),
        ),
        (fs5, fs3, fs5),
        (fs3, fs5, fs5),
        (fs4, fs3, fs3),
        (fs3, fs4, fs3),
        (
            BoseSentence({fw2: 1, fw3: 1, fw4: 1}),
            BoseSentence({fw4: 1, fw2: 1}),
            BoseSentence({fw2: 2, fw3: 1, fw4: 1, fw2 * fw2: 1, fw3 * fw2: 1}),
        ),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_fs_mult)
    def test_mul_bose_sentences(self, f1, f2, result):
        """Test that the correct result of multiplication between two
        BoseSentences is produced."""

        simplified_product = f1 * f2
        simplified_product.simplify()

        assert simplified_product == result

    SENTENCES_AND_WORDS_MUL = (
        (
            fw1,
            BoseSentence({fw3: 1.2}),
            BoseSentence({fw3 * fw1: 1.2}),
        ),
        (
            fw2,
            BoseSentence({fw3: 1.2, fw1: 3.7}),
            BoseSentence({fw3 * fw2: 1.2, fw1 * fw2: 3.7}),
        ),
    )

    @pytest.mark.parametrize("fw, fs, result", SENTENCES_AND_WORDS_MUL)
    def test_mul_bose_word_and_sentence(self, fw, fs, result):
        """Test that a BoseWord and a BoseSentence can be multiplied together
        and return a new BoseSentence"""
        assert fs * fw == result

    SENTENCES_AND_NUMBERS_MUL = (
        (fs1, 2, BoseSentence({fw1: 1.23 * 2, fw2: 4j * 2, fw3: -0.5 * 2})),  # int
        (fs2, 3.4, BoseSentence({fw1: -1.23 * 3.4, fw2: -4j * 3.4, fw3: 0.5 * 3.4})),  # float
        (fs1, 3j, BoseSentence({fw1: 3.69j, fw2: -12, fw3: -1.5j})),  # complex
        (fs5, 10, BoseSentence({})),  # null operator times constant
        (
            fs1,
            np.array([2]),
            BoseSentence({fw1: 1.23 * 2, fw2: 4j * 2, fw3: -0.5 * 2}),
        ),  # numpy array
        (
            fs1,
            pnp.array([2]),
            BoseSentence({fw1: 1.23 * 2, fw2: 4j * 2, fw3: -0.5 * 2}),
        ),  # pennylane numpy array
        (
            fs1,
            pnp.array([2, 2])[0],
            BoseSentence({fw1: 1.23 * 2, fw2: 4j * 2, fw3: -0.5 * 2}),
        ),  # pennylane tensor with no length
    )

    @pytest.mark.parametrize("fs, number, result", SENTENCES_AND_NUMBERS_MUL)
    def test_mul_number(self, fs, number, result):
        """Test that a BoseSentence can be multiplied onto a number (from the left)
        and return a BoseSentence"""
        assert fs * number == result

    @pytest.mark.parametrize("fs, number, result", SENTENCES_AND_NUMBERS_MUL)
    def test_rmul_number(self, fs, number, result):
        """Test that a BoseSentence can be multiplied onto a number (from the right)
        and return a BoseSentence"""
        assert number * fs == result

    tup_fs_add = (  # computed by hand
        (fs1, fs1, BoseSentence({fw1: 2.46, fw2: 8j, fw3: -1})),
        (fs1, fs2, BoseSentence({})),
        (fs1, fs3, BoseSentence({fw1: 1.23, fw2: 4j, fw3: -1, fw4: 1})),
        (fs2, fs5, fs2),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_fs_add)
    def test_add_bose_sentences(self, f1, f2, result):
        """Test that the correct result of addition is produced for two BoseSentences."""

        simplified_product = f1 + f2
        simplified_product.simplify()

        assert simplified_product == result

    SENTENCES_AND_WORDS_ADD = [
        (fw1, BoseSentence({fw1: 1.2, fw3: 3j}), BoseSentence({fw1: 2.2, fw3: 3j})),
        (fw3, BoseSentence({fw1: 1.2, fw3: 3j}), BoseSentence({fw1: 1.2, fw3: (1 + 3j)})),
        (fw1, BoseSentence({fw1: -1.2, fw3: 3j}), BoseSentence({fw1: -0.2, fw3: 3j})),
    ]

    @pytest.mark.parametrize("w, s, res", SENTENCES_AND_WORDS_ADD)
    def test_add_bose_words_and_sentences(self, w, s, res):
        """Test that adding a BoseWord to a BoseSentence returns the expected BoseSentence"""
        sum = s + w
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    SENTENCES_AND_CONSTANTS_ADD = [
        (BoseSentence({fw1: 1.2, fw3: 3j}), 3, BoseSentence({fw1: 1.2, fw3: 3j, fw4: 3})),  # int
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            1.3,
            BoseSentence({fw1: 1.2, fw3: 3j, fw4: 1.3}),
        ),  # float
        (
            BoseSentence({fw1: -1.2, fw3: 3j}),  # complex
            (1 + 2j),
            BoseSentence({fw1: -1.2, fw3: 3j, fw4: (1 + 2j)}),
        ),
        (BoseSentence({}), 5, BoseSentence({fw4: 5})),  # null sentence
        (BoseSentence({fw4: 3}), 1j, BoseSentence({fw4: 3 + 1j})),  # identity only
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            np.array([3]),
            BoseSentence({fw1: 1.2, fw3: 3j, fw4: 3}),
        ),  # numpy array
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3]),
            BoseSentence({fw1: 1.2, fw3: 3j, fw4: 3}),
        ),  # pennylane numpy array
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3, 0])[0],
            BoseSentence({fw1: 1.2, fw3: 3j, fw4: 3}),
        ),  # pennylane tensor with no length
    ]

    @pytest.mark.parametrize("s, c, res", SENTENCES_AND_CONSTANTS_ADD)
    def test_add_bose_sentences_and_constants(self, s, c, res):
        """Test that adding a constant to a BoseSentence returns the expected BoseSentence"""
        sum = s + c
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    @pytest.mark.parametrize("s, c, res", SENTENCES_AND_CONSTANTS_ADD)
    def test_radd_bose_sentences_and_constants(self, s, c, res):
        """Test that adding a BoseSentence to a constant (__radd___) returns the expected BoseSentence"""
        sum = c + s
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        sum_rounded = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in sum.items()}
        )
        assert sum_rounded == res

    SENTENCE_MINUS_WORD = (  # computed by hand
        (fs1, fw1, BoseSentence({fw1: 0.23, fw2: 4j, fw3: -0.5})),
        (fs2, fw3, BoseSentence({fw1: -1.23, fw2: -4j, fw3: -0.5})),
        (fs3, fw4, BoseSentence({fw3: -0.5})),
        (BoseSentence({fw1: 1.2, fw3: 3j}), fw1, BoseSentence({fw1: 0.2, fw3: 3j})),
        (BoseSentence({fw1: 1.2, fw3: 3j}), fw3, BoseSentence({fw1: 1.2, fw3: (-1 + 3j)})),
        (BoseSentence({fw1: -1.2, fw3: 3j}), fw1, BoseSentence({fw1: -2.2, fw3: 3j})),
    )

    @pytest.mark.parametrize("fs, fw, result", SENTENCE_MINUS_WORD)
    def test_subtract_bose_word_from_bose_sentence(self, fs, fw, result):
        """Test that the correct result is produced if a BoseWord is
        subtracted from a BoseSentence"""

        simplified_diff = fs - fw
        simplified_diff.simplify()
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        simplified_diff = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in simplified_diff.items()}
        )

        assert simplified_diff == result

    SENTENCE_MINUS_CONSTANT = (  # computed by hand
        (fs1, 3, BoseSentence({fw1: 1.23, fw2: 4j, fw3: -0.5, fw4: -3})),  # int
        (fs2, -2.7, BoseSentence({fw1: -1.23, fw2: -4j, fw3: 0.5, fw4: 2.7})),  # float
        (fs3, 2j, BoseSentence({fw3: -0.5, fw4: (1 - 2j)})),  # complex
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            -4,
            BoseSentence({fw1: 1.2, fw3: 3j, fw4: 4}),
        ),  # negative int
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            np.array([3]),
            BoseSentence({fw1: 1.2, fw3: 3j, fw4: -3}),
        ),  # numpy array
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3]),
            BoseSentence({fw1: 1.2, fw3: 3j, fw4: -3}),
        ),  # pennylane numpy array
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3, 2])[0],
            BoseSentence({fw1: 1.2, fw3: 3j, fw4: -3}),
        ),  # pennylane tensor with no len
    )

    @pytest.mark.parametrize("fs, c, result", SENTENCE_MINUS_CONSTANT)
    def test_subtract_constant_from_bose_sentence(self, fs, c, result):
        """Test that the correct result is produced if a BoseWord is
        subtracted from a BoseSentence"""

        simplified_diff = fs - c
        simplified_diff.simplify()
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        simplified_diff = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in simplified_diff.items()}
        )
        assert simplified_diff == result

    CONSTANT_MINUS_SENTENCE = (  # computed by hand
        (fs1, 3, BoseSentence({fw1: -1.23, fw2: -4j, fw3: 0.5, fw4: 3})),
        (fs2, -2.7, BoseSentence({fw1: 1.23, fw2: 4j, fw3: -0.5, fw4: -2.7})),
        (fs3, 2j, BoseSentence({fw3: 0.5, fw4: (-1 + 2j)})),
        (BoseSentence({fw1: 1.2, fw3: 3j}), -4, BoseSentence({fw1: -1.2, fw3: -3j, fw4: -4})),
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            np.array([3]),
            BoseSentence({fw1: -1.2, fw3: -3j, fw4: 3}),
        ),  # numpy array
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3]),
            BoseSentence({fw1: -1.2, fw3: -3j, fw4: 3}),
        ),  # pennylane numpy array
        (
            BoseSentence({fw1: 1.2, fw3: 3j}),
            pnp.array([3, 3])[0],
            BoseSentence({fw1: -1.2, fw3: -3j, fw4: 3}),
        ),  # pennylane tensor with to len
    )

    @pytest.mark.parametrize("fs, c, result", CONSTANT_MINUS_SENTENCE)
    def test_subtract_bose_sentence_from_constant(self, fs, c, result):
        """Test that the correct result is produced if a BoseWord is
        subtracted from a BoseSentence"""

        simplified_diff = c - fs
        simplified_diff.simplify()
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        simplified_diff = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in simplified_diff.items()}
        )
        assert simplified_diff == result

    tup_fs_subtract = (  # computed by hand
        (fs1, fs1, BoseSentence({})),
        (fs1, fs2, BoseSentence({fw1: 2.46, fw2: 8j, fw3: -1})),
        (fs1, fs3, BoseSentence({fw1: 1.23, fw2: 4j, fw4: -1})),
        (fs2, fs5, fs2),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_fs_subtract)
    def test_subtract_bose_sentences(self, f1, f2, result):
        """Test that the correct result of subtraction is produced for two BoseSentences."""

        simplified_product = f1 - f2
        simplified_product.simplify()

        assert simplified_product == result

    tup_fs_pow = (
        (fs1, 0, BoseSentence({BoseWord({}): 1})),
        (fs1, 1, fs1),
        (fs1, 2, fs1_x_fs2),
        (fs3, 0, BoseSentence({BoseWord({}): 1})),
        (fs3, 1, fs3),
        (fs4, 0, BoseSentence({BoseWord({}): 1})),
        (fs4, 3, fs4),
    )

    @pytest.mark.parametrize("f1, pow, result", tup_fs_pow)
    def test_pow(self, f1, pow, result):
        """Test that raising a BoseWord to an integer power returns the expected BoseWord"""
        assert f1**pow == result

    tup_fs_pow_error = ((fs1, -1), (fs3, 1.5))

    @pytest.mark.parametrize("f1, pow", tup_fs_pow_error)
    def test_pow_error(self, f1, pow):
        """Test that invalid values for the exponent raises an error"""
        with pytest.raises(ValueError, match="The exponent must be a positive integer."):
            f1**pow  # pylint: disable=pointless-statement

    TYPE_ERRORS = ((fs4, "string"),)

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_add_error(self, fs, bad_type):
        """Test __add__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a BoseSentence."):
            fs + bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_radd_error(self, fs, bad_type):
        """Test __radd__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a BoseSentence."):
            bad_type + fs  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_sub_error(self, fs, bad_type):
        """Test __sub__ with unsupported type raises an error"""
        with pytest.raises(
            TypeError, match=f"Cannot subtract {type(bad_type)} from a BoseSentence."
        ):
            fs - bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_rsub_error(self, fs, bad_type):
        """Test __rsub__ with unsupported type raises an error"""
        with pytest.raises(
            TypeError, match=f"Cannot subtract a BoseSentence from {type(bad_type)}."
        ):
            bad_type - fs  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_mul_error(self, fs, bad_type):
        """Test __sub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply BoseSentence by {type(bad_type)}."):
            fs * bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("fs, bad_type", TYPE_ERRORS)
    def test_rmul_error(self, fs, bad_type):
        """Test __rsub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply {type(bad_type)} by BoseSentence."):
            bad_type * fs  # pylint: disable=pointless-statement

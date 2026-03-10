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
"""Unit Tests for the Bosonic representation classes."""
import pickle
from copy import copy, deepcopy

import numpy as np
import pytest

from pennylane import numpy as pnp
from pennylane.bose import BoseSentence, BoseWord

bw1 = BoseWord({(0, 0): "+", (1, 1): "-"})
bw1_dag = BoseWord({(0, 1): "+", (1, 0): "-"})

bw2 = BoseWord({(0, 0): "+", (1, 0): "-"})
bw2_dag = BoseWord({(0, 0): "+", (1, 0): "-"})

bw3 = BoseWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
bw3_dag = BoseWord({(0, 4): "+", (1, 0): "-", (2, 3): "+", (3, 0): "-"})

bw4 = BoseWord({})
bw4_dag = BoseWord({})

bw5 = BoseWord({(0, 10): "+", (1, 30): "-", (2, 0): "+", (3, 400): "-"})
bw5_dag = BoseWord({(0, 400): "+", (1, 0): "-", (2, 30): "+", (3, 10): "-"})

bw6 = BoseWord({(0, 10): "+", (1, 30): "+", (2, 0): "-", (3, 400): "-"})
bw6_dag = BoseWord({(0, 400): "+", (1, 0): "+", (2, 30): "-", (3, 10): "-"})

bw7 = BoseWord({(0, 10): "-", (1, 30): "+", (2, 0): "-", (3, 400): "+"})
bw7_dag = BoseWord({(0, 400): "-", (1, 0): "+", (2, 30): "-", (3, 10): "+"})

bw8 = BoseWord({(0, 0): "-", (1, 1): "+"})
bw8c = BoseWord({(0, 1): "+", (1, 0): "-"})
bw8cs = BoseSentence({bw8c: -1})

bw9 = BoseWord({(0, 0): "-", (1, 1): "-"})
bw9c = BoseWord({(0, 1): "-", (1, 0): "-"})
bw9cs = BoseSentence({bw9c: -1})

bw10 = BoseWord({(0, 0): "+", (1, 1): "+"})
bw10c = BoseWord({(0, 1): "+", (1, 0): "+"})
bw10cs = BoseSentence({bw10c: -1})

bw11 = BoseWord({(0, 0): "-", (1, 0): "+"})
bw11c = BoseWord({(0, 0): "+", (1, 0): "-"})
bw11cs = 1 + BoseSentence({bw11c: -1})

bw12 = BoseWord({(0, 0): "+", (1, 0): "+"})
bw12c = BoseWord({(0, 0): "+", (1, 0): "+"})
bw12cs = BoseSentence({bw12c: 1})

bw13 = BoseWord({(0, 0): "-", (1, 0): "-"})
bw13c = BoseWord({(0, 0): "-", (1, 0): "-"})
bw13cs = BoseSentence({bw13c: 1})

bw14 = BoseWord({(0, 0): "+", (1, 0): "-"})
bw14c = BoseWord({(0, 0): "-", (1, 0): "+"})
bw14cs = 1 + BoseSentence({bw14c: -1})

bw15 = BoseWord({(0, 0): "-", (1, 1): "+", (2, 2): "+"})
bw15c = BoseWord({(0, 1): "+", (1, 0): "-", (2, 2): "+"})
bw15cs = BoseSentence({bw15c: -1})

bw16 = BoseWord({(0, 0): "-", (1, 1): "+", (2, 2): "-"})
bw16c = BoseWord({(0, 0): "-", (1, 2): "-", (2, 1): "+"})
bw16cs = BoseSentence({bw16c: -1})

bw17 = BoseWord({(0, 0): "-", (1, 0): "+", (2, 2): "-"})
bw17c1 = BoseWord({(0, 2): "-"})
bw17c2 = BoseWord({(0, 0): "+", (1, 0): "-", (2, 2): "-"})
bw17cs = bw17c1 - bw17c2

bw18 = BoseWord({(0, 0): "+", (1, 1): "+", (2, 2): "-", (3, 3): "-"})
bw18c = BoseWord({(0, 0): "+", (1, 3): "-", (2, 1): "+", (3, 2): "-"})
bw18cs = BoseSentence({bw18c: 1})

bw19 = BoseWord({(0, 0): "+", (1, 1): "+", (2, 2): "-", (3, 2): "+"})
bw19c1 = BoseWord({(0, 0): "+", (1, 1): "+"})
bw19c2 = BoseWord({(0, 2): "+", (1, 0): "+", (2, 1): "+", (3, 2): "-"})
bw19cs = BoseSentence({bw19c1: 1, bw19c2: -1})

bw20 = BoseWord({(0, 0): "-", (1, 0): "+", (2, 1): "-", (3, 0): "-", (4, 0): "+"})
bw20c1 = BoseWord({(0, 0): "-", (1, 0): "+", (2, 1): "-"})
bw20c2 = BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "-"})
bw20c3 = BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-", (4, 0): "-"})
bw20cs = bw20c1 + bw20c2 - bw20c3

bw21 = BoseWord({(0, 0): "-", (1, 1): "-", (2, 0): "+", (3, 1): "+", (4, 0): "+", (5, 2): "+"})
bw22 = BoseWord({(0, 0): "-", (1, 0): "+"})


# pylint: disable=too-many-public-methods
class TestBoseWord:
    """
    Tests for BoseWord
    """

    # Expected bose sentences were computed manually or with openfermion
    @pytest.mark.parametrize(
        ("bose_sentence", "expected"),
        [
            (
                BoseSentence(
                    {
                        BoseWord({(0, 0): "+", (1, 0): "+"}): 5.051e-06,
                        BoseWord({(0, 0): "+", (1, 0): "-"}): 5.051e-06,
                        BoseWord({(0, 0): "-", (1, 0): "+"}): 5.051e-06,
                        BoseWord({(0, 0): "-", (1, 0): "-"}): 5.051e-06,
                    }
                ),
                BoseSentence(
                    {
                        BoseWord({(0, 0): "+", (1, 0): "+"}): 5.051e-06,
                        BoseWord({(0, 0): "+", (1, 0): "-"}): 1.0102e-05,
                        BoseWord({}): 5.051e-06,
                        BoseWord({(0, 0): "-", (1, 0): "-"}): 5.051e-06,
                    }
                ),
            ),
            (
                bw21,
                BoseSentence(
                    {
                        BoseWord({(0, 0): "+", (1, 2): "+"}): 2.0,
                        BoseWord({(0, 0): "+", (1, 1): "+", (2, 2): "+", (3, 1): "-"}): 2.0,
                        BoseWord({(0, 0): "+", (1, 0): "+", (2, 2): "+", (3, 0): "-"}): 1.0,
                        BoseWord(
                            {
                                (0, 0): "+",
                                (1, 0): "+",
                                (2, 1): "+",
                                (3, 2): "+",
                                (4, 0): "-",
                                (5, 1): "-",
                            }
                        ): 1.0,
                    }
                ),
            ),
        ],
    )
    def test_normal_order(self, bose_sentence, expected):
        """Test that normal_order correctly normal orders the BoseWord"""
        assert bose_sentence.normal_order() == expected

    @pytest.mark.parametrize(
        ("bw", "i", "j", "bs"),
        [
            (
                bw22,
                0,
                1,
                BoseSentence({BoseWord({(0, 0): "+", (1, 0): "-"}): 1.0, BoseWord({}): 1.0}),
            ),
            (
                bw22,
                0,
                0,
                BoseSentence({bw22: 1}),
            ),
            (
                BoseWord({(0, 0): "-", (1, 0): "-", (2, 0): "+", (3, 0): "+"}),
                3,
                0,
                BoseSentence(
                    {
                        BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "-", (3, 0): "+"}): 1.0,
                        BoseWord({(0, 0): "-", (1, 0): "+"}): 2.0,
                    }
                ),
            ),
            (
                BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "-", (3, 0): "+"}),
                3,
                1,
                BoseSentence(
                    {
                        BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "-", (3, 0): "-"}): 1.0,
                        BoseWord({(0, 0): "+", (1, 0): "-"}): 2.0,
                    }
                ),
            ),
            (
                BoseWord({(0, 0): "-", (1, 1): "+"}),
                0,
                1,
                BoseSentence({BoseWord({(0, 1): "+", (1, 0): "-"}): 1}),
            ),
            (
                BoseWord({(0, 0): "-", (1, 0): "+", (2, 0): "+", (3, 0): "-"}),
                0,
                2,
                BoseSentence(
                    {
                        BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "-", (3, 0): "-"}): 1.0,
                        BoseWord({(0, 0): "+", (1, 0): "-"}): 2.0,
                    }
                ),
            ),
        ],
    )
    def test_shift_operator(self, bw, i, j, bs):
        """Test that the shift_operator method correctly applies the commutator relations."""
        assert bw.shift_operator(i, j) == bs

    def test_shift_operator_errors(self):
        """Test that the shift_operator method correctly raises exceptions."""
        with pytest.raises(TypeError, match="Positions must be integers."):
            bw1.shift_operator(0.5, 1)

        with pytest.raises(ValueError, match="Positions must be positive integers."):
            bw1.shift_operator(-1, 0)

        with pytest.raises(ValueError, match="Positions are out of range."):
            bw1.shift_operator(1, 6)

    def test_missing(self):
        """Test that empty string is returned for missing key."""
        bw = BoseWord({(0, 0): "+", (1, 1): "-"})
        assert (2, 3) not in bw.keys()
        assert bw[(2, 3)] == ""

    def test_set_items(self):
        """Test that setting items raises an error"""
        bw = BoseWord({(0, 0): "+", (1, 1): "-"})
        with pytest.raises(TypeError, match="BoseWord object does not support assignment"):
            bw[(2, 2)] = "+"

    def test_update_items(self):
        """Test that updating items raises an error"""
        bw = BoseWord({(0, 0): "+", (1, 1): "-"})
        with pytest.raises(TypeError, match="BoseWord object does not support assignment"):
            bw.update({(2, 2): "+"})

        with pytest.raises(TypeError, match="BoseWord object does not support assignment"):
            bw[(1, 1)] = "+"

    def test_hash(self):
        """Test that a unique hash exists for different BoseWords."""
        bw_1 = BoseWord({(0, 0): "+", (1, 1): "-"})
        bw_2 = BoseWord({(0, 0): "+", (1, 1): "-"})  # same as 1
        bw_3 = BoseWord({(1, 1): "-", (0, 0): "+"})  # same as 1 but reordered
        bw_4 = BoseWord({(0, 0): "+", (1, 2): "-"})  # distinct from above

        assert hash(bw_1) == hash(bw_2)
        assert hash(bw_1) == hash(bw_3)
        assert hash(bw_1) != hash(bw_4)

    @pytest.mark.parametrize("bw", (bw1, bw2, bw3, bw4))
    def test_copy(self, bw):
        """Test that the copy is identical to the original."""
        copy_bw = copy(bw)
        deep_copy_bw = deepcopy(bw)

        assert copy_bw == bw
        assert deep_copy_bw == bw
        assert copy_bw is not bw
        assert deep_copy_bw is not bw

    tup_bws_wires = ((bw1, {0, 1}), (bw2, {0}), (bw3, {0, 3, 4}), (bw4, set()))

    @pytest.mark.parametrize("bw, wires", tup_bws_wires)
    def test_wires(self, bw, wires):
        """Test that the wires are tracked correctly."""
        assert bw.wires == wires

    tup_bw_compact = (
        (bw1, "b\u207a(0) b(1)"),
        (bw2, "b\u207a(0) b(0)"),
        (bw3, "b\u207a(0) b(3) b\u207a(0) b(4)"),
        (bw4, "I"),
        (bw5, "b\u207a(10) b(30) b\u207a(0) b(400)"),
        (bw6, "b\u207a(10) b\u207a(30) b(0) b(400)"),
        (bw7, "b(10) b\u207a(30) b(0) b\u207a(400)"),
    )

    @pytest.mark.parametrize("bw, str_rep", tup_bw_compact)
    def test_compact(self, bw, str_rep):
        """Test string representation from to_string"""
        assert bw.to_string() == str_rep

    @pytest.mark.parametrize("bw, str_rep", tup_bw_compact)
    def test_str(self, bw, str_rep):
        """Test __str__ and __repr__ methods"""
        assert str(bw) == str_rep
        assert repr(bw) == f"BoseWord({bw.sorted_dic})"

    def test_pickling(self):
        """Check that BoseWords can be pickled and unpickled."""
        bw = BoseWord({(0, 0): "+", (1, 1): "-"})
        serialization = pickle.dumps(bw)
        new_bw = pickle.loads(serialization)
        assert bw == new_bw

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

    tup_bw_dag = (
        (bw1, bw1_dag),
        (bw2, bw2_dag),
        (bw3, bw3_dag),
        (bw4, bw4_dag),
        (bw5, bw5_dag),
        (bw6, bw6_dag),
        (bw7, bw7_dag),
    )

    @pytest.mark.parametrize("bw, bw_dag", tup_bw_dag)
    def test_adjoint(self, bw, bw_dag):
        assert bw.adjoint() == bw_dag


class TestBoseWordArithmetic:
    WORDS_MUL = (
        (
            bw1,
            bw1,
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
        ),
        (
            bw1,
            bw1,
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"}),
        ),
        (
            bw1,
            bw3,
            BoseWord(
                {(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 3): "-", (4, 0): "+", (5, 4): "-"}
            ),
            BoseWord(
                {(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-", (4, 0): "+", (5, 1): "-"}
            ),
        ),
        (
            bw2,
            bw1,
            BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-"}),
            BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 0): "-"}),
        ),
        (bw1, bw4, bw1, bw1),
        (bw4, bw3, bw3, bw3),
        (bw4, bw4, bw4, bw4),
    )

    @pytest.mark.parametrize("f1, f2, result_bw_right, result_bw_left", WORDS_MUL)
    def test_mul_bose_words(self, f1, f2, result_bw_right, result_bw_left):
        """Test that two BoseWords can be multiplied together and return a new
        BoseWord, with operators in the expected order"""
        assert f1 * f2 == result_bw_right
        assert f2 * f1 == result_bw_left

    WORDS_AND_SENTENCES_MUL = (
        (
            bw1,
            BoseSentence({bw3: 1.2}),
            BoseSentence({bw1 * bw3: 1.2}),
        ),
        (
            bw2,
            BoseSentence({bw3: 1.2, bw1: 3.7}),
            BoseSentence({bw2 * bw3: 1.2, bw2 * bw1: 3.7}),
        ),
    )

    @pytest.mark.parametrize("bw, bs, result", WORDS_AND_SENTENCES_MUL)
    def test_mul_bose_word_and_sentence(self, bw, bs, result):
        """Test that a BoseWord can be multiplied by a BoseSentence
        and return a new BoseSentence"""
        assert bw * bs == result

    WORDS_AND_NUMBERS_MUL = (
        (bw1, 2, BoseSentence({bw1: 2})),  # int
        (bw2, 3.7, BoseSentence({bw2: 3.7})),  # float
        (bw2, 2j, BoseSentence({bw2: 2j})),  # complex
        (bw2, np.array([2]), BoseSentence({bw2: 2})),  # numpy array
        (bw1, pnp.array([2]), BoseSentence({bw1: 2})),  # pennylane numpy array
        (bw1, pnp.array([2, 2])[0], BoseSentence({bw1: 2})),  # pennylane tensor with no length
    )

    @pytest.mark.parametrize("bw, number, result", WORDS_AND_NUMBERS_MUL)
    def test_mul_number(self, bw, number, result):
        """Test that a BoseWord can be multiplied onto a number (from the left)
        and return a BoseSentence"""
        assert bw * number == result

    @pytest.mark.parametrize("bw, number, result", WORDS_AND_NUMBERS_MUL)
    def test_rmul_number(self, bw, number, result):
        """Test that a BoseWord can be multiplied onto a number (from the right)
        and return a BoseSentence"""
        assert number * bw == result

    tup_bw_mult_error = ((bw4, "string"),)

    @pytest.mark.parametrize("bw, bad_type", tup_bw_mult_error)
    def test_mul_error(self, bw, bad_type):
        """Test multiply with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply BoseWord by {type(bad_type)}."):
            bw * bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bw, bad_type", tup_bw_mult_error)
    def test_rmul_error(self, bw, bad_type):
        """Test __rmul__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply BoseWord by {type(bad_type)}."):
            bad_type * bw  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bw, bad_type", tup_bw_mult_error)
    def test_add_error(self, bw, bad_type):
        """Test __add__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a BoseWord"):
            bw + bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bw, bad_type", tup_bw_mult_error)
    def test_radd_error(self, bw, bad_type):
        """Test __radd__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a BoseWord"):
            bad_type + bw  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bw, bad_type", tup_bw_mult_error)
    def test_sub_error(self, bw, bad_type):
        """Test __sub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot subtract {type(bad_type)} from a BoseWord"):
            bw - bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bw, bad_type", tup_bw_mult_error)
    def test_rsub_error(self, bw, bad_type):
        """Test __rsub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot subtract a BoseWord from {type(bad_type)}"):
            bad_type - bw  # pylint: disable=pointless-statement

    WORDS_ADD = [
        (bw1, bw2, BoseSentence({bw1: 1, bw2: 1})),
        (bw3, bw2, BoseSentence({bw2: 1, bw3: 1})),
        (bw2, bw2, BoseSentence({bw2: 2})),
    ]

    @pytest.mark.parametrize("f1, f2, res", WORDS_ADD)
    def test_add_bose_words(self, f1, f2, res):
        """Test that adding two BoseWords returns the expected BoseSentence"""
        assert f1 + f2 == res
        assert f2 + f1 == res

    WORDS_AND_SENTENCES_ADD = [
        (bw1, BoseSentence({bw1: 1.2, bw3: 3j}), BoseSentence({bw1: 2.2, bw3: 3j})),
        (bw3, BoseSentence({bw1: 1.2, bw3: 3j}), BoseSentence({bw1: 1.2, bw3: (1 + 3j)})),
        (bw1, BoseSentence({bw1: -1.2, bw3: 3j}), BoseSentence({bw1: -0.2, bw3: 3j})),
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
        (bw1, 5, BoseSentence({bw1: 1, bw4: 5})),  # int
        (bw2, 2.8, BoseSentence({bw2: 1, bw4: 2.8})),  # float
        (bw3, (1 + 3j), BoseSentence({bw3: 1, bw4: (1 + 3j)})),  # complex
        (bw1, np.array([5]), BoseSentence({bw1: 1, bw4: 5})),  # numpy array
        (bw2, pnp.array([2.8]), BoseSentence({bw2: 1, bw4: 2.8})),  # pennylane numpy array
        (
            bw1,
            pnp.array([2, 2])[0],
            BoseSentence({bw1: 1, bw4: 2}),
        ),  # pennylane tensor with no length
        (bw4, 2, BoseSentence({bw4: 3})),  # BoseWord is Identity
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
        (bw1, bw2, BoseSentence({bw1: 1, bw2: -1}), BoseSentence({bw1: -1, bw2: 1})),
        (bw2, bw3, BoseSentence({bw2: 1, bw3: -1}), BoseSentence({bw2: -1, bw3: 1})),
        (bw2, bw2, BoseSentence({bw2: 0}), BoseSentence({bw2: 0})),
    ]

    @pytest.mark.parametrize("f1, f2, res1, res2", WORDS_SUB)
    def test_subtract_bose_words(self, f1, f2, res1, res2):
        """Test that subtracting one BoseWord from another returns the expected BoseSentence"""
        assert f1 - f2 == res1
        assert f2 - f1 == res2

    WORDS_AND_SENTENCES_SUB = [
        (bw1, BoseSentence({bw1: 1.2, bw3: 3j}), BoseSentence({bw1: -0.2, bw3: -3j})),
        (bw3, BoseSentence({bw1: 1.2, bw3: 3j}), BoseSentence({bw1: -1.2, bw3: (1 - 3j)})),
        (bw1, BoseSentence({bw1: -1.2, bw3: 3j}), BoseSentence({bw1: 2.2, bw3: -3j})),
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
        (bw1, 5, BoseSentence({bw1: 1, bw4: -5})),  # int
        (bw2, 2.8, BoseSentence({bw2: 1, bw4: -2.8})),  # float
        (bw3, (1 + 3j), BoseSentence({bw3: 1, bw4: -(1 + 3j)})),  # complex
        (bw1, np.array([5]), BoseSentence({bw1: 1, bw4: -5})),  # numpy array
        (bw2, pnp.array([2.8]), BoseSentence({bw2: 1, bw4: -2.8})),  # pennylane numpy array
        (
            bw1,
            pnp.array([2, 2])[0],
            BoseSentence({bw1: 1, bw4: -2}),
        ),  # pennylane tensor with no length
        (bw4, 2, BoseSentence({bw4: -1})),  # BoseWord is Identity
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
        (bw1, 5, BoseSentence({bw1: -1, bw4: 5})),  # int
        (bw2, 2.8, BoseSentence({bw2: -1, bw4: 2.8})),  # float
        (bw3, (1 + 3j), BoseSentence({bw3: -1, bw4: (1 + 3j)})),  # complex
        (bw1, np.array([5]), BoseSentence({bw1: -1, bw4: 5})),  # numpy array
        (bw2, pnp.array([2.8]), BoseSentence({bw2: -1, bw4: 2.8})),  # pennylane numpy array
        (
            bw1,
            pnp.array([2, 2])[0],
            BoseSentence({bw1: -1, bw4: 2}),
        ),  # pennylane tensor with no length
        (bw4, 2, BoseSentence({bw4: 1})),  # BoseWord is Identity
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

    tup_bw_pow = (
        (bw1, 0, BoseWord({})),
        (bw1, 1, BoseWord({(0, 0): "+", (1, 1): "-"})),
        (bw1, 2, BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"})),
        (
            bw2,
            3,
            BoseWord(
                {(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 0): "-", (4, 0): "+", (5, 0): "-"}
            ),
        ),
    )

    @pytest.mark.parametrize("f1, pow, result_bw", tup_bw_pow)
    def test_pow(self, f1, pow, result_bw):
        """Test that raising a BoseWord to an integer power returns the expected BoseWord"""
        assert f1**pow == result_bw

    tup_bw_pow_error = ((bw1, -1), (bw3, 1.5))

    @pytest.mark.parametrize("f1, pow", tup_bw_pow_error)
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
            method_to_test = getattr(bw1, method_name)
            _ = method_to_test(np.array([1, 2]))


bs1 = BoseSentence({bw1: 1.23, bw2: 4j, bw3: -0.5})
bs1_dag = BoseSentence({bw1_dag: 1.23, bw2_dag: -4j, bw3_dag: -0.5})

bs2 = BoseSentence({bw1: -1.23, bw2: -4j, bw3: 0.5})
bs2_dag = BoseSentence({bw1_dag: -1.23, bw2_dag: 4j, bw3_dag: 0.5})

bs1_hamiltonian = BoseSentence({bw1: 1.23, bw2: 4, bw3: -0.5})
bs1_hamiltonian_dag = BoseSentence({bw1_dag: 1.23, bw2_dag: 4, bw3_dag: -0.5})

bs2_hamiltonian = BoseSentence({bw1: -1.23, bw2: -4, bw3: 0.5})
bs2_hamiltonian_dag = BoseSentence({bw1_dag: -1.23, bw2_dag: -4, bw3_dag: 0.5})

bs3 = BoseSentence({bw3: -0.5, bw4: 1})
bs3_dag = BoseSentence({bw3_dag: -0.5, bw4_dag: 1})

bs4 = BoseSentence({bw4: 1})
bs4_dag = BoseSentence({bw4_dag: 1})

bs5 = BoseSentence({})
bs5_dag = BoseSentence({})

bs6 = BoseSentence({bw1: 1.2, bw2: 3.1})
bs6_dag = BoseSentence({bw1_dag: 1.2, bw2_dag: 3.1})

bs7 = BoseSentence(
    {
        BoseWord({(0, 0): "+", (1, 1): "-"}): 1.23,  # b+(0) b(1)
        BoseWord({(0, 0): "+", (1, 0): "-"}): 4.0j,  # b+(0) b(0) = n(0) (number operator)
        BoseWord({(0, 0): "+", (1, 2): "-", (2, 1): "+"}): -0.5,  # b+(0) b(2) b+(1)
    }
)

bs1_x_bs2 = BoseSentence(  # bs1 * bs1, computed by hand
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

bs8 = bw8 + bw9
bs8c = bw8 + bw9cs

bs9 = 1.3 * bw8 + (1.4 + 3.8j) * bw9
bs9c = 1.3 * bw8 + (1.4 + 3.8j) * bw9cs

bs10 = -1.3 * bw11 + 2.3 * bw9
bs10c = -1.3 * bw11cs + 2.3 * bw9


class TestBoseSentence:
    def test_missing(self):
        """Test the result when a missing key is indexed."""
        bw = BoseWord({(0, 0): "+", (1, 1): "-"})
        new_bw = BoseWord({(0, 2): "+", (1, 3): "-"})
        bs = BoseSentence({bw: 1.0})

        assert new_bw not in bs.keys()
        assert bs[new_bw] == 0.0

    def test_set_items(self):
        """Test that we can add a new key to a BoseSentence."""
        bw = BoseWord({(0, 0): "+", (1, 1): "-"})
        bs = BoseSentence({bw: 1.0})

        new_bw = BoseWord({(0, 2): "+", (1, 3): "-"})
        assert new_bw not in bs.keys()

        bs[new_bw] = 3.45
        assert new_bw in bs.keys() and bs[new_bw] == 3.45

    tup_bs_str = (
        (
            bs1,
            "1.23 * b\u207a(0) b(1)\n"
            + "+ 4j * b\u207a(0) b(0)\n"
            + "+ -0.5 * b\u207a(0) b(3) b\u207a(0) b(4)",
        ),
        (
            bs2,
            "-1.23 * b\u207a(0) b(1)\n"
            + "+ (-0-4j) * b\u207a(0) b(0)\n"
            + "+ 0.5 * b\u207a(0) b(3) b\u207a(0) b(4)",
        ),
        (bs3, "-0.5 * b\u207a(0) b(3) b\u207a(0) b(4)\n" + "+ 1 * I"),
        (bs4, "1 * I"),
        (bs5, "0 * I"),
    )

    @pytest.mark.parametrize("bs, str_rep", tup_bs_str)
    def test_str(self, bs, str_rep):
        """Test the string representation of the BoseSentence."""
        assert str(bs) == str_rep
        assert repr(bs) == f"BoseSentence({dict(bs)})"

    tup_bs_wires = (
        (bs1, {0, 1, 3, 4}),
        (bs2, {0, 1, 3, 4}),
        (bs3, {0, 3, 4}),
        (bs4, set()),
    )

    @pytest.mark.parametrize("bs, wires", tup_bs_wires)
    def test_wires(self, bs, wires):
        """Test the correct wires are given for the BoseSentence."""
        assert bs.wires == wires

    @pytest.mark.parametrize("bs", (bs1, bs2, bs3, bs4))
    def test_copy(self, bs):
        """Test that the copy is identical to the original."""
        copy_bs = copy(bs)
        deep_copy_bs = deepcopy(bs)

        assert copy_bs == bs
        assert deep_copy_bs == bs
        assert copy_bs is not bs
        assert deep_copy_bs is not bs

    def test_simplify(self):
        """Test that simplify removes terms in the BoseSentence with coefficient less than the
        threshold."""
        un_simplified_bs = BoseSentence({bw1: 0.001, bw2: 0.05, bw3: 1})

        expected_simplified_bs0 = BoseSentence({bw1: 0.001, bw2: 0.05, bw3: 1})
        expected_simplified_bs1 = BoseSentence({bw2: 0.05, bw3: 1})
        expected_simplified_bs2 = BoseSentence({bw3: 1})

        un_simplified_bs.simplify()
        assert un_simplified_bs == expected_simplified_bs0  # default tol = 1e-8
        un_simplified_bs.simplify(tol=1e-2)
        assert un_simplified_bs == expected_simplified_bs1
        un_simplified_bs.simplify(tol=1e-1)
        assert un_simplified_bs == expected_simplified_bs2

    def test_pickling(self):
        """Check that BoseSentences can be pickled and unpickled."""
        f1 = BoseWord({(0, 0): "+", (1, 1): "-"})
        f2 = BoseWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
        bs = BoseSentence({f1: 1.5, f2: -0.5})

        serialization = pickle.dumps(bs)
        new_bs = pickle.loads(serialization)
        assert bs == new_bs

    bs_dag_tup = (
        (bs1, bs1_dag),
        (bs2, bs2_dag),
        (bs3, bs3_dag),
        (bs4, bs4_dag),
        (bs5, bs5_dag),
        (bs6, bs6_dag),
        (bs1_hamiltonian, bs1_hamiltonian_dag),
        (bs2_hamiltonian, bs2_hamiltonian_dag),
    )

    @pytest.mark.parametrize("bs, bs_dag", bs_dag_tup)
    def test_adjoint(self, bs, bs_dag):
        assert bs.adjoint() == bs_dag


class TestBoseSentenceArithmetic:
    tup_bs_mult = (  # computed by hand
        (
            bs1,
            bs1,
            bs1_x_bs2,
        ),
        (
            bs3,
            bs4,
            BoseSentence(
                {
                    BoseWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"}): -0.5,
                    BoseWord({}): 1,
                }
            ),
        ),
        (
            bs4,
            bs4,
            BoseSentence(
                {
                    BoseWord({}): 1,
                }
            ),
        ),
        (bs5, bs3, bs5),
        (bs3, bs5, bs5),
        (bs4, bs3, bs3),
        (bs3, bs4, bs3),
        (
            BoseSentence({bw2: 1, bw3: 1, bw4: 1}),
            BoseSentence({bw4: 1, bw2: 1}),
            BoseSentence({bw2: 2, bw3: 1, bw4: 1, bw2 * bw2: 1, bw3 * bw2: 1}),
        ),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_bs_mult)
    def test_mul_bose_sentences(self, f1, f2, result):
        """Test that the correct result of multiplication between two
        BoseSentences is produced."""

        simplified_product = f1 * f2
        simplified_product.simplify()

        assert simplified_product == result

    SENTENCES_AND_WORDS_MUL = (
        (
            bw1,
            BoseSentence({bw3: 1.2}),
            BoseSentence({bw3 * bw1: 1.2}),
        ),
        (
            bw2,
            BoseSentence({bw3: 1.2, bw1: 3.7}),
            BoseSentence({bw3 * bw2: 1.2, bw1 * bw2: 3.7}),
        ),
    )

    @pytest.mark.parametrize("bw, bs, result", SENTENCES_AND_WORDS_MUL)
    def test_mul_bose_word_and_sentence(self, bw, bs, result):
        """Test that a BoseWord and a BoseSentence can be multiplied together
        and return a new BoseSentence"""
        assert bs * bw == result

    SENTENCES_AND_NUMBERS_MUL = (
        (bs1, 2, BoseSentence({bw1: 1.23 * 2, bw2: 4j * 2, bw3: -0.5 * 2})),  # int
        (bs2, 3.4, BoseSentence({bw1: -1.23 * 3.4, bw2: -4j * 3.4, bw3: 0.5 * 3.4})),  # float
        (bs1, 3j, BoseSentence({bw1: 3.69j, bw2: -12, bw3: -1.5j})),  # complex
        (bs5, 10, BoseSentence({})),  # null operator times constant
        (
            bs1,
            np.array([2]),
            BoseSentence({bw1: 1.23 * 2, bw2: 4j * 2, bw3: -0.5 * 2}),
        ),  # numpy array
        (
            bs1,
            pnp.array([2]),
            BoseSentence({bw1: 1.23 * 2, bw2: 4j * 2, bw3: -0.5 * 2}),
        ),  # pennylane numpy array
        (
            bs1,
            pnp.array([2, 2])[0],
            BoseSentence({bw1: 1.23 * 2, bw2: 4j * 2, bw3: -0.5 * 2}),
        ),  # pennylane tensor with no length
    )

    @pytest.mark.parametrize("bs, number, result", SENTENCES_AND_NUMBERS_MUL)
    def test_mul_number(self, bs, number, result):
        """Test that a BoseSentence can be multiplied onto a number (from the left)
        and return a BoseSentence"""
        assert bs * number == result

    @pytest.mark.parametrize("bs, number, result", SENTENCES_AND_NUMBERS_MUL)
    def test_rmul_number(self, bs, number, result):
        """Test that a BoseSentence can be multiplied onto a number (from the right)
        and return a BoseSentence"""
        assert number * bs == result

    tup_bs_add = (  # computed by hand
        (bs1, bs1, BoseSentence({bw1: 2.46, bw2: 8j, bw3: -1})),
        (bs1, bs2, BoseSentence({})),
        (bs1, bs3, BoseSentence({bw1: 1.23, bw2: 4j, bw3: -1, bw4: 1})),
        (bs2, bs5, bs2),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_bs_add)
    def test_add_bose_sentences(self, f1, f2, result):
        """Test that the correct result of addition is produced for two BoseSentences."""

        simplified_product = f1 + f2
        simplified_product.simplify()

        assert simplified_product == result

    SENTENCES_AND_WORDS_ADD = [
        (bw1, BoseSentence({bw1: 1.2, bw3: 3j}), BoseSentence({bw1: 2.2, bw3: 3j})),
        (bw3, BoseSentence({bw1: 1.2, bw3: 3j}), BoseSentence({bw1: 1.2, bw3: (1 + 3j)})),
        (bw1, BoseSentence({bw1: -1.2, bw3: 3j}), BoseSentence({bw1: -0.2, bw3: 3j})),
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
        (BoseSentence({bw1: 1.2, bw3: 3j}), 3, BoseSentence({bw1: 1.2, bw3: 3j, bw4: 3})),  # int
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            1.3,
            BoseSentence({bw1: 1.2, bw3: 3j, bw4: 1.3}),
        ),  # float
        (
            BoseSentence({bw1: -1.2, bw3: 3j}),  # complex
            (1 + 2j),
            BoseSentence({bw1: -1.2, bw3: 3j, bw4: (1 + 2j)}),
        ),
        (BoseSentence({}), 5, BoseSentence({bw4: 5})),  # null sentence
        (BoseSentence({bw4: 3}), 1j, BoseSentence({bw4: 3 + 1j})),  # identity only
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            np.array([3]),
            BoseSentence({bw1: 1.2, bw3: 3j, bw4: 3}),
        ),  # numpy array
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            pnp.array([3]),
            BoseSentence({bw1: 1.2, bw3: 3j, bw4: 3}),
        ),  # pennylane numpy array
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            pnp.array([3, 0])[0],
            BoseSentence({bw1: 1.2, bw3: 3j, bw4: 3}),
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
        (bs1, bw1, BoseSentence({bw1: 0.23, bw2: 4j, bw3: -0.5})),
        (bs2, bw3, BoseSentence({bw1: -1.23, bw2: -4j, bw3: -0.5})),
        (bs3, bw4, BoseSentence({bw3: -0.5})),
        (BoseSentence({bw1: 1.2, bw3: 3j}), bw1, BoseSentence({bw1: 0.2, bw3: 3j})),
        (BoseSentence({bw1: 1.2, bw3: 3j}), bw3, BoseSentence({bw1: 1.2, bw3: (-1 + 3j)})),
        (BoseSentence({bw1: -1.2, bw3: 3j}), bw1, BoseSentence({bw1: -2.2, bw3: 3j})),
    )

    @pytest.mark.parametrize("bs, bw, result", SENTENCE_MINUS_WORD)
    def test_subtract_bose_word_from_bose_sentence(self, bs, bw, result):
        """Test that the correct result is produced if a BoseWord is
        subtracted from a BoseSentence"""

        simplified_diff = bs - bw
        simplified_diff.simplify()
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        simplified_diff = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in simplified_diff.items()}
        )

        assert simplified_diff == result

    SENTENCE_MINUS_CONSTANT = (  # computed by hand
        (bs1, 3, BoseSentence({bw1: 1.23, bw2: 4j, bw3: -0.5, bw4: -3})),  # int
        (bs2, -2.7, BoseSentence({bw1: -1.23, bw2: -4j, bw3: 0.5, bw4: 2.7})),  # float
        (bs3, 2j, BoseSentence({bw3: -0.5, bw4: (1 - 2j)})),  # complex
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            -4,
            BoseSentence({bw1: 1.2, bw3: 3j, bw4: 4}),
        ),  # negative int
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            np.array([3]),
            BoseSentence({bw1: 1.2, bw3: 3j, bw4: -3}),
        ),  # numpy array
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            pnp.array([3]),
            BoseSentence({bw1: 1.2, bw3: 3j, bw4: -3}),
        ),  # pennylane numpy array
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            pnp.array([3, 2])[0],
            BoseSentence({bw1: 1.2, bw3: 3j, bw4: -3}),
        ),  # pennylane tensor with no len
    )

    @pytest.mark.parametrize("bs, c, result", SENTENCE_MINUS_CONSTANT)
    def test_subtract_constant_from_bose_sentence(self, bs, c, result):
        """Test that the correct result is produced if a BoseWord is
        subtracted from a BoseSentence"""

        simplified_diff = bs - c
        simplified_diff.simplify()
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        simplified_diff = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in simplified_diff.items()}
        )
        assert simplified_diff == result

    CONSTANT_MINUS_SENTENCE = (  # computed by hand
        (bs1, 3, BoseSentence({bw1: -1.23, bw2: -4j, bw3: 0.5, bw4: 3})),
        (bs2, -2.7, BoseSentence({bw1: 1.23, bw2: 4j, bw3: -0.5, bw4: -2.7})),
        (bs3, 2j, BoseSentence({bw3: 0.5, bw4: (-1 + 2j)})),
        (BoseSentence({bw1: 1.2, bw3: 3j}), -4, BoseSentence({bw1: -1.2, bw3: -3j, bw4: -4})),
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            np.array([3]),
            BoseSentence({bw1: -1.2, bw3: -3j, bw4: 3}),
        ),  # numpy array
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            pnp.array([3]),
            BoseSentence({bw1: -1.2, bw3: -3j, bw4: 3}),
        ),  # pennylane numpy array
        (
            BoseSentence({bw1: 1.2, bw3: 3j}),
            pnp.array([3, 3])[0],
            BoseSentence({bw1: -1.2, bw3: -3j, bw4: 3}),
        ),  # pennylane tensor with to len
    )

    @pytest.mark.parametrize("bs, c, result", CONSTANT_MINUS_SENTENCE)
    def test_subtract_bose_sentence_from_constant(self, bs, c, result):
        """Test that the correct result is produced if a BoseWord is
        subtracted from a BoseSentence"""

        simplified_diff = c - bs
        simplified_diff.simplify()
        # due to rounding, the actual result for floats is
        # e.g. -0.19999999999999... instead of 0.2, so we round to compare
        simplified_diff = BoseSentence(
            {k: round(v, 10) if isinstance(v, float) else v for k, v in simplified_diff.items()}
        )
        assert simplified_diff == result

    tup_bs_subtract = (  # computed by hand
        (bs1, bs1, BoseSentence({})),
        (bs1, bs2, BoseSentence({bw1: 2.46, bw2: 8j, bw3: -1})),
        (bs1, bs3, BoseSentence({bw1: 1.23, bw2: 4j, bw4: -1})),
        (bs2, bs5, bs2),
    )

    @pytest.mark.parametrize("f1, f2, result", tup_bs_subtract)
    def test_subtract_bose_sentences(self, f1, f2, result):
        """Test that the correct result of subtraction is produced for two BoseSentences."""

        simplified_product = f1 - f2
        simplified_product.simplify()

        assert simplified_product == result

    tup_bs_pow = (
        (bs1, 0, BoseSentence({BoseWord({}): 1})),
        (bs1, 1, bs1),
        (bs1, 2, bs1_x_bs2),
        (bs3, 0, BoseSentence({BoseWord({}): 1})),
        (bs3, 1, bs3),
        (bs4, 0, BoseSentence({BoseWord({}): 1})),
        (bs4, 3, bs4),
    )

    @pytest.mark.parametrize("f1, pow, result", tup_bs_pow)
    def test_pow(self, f1, pow, result):
        """Test that raising a BoseWord to an integer power returns the expected BoseWord"""
        assert f1**pow == result

    tup_bs_pow_error = ((bs1, -1), (bs3, 1.5))

    @pytest.mark.parametrize("f1, pow", tup_bs_pow_error)
    def test_pow_error(self, f1, pow):
        """Test that invalid values for the exponent raises an error"""
        with pytest.raises(ValueError, match="The exponent must be a positive integer."):
            f1**pow  # pylint: disable=pointless-statement

    TYPE_ERRORS = ((bs4, "string"),)

    @pytest.mark.parametrize("bs, bad_type", TYPE_ERRORS)
    def test_add_error(self, bs, bad_type):
        """Test __add__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a BoseSentence."):
            bs + bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bs, bad_type", TYPE_ERRORS)
    def test_radd_error(self, bs, bad_type):
        """Test __radd__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot add {type(bad_type)} to a BoseSentence."):
            bad_type + bs  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bs, bad_type", TYPE_ERRORS)
    def test_sub_error(self, bs, bad_type):
        """Test __sub__ with unsupported type raises an error"""
        with pytest.raises(
            TypeError, match=f"Cannot subtract {type(bad_type)} from a BoseSentence."
        ):
            bs - bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bs, bad_type", TYPE_ERRORS)
    def test_rsub_error(self, bs, bad_type):
        """Test __rsub__ with unsupported type raises an error"""
        with pytest.raises(
            TypeError, match=f"Cannot subtract a BoseSentence from {type(bad_type)}."
        ):
            bad_type - bs  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bs, bad_type", TYPE_ERRORS)
    def test_mul_error(self, bs, bad_type):
        """Test __sub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply BoseSentence by {type(bad_type)}."):
            bs * bad_type  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("bs, bad_type", TYPE_ERRORS)
    def test_rmul_error(self, bs, bad_type):
        """Test __rsub__ with unsupported type raises an error"""
        with pytest.raises(TypeError, match=f"Cannot multiply {type(bad_type)} by BoseSentence."):
            bad_type * bs  # pylint: disable=pointless-statement

    @pytest.mark.parametrize(
        "method_name", ("__add__", "__sub__", "__mul__", "__radd__", "__rsub__", "__rmul__")
    )
    def test_array_must_not_exceed_length_1(self, method_name):
        with pytest.raises(
            ValueError, match="Arithmetic Bose operations can only accept an array of length 1"
        ):
            method_to_test = getattr(bs1, method_name)
            _ = method_to_test(np.array([1, 2]))

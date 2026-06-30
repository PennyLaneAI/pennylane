# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the fermionic algebra primitives in ``fermi.py``.

Covers FermiOp factories, FermiWord (normal ordering, the zero test, products,
hashing/equality), and FermiSentence arithmetic. Several FermiSentence tests
target previously-identified bugs (``__mul__`` raising ``KeyError`` and
``__add__`` / ``__matmul__`` returning bare dicts instead of FermiSentences);
those tests are written against the *correct* behavior, so they fail on the
unfixed implementation and pass once it is corrected.
"""

import pytest

from pennylane.labs.trotter_error.fragments.gan_fragments.fermi import (
    FermiOp,
    FermiSentence,
    FermiSpace,
    FermiType,
    FermiWord,
)


def test_fermiop_factories_set_type_space_mode():
    """Test that the FermiSpace is set correctly"""
    cm = FermiOp.creation_mol(2)
    assert cm.op_type == FermiType.CREATION
    assert cm.space == FermiSpace.MOLECULAR
    assert cm.mode == 2

    am = FermiOp.annihilation_mol(3)
    assert am.op_type == FermiType.ANNIHILATION
    assert am.space == FermiSpace.MOLECULAR

    ck = FermiOp.creation_met(1)
    assert ck.op_type == FermiType.CREATION
    assert ck.space == FermiSpace.METALLIC

    ak = FermiOp.annihilation_met(0)
    assert ak.op_type == FermiType.ANNIHILATION
    assert ak.space == FermiSpace.METALLIC


def test_fermiop_is_frozen_and_hashable():
    """Test FermiOp is hashable"""
    op = FermiOp.creation_mol(0)
    assert hash(op) == hash(FermiOp.creation_mol(0))
    with pytest.raises(Exception):
        op.mode = 5


def test_fermiop_equality():
    """Test equality operator"""
    assert FermiOp.creation_mol(0) == FermiOp.creation_mol(0)
    assert FermiOp.creation_mol(0) != FermiOp.annihilation_mol(0)
    assert FermiOp.creation_mol(0) != FermiOp.creation_met(0)
    assert FermiOp.creation_mol(0) != FermiOp.creation_mol(1)


def test_word_getitem_and_equality_and_hash():
    """Test that FermiWord can be built from its components"""
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    w = FermiWord([c0, a0])
    assert w[0] == c0
    assert w[1] == a0
    assert w == FermiWord([c0, a0])
    assert hash(w) == hash(FermiWord([c0, a0]))


def test_word_setitem_type_checked():
    """Test setitem"""
    w = FermiWord([FermiOp.creation_mol(0)])
    w[0] = FermiOp.annihilation_mol(1)
    assert w[0] == FermiOp.annihilation_mol(1)
    with pytest.raises(TypeError):
        w[0] = "not an op"


def test_word_matmul_concatenates():
    """Test matmul operation"""
    c0, a1 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(1)
    prod = FermiWord([c0]) @ FermiWord([a1])
    assert prod == FermiWord([c0, a1])
    with pytest.raises(TypeError):
        _ = FermiWord([c0]) @ c0  # not a FermiWord


def test_identity_word_is_empty():
    """Test the identity operator"""
    assert FermiWord.identity().ops == []


def test_is_zero_on_repeated_adjacent_operator():
    """Test is_zero"""
    c0 = FermiOp.creation_mol(0)
    assert FermiWord([c0, c0]).is_zero()  # c_0 c_0 = 0
    assert not FermiWord([c0]).is_zero()
    assert not FermiWord.identity().is_zero()  # empty word is the identity, not zero

    # repeated but non-adjacent is NOT flagged zero by this (syntactic) check
    a1 = FermiOp.annihilation_mol(1)
    assert not FermiWord([c0, a1, c0]).is_zero()


def test_normal_order_same_mode_contraction():
    """Test normal order on same mode"""
    # a_0 c_0 = 1 - c_0 a_0  (since {a_0, c_0} = 1)
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    result = FermiWord([a0, c0]).normal_order()
    expected = FermiSentence(
        {
            FermiWord([c0, a0]): -1.0,
            FermiWord.identity(): 1.0,
        }
    )
    assert result == expected


def test_normal_order_already_ordered_is_fixed_point():
    """Test normal_order on normal ordered string"""
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    result = FermiWord([c0, a0]).normal_order()
    assert result == FermiSentence({FermiWord([c0, a0]): 1.0})


def test_normal_order_distinct_mode_anticommute_sign():
    """Test normal order on different modes"""
    # a_0 c_1 = - c_1 a_0  (different modes, no contraction)
    a0, c1 = FermiOp.annihilation_mol(0), FermiOp.creation_mol(1)
    result = FermiWord([a0, c1]).normal_order()
    expected = FermiSentence({FermiWord([c1, a0]): -1.0})
    assert result == expected


def test_normal_order_swapping_equal_creation_modes_gives_zero():
    """Test normal order on cancellation"""
    # c_0 c_0 vanishes
    c0 = FermiOp.creation_mol(0)
    result = FermiWord([c0, c0]).normal_order()
    assert result == FermiSentence({})


def test_word_add_word_returns_sentence():
    """Test word addition"""
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    s = FermiWord([c0]) + FermiWord([a0])
    assert isinstance(s, FermiSentence)
    assert s == FermiSentence({FermiWord([c0]): 1, FermiWord([a0]): 1})


def test_word_add_float_adds_identity_term():
    """Test addition between word and float"""
    c0 = FermiOp.creation_mol(0)
    s = FermiWord([c0]) + 2.0
    assert isinstance(s, FermiSentence)
    assert s == FermiSentence({FermiWord([c0]): 1, FermiWord.identity(): 2.0})


def test_word_add_bad_type_raises():
    """Test word addition raises error"""
    with pytest.raises(TypeError):
        _ = FermiWord([FermiOp.creation_mol(0)]) + "x"


def test_word_scalar_mul_returns_sentence():
    """Test word scalar multiplication"""
    c0 = FermiOp.creation_mol(0)
    s = FermiWord([c0]) * 3.0
    assert isinstance(s, FermiSentence)
    assert s == FermiSentence({FermiWord([c0]): 3.0})


def test_sentence_scalar_mul_scales_and_returns_sentence():
    """Test sentence scalar multiplication"""
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    s = FermiSentence({FermiWord([c0]): 1.0, FermiWord([a0]): 2.0})
    scaled = s * 2.0
    assert isinstance(scaled, FermiSentence)
    assert scaled == FermiSentence({FermiWord([c0]): 2.0, FermiWord([a0]): 4.0})


def test_sentence_add_word_returns_sentence():
    """Test sentence/word addition"""
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    s = FermiSentence({FermiWord([c0]): 1.0})
    out = s + FermiWord([a0])
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0]): 1.0, FermiWord([a0]): 1})


def test_sentence_add_sentence_returns_sentence():
    """Test sentence/sentence addition"""
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    s1 = FermiSentence({FermiWord([c0]): 1.0})
    s2 = FermiSentence({FermiWord([c0]): 2.0, FermiWord([a0]): 1.0})
    out = s1 + s2
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0]): 3.0, FermiWord([a0]): 1.0})


def test_sentence_add_float_returns_sentence():
    """Test sentence/float addition"""
    c0 = FermiOp.creation_mol(0)
    s = FermiSentence({FermiWord([c0]): 1.0})
    out = s + 5.0
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0]): 1.0, FermiWord.identity(): 5.0})


def test_sentence_add_bad_type_raises():
    """Test sentence addition raises error"""
    s = FermiSentence({FermiWord([FermiOp.creation_mol(0)]): 1.0})
    with pytest.raises(TypeError):
        _ = s + "x"


def test_sentence_matmul_word_returns_sentence():
    """Test sentence/word matmul"""
    c0, a1 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(1)
    s = FermiSentence({FermiWord([c0]): 2.0})
    out = s @ FermiWord([a1])
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0, a1]): 2.0})


def test_sentence_matmul_sentence_multiplies_coeffs_and_returns_sentence():
    """Test sentence/sentence matmul"""
    c0, a1 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(1)
    s1 = FermiSentence({FermiWord([c0]): 2.0})
    s2 = FermiSentence({FermiWord([a1]): 3.0})
    out = s1 @ s2
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0, a1]): 6.0})


def test_sentence_matmul_bad_type_raises():
    """Test sentence matmul raises error"""
    s = FermiSentence({FermiWord([FermiOp.creation_mol(0)]): 1.0})
    with pytest.raises(TypeError):
        _ = s @ 5

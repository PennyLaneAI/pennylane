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


# --------------------------------------------------------------------------- #
# FermiOp
# --------------------------------------------------------------------------- #
def test_fermiop_factories_set_type_space_mode():
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
    op = FermiOp.creation_mol(0)
    # frozen dataclass -> hashable and immutable
    assert hash(op) == hash(FermiOp.creation_mol(0))
    with pytest.raises(Exception):
        op.mode = 5  # type: ignore[misc]


def test_fermiop_equality():
    assert FermiOp.creation_mol(0) == FermiOp.creation_mol(0)
    assert FermiOp.creation_mol(0) != FermiOp.annihilation_mol(0)
    assert FermiOp.creation_mol(0) != FermiOp.creation_met(0)
    assert FermiOp.creation_mol(0) != FermiOp.creation_mol(1)


# --------------------------------------------------------------------------- #
# FermiWord: structure
# --------------------------------------------------------------------------- #
def test_word_getitem_and_equality_and_hash():
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    w = FermiWord([c0, a0])
    assert w[0] == c0
    assert w[1] == a0
    assert w == FermiWord([c0, a0])
    assert hash(w) == hash(FermiWord([c0, a0]))


def test_word_setitem_type_checked():
    w = FermiWord([FermiOp.creation_mol(0)])
    w[0] = FermiOp.annihilation_mol(1)
    assert w[0] == FermiOp.annihilation_mol(1)
    with pytest.raises(TypeError):
        w[0] = "not an op"


def test_word_matmul_concatenates():
    c0, a1 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(1)
    prod = FermiWord([c0]) @ FermiWord([a1])
    assert prod == FermiWord([c0, a1])
    with pytest.raises(TypeError):
        _ = FermiWord([c0]) @ c0  # not a FermiWord


def test_identity_word_is_empty():
    assert FermiWord.identity().ops == []


# --------------------------------------------------------------------------- #
# FermiWord.is_zero
# --------------------------------------------------------------------------- #
def test_is_zero_on_repeated_adjacent_operator():
    c0 = FermiOp.creation_mol(0)
    assert FermiWord([c0, c0]).is_zero()  # c_0 c_0 = 0
    assert not FermiWord([c0]).is_zero()
    assert not FermiWord.identity().is_zero()  # empty word is the identity, not zero
    # repeated but non-adjacent is NOT flagged zero by this (syntactic) check
    a1 = FermiOp.annihilation_mol(1)
    assert not FermiWord([c0, a1, c0]).is_zero()


# --------------------------------------------------------------------------- #
# FermiWord.normal_order  (canonical: creation left of annihilation)
# --------------------------------------------------------------------------- #
def test_normal_order_same_mode_contraction():
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
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    result = FermiWord([c0, a0]).normal_order()
    assert result == FermiSentence({FermiWord([c0, a0]): 1.0})


def test_normal_order_distinct_mode_anticommute_sign():
    # a_0 c_1 = - c_1 a_0  (different modes, no contraction)
    a0, c1 = FermiOp.annihilation_mol(0), FermiOp.creation_mol(1)
    result = FermiWord([a0, c1]).normal_order()
    expected = FermiSentence({FermiWord([c1, a0]): -1.0})
    assert result == expected


def test_normal_order_swapping_equal_creation_modes_gives_zero():
    # c_0 c_0 vanishes
    c0 = FermiOp.creation_mol(0)
    result = FermiWord([c0, c0]).normal_order()
    assert result == FermiSentence({})


# --------------------------------------------------------------------------- #
# FermiWord additive / scalar ops (these already return FermiSentence)
# --------------------------------------------------------------------------- #
def test_word_add_word_returns_sentence():
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    s = FermiWord([c0]) + FermiWord([a0])
    assert isinstance(s, FermiSentence)
    assert s == FermiSentence({FermiWord([c0]): 1, FermiWord([a0]): 1})


def test_word_add_float_adds_identity_term():
    c0 = FermiOp.creation_mol(0)
    s = FermiWord([c0]) + 2.0
    assert isinstance(s, FermiSentence)
    assert s == FermiSentence({FermiWord([c0]): 1, FermiWord.identity(): 2.0})


def test_word_add_bad_type_raises():
    with pytest.raises(TypeError):
        _ = FermiWord([FermiOp.creation_mol(0)]) + "x"


def test_word_scalar_mul_returns_sentence():
    c0 = FermiOp.creation_mol(0)
    s = FermiWord([c0]) * 3.0
    assert isinstance(s, FermiSentence)
    assert s == FermiSentence({FermiWord([c0]): 3.0})


# --------------------------------------------------------------------------- #
# FermiSentence  -- BUG-TARGETING TESTS
# These assert the correct contract and must fail on the unfixed code.
# --------------------------------------------------------------------------- #
def test_sentence_scalar_mul_scales_and_returns_sentence():
    # BUG: __mul__ used `d = {}` then `d[key] += ...`, raising KeyError.
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    s = FermiSentence({FermiWord([c0]): 1.0, FermiWord([a0]): 2.0})
    scaled = s * 2.0
    assert isinstance(scaled, FermiSentence)
    assert scaled == FermiSentence({FermiWord([c0]): 2.0, FermiWord([a0]): 4.0})


def test_sentence_add_word_returns_sentence():
    # BUG: __add__ returned a bare defaultdict instead of a FermiSentence.
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    s = FermiSentence({FermiWord([c0]): 1.0})
    out = s + FermiWord([a0])
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0]): 1.0, FermiWord([a0]): 1})


def test_sentence_add_sentence_returns_sentence():
    c0, a0 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(0)
    s1 = FermiSentence({FermiWord([c0]): 1.0})
    s2 = FermiSentence({FermiWord([c0]): 2.0, FermiWord([a0]): 1.0})
    out = s1 + s2
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0]): 3.0, FermiWord([a0]): 1.0})


def test_sentence_add_float_returns_sentence():
    c0 = FermiOp.creation_mol(0)
    s = FermiSentence({FermiWord([c0]): 1.0})
    out = s + 5.0
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0]): 1.0, FermiWord.identity(): 5.0})


def test_sentence_add_bad_type_raises():
    s = FermiSentence({FermiWord([FermiOp.creation_mol(0)]): 1.0})
    with pytest.raises(TypeError):
        _ = s + "x"


def test_sentence_matmul_word_returns_sentence():
    c0, a1 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(1)
    s = FermiSentence({FermiWord([c0]): 2.0})
    out = s @ FermiWord([a1])
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0, a1]): 2.0})


def test_sentence_matmul_sentence_multiplies_coeffs_and_returns_sentence():
    c0, a1 = FermiOp.creation_mol(0), FermiOp.annihilation_mol(1)
    s1 = FermiSentence({FermiWord([c0]): 2.0})
    s2 = FermiSentence({FermiWord([a1]): 3.0})
    out = s1 @ s2
    assert isinstance(out, FermiSentence)
    assert out == FermiSentence({FermiWord([c0, a1]): 6.0})


def test_sentence_matmul_bad_type_raises():
    s = FermiSentence({FermiWord([FermiOp.creation_mol(0)]): 1.0})
    with pytest.raises(TypeError):
        _ = s @ 5

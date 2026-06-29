"""Tests for the GAN operator classes in ``gan_fragments.py``.

Covers FuncSymbol (factories, base, norm), GanMonomial (normal ordering,
exponent merging, products, norm), GanCoeff (sparsity, arithmetic, equality,
norm), and GanFragment (term pruning, arithmetic, the operator product, norm).
"""

import math

import pytest

from pennylane.labs.trotter_error.fragments.gan_fragments.gan_fragments import (
    FuncType,
    FuncSymbol,
    GanMonomial,
    GanCoeff,
    GanFragment,
)
from pennylane.labs.trotter_error.fragments.gan_fragments.fermi import FermiOp, FermiWord


# --------------------------------------------------------------------------- #
# FuncSymbol
# --------------------------------------------------------------------------- #
def test_funcsymbol_factory_defaults():
    assert FuncSymbol.momentum(0).exponent == 2
    assert FuncSymbol.momentum(0).f_type == FuncType.MOMENTUM
    assert FuncSymbol.position(0).exponent == 1
    assert FuncSymbol.position(0).f_type == FuncType.POSITION
    ident = FuncSymbol.identity()
    assert ident.f_type == FuncType.IDENTITY
    assert repr(ident) == "I"


def test_funcsymbol_base_ignores_exponent():
    assert FuncSymbol.position(2, 1).base() == FuncSymbol.position(2, 5).base()
    assert FuncSymbol.position(2).base() != FuncSymbol.position(3).base()
    assert FuncSymbol.position(2).base() != FuncSymbol.momentum(2).base()


def test_funcsymbol_norm():
    g = 10
    expected = math.sqrt(g * math.pi / 2)
    assert FuncSymbol.position(0, 1).norm(g) == pytest.approx(expected)
    assert FuncSymbol.momentum(0, 1).norm(g) == pytest.approx(expected)
    # documented behavior: linear in exponent
    assert FuncSymbol.position(0, 3).norm(g) == pytest.approx(3 * expected)
    assert FuncSymbol.identity().norm(g) == 1


def test_funcsymbol_is_hashable_frozen():
    assert hash(FuncSymbol.position(1)) == hash(FuncSymbol.position(1))
    with pytest.raises(Exception):
        FuncSymbol.position(1).mode = 2  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# GanMonomial
# --------------------------------------------------------------------------- #
def test_monomial_identity_is_empty():
    assert GanMonomial.identity().funcs == []


def test_monomial_matmul_concatenates():
    m = GanMonomial([FuncSymbol.position(0)]) @ GanMonomial([FuncSymbol.position(1)])
    assert m.funcs == [FuncSymbol.position(0), FuncSymbol.position(1)]


def test_monomial_normal_order_sorts_positions_by_mode():
    m = GanMonomial([FuncSymbol.position(1), FuncSymbol.position(0)]).normal_order()
    assert m.funcs == [FuncSymbol.position(0), FuncSymbol.position(1)]


def test_monomial_normal_order_merges_equal_base_exponents():
    m = GanMonomial([FuncSymbol.position(0), FuncSymbol.position(0)]).normal_order()
    assert m.funcs == [FuncSymbol.position(0, 2)]


def test_monomial_equality_and_hash():
    a = GanMonomial([FuncSymbol.position(0)])
    b = GanMonomial([FuncSymbol.position(0)])
    assert a == b
    assert hash(a) == hash(b)


def test_monomial_norm_is_product_of_factor_norms():
    g = 8
    m = GanMonomial([FuncSymbol.position(0), FuncSymbol.position(1)])
    expected = abs(FuncSymbol.position(0).norm(g)) * abs(FuncSymbol.position(1).norm(g))
    assert m.norm(g) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# GanCoeff
# --------------------------------------------------------------------------- #
def _mono(mode=0):
    return GanMonomial([FuncSymbol.position(mode)])


def test_gancoeff_drops_negligible_terms():
    c = GanCoeff({_mono(0): 1e-12})
    assert c.is_zero()
    c2 = GanCoeff({_mono(0): 1.0})
    assert not c2.is_zero()


def test_gancoeff_identity_is_empty_and_zero():
    ident = GanCoeff.identity()
    assert ident.monomials == {}
    assert ident.is_zero()


def test_gancoeff_add_combines_like_monomials():
    c = GanCoeff({_mono(0): 1.0}) + GanCoeff({_mono(0): 2.0, _mono(1): 1.0})
    assert c == GanCoeff({_mono(0): 3.0, _mono(1): 1.0})


def test_gancoeff_sub():
    c = GanCoeff({_mono(0): 3.0}) - GanCoeff({_mono(0): 1.0})
    assert c == GanCoeff({_mono(0): 2.0})


def test_gancoeff_scalar_mul_both_sides():
    c = GanCoeff({_mono(0): 2.0})
    assert (3.0 * c) == GanCoeff({_mono(0): 6.0})
    assert (c * 3.0) == GanCoeff({_mono(0): 6.0})


def test_gancoeff_matmul_multiplies_monomials_and_coeffs():
    # Q(0) * Q(1) -> Q(0)Q(1), coeff 2*3 = 6
    c = GanCoeff({_mono(0): 2.0}) @ GanCoeff({_mono(1): 3.0})
    expected_mono = (_mono(0) @ _mono(1)).normal_order()
    assert c == GanCoeff({expected_mono: 6.0})


def test_gancoeff_equality_uses_isclose():
    c1 = GanCoeff({_mono(0): 1.0})
    c2 = GanCoeff({_mono(0): 1.0 + 1e-12})
    assert c1 == c2
    assert GanCoeff({_mono(0): 1.0}) != GanCoeff({_mono(1): 1.0})


def test_gancoeff_norm():
    g = 8
    c = GanCoeff({_mono(0): 2.0})
    assert c.norm(g) == pytest.approx(2.0 * _mono(0).norm(g))


# --------------------------------------------------------------------------- #
# GanFragment
# --------------------------------------------------------------------------- #
def _num_op(mode=0):
    """A simple number-operator fermionic word c^dag_mol(mode) a_mol(mode)."""
    return FermiWord([FermiOp.creation_mol(mode), FermiOp.annihilation_mol(mode)])


def test_fragment_prunes_zero_fermi_and_zero_coeff():
    c0 = FermiOp.creation_mol(0)
    # zero fermionic word (c_0 c_0) -> dropped
    frag = GanFragment({FermiWord([c0, c0]): GanCoeff({_mono(0): 1.0})})
    assert frag.fragment == {}
    # zero coefficient -> dropped
    frag2 = GanFragment({_num_op(0): GanCoeff.identity()})
    assert frag2.fragment == {}


def test_fragment_add_combines_terms():
    f1 = GanFragment({_num_op(0): GanCoeff({_mono(0): 1.0})})
    f2 = GanFragment({_num_op(0): GanCoeff({_mono(0): 2.0})})
    out = f1 + f2
    assert out == GanFragment({_num_op(0): GanCoeff({_mono(0): 3.0})})


def test_fragment_sub():
    f1 = GanFragment({_num_op(0): GanCoeff({_mono(0): 3.0})})
    f2 = GanFragment({_num_op(0): GanCoeff({_mono(0): 1.0})})
    assert (f1 - f2) == GanFragment({_num_op(0): GanCoeff({_mono(0): 2.0})})


def test_fragment_scalar_mul_both_sides():
    f = GanFragment({_num_op(0): GanCoeff({_mono(0): 2.0})})
    expected = GanFragment({_num_op(0): GanCoeff({_mono(0): 6.0})})
    assert (f * 3) == expected
    assert (3 * f) == expected


def test_fragment_equality():
    f1 = GanFragment({_num_op(0): GanCoeff({_mono(0): 1.0})})
    f2 = GanFragment({_num_op(0): GanCoeff({_mono(0): 1.0})})
    assert f1 == f2
    f3 = GanFragment({_num_op(1): GanCoeff({_mono(0): 1.0})})
    assert f1 != f3


def test_fragment_matmul_number_operator_is_idempotent():
    # The molecular number operator n_0 = c_0^dag a_0 satisfies n_0^2 = n_0.
    # GanFragment.__matmul__ concatenates the fermionic words and normal-orders
    # the product, which must collapse n_0 n_0 back to n_0 with coefficient 1.
    f = GanFragment({_num_op(0): GanCoeff({GanMonomial.identity(): 1.0})})
    prod = f @ f
    expected = GanFragment({_num_op(0): GanCoeff({GanMonomial.identity(): 1.0})})
    assert prod == expected


def test_fragment_norm_sums_coeff_norms():
    g = 8
    f = GanFragment(
        {
            _num_op(0): GanCoeff({_mono(0): 2.0}),
            _num_op(1): GanCoeff({_mono(1): 3.0}),
        }
    )
    expected = GanCoeff({_mono(0): 2.0}).norm(g) + GanCoeff({_mono(1): 3.0}).norm(g)
    assert f.norm(g) == pytest.approx(expected)

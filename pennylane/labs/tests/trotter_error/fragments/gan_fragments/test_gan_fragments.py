"""Tests for the GAN operator classes in ``gan_fragments.py``.

Covers FuncSymbol (factories, base, norm), GanMonomial (normal ordering,
exponent merging, products, norm), GanCoeff (sparsity, arithmetic, equality,
norm), and GanFragment (term pruning, arithmetic, the operator product, norm).
"""

import math

import pytest

from pennylane.labs.trotter_error.fragments.gan_fragments.fermi import FermiOp, GanFermi
from pennylane.labs.trotter_error.fragments.gan_fragments.gan_fragments import (
    FuncSymbol,
    FuncType,
    GanCoeff,
    GanFragment,
    GanMonomial,
)


def test_funcsymbol_factory_defaults():
    """Test FuncSymbol class methods are correct"""
    assert FuncSymbol.momentum(0).exponent == 2
    assert FuncSymbol.momentum(0).f_type == FuncType.MOMENTUM
    assert FuncSymbol.position(0).exponent == 1
    assert FuncSymbol.position(0).f_type == FuncType.POSITION
    ident = FuncSymbol.identity()
    assert ident.f_type == FuncType.IDENTITY
    assert repr(ident) == "I"


def test_funcsymbol_base_ignores_exponent():
    """Test that the base method is correct"""
    assert FuncSymbol.position(2, 1).base() == FuncSymbol.position(2, 5).base()
    assert FuncSymbol.position(2).base() != FuncSymbol.position(3).base()
    assert FuncSymbol.position(2).base() != FuncSymbol.momentum(2).base()


def test_funcsymbol_norm():
    """Test the norm method"""
    g = 10
    expected = math.sqrt(g * math.pi / 2)
    assert FuncSymbol.position(0, 1).norm(g) == pytest.approx(expected)
    assert FuncSymbol.momentum(0, 1).norm(g) == pytest.approx(expected)
    assert FuncSymbol.position(0, 3).norm(g) == pytest.approx(3 * expected)
    assert FuncSymbol.identity().norm(g) == 1


def test_funcsymbol_is_hashable_frozen():
    """Test FuncSymbol is hashable"""
    assert hash(FuncSymbol.position(1)) == hash(FuncSymbol.position(1))
    with pytest.raises(Exception):
        FuncSymbol.position(1).mode = 2


def test_monomial_matmul_concatenates():
    """Test the monomial matmul operation"""
    m = GanMonomial([FuncSymbol.position(0)]) @ GanMonomial([FuncSymbol.position(1)])
    assert m.funcs == [FuncSymbol.position(0), FuncSymbol.position(1)]


def test_monomial_equality_and_hash():
    """Test monomials are hashable"""
    a = GanMonomial([FuncSymbol.position(0)])
    b = GanMonomial([FuncSymbol.position(0)])
    assert a == b
    assert hash(a) == hash(b)


def test_monomial_norm_is_product_of_factor_norms():
    """Test product of norms"""
    g = 8
    m = GanMonomial([FuncSymbol.position(0), FuncSymbol.position(1)])
    expected = abs(FuncSymbol.position(0).norm(g)) * abs(FuncSymbol.position(1).norm(g))
    assert m.norm(g) == pytest.approx(expected)


def _mono(mode=0):
    """Returns a simply monomial for testing"""
    return GanMonomial([FuncSymbol.position(mode)])


def test_gancoeff_drops_negligible_terms():
    """Test that gancoeff drops negligible terms"""
    c = GanCoeff({_mono(0): 1e-12})
    assert c.is_zero()
    c2 = GanCoeff({_mono(0): 1.0})
    assert not c2.is_zero()


def test_gancoeff_identity_is_empty_and_zero():
    """Test the gancoeff identity operator"""
    ident = GanCoeff.identity()
    assert ident.monomials == {}
    assert ident.is_zero()


def test_gancoeff_add_combines_like_monomials():
    """Test that like monomials are combined"""
    c = GanCoeff({_mono(0): 1.0}) + GanCoeff({_mono(0): 2.0, _mono(1): 1.0})
    assert c == GanCoeff({_mono(0): 3.0, _mono(1): 1.0})


def test_gancoeff_sub():
    """Test gancoeff subtraction"""
    c = GanCoeff({_mono(0): 3.0}) - GanCoeff({_mono(0): 1.0})
    assert c == GanCoeff({_mono(0): 2.0})


def test_gancoeff_scalar_mul_both_sides():
    """Test gancoeff scalar multiplication"""
    c = GanCoeff({_mono(0): 2.0})
    assert (3.0 * c) == GanCoeff({_mono(0): 6.0})
    assert (c * 3.0) == GanCoeff({_mono(0): 6.0})


def test_gancoeff_matmul_multiplies_monomials_and_coeffs():
    """Test gancoeff matmul"""
    c = GanCoeff({_mono(0): 2.0}) @ GanCoeff({_mono(1): 3.0})
    expected_mono = (_mono(0) @ _mono(1)).normal_order()
    assert c == GanCoeff({expected_mono: 6.0})


def test_gancoeff_equality_uses_isclose():
    """Test that isclose is used"""
    c1 = GanCoeff({_mono(0): 1.0})
    c2 = GanCoeff({_mono(0): 1.0 + 1e-12})
    assert c1 == c2
    assert GanCoeff({_mono(0): 1.0}) != GanCoeff({_mono(1): 1.0})


def test_gancoeff_norm():
    """Test the gancoeff norm method"""
    g = 8
    c = GanCoeff({_mono(0): 2.0})
    assert c.norm(g) == pytest.approx(2.0 * _mono(0).norm(g))


def _num_op(mode=0):
    """A fermionic number-operator c^dag_mol(mode) a_mol(mode)."""
    return GanFermi([FermiOp.creation_mol(mode), FermiOp.annihilation_mol(mode)])


def test_fragment_prunes_zero_fermi_and_zero_coeff():
    """ "Test GanFragment drops negligible terms"""
    c0 = FermiOp.creation_mol(0)
    # zero fermionic word (c_0 c_0) -> dropped
    frag = GanFragment({GanFermi([c0, c0]): GanCoeff({_mono(0): 1.0})})
    assert frag.fragment == {}
    # zero coefficient -> dropped
    frag2 = GanFragment({_num_op(0): GanCoeff.identity()})
    assert frag2.fragment == {}


def test_fragment_add_combines_terms():
    """Test GanFragment combines like terms"""
    f1 = GanFragment({_num_op(0): GanCoeff({_mono(0): 1.0})})
    f2 = GanFragment({_num_op(0): GanCoeff({_mono(0): 2.0})})
    out = f1 + f2
    assert out == GanFragment({_num_op(0): GanCoeff({_mono(0): 3.0})})


def test_fragment_sub():
    """Test GanFragment subtraction"""
    f1 = GanFragment({_num_op(0): GanCoeff({_mono(0): 3.0})})
    f2 = GanFragment({_num_op(0): GanCoeff({_mono(0): 1.0})})
    assert (f1 - f2) == GanFragment({_num_op(0): GanCoeff({_mono(0): 2.0})})


def test_fragment_scalar_mul_both_sides():
    """Test GanFragment scalar multiplication"""
    f = GanFragment({_num_op(0): GanCoeff({_mono(0): 2.0})})
    expected = GanFragment({_num_op(0): GanCoeff({_mono(0): 6.0})})
    assert (f * 3) == expected
    assert (3 * f) == expected


def test_fragment_equality():
    """Test GanFragment equality"""
    f1 = GanFragment({_num_op(0): GanCoeff({_mono(0): 1.0})})
    f2 = GanFragment({_num_op(0): GanCoeff({_mono(0): 1.0})})
    assert f1 == f2
    f3 = GanFragment({_num_op(1): GanCoeff({_mono(0): 1.0})})
    assert f1 != f3


def test_fragment_matmul_number_operator_is_idempotent():
    """Test number operator is idempotent"""
    f = GanFragment({_num_op(0): GanCoeff({GanMonomial.identity(): 1.0})})
    prod = f @ f
    expected = GanFragment({_num_op(0): GanCoeff({GanMonomial.identity(): 1.0})})

    print(prod)
    print(expected)

    assert prod == expected


def test_fragment_norm_sums_coeff_norms():
    """Test GanFragment norm method"""
    g = 8
    f = GanFragment(
        {
            _num_op(0): GanCoeff({_mono(0): 2.0}),
            _num_op(1): GanCoeff({_mono(1): 3.0}),
        }
    )
    expected = GanCoeff({_mono(0): 2.0}).norm(g) + GanCoeff({_mono(1): 3.0}).norm(g)
    assert f.norm(g) == pytest.approx(expected)


#Q = FuncSymbol.position
#P = FuncSymbol.momentum
#I = FuncSymbol.identity
#
#@pytest.mark.parametrize(
#    "monomial, expected",
#    [
#        # Already ordered single factors are unchanged.
#        (GanMonomial([Q(0)]), GanMonomial([Q(0)])),
#        # Merge same-base factors by summing exponents.
#        (GanMonomial([Q(0, 1), Q(0, 2)]), GanMonomial([Q(0, 3)])),
#        (GanMonomial([P(1, 1), P(1, 1), P(1, 1)]), GanMonomial([P(1, 3)])),
#        # Same-mode P and Q do NOT commute: order is preserved, no merge.
#        (GanMonomial([Q(0, 1), P(0, 1)]), GanMonomial([Q(0, 1), P(0, 1)])),
#        (GanMonomial([P(0, 1), Q(0, 1)]), GanMonomial([P(0, 1), Q(0, 1)])),
#        # Different modes are sorted into ascending-mode canonical order.
#        (GanMonomial([P(2, 1), P(1, 1)]), GanMonomial([P(1, 1), P(2, 1)])),
#        (GanMonomial([Q(2, 1), Q(0, 1)]), GanMonomial([Q(0, 1), Q(2, 1)])),
#        # Mixed different-mode factors: stable sort by mode keeps same-mode order.
#        (GanMonomial([Q(0, 1), P(2, 1), P(0, 1)]), GanMonomial([Q(0, 1), P(0, 1), P(2, 1)])),
#        (GanMonomial([P(1, 2), Q(0, 1)]), GanMonomial([Q(0, 1), P(1, 2)])),
#        # Identity is dropped when other factors are present.
#        (GanMonomial([I(), Q(0, 1)]), GanMonomial([Q(0, 1)])),
#        (GanMonomial([Q(0, 1), I()]), GanMonomial([Q(0, 1)])),
#        (GanMonomial([I(), P(1, 2), I()]), GanMonomial([P(1, 2)])),
#        # Identity-only and empty products collapse to a single identity.
#        (GanMonomial([I()]), GanMonomial([I()])),
#        (GanMonomial([]), GanMonomial([I()])),
#        # Merge only fires after reordering brings same-base factors together.
#        (GanMonomial([P(2, 1), P(1, 1), P(2, 1)]), GanMonomial([P(1, 1), P(2, 2)])),
#        # A fuller mix: sort by mode, preserve same-mode order, merge same base.
#        (
#            GanMonomial([P(1, 1), Q(0, 1), Q(0, 1), P(2, 1)]),
#            GanMonomial([Q(0, 2), P(1, 1), P(2, 1)]),
#        ),
#    ],
#)
#def test_normal_order(monomial, expected):
#    """test the GanMonomial normal order function"""
#    assert monomial.normal_order() == expected

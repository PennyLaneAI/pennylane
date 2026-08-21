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
"""
Test Expression class and its associated methods
"""

# pylint: disable=unnecessary-dunder-call,protected-access,too-many-public-methods
import math

import pytest

from pennylane.resource.expression import (
    Expression,
    _cast_if_constant,
    _term_to_str,
)


def test_cast_if_constant():
    """Test that _cast_if_constant returns the expected type based on the input data."""
    zero = _cast_if_constant({}, set(), skip_copy=True, skip_normalization=False)
    assert zero == 0
    assert isinstance(zero, int)

    # Check for literal 0
    lit_zero = _cast_if_constant({(): 0}, set(), skip_copy=True, skip_normalization=False)
    assert lit_zero == 0
    assert isinstance(lit_zero, int)

    three = _cast_if_constant({(): 3}, set(), skip_copy=True, skip_normalization=False)
    assert three == 3
    assert isinstance(three, int)

    nega = _cast_if_constant({(): -1}, set(), skip_copy=True, skip_normalization=False)
    assert nega == -1
    assert isinstance(nega, int)

    exp = _cast_if_constant({("x",): 1}, set(), skip_copy=True, skip_normalization=False)
    assert isinstance(exp, Expression)
    assert exp == Expression({("x",): 1})


def test_cast_if_constant_floats():
    """Test that _cast_if_constant handles float coefficients the same way as int coefficients."""
    # A float constant collapses to a float rather than an Expression
    two_and_a_half = _cast_if_constant({(): 2.5}, set(), skip_copy=True, skip_normalization=False)
    assert two_and_a_half == 2.5
    assert isinstance(two_and_a_half, float)

    # Check for literal 0.0; the type of the coefficient is preserved
    lit_zero = _cast_if_constant({(): 0.0}, set(), skip_copy=True, skip_normalization=False)
    assert lit_zero == 0
    assert isinstance(lit_zero, float)

    nega = _cast_if_constant({(): -1.5}, set(), skip_copy=True, skip_normalization=False)
    assert nega == -1.5
    assert isinstance(nega, float)

    exp = _cast_if_constant({("x",): 1.5}, set(), skip_copy=True, skip_normalization=False)
    assert isinstance(exp, Expression)
    assert exp == Expression({("x",): 1.5})


@pytest.mark.parametrize(
    "vars, coeff, expected",
    [
        ((), 0, "0"),
        ((), 0.0, "0"),
        ((), 1, "1"),
        ((), 1.0, "1"),
        ((), 5, "5"),
        ((), -0.5, "-0.5"),
        (("x",), 1, "x"),
        (("x",), 1.0, "x"),
        (("x",), 3, "3*x"),
        (("x",), 3.0, "3*x"),
        (("x",), 2.5, "2.5*x"),
        (("x", "x"), 1, "x*x"),
        (("x", "y"), 1, "x*y"),
        (("x", "y"), 2, "2*x*y"),
        (("x", "y"), 0.5, "0.5*x*y"),
        (("bar", "foo"), 3, "3*bar*foo"),
    ],
)
def test_term_to_str(vars, coeff, expected):
    """Test that _term_to_str returns the expected string representation of a term."""
    assert _term_to_str(vars, coeff) == expected


@pytest.fixture(name="sample_expr")
def fixture_sample_expr() -> Expression:
    """Helper method to create a simple expression for testing."""
    return Expression(
        {
            ("z", "z"): 1,
            ("x", "y"): 1,
            ("x",): 2,
            (): 5,
        }
    )


@pytest.fixture(name="sample_float_expr")
def fixture_sample_float_expr() -> Expression:
    """Helper method to create a simple expression with float coefficients for testing."""
    return Expression(
        {
            ("z", "z"): 1.5,
            ("x", "y"): 1.0,
            ("x",): 2.5,
            (): 5.0,
        }
    )


class TestExpression:
    """Test the methods and attributes of the Expression class"""

    def test_init_as_int(self):
        """Test that the __init__ method can handle an integer input for a constant expression."""
        expr = Expression(5)
        assert expr._data == {(): 5}
        assert expr.vars == set()

        expr = Expression(0)
        assert expr._data == {}

    def test_init_as_float(self):
        """Test that the __init__ method can handle a float input for a constant expression."""
        expr = Expression(2.5)
        assert expr._data == {(): 2.5}
        assert expr.vars == set()

        # A float zero is handled the same way as an int zero
        expr = Expression(0.0)
        assert expr._data == {}
        assert expr.vars == set()

    def test_init_invalid_input(self):
        """Test that the __init__ method raises a TypeError for invalid input types."""
        with pytest.raises(TypeError, match="must be a dictionary of tuples or a real number"):
            Expression("foo")

        with pytest.raises(TypeError, match="must be a dictionary of tuples or a real number"):
            Expression(1.5j)

    def test_init_skip_normalization(self):
        """Test that the __init__ method can skip normalization when _skip_normalization is True."""
        data = {("x", "y"): 2, ("y", "x"): 3, ("z",): 1, ("foo",): 0, (): 4}
        expr = Expression(data, _skip_normalization=True)
        assert expr._data == data

    def test_init_normalizes(self):
        """Test that the __init__ method normalizes the input data correctly."""
        expr = Expression(
            {
                ("x",): 1,
                ("x", "y"): 1.5,  # Should be combined with the next term
                ("y", "x"): 1.0,
                ("z",): 0.5,
                ("foo",): 0.0,  # Should disappear
                (): 4,  # int and float coefficients can coexist
            }
        )
        assert expr._data == {
            ("x",): 1,
            ("x", "y"): 2.5,
            ("z",): 0.5,
            (): 4,
        }

    def test_normalize(self):
        """Test that the _normalize method correctly combines like terms and removes zero terms."""
        expr = Expression(
            {
                ("x", "y"): 2,
                ("y", "x"): -2,
                ("z",): 1,
                ("foo",): 0,
                (): 4,
            },
            _skip_normalization=True,
        )
        expr._normalize()
        assert expr._data == {
            ("z",): 1,
            (): 4,
        }

    def test_normalize_cancels_floats(self):
        """Test that _normalize removes float terms that cancel out exactly."""
        expr = Expression(
            {
                ("x", "y"): 1.5,
                ("y", "x"): -1.5,
                ("z",): 0.5,
                (): 4.0,
            },
            _skip_normalization=True,
        )
        expr._normalize()
        assert expr._data == {
            ("z",): 0.5,
            (): 4.0,
        }

    @pytest.mark.parametrize("val", [1j, "foo", None])
    def test_normalize_invalid(self, val):
        """Test that _normalize raises on invalid coefficients."""
        expr = Expression(
            {
                ("x",): val,
            },
            _skip_normalization=True,
        )

        with pytest.raises(TypeError, match="Expression coefficients must be int or float"):
            expr._normalize()

    def test_vars(self, sample_expr):
        """Test that the vars property returns the expected set of variables."""
        assert sample_expr.vars == {"x", "y", "z"}

    def test_subs(self, sample_expr):
        """Test that the subs method correctly substitutes values for variables."""
        substitutions = {"x": 3}
        new_expr = sample_expr.subs(substitutions)
        assert new_expr._data == {
            ("z", "z"): 1,  # Unchanged
            ("y",): 3,  # 1*x*y becomes 3*y
            (): 11,  # 2*x becomes 6, plus the original constant term of 5
        }
        assert new_expr.vars == {"y", "z"}

        new_expr2 = new_expr.subs({"y": 4})
        assert new_expr2._data == {("z", "z"): 1, (): 23}  # Unchanged
        assert new_expr2.vars == {"z"}

        # Substituting the last variable should yield an integer
        new_expr3 = new_expr2.subs({"z": 3})
        assert isinstance(new_expr3, int)
        assert new_expr3 == 32

    def test_subs_float(self, sample_float_expr):
        """Test that the subs method correctly substitutes float values for variables."""
        new_expr = sample_float_expr.subs({"x": 2.0})
        assert new_expr._data == {
            ("z", "z"): 1.5,  # Unchanged
            ("y",): 2.0,  # 1.0*x*y becomes 2.0*y
            (): 10.0,  # 2.5*x becomes 5.0, plus the original constant term of 5.0
        }
        assert new_expr.vars == {"y", "z"}

        # Test that substituting float values into an int-valued expression yields floats
        expr = Expression({("x",): 2, ("y",): 1})
        assert expr.subs(x=1.5) == Expression({("y",): 1, (): 3.0})

        # Test that substituting the last variable of a float expression yields a float
        expr = Expression({("x",): 1.5, (): 0.5})
        result = expr.subs(x=2)
        assert isinstance(result, float)
        assert result == 3.5

    def test_subs_doesnt_mutate(self):
        """Test that the subs method doesn't mutate the incoming dictionary."""

        s = Expression({("x",): 1, ("y",): -1})
        input = {"x": 5}
        _ = s.subs(input, y=5)
        assert input == {"x": 5}

    def test_subs_cancels_out(self):
        s = Expression({("x",): 1, ("y",): -1})
        subbed = s.subs({"x": 5, "y": 5})
        assert subbed == 0
        assert isinstance(subbed, int)

    def test_subs_kwargs(self, sample_expr):
        assert sample_expr.subs(x=2, z=3) == sample_expr.subs({"x": 2, "z": 3})

    @pytest.mark.parametrize(
        "expr, expected",
        [
            (Expression({}), "0"),
            (Expression({(): 5}), "5"),
            (Expression({("x",): 1}), "x"),
            (Expression({("x",): 3}), "3*x"),
            (Expression({("x", "y"): 2}), "2*x*y"),
        ],
    )
    def test_str(self, expr, expected):
        """Test that the __str__ method returns the expected string representation of the expression."""
        assert str(expr) == expected

    def test_str2(self, sample_expr):
        # Needs to be separate since a fixture can't be used within parametrize
        assert str(sample_expr) == "z*z + x*y + 2*x + 5"

    def test_str3(self, sample_float_expr):
        # Needs to be separate since a fixture can't be used within parametrize
        assert str(sample_float_expr) == "1.5*z*z + x*y + 2.5*x + 5"

    def test_repr(self, sample_expr):
        assert repr(sample_expr) == f"Expression({sample_expr._data})"

    def test_eq(self):
        """Test that the __eq__ method correctly determines equality of expressions."""
        expr1 = Expression({("x",): 1, (): 2})
        expr2 = Expression({("x",): 1, (): 2})
        expr3 = Expression({("x",): 1, (): 3})
        expr4 = Expression({("y",): 1, (): 2})
        assert expr1 == expr2
        assert expr1 != expr3
        assert expr1 != expr4
        assert expr1 != 1
        assert expr1 != 2
        assert expr1 != 3
        assert expr1 != "not an expression"

    def test_eq_with_constants(self):
        # Test comparison of constant-valued Expression with ints
        assert Expression({(): 2}) == 2
        assert Expression({(): 2}) != 3
        assert Expression({}) == 0
        assert Expression({}) != 1
        assert Expression({("x",): 1}) != 1

        # Test comparison of constant-valued Expression with floats
        assert Expression({(): 2.5}) == 2.5
        assert Expression({(): 2.5}) != 2
        assert Expression({}) == 0.0
        assert Expression({}) != 0.5

        # Integral floats compare equal to their int counterparts
        assert Expression({(): 2.0}) == 2
        assert Expression({(): 2}) == 2.0
        assert Expression({(): 2}) == Expression({(): 2.0})
        assert Expression({("x",): 1.0}) == Expression({("x",): 1})

    def test_hash(self):
        """Test that the __hash__ method returns consistent hash values for equal expressions."""
        expr1 = Expression({("x",): 1, (): 2})
        expr2 = Expression({("x",): 1, (): 2})
        expr3 = Expression({("x",): 1, ("y",): 1})

        assert hash(expr1) == hash(expr2)
        assert hash(expr1) == hash(expr3.subs(y=2))

        # Test that hashing remains consistent with the equivalent numeric values
        assert hash(Expression({(): 3})) == hash(3)
        assert hash(Expression({(): 2.5})) == hash(2.5)
        assert hash(Expression({(): 0.0})) == hash(0)
        assert hash(Expression({(): 2.0})) == hash(2)
        assert hash(Expression({("x",): 1.0})) == hash(Expression({("x",): 1}))

        # Consistent hashing and equality means lookups work across numeric types
        assert {Expression({(): 2.5}): "a"}[2.5] == "a"


class TestExpressionMath:
    def test_int(self):
        assert Expression({}).__int__() == 0
        assert Expression({(): 5}).__int__() == 5
        with pytest.raises(ValueError):
            Expression({("x",): 1}).__int__()
        with pytest.raises(ValueError):
            Expression({("x",): 1, ("y",): 1}).__int__()

    def test_add(self):
        expr1 = Expression({("x",): 1, (): 1})
        expr2 = Expression({("y",): 2, (): 2})
        expr3 = expr1 + expr2
        assert expr3._data == {("x",): 1, ("y",): 2, (): 3}
        assert expr1 + expr2 == expr2 + expr1

    def test_add_with_overlapping_vars(self):
        expr1 = Expression({("x",): 1, (): 1})
        expr2 = Expression({("x",): 2, (): 2})
        expr3 = expr1 + expr2
        assert expr3._data == {("x",): 3, (): 3}
        assert expr1 + expr2 == expr2 + expr1

    def test_add_int(self):
        expr = Expression({("x",): 1, (): 2})
        new_expr = expr + 3
        assert new_expr._data == {("x",): 1, (): 5}
        assert expr + 3 == 3 + expr

    def test_add_cancels(self):
        expr1 = Expression({("x",): 1, (): 1})
        expr2 = Expression({("x",): -1, (): 2})
        expr3 = Expression({("x",): -1, (): -1})

        assert expr1 + expr2 == 3
        assert isinstance(expr1 + expr2, int)

        assert expr1 + expr3 == 0
        assert isinstance(expr1 + expr3, int)

    def test_add_casts_to_int(self):
        expr = Expression({(): 2})
        new_expr = expr + 3
        assert isinstance(new_expr, int)
        assert new_expr == 5

        new_expr = expr + Expression({(): 3})
        assert isinstance(new_expr, int)
        assert new_expr == 5

    def test_add_invalid(self, sample_expr):
        # pylint: disable=pointless-statement
        with pytest.raises(TypeError):
            sample_expr + "not an expression"
        with pytest.raises(TypeError):
            "not an expression" + sample_expr

    def test_mul(self):
        expr1 = Expression({("x",): 1, (): 2})
        expr2 = Expression({("y",): 3, (): 4})
        expr3 = expr1 * expr2
        assert expr3._data == {("x", "y"): 3, ("x",): 4, ("y",): 6, (): 8}
        assert expr1 * expr2 == expr2 * expr1

    def test_mul_with_overlapping_vars(self):
        expr1 = Expression({("x",): 1, (): 2})
        expr2 = Expression({("x",): 3, (): 4})
        expr3 = expr1 * expr2
        assert expr3._data == {("x", "x"): 3, ("x",): 10, (): 8}
        assert expr1 * expr2 == expr2 * expr1

    def test_mul_int(self):
        expr = Expression({("x",): 1, (): 2})
        new_expr = expr * 3
        assert new_expr._data == {("x",): 3, (): 6}
        assert expr * 3 == 3 * expr

    def test_mul_zero(self):
        expr = Expression({("x",): 1, (): 2})
        new_expr = expr * 0
        assert isinstance(new_expr, int)
        assert expr * 0 == 0 * expr == 0

    def test_mul_casts_to_int(self):
        expr = Expression({(): 2})
        new_expr = expr * 3
        assert isinstance(new_expr, int)
        assert new_expr == 6

        new_expr = expr * Expression({(): 3})
        assert isinstance(new_expr, int)
        assert new_expr == 6

    def test_mul_invalid(self, sample_expr):
        # pylint: disable=pointless-statement
        with pytest.raises(TypeError):
            sample_expr * "not an expression"
        with pytest.raises(TypeError):
            "not an expression" * sample_expr

    @pytest.mark.parametrize("func", [math.ceil, math.floor, round, math.trunc])
    def test_rounding(self, func):
        """Test that an int-valued expression is unaffected."""
        expr = Expression({("x",): 2, (): 3})
        result = func(expr)
        assert result._data == {("x",): 2, (): 3}

    def test_round_ndigits(self):
        """Test that rounding an int-valued expression with ndigits is unaffected."""
        expr = Expression({("x",): 2, (): 3.14159})
        result = round(expr, ndigits=2)
        assert result._data == {("x",): 2, (): 3.14}


class TestExpressionFloatMath:
    """Test arithmetic on expressions with float coefficients."""

    def test_int_truncates(self):
        """Test that __int__ truncates a float constant term towards zero."""
        result = Expression({(): 2.9}).__int__()
        assert result == 2
        assert isinstance(result, int)

        assert Expression({(): -2.9}).__int__() == -2
        assert int(Expression({(): 2.5})) == 2

        with pytest.raises(ValueError, match="contains variables"):
            int(Expression({("x",): 1.5}))
        with pytest.raises(ValueError, match="more than one term"):
            int(Expression({("x",): 1.5, (): 1.5}))

    def test_float(self):
        """Test that __float__ returns the constant term as a float."""
        assert float(Expression({})) == 0.0
        assert isinstance(float(Expression({})), float)

        assert float(Expression({(): 2.5})) == 2.5

        # Int-valued expressions can also be cast to float
        result = float(Expression({(): 5}))
        assert result == 5.0
        assert isinstance(result, float)

        with pytest.raises(ValueError, match="contains variables"):
            float(Expression({("x",): 1.5}))
        with pytest.raises(ValueError, match="more than one term"):
            float(Expression({("x",): 1.5, (): 1.5}))

    def test_add(self):
        """Test addition of two float-valued expressions."""
        expr1 = Expression({("x",): 1.5, (): 0.5})
        expr2 = Expression({("y",): 2.5, (): 2.0})
        expr3 = expr1 + expr2
        assert expr3._data == {("x",): 1.5, ("y",): 2.5, (): 2.5}
        assert expr1 + expr2 == expr2 + expr1

    def test_add_with_overlapping_vars(self):
        """Test addition of float-valued expressions that share variables."""
        expr1 = Expression({("x",): 1.5, (): 0.5})
        expr2 = Expression({("x",): 2.0, (): 2.0})
        expr3 = expr1 + expr2
        assert expr3._data == {("x",): 3.5, (): 2.5}
        assert expr1 + expr2 == expr2 + expr1

    def test_add_float(self):
        """Test that a float can be added to an int-valued expression."""
        expr = Expression({("x",): 1, (): 2})
        new_expr = expr + 0.5
        assert new_expr._data == {("x",): 1, (): 2.5}
        assert expr + 0.5 == 0.5 + expr

    def test_add_cancels(self):
        """Test that exactly cancelling float terms are removed."""
        expr1 = Expression({("x",): 1.5, (): 0.5})

        result = expr1 + Expression({("x",): -1.5, (): 2.0})
        assert result == 2.5
        assert isinstance(result, float)

        # Every term cancelling yields an int zero, as it does for int-valued expressions
        result = expr1 + Expression({("x",): -1.5, (): -0.5})
        assert result == 0
        assert isinstance(result, int)

    def test_add_casts_to_float(self):
        """Test that adding to a constant float expression yields a float."""
        expr = Expression({(): 2.5})

        new_expr = expr + 1
        assert isinstance(new_expr, float)
        assert new_expr == 3.5

        new_expr = expr + Expression({(): 0.5})
        assert isinstance(new_expr, float)
        assert new_expr == 3.0

    def test_add_invalid(self, sample_float_expr):
        """Test that adding a non-real value still raises a TypeError."""
        # pylint: disable=pointless-statement
        with pytest.raises(TypeError):
            sample_float_expr + "not an expression"
        with pytest.raises(TypeError):
            sample_float_expr + 1.5j

    def test_mul(self):
        """Test multiplication of two float-valued expressions."""
        expr1 = Expression({("x",): 1.5, (): 2.0})
        expr2 = Expression({("y",): 0.5, (): 4.0})
        expr3 = expr1 * expr2
        assert expr3._data == {("x", "y"): 0.75, ("x",): 6.0, ("y",): 1.0, (): 8.0}
        assert expr1 * expr2 == expr2 * expr1

    def test_mul_with_overlapping_vars(self):
        """Test multiplication of float-valued expressions that share variables."""
        expr1 = Expression({("x",): 1.5, (): 2.0})
        expr2 = Expression({("x",): 0.5, (): 4.0})
        expr3 = expr1 * expr2
        assert expr3._data == {("x", "x"): 0.75, ("x",): 7.0, (): 8.0}
        assert expr1 * expr2 == expr2 * expr1

    def test_mul_float(self):
        """Test that an int-valued expression can be scaled by a float."""
        expr = Expression({("x",): 2, (): 3})
        new_expr = expr * 0.5
        assert new_expr._data == {("x",): 1.0, (): 1.5}
        assert expr * 0.5 == 0.5 * expr

    def test_mul_float_zero(self):
        """Test that scaling by 0.0 collapses to an int zero, as scaling by 0 does."""
        expr = Expression({("x",): 1.5, (): 2.5})
        new_expr = expr * 0.0
        assert isinstance(new_expr, int)
        assert expr * 0.0 == 0.0 * expr == 0

    def test_mul_casts_to_float(self):
        """Test that multiplying a constant float expression yields a float."""
        expr = Expression({(): 2.5})

        new_expr = expr * 2
        assert isinstance(new_expr, float)
        assert new_expr == 5.0

        new_expr = expr * Expression({(): 2.0})
        assert isinstance(new_expr, float)
        assert new_expr == 5.0

    def test_mul_underflow_normalizes(self):
        """Test that coefficients underflowing to zero during float scaling are removed."""
        expr = Expression({("x",): 5e-324, (): 2.0})
        new_expr = expr * 0.5
        assert new_expr == 1.0
        assert isinstance(new_expr, float)

    @pytest.mark.parametrize("func", [math.ceil, math.floor, round, math.trunc])
    def test_rounding_no_constant_term(self, func):
        """Test that an expression without a constant term is returned unchanged."""
        expr = Expression({("x",): 1.5, ("y",): 2.5})
        assert func(expr) is expr

    @pytest.mark.parametrize(
        "func, expected",
        [(math.ceil, 3), (math.floor, 2), (round, 2), (math.trunc, 2)],
    )
    def test_rounding_float_constant_casts_to_int(self, func, expected):
        """Test that a constant float expression collapses to an int."""
        result = func(Expression({(): 2.5}))
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize(
        "func, expected",
        [(math.ceil, -2), (math.floor, -3), (round, -2), (math.trunc, -2)],
    )
    def test_rounding_negative_float_constant(self, func, expected):
        """Test that negative constant terms are handled with the correct rounding direction."""
        result = func(Expression({(): -2.5}))
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize(
        "func, expected",
        [(math.ceil, 2), (math.floor, 1), (round, 2), (math.trunc, 1)],
    )
    def test_rounding_only_constant_term_is_affected(self, func, expected):
        """Test that only the constant term is made integral, and the original is not mutated."""
        expr = Expression({("x", "y"): 1.5, ("z",): 2.5, (): 1.5})
        result = func(expr)
        assert result._data == {("x", "y"): 1.5, ("z",): 2.5, (): expected}
        assert result.vars == {"x", "y", "z"}
        assert expr._data == {("x", "y"): 1.5, ("z",): 2.5, (): 1.5}

    @pytest.mark.parametrize(
        "func, coeff",
        [(math.ceil, -0.25), (math.floor, 0.25), (round, 0.25), (math.trunc, 0.25)],
    )
    def test_rounding_zeroed_constant_term_is_dropped(self, func, coeff):
        """Test that a constant term rounding to zero is removed from the expression."""
        expr = Expression({("x",): 1.5, (): coeff})
        result = func(expr)
        assert isinstance(result, Expression)
        assert result._data == {("x",): 1.5}

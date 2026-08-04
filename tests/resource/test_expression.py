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

# pylint: disable=unnecessary-dunder-call,protected-access
import pytest

from pennylane.resource.expression import (
    Expression,
    _term_to_str,
)


@pytest.mark.parametrize(
    "vars, coeff, expected",
    [
        ((), 0, "0"),
        ((), 1, "1"),
        ((), 5, "5"),
        (("x",), 1, "x"),
        (("x",), 3, "3*x"),
        (("x", "x"), 1, "x*x"),
        (("x", "y"), 1, "x*y"),
        (("x", "y"), 2, "2*x*y"),
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


class TestExpression:
    """Test the methods and attributes of the Expression class"""

    def test_init_as_int(self):
        """Test that the __init__ method can handle an integer input for a constant expression."""
        expr = Expression(5)
        assert expr._data == {(): 5}
        assert expr.vars == set()

        expr = Expression(0)
        assert expr._data == {}

    def test_init_invalid_input(self):
        """Test that the __init__ method raises a TypeError for invalid input types."""
        with pytest.raises(TypeError):
            Expression("foo")

    def test_init_skip_normalization(self):
        """Test that the __init__ method can skip normalization when _skip_normalization is True."""
        data = {("x", "y"): 2, ("y", "x"): 3, ("z",): 1, ("foo",): 0, (): 4}
        expr = Expression(data, _skip_normalization=True)
        assert expr._data == data

    def test_init_normalizes(self):
        """Test that the __init__ method normalizes the input data correctly."""
        expr = Expression(
            {
                ("x", "y"): 2,  # Should be combined with the next term
                ("y", "x"): 3,
                ("z",): 1,
                ("foo",): 0,  # Should disappear
                (): 4,
            }
        )
        assert expr._data == {
            ("x", "y"): 5,
            ("z",): 1,
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

        # Test comparison of constant-valued Expression with ints
        assert Expression({(): 2}) == 2
        assert Expression({(): 2}) != 3
        assert Expression({}) == 0
        assert Expression({}) != 1


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
        assert new_expr._data == {}
        assert expr * 0 == 0 * expr == Expression({})

    def test_mul_invalid(self, sample_expr):
        # pylint: disable=pointless-statement
        with pytest.raises(TypeError):
            sample_expr * "not an expression"
        with pytest.raises(TypeError):
            "not an expression" * sample_expr

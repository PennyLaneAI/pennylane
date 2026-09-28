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
Helper classes and functions for symbolic resource expressions.
"""

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from math import ceil, floor, trunc
from numbers import Real
from typing import Union

# Type objects that represent either an Expression or a numeric constant
IntExprResult = Union["Expression", int]
ExprResult = Union["Expression", Real]


def _cast_if_constant(
    new_data: dict[tuple[str, ...], Real],
    vars: set[str],
    skip_copy: bool,
    skip_normalization: bool,
) -> ExprResult:
    """Collapse the new data for creating an Expression into a constant if possible.

    Args:
        new_data (dict[tuple[str, ...], Real]): The new data for the Expression.
        vars (set[str]): The set of variables in the Expression.
        skip_copy (bool): Whether to skip copying the new data when creating the Expression.
        skip_normalization (bool): Whether to skip normalization when creating the Expression.

    Returns:
        Expression | Real: An real number if the result is a constant, otherwise a new
        :class:`~.resource.Expression` instance. The type of a constant result matches the type of
        the corresponding coefficient.
    """
    if len(new_data) == 0:
        return 0
    if len(new_data) == 1 and () in new_data:
        # Return as int/float rather than Expression if the result is a constant
        return new_data[()]
    return Expression(
        new_data, vars=vars, _skip_copy=skip_copy, _skip_normalization=skip_normalization
    )


@lru_cache
def _term_to_str(vars: tuple[str, ...], coeff: Real) -> str:
    if isinstance(coeff, (float, Fraction)) and coeff.is_integer():
        coeff = int(coeff)
    if not vars:
        return str(coeff)
    if coeff == 1:
        return "*".join(vars)
    return f"{coeff}*{'*'.join(vars)}"


class Expression:
    """
    Internal class for representing symbolic expressions of resources.
    Specifically, each expression is a polynomial in the variables with real (numeric)
    coefficients, where the variables represent symbolic parameters of the resources.
    The expressions are represented as a dictionary mapping tuples of variable names to their
    coefficients.

    .. warning::

        This class is intended to be immutable. Do not modify the internal state of an expression
        after it is created, as this may lead to incorrect behaviour.
    """

    __slots__ = ("_hashval", "_str", "_repr", "_data", "_vars")

    _data: dict[tuple[str, ...], Real]
    _vars: set[str]

    def __init__(
        self,
        data: dict[tuple[str, ...], Real] | Real,
        vars: set[str] | None = None,
        _skip_copy: bool = False,
        _skip_normalization: bool = False,
    ) -> None:
        """
        Initializes the expression with the given data.

        Args:
            data (dict[tuple[str, ...], Real] | Real): A dictionary mapping tuples of
                variable names to their coefficients, or a real number for a constant expression.
            vars (set[str] | None): An optional set of variables that appear in the expression.
                These must be a superset of the variables that appear in the keys of the data
                dictionary. If not provided, the variables will be inferred from the keys of the
                data dictionary.
        """
        if not isinstance(data, (dict, Real)):
            raise TypeError("Expression data must be a dictionary of tuples or a real number")

        self._hashval = None
        self._str = None
        self._repr = None

        if isinstance(data, Real):
            if data == 0:
                self._data = {}
            else:
                self._data = {(): data}
        elif _skip_copy:
            self._data = data
        else:
            self._data = data.copy()

        if not _skip_normalization:
            self._normalize()
            # Sort order of variable tuples for deterministic display and testing
            self._data = {
                vars: self._data[vars]
                for vars in sorted(
                    self._data.keys(),
                    reverse=True,
                    key=lambda var_tuple: (len(var_tuple), var_tuple),
                )
            }

        if vars is not None:
            self._vars = vars
        else:
            self._vars = frozenset(var for vars in self._data.keys() for var in vars)

    def _normalize(self) -> None:
        """
        Normalizes the expression by sorting the variable tuples and combining like terms.
        Also removes any terms with a zero coefficient.
        """
        for vars in list(self._data.keys()):
            if not isinstance(self._data[vars], Real):
                raise TypeError(
                    f"Expression coefficients must be real numbers, got '{self._data[vars]}'"
                )
            if self._data[vars] == 0:
                del self._data[vars]
                continue
            sorted_vars = tuple(sorted(vars))
            if sorted_vars != vars:
                if sorted_vars not in self._data:
                    self._data[sorted_vars] = self._data[vars]
                else:
                    self._data[sorted_vars] += self._data[vars]
                    if self._data[sorted_vars] == 0:
                        del self._data[sorted_vars]
                del self._data[vars]

    @property
    def vars(self) -> set[str]:
        """
        The set of variables that appear in the expression.
        """
        return self._vars

    def subs(self, substitutions: dict[str, Real] | None = None, **kwargs) -> ExprResult:
        """
        Substitutes the given values for the variables in the expression.

        Args:
            substitutions (dict[str, Real] | None): A dictionary mapping variable names to their
                values. If None, an empty dictionary is used. Additional keyword arguments can
                also be provided as substitutions.

        Returns:
            Expression | Real: A new expression with the variables substituted, or a number if
            the result is a constant.
        """
        if substitutions is None:
            substitutions = {}

        # NOTE: don't mutate incoming dict
        substitutions_copy = {**substitutions, **kwargs}

        new_data = defaultdict(int)
        for vars, coeff in self._data.items():
            new_k = []
            mult = 1
            for var in vars:
                if var in substitutions_copy:
                    mult *= substitutions_copy[var]
                else:
                    new_k.append(var)

            new_k = tuple(new_k)
            new_data[new_k] += coeff * mult

        return _cast_if_constant(
            new_data,
            vars=self._vars.difference(substitutions_copy.keys()),
            skip_copy=False,
            skip_normalization=False,
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the expression.

        The format of this string is a sum of terms, where each term is of the form
        "coeff*var1*var2*...". If a term has no variables, it is just the coefficient.
        If a term has a coefficient of 1, the coefficient is omitted.
        If the expression is zero, it is "0".
        """
        if self._str is not None:
            return self._str
        if len(self._data) == 0:
            self._str = "0"
        else:
            self._str = " + ".join(
                [_term_to_str(vars, coeff) for vars, coeff in self._data.items()]
            )
        return self._str

    def __repr__(self) -> str:
        if self._repr is None:
            self._repr = f"Expression({self._data})"
        return self._repr

    def __eq__(self, other) -> bool:
        if not isinstance(other, (Expression, Real)):
            return NotImplemented
        if isinstance(other, Real):
            match len(self._data):
                case 0:
                    return other == 0
                case 1 if () in self._data:
                    return self._data[()] == other
                case _:
                    return False
        return self._data == other._data

    def __hash__(self) -> int:
        # NOTE: `lru_cache` and related methods can't be used here since they rely on a hash value existing
        if self._hashval is None:
            # Extra cases to make sure that numeric hashes are consistent with Expression hashes
            if len(self._data) == 0:
                self._hashval = hash(0)
            elif len(self._data) == 1 and () in self._data:
                self._hashval = hash(self._data[()])
            else:
                self._hashval = hash(frozenset(self._data.items()))

        return self._hashval

    def __int__(self) -> int:
        if len(self._data) == 0:
            return 0
        if len(self._data) > 1:
            raise ValueError("Expression cannot be converted to int, more than one term")
        if () not in self._data:
            raise ValueError("Expression cannot be converted to int, contains variables")
        return int(self._data[()])

    def __float__(self) -> float:
        if len(self._data) == 0:
            return 0.0
        if len(self._data) > 1:
            raise ValueError("Expression cannot be converted to float, more than one term")
        if () not in self._data:
            raise ValueError("Expression cannot be converted to float, contains variables")
        return float(self._data[()])

    def __mul__(self, other) -> ExprResult:
        if not isinstance(other, (Expression, Real)):
            return NotImplemented

        if isinstance(other, Real):
            if other == 0:
                return 0
            # Scaling by a non-zero int can never zero out a coefficient, but float
            # multiplication can underflow to zero. Such terms are dropped here to preserve the
            # invariant that zero coefficients are never stored.
            return _cast_if_constant(
                {
                    vars: new_coeff
                    for vars, coeff in self._data.items()
                    if (new_coeff := coeff * other) != 0
                },
                vars=self._vars,
                skip_copy=True,
                skip_normalization=True,
            )

        new_data = defaultdict(int)
        for vars1, coeff1 in self._data.items():
            for vars2, coeff2 in other._data.items():
                new_data[vars1 + vars2] += coeff1 * coeff2
        return _cast_if_constant(
            new_data, self._vars.union(other._vars), skip_copy=False, skip_normalization=False
        )

    def __rmul__(self, other) -> ExprResult:
        return self.__mul__(other)

    def __add__(self, other) -> ExprResult:
        if not isinstance(other, (Expression, Real)):
            return NotImplemented

        vars = self._vars
        new_data = self._data.copy()

        if isinstance(other, Real):
            new_data[()] = new_data.get((), 0) + other
        else:
            for other_vars, coeff in other._data.items():
                new_val = new_data.get(other_vars, 0) + coeff
                if new_val == 0:
                    del new_data[other_vars]
                else:
                    new_data[other_vars] = new_val
            vars = vars.union(other._vars)

        return _cast_if_constant(new_data, vars, skip_copy=True, skip_normalization=False)

    def __radd__(self, other) -> ExprResult:
        return self.__add__(other)

    def _builtin_to_int_helper(self, func) -> ExprResult:
        """Helper function to implement ``__floor__``, ``__ceil__``, ``__round__``, and
        ``__trunc__`` for :class:`~.resource.Expression` objects.

        Since the contents of the object are symbolic, it is not possible to fully cast to an int.
        Instead, the chosen function is applied only to the constant term of the expression, if it
        exists. If the expression has no constant term, the function has no effect.

        Args:
            func (Callable[[Real], int | float]): The function to apply to the constant term.

        Returns:
            Expression | int | float: An int if the result is a constant, otherwise a
            :class:`~.resource.Expression` instance where the constant term is integral.
            May also return a float if using ``round`` with ``ndigits`` specified.
        """
        if len(self._data) == 0:
            return 0
        if () not in self._data:
            return self

        new_data = self._data.copy()
        new_data[()] = func(new_data[()])
        return _cast_if_constant(
            new_data, vars=self._vars, skip_copy=True, skip_normalization=False
        )

    def __ceil__(self) -> IntExprResult:
        return self._builtin_to_int_helper(ceil)

    def __floor__(self) -> IntExprResult:
        return self._builtin_to_int_helper(floor)

    def __round__(self, ndigits=None) -> ExprResult:
        return self._builtin_to_int_helper(lambda x: round(x, ndigits))

    def __trunc__(self) -> IntExprResult:
        return self._builtin_to_int_helper(trunc)

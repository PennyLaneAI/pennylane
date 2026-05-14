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
from functools import cache


def _term_to_str(vars: tuple[str, ...], coeff: int) -> str:
    if not vars:
        return str(coeff)
    elif coeff == 1:
        return "*".join(vars)
    else:
        return f"{coeff}*{'*'.join(vars)}"


class Expression:
    """
    Internal class for representing symbolic expressions of resources.
    Specifically, each expression is an integral polynomial in the variables, where the variables
    represent symbolic parameters of the resources. The expressions are represented as a dictionary
    mapping tuples of variable names to their coefficients.

    NOTE: This class is intended to be immutable. Do not modify the internal state of an expression
    after it is created, as this may lead to incorrect behavior.
    """

    __slots__ = ("_hashval", "_data")

    _data: dict[tuple[str, ...], int]

    def __init__(
        self,
        data: dict[tuple[str, ...], int],
        _skip_copy: bool = False,
        _skip_normalization: bool = False,
    ) -> None:
        """
        Initializes the expression with the given data.

        Args:
            data (dict[tuple[str, ...], int]): A dictionary mapping tuples of variable names to their coefficients.
        """
        self._hashval = None
        if _skip_copy:
            self._data = data
        else:
            self._data = data.copy()
        if not _skip_normalization:
            self._normalize()

    def _normalize(self) -> None:
        """
        Normalizes the expression by sorting the variable tuples and combining like terms.
        Also removes any terms with a zero coefficient.
        """
        for vars in list(self._data.keys()):
            if self._data[vars] == 0:
                del self._data[vars]
                continue
            sorted_vars = tuple(sorted(vars))
            if sorted_vars != vars:
                if sorted_vars not in self._data:
                    self._data[sorted_vars] = 0
                self._data[sorted_vars] += self._data[vars]
                del self._data[vars]

    @property
    @cache
    def vars(self) -> set[str]:
        """
        Returns the set of variables that appear in the expression.

        Returns:
            set[str]: The set of variable names that appear in the expression.
        """
        return set(var for vars in self._data.keys() for var in vars)

    def subs(self, substitutions: dict[str, int]) -> "Expression":
        """
        Substitutes the given values for the variables in the expression.

        Args:
            substitutions (dict[str, int]): A dictionary mapping variable names to their values.

        Returns:
            Expression: A new expression with the variables substituted.
        """
        new_data = defaultdict(int)
        for vars, coeff in self._data.items():
            new_k = []
            mult = 1
            for var in vars:
                if var in substitutions:
                    mult *= substitutions[var]
                else:
                    new_k.append(var)

            new_k = tuple(new_k)
            new_data[new_k] += coeff * mult

        if len(new_data) == 0:
            return 0
        if len(new_data) == 1 and () in new_data:
            return new_data[()]
        return Expression(new_data, _skip_normalization=True)

    @cache
    def __str__(self) -> str:
        """
        Returns a string representation of the expression.

        The format of this string is a sum of terms, where each term is of the form
        "coeff*var1*var2*...". If a term has no variables, it is just the coefficient.
        If a term has a coefficient of 1, the coefficient is omitted.
        If the expression is zero, it is "0".
        """
        if len(self._data) == 0:
            return "0"
        return " + ".join([_term_to_str(vars, coeff) for vars, coeff in self._data.items()])

    @cache
    def __repr__(self) -> str:
        return f"Expression({self._data})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, (int, Expression)):
            return False
        if isinstance(other, int):
            return len(self._data) == 1 and () in self._data and self._data[()] == other
        return self._data == other._data

    def __hash__(self) -> int:
        # NOTE: `lru_cache` and related methods can't be used here since they rely on a hash value existing
        if self._hashval is None:
            self._hashval = hash(frozenset(self._data.items()))

        return self._hashval

    def __int__(self) -> int:
        if len(self._data) == 0:
            return 0
        if len(self._data) > 1:
            raise ValueError("Expression cannot be converted to int, more than one term")
        if () not in self._data:
            raise ValueError("Expression cannot be converted to int, contains variables")
        return self._data[()]

    def __mul__(self, other) -> "Expression":
        if not isinstance(other, (int, Expression)):
            raise TypeError("Invalid type for multiplication with expression: " + str(type(other)))

        if isinstance(other, int):
            return Expression(
                {vars: coeff * other for vars, coeff in self._data.items()},
                _skip_copy=True,
                _skip_normalization=True,
            )

        new_data = defaultdict(int)
        for vars1, coeff1 in self._data.items():
            for vars2, coeff2 in other._data.items():
                new_data[vars1 + vars2] += coeff1 * coeff2
        return Expression(dict(new_data))

    def __rmul__(self, other) -> "Expression":
        return self.__mul__(other)

    def __add__(self, other) -> "Expression":
        if not isinstance(other, (int, Expression)):
            raise TypeError("Invalid type for addition with expression: " + str(type(other)))

        if isinstance(other, int):
            new_data = self._data.copy()
            new_data[()] = new_data.get((), 0) + other
            return Expression(new_data, _skip_copy=True, _skip_normalization=True)

        new_data = defaultdict(int, self._data)
        for vars, coeff in other._data.items():
            new_data[vars] += coeff
        return Expression(new_data, _skip_normalization=True)

    def __radd__(self, other) -> "Expression":
        return self.__add__(other)

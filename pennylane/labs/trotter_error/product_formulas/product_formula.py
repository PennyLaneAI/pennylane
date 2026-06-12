# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The ProductFormula class"""

from __future__ import annotations

import copy
from collections.abc import Hashable
from typing import Sequence


class ProductFormula:
    r"""A symbolic representation of a product formula for Hamiltonian simulation.

    Product formulas (also called Trotter--Suzuki formulas) approximate the
    time-evolution operator of a Hamiltonian
    :math:`H = \sum_j H_j` by a product of exponentials of the individual
    fragments :math:`H_j`. For example, the first-order (Lie--Trotter) formula is

    .. math::

        e^{-i H t} \approx \prod_j e^{-i H_j t},

    and the symmetric second-order (Strang) formula is

    .. math::

        e^{-i H t} \approx \Big(\prod_j e^{-i H_j t / 2}\Big)
                           \Big(\prod_j e^{-i H_j t / 2}\Big)^{\!\dagger}.

    This class stores such a formula symbolically rather than numerically: each
    factor :math:`e^{c_j H_j}` is recorded as a ``(symbol, coefficient)`` pair,
    where ``symbol`` identifies the fragment :math:`H_j` and ``coefficient`` is
    the (possibly complex) splitting coefficient :math:`c_j` multiplying it. The
    overall formula may additionally be raised to a power via ``exponent`` to
    represent repeated Trotter steps, e.g. :math:`\big(S(t/r)\big)^r`.

    Symbols may themselves be ``ProductFormula`` instances, allowing higher-order
    formulas to be composed recursively from lower-order ones (see
    :meth:`prod`).

    Args:
        symbols (Sequence[tuple[Hashable, complex]]): an ordered sequence of
            ``(symbol, coefficient)`` pairs. Each ``symbol`` is a hashable object
            naming a Hamiltonian fragment and each ``coefficient`` is the scalar
            multiplying it in the exponent. Order is significant because
            fragment exponentials generally do not commute.
        exponent (float): the power to which the entire formula is raised,
            typically the number of repeated steps. Defaults to ``1.0``.

    Raises:
        TypeError: if any symbol is not hashable, any coefficient is not an
            ``int``, ``float``, or ``complex``, the ``exponent`` is not an
            ``int`` or ``float``, or any symbol is itself a ``ProductFormula``
            (recursive formulas must be built with :meth:`prod`).

    **Example**

    A first-order formula for :math:`H = H_A + H_B`:

    >>> pf = ProductFormula([("A", 1.0), ("B", 1.0)])
    >>> pf
    PF[(('A', 1.0), ('B', 1.0))]**1.0

    A symmetric second-order formula:

    >>> sym = ProductFormula([("A", 0.5), ("B", 1.0), ("A", 0.5)])
    >>> sym.is_symmetric
    True
    """

    def __init__(self, symbols: Sequence[tuple[Hashable, complex]], exponent: float = 1.0):
        if not all(isinstance(symbol[0], Hashable) for symbol in symbols):
            raise TypeError("Symbols must be Hashable objects.")
        if not all(isinstance(symbol[1], (int, float, complex)) for symbol in symbols):
            raise TypeError("Coefficients must be int, float, or complex.")
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be int or float.")
        if any(isinstance(symbol[0], ProductFormula) for symbol in symbols):
            raise TypeError("Symbols cannot be type `ProductFormula`.")

        symbols, coeffs = zip(*symbols)
        self.symbols, self.coeffs = list(symbols), list(coeffs)

        self.exponent = exponent
        self._recursive = False

    @property
    def symbol_set(self) -> set:
        """The set of all base fragment symbols appearing in the formula.

        For a recursive formula (one whose symbols are themselves
        ``ProductFormula`` instances), the base symbols are collected by
        descending into every sub-formula, so the returned set contains only
        leaf-level (non-``ProductFormula``) symbols.

        Returns:
            set: the distinct fragment symbols referenced by this formula.
        """
        symbol_set = set()

        for symbol in self.symbols:
            if isinstance(symbol, ProductFormula):
                symbol_set = set.union(symbol_set, symbol.symbol_set)
            else:
                symbol_set.add(symbol)

        return symbol_set

    @property
    def is_recursive(self) -> bool:
        """Whether the formula is composed of sub-formulas.

        Returns ``True`` if this formula was built from other
        ``ProductFormula`` instances via :meth:`prod`, and ``False`` for a flat
        formula constructed directly from ``(symbol, coefficient)`` pairs.

        Returns:
            bool: ``True`` if the symbols are themselves product formulas.
        """
        return self._recursive

    @property
    def is_symmetric(self) -> bool:
        """Whether the formula is palindromic (symmetric).

        A product formula is symmetric when its ordered factors read the same
        forwards and backwards, i.e. both the ``symbols`` and ``coeffs``
        sequences are palindromes. Symmetric formulas (such as the Strang
        splitting) are even-order and self-inverse up to time reversal, which is
        what gives them their improved error scaling.

        Note:
            The check is performed only on the top-level factor ordering; it
            does not recurse into sub-formulas.

        Returns:
            bool: ``True`` if the symbol and coefficient sequences are
            palindromic.
        """
        return self.symbols == self.symbols[::-1] and self.coeffs == self.coeffs[::-1]

    @classmethod
    def prod(cls, product_formulas: Sequence[ProductFormula]) -> ProductFormula:
        """Compose several product formulas into a single recursive formula.

        This represents the ordered product of the given formulas,
        :math:`S = S_1 S_2 \\cdots S_n`, by storing them as the symbols of a new
        ``ProductFormula``. The resulting formula is flagged as recursive (see
        :attr:`is_recursive`), each sub-formula is assigned a unit coefficient,
        and the overall ``exponent`` is set to ``1.0``.

        Unlike the constructor, this bypasses ``__init__`` and so permits
        ``ProductFormula`` instances as symbols; it is the supported way to nest
        formulas (e.g. building higher-order Suzuki recursions from lower-order
        ones).

        Args:
            product_formulas (Sequence[ProductFormula]): the ordered sequence of
                formulas to compose. Order is significant.

        Raises:
            TypeError: if any element is not a ``ProductFormula``.

        Returns:
            ProductFormula: a recursive formula whose symbols are (deep copies
            of) the supplied formulas.
        """
        if not all(isinstance(pf, ProductFormula) for pf in product_formulas):
            raise TypeError("Product formulas must be type `ProductFormula`.")

        product_formula = super().__new__(cls)
        product_formula.symbols = copy.deepcopy(product_formulas)
        product_formula.coeffs = [1] * len(product_formulas)
        product_formula.exponent = 1.0
        product_formula._recursive = True

        return product_formula

    def __call__(self, t: float) -> ProductFormula:
        """Evaluate the formula at evolution time ``t``.

        Scaling by the simulation time ``t`` propagates into the splitting
        coefficients. For a flat formula, every coefficient :math:`c_j` is
        replaced by :math:`t\\, c_j`. For a recursive formula, ``t`` is passed
        down to each sub-formula instead (the recursive formula's own unit
        coefficients are left unchanged). The original formula is not mutated; a
        deep copy is returned.

        Args:
            t (float): the evolution time to substitute into the formula.

        Returns:
            ProductFormula: a new formula evaluated at time ``t``.
        """
        pf = copy.deepcopy(self)

        if pf.is_recursive:
            pf.symbols = [symbol(t) for symbol in pf.symbols]
            return pf

        pf.coeffs = [t * coeff for coeff in pf.coeffs]

        return pf

    def __eq__(self, other: ProductFormula) -> bool:
        """Whether two formulas are identical.

        Two formulas compare equal when they have the same ordered ``symbols``,
        the same ordered ``coeffs``, and the same ``exponent``.

        Args:
            other (ProductFormula): the formula to compare against.

        Returns:
            bool: ``True`` if the formulas are equal.
        """
        if self.symbols != other.symbols:
            return False

        if self.coeffs != other.coeffs:
            return False

        return self.exponent == other.exponent

    def __hash__(self) -> int:
        """Return a hash based on the ordered symbols and coefficients.

        Returns:
            int: a hash of the ``(symbols, coeffs)`` pair.
        """
        return hash((tuple(self.symbols), tuple(self.coeffs)))

    def __pow__(self, z: float) -> ProductFormula:
        """Raise the formula to the power ``z``.

        This scales the existing :attr:`exponent` by ``z`` (the powers
        multiply), modelling repetition of the formula such as the ``r`` Trotter
        steps in :math:`\\big(S(t/r)\\big)^r`. The original formula is not
        mutated; a deep copy is returned.

        Args:
            z (float): the power to raise the formula to.

        Raises:
            TypeError: if ``z`` is not an ``int`` or ``float``.

        Returns:
            ProductFormula: a new formula with exponent ``z * self.exponent``.
        """
        if not isinstance(z, (int, float)):
            raise TypeError("Exponent must be int or float.")

        ret = copy.deepcopy(self)
        ret.exponent = z * self.exponent

        return ret

    def __repr__(self) -> str:
        """Return a string representation of the formula.

        Returns:
            str: a representation of the form
            ``PF[((symbol, coeff), ...)]**exponent``.
        """
        return f"PF[{tuple(zip(self.symbols, self.coeffs))}]**{self.exponent}"

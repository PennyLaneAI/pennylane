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
r"""Operator classes for the GAN Hamiltonian and its fragments.

This module defines the data structures that represent a GAN Hamiltonian (or a
fragment of one) so it can be consumed by the Trotter-error machinery. The GAN
Hamiltonian mixes a fermionic (electronic) part with bosonic (nuclear)
vibrational degrees of freedom, so an operator is stored as a mapping from a
fermionic string to a coefficient that is itself a polynomial in the nuclear
position/momentum functions.

The class hierarchy is built bottom-up:

* :class:`FuncSymbol` --- a single nuclear function (a power of a position or
  momentum on one mode, or the identity).
* :class:`GanMonomial` --- an ordered product of :class:`FuncSymbol` factors,
  with a canonical normal ordering.
* :class:`GanCoeff` --- a linear combination of monomials; this plays the role
  of a (operator-valued) scalar multiplying a fermionic string.
* :class:`GanFragment` --- a :class:`~.Fragment` represented as a mapping from
  :class:`~.FermiWord` to :class:`GanCoeff`, i.e. the full GAN operator.

Each class exposes a :meth:`norm` method returning an upper bound on the
spectral norm (used for Trotter-error bounds), and the arithmetic dunder methods
needed to add and multiply operators.
"""

from __future__ import annotations

import math

from collections import defaultdict, Counter
from collections.abc import Hashable
from dataclasses import dataclass
from enum import IntEnum
from functools import cache
from typing import Sequence

import numpy as np

from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.fragments.gan_fragments.fermi import FermiWord

class FuncType(IntEnum):
    """The kind of nuclear function: momentum, position, or identity.

    Defined as an :class:`~enum.IntEnum` so the values impose a total order,
    which the monomial normal ordering uses to sort same-mode factors.
    """

    MOMENTUM = 0
    POSITION = 1
    IDENTITY = 2


@dataclass(frozen=True)
class FuncSymbol:
    """A single nuclear function appearing in a :class:`GanMonomial`.

    Represents one factor such as a position :math:`Q(\\text{mode})`, a momentum
    :math:`P(\\text{mode})`, or the identity, raised to an integer power. The
    dataclass is frozen so symbols are hashable and usable inside monomials.

    Args:
        f_type (FuncType): whether this is a momentum, position, or identity
            function.
        mode (int): the vibrational mode the function acts on.
        symbol (Hashable): the display symbol for the function (e.g. ``"P"``,
            ``"Q"``, ``"I"``).
        exponent (int): the power the function is raised to. Defaults to ``1``.
    """

    f_type: FuncType
    mode: int
    symbol: Hashable
    exponent: int = 1

    @staticmethod
    def momentum(mode, exponent=2):
        """Return a momentum function on the given mode.

        Args:
            mode (int): the vibrational mode.
            exponent (int): the power of the momentum. Defaults to ``2``.

        Returns:
            FuncSymbol: the momentum function.
        """
        return FuncSymbol(FuncType.MOMENTUM, mode, symbol="P", exponent=exponent)

    @staticmethod
    def position(mode, exponent=1):
        """Return a position function on the given mode.

        Args:
            mode (int): the vibrational mode.
            exponent (int): the power of the position. Defaults to ``1``.

        Returns:
            FuncSymbol: the position function.
        """
        return FuncSymbol(FuncType.POSITION, mode, symbol="Q", exponent=exponent)

    @staticmethod
    def identity():
        """Return the identity function.

        Returns:
            FuncSymbol: the identity function (mode ``1``, exponent ``1``).
        """
        return FuncSymbol(FuncType.IDENTITY, mode=1, symbol="I", exponent=1)

    def base(self):
        """Return the identity of the function ignoring its exponent.

        Two factors with the same base can be merged by adding their exponents
        (see monomial simplification).

        Returns:
            tuple: the ``(f_type, mode, symbol)`` triple.
        """
        return (self.f_type, self.mode, self.symbol)

    def __repr__(self):

        if self.f_type == FuncType.IDENTITY:
            return "I"

        if self.exponent == 1:
            return f"{self.symbol}({self.mode})"

        return f"{self.symbol}({self.mode})^{self.exponent}"

    def norm(self, gridpoints: int) -> float:
        r"""Return an upper bound on the spectral norm of the function.

        Position and momentum functions are bounded on a grid of the given size
        by :math:`\sqrt{\text{gridpoints}\,\pi/2}` times the exponent; the
        identity has norm one.

        Args:
            gridpoints (int): the number of grid points discretizing the
                vibrational mode.

        Returns:
            float: the norm bound for this function.
        """

        match self.f_type:
            case FuncType.POSITION:
                return math.sqrt(gridpoints * math.pi / 2) * self.exponent

            case FuncType.MOMENTUM:
                return math.sqrt(gridpoints * math.pi / 2) * self.exponent

            case FuncType.IDENTITY:
                return 1


class GanMonomial:
    """An ordered product of nuclear functions (:class:`FuncSymbol` factors).

    A monomial is a single basis term in the nuclear (vibrational) degrees of
    freedom, e.g. :math:`Q(0)\\,P(1)^2`. It supports a canonical normal ordering
    (:meth:`normal_order`) so that equivalent products compare and hash equally.

    The hash is computed once at construction; when monomials are produced by
    multiplication a precomputed hash may be supplied to avoid recomputation.

    Args:
        funcs (Sequence[FuncSymbol]): the ordered function factors.
        _hash (int): an optional precomputed hash. If ``0`` (the default), the
            hash is computed from ``funcs``.
    """

    def __init__(self, funcs: Sequence[FuncSymbol], _hash: int = 0):

        self.funcs = funcs
        self._hash = hash(tuple(self.funcs)) if _hash == 0 else _hash

    def normal_order(self) -> GanMonomial:
        """Return the monomial in its canonical normal ordering.

        Sorts commuting factors into a fixed order and merges repeated factors
        by summing exponents (see the module-level ``_normal_order`` helper).

        Returns:
            GanMonomial: the normal-ordered monomial.
        """
        return _normal_order(self)

    @staticmethod
    def identity():
        """Return the identity monomial (the empty product).

        Returns:
            GanMonomial: a monomial with no factors.
        """
        return GanMonomial([])

    def __matmul__(self, other: GanMonomial):
        """Concatenate two monomials into their product.

        Args:
            other (GanMonomial): the monomial to multiply on the right.

        Returns:
            GanMonomial: the product monomial (not yet normal-ordered).
        """
        return GanMonomial(self.funcs + other.funcs, hash((self, other)))

    def __eq__(self, other: GanMonomial):
        """Whether two monomials have identical ordered factors."""
        return self.funcs == other.funcs

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return str(self.funcs)

    def norm(self, gridpoints: int) -> float:
        """Return an upper bound on the spectral norm of the monomial.

        Computed as the product of the per-factor norm bounds.

        Args:
            gridpoints (int): the number of grid points per vibrational mode.

        Returns:
            float: the norm bound for the monomial.
        """
        return math.prod(abs(func.norm(gridpoints)) for func in self.funcs)


class GanCoeff:
    """A linear combination of :class:`GanMonomial` objects.

    A ``GanCoeff`` acts as an (operator-valued) coefficient multiplying a
    fermionic string in a :class:`GanFragment`: it is a polynomial in the
    nuclear position/momentum functions. Monomials with coefficient below a
    small tolerance are dropped at construction so the representation stays
    sparse.

    Args:
        monomials (dict[GanMonomial, float]): the monomial-to-coefficient
            mapping.
    """

    def __init__(self, monomials: dict[GanMonomial, float]):
        self.monomials = {monomial: coeff for monomial, coeff in monomials.items() if abs(coeff) > 1e-08}

    def __add__(self, other: GanCoeff):
        """Add two coefficients by summing shared monomials.

        Args:
            other (GanCoeff): the coefficient to add.

        Returns:
            GanCoeff: the summed coefficient.
        """
        combined = Counter(self.monomials)
        combined.update(other.monomials)

        return GanCoeff(combined)

    def __sub__(self, other: GanCoeff):
        """Subtract ``other`` from this coefficient.

        Args:
            other (GanCoeff): the coefficient to subtract.

        Returns:
            GanCoeff: the difference.
        """
        return self + (-1)*other

    def __mul__(self, scalar: float):
        """Scale every monomial coefficient by ``scalar`` (dropping zeros).

        Args:
            scalar (float): the scalar multiplier.

        Returns:
            GanCoeff: the scaled coefficient.
        """
        return GanCoeff(
            {
                key: scalar * value
                for key, value in self.monomials.items()
                if not np.isclose(scalar * value, 0)
            }
        )

    __rmul__ = __mul__

    def __matmul__(self, other: GanCoeff):
        """Multiply two coefficients (polynomial product).

        Distributes over all monomial pairs, multiplying the monomials (with
        normal ordering, via a cached product) and their scalar coefficients.

        Args:
            other (GanCoeff): the right operand.

        Returns:
            GanCoeff: the product coefficient.
        """
        d = {}

        for l_key, l_value in self.monomials.items():
            for r_key, r_value in other.monomials.items():
                new_key = _cached_mon_matmul(l_key, r_key)
                coeff = l_value * r_value
                d[new_key] = d.get(new_key, 0) + coeff

        return GanCoeff(d)

    @staticmethod
    def identity():
        """Return the empty (zero) coefficient.

        Used as the ``defaultdict`` factory when accumulating fragment terms.

        Returns:
            GanCoeff: a coefficient with no monomials.
        """
        return GanCoeff({})

    def is_zero(self):
        """Whether the coefficient has no surviving monomials.

        Returns:
            bool: ``True`` if the coefficient is empty.
        """
        return len(self.monomials) == 0

    def __repr__(self):
        return " + ".join(f"{coeff}*{monomial}" for monomial, coeff in self.monomials.items())

    def __eq__(self, other):
        """Whether two coefficients have the same monomials and (close) values."""

        if self.monomials.keys() != other.monomials.keys():
            return False

        return all(np.isclose(self.monomials[k], other.monomials[k]) for k in self.monomials.keys())

    def norm(self, gridpoints: int) -> float:
        """Return a triangle-inequality upper bound on the spectral norm.

        Computed as the sum over monomials of the absolute coefficient times the
        monomial norm bound.

        Args:
            gridpoints (int): the number of grid points per vibrational mode.

        Returns:
            float: the norm bound for the coefficient.
        """
        return sum(abs(coeff * monomial.norm(gridpoints)) for monomial, coeff in self.monomials.items())


class GanFragment(Fragment):
    """A GAN Hamiltonian (or fragment) as a mapping of fermionic strings to coefficients.

    Implements the :class:`~.Fragment` interface required by the Trotter-error
    module. The operator is stored as a dictionary from :class:`~.FermiWord` to
    :class:`GanCoeff`, i.e. each fermionic string carries a nuclear-function
    polynomial coefficient. Terms whose fermionic string vanishes or whose
    coefficient is zero are dropped at construction.

    Args:
        fragment (dict[FermiWord, GanCoeff]): the fermionic-string-to-coefficient
            mapping defining the operator.
    """

    def __init__(self, fragment: dict[FermiWord, GanCoeff]):
        self.fragment = {fermi: coeff for fermi, coeff in fragment.items() if not (fermi.is_zero() or coeff.is_zero())}

    def __add__(self, other: GanFragment):
        """Add two fragments by summing coefficients of shared fermionic strings.

        Args:
            other (GanFragment): the fragment to add.

        Returns:
            GanFragment: the summed fragment.
        """
        d = defaultdict(lambda: GanCoeff({}))

        for key, value in self.fragment.items():
            d[key] += value

        for key, value in other.fragment.items():
            d[key] += value

        return GanFragment(d)

    def __sub__(self, other: GanFragment):
        """Subtract ``other`` from this fragment.

        Args:
            other (GanFragment): the fragment to subtract.

        Returns:
            GanFragment: the difference.
        """
        return self + (-1)*other

    def __mul__(self, scalar: complex):
        """Scale every coefficient by ``scalar``.

        Args:
            scalar (complex): the scalar multiplier.

        Returns:
            GanFragment: the scaled fragment.
        """
        d = {}

        for key, value in self.fragment.items():
            d[key] = scalar * value

        return GanFragment(d)

    __rmul__ = __mul__

    def __matmul__(self, other: GanFragment):
        """Multiply two fragments (operator product).

        Distributes over all term pairs: the fermionic strings are concatenated
        and normal-ordered (which may produce several words), and the resulting
        word coefficients are multiplied by the product of the two nuclear
        coefficients.

        Args:
            other (GanFragment): the right operand.

        Returns:
            GanFragment: the product fragment.
        """
        d = defaultdict(GanCoeff.identity)

        for l_key, l_value in self.fragment.items():
            for r_key, r_value in other.fragment.items():
                new_fermi = (l_key @ r_key).normal_order()
                for term, coeff in new_fermi.words.items():
                    d[term] += coeff * (l_value @ r_value)

        return GanFragment(d)

    def __repr__(self):
        return " + ".join(f"({coeff})({fermi})" for fermi, coeff in self.fragment.items())

    def __eq__(self, other: GanFragment):
        """Whether two fragments have the same terms and coefficients."""
        return self.fragment == other.fragment

    def norm(self, gridpoints: int) -> float:
        """Return a triangle-inequality upper bound on the spectral norm.

        Computed as the sum of the per-term coefficient norm bounds.

        Args:
            gridpoints (int): the number of grid points per vibrational mode.

        Returns:
            float: the norm bound for the fragment.
        """
        return sum(coeff.norm(gridpoints) for coeff in self.fragment.values())


@cache
def _normal_order(monomial):
    """Sort and simplify a monomial into its canonical normal ordering.

    Commuting factors (position/identity, and momentum/position on different
    modes) are bubbled into a canonical order keyed by mode and function type;
    adjacent factors sharing a base are then merged by summing exponents. The
    result is cached because the same monomials recur heavily during fragment
    multiplication.

    Args:
        monomial (GanMonomial): the monomial to normal-order.

    Returns:
        GanMonomial: the normal-ordered, simplified monomial.
    """
    funcs = list(monomial.funcs)
    n_ops = len(funcs)

    basis_types = {FuncType.POSITION, FuncType.IDENTITY}

    for i in range(n_ops):
        curr = i
        for j in reversed(range(i)):
            l_type = funcs[j].f_type
            l_mode = funcs[j].mode
            r_type = funcs[curr].f_type
            r_mode = funcs[curr].mode

            if l_type in basis_types and r_type in basis_types and l_mode > r_mode:
                funcs[curr], funcs[j] = funcs[j], funcs[curr]
                curr -= 1
                continue

            if l_type in basis_types and r_type in basis_types and l_mode == r_mode and l_type > r_type:
                funcs[curr], funcs[j] = funcs[j], funcs[curr]
                curr -= 1
                continue

            if l_type in basis_types and r_type == FuncType.MOMENTUM and l_mode != r_mode:
                funcs[curr], funcs[j] = funcs[j], funcs[curr]
                curr -= 1
                continue

            break

    funcs = _simplify_monomial(funcs)

    return GanMonomial(funcs)


@cache
def _cached_mon_matmul(l_mon: GanMonomial, r_mon: GanMonomial):
    """Return the normal-ordered product of two monomials, with caching.

    Args:
        l_mon (GanMonomial): the left monomial.
        r_mon (GanMonomial): the right monomial.

    Returns:
        GanMonomial: the normal-ordered product ``l_mon @ r_mon``.
    """
    return (l_mon @ r_mon).normal_order()


def _simplify_monomial(funcs):
    """Merge consecutive factors that share a base by summing their exponents.

    Args:
        funcs (list[FuncSymbol]): the (already-sorted) factors of a monomial.

    Returns:
        list[FuncSymbol]: the simplified factors, with runs of equal-base
        factors collapsed into a single factor.
    """

    new_funcs = []
    i = 0

    while i < len(funcs):
        current_func = funcs[i]
        exponent = 0

        while i < len(funcs) and current_func.base() == funcs[i].base():
            exponent += funcs[i].exponent
            i += 1

        new_funcs.append(
            FuncSymbol(current_func.f_type, current_func.mode, current_func.symbol, exponent)
        )

    return new_funcs

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
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
from scipy.linalg import expm, fractional_matrix_power


class ProductFormula:
    r"""Class for representing product formulas.

    For a set of Hermitian operators :math:`H_1,\dots,H_n`
    a product formula is any function in the form :math:`U(t) = \prod_{k=1}^n e^{it\alpha_k H_k}` with
    :math:`\alpha_k \in \mathbb{R}`.

    Args:
        terms (Sequence[Hashable] | Sequence[``ProductFormula``]): Either a list of labels for the
            Hermitian operators or a list of ``ProductFormula`` objcts. When a list of labels is given,
            the product formula returned is the product of exponentials of the lables. When a list of product
            formulas is given the product formula returned is the product of the given product formulas.
        coeffs: (Sequence[float]): A list of coefficients corresponding to the given terms. This argument
            is not needed when the terms are ``ProductFormula`` objects.
        exponent (float): Raises the product formula to the power of ``exponent``. Defaults to 1.0.
        label (str): Optional parameter used for pretty printing.

    **Example**

    This example uses :class:`~pennylane.labs.troter_error.ProductFormula` to build the fourth order
    Troter-Suzuki formula on three fragments. First we build the second order formula.

    >>> from pennylane.labs.trotter_error import ProductFormula
    >>>
    >>> frag_labels = ["A", "B", "C", "B", "A"]
    >>> frag_coeffs = [1/2, 1/2, 1, 1/2, 1/2]
    >>> second_order = ProductFormula(frag_labels, frag_coeffs)

    Now we build the fourth order formula out of the second order formula using arithmetic operations.

    >>> u = 1 / (4 - 4**(1/3))
    >>> v = 1 - 4*u
    >>>
    >>> fourth_order = second_order(u)**2 @ second_order(v) @ second_order(u)**2

    """

    def __init__(
        self,
        terms: Sequence[Hashable] | Sequence[ProductFormula],
        coeffs: Sequence[float] = None,
        exponent: float = 1.0,
        label: str = None,
    ):

        if any(not isinstance(term, type(terms[0])) for term in terms):
            raise TypeError("All terms must have the same type.")

        if not isinstance(terms[0], (Hashable, ProductFormula)):
            raise TypeError(
                f"Terms must either be Hashable type or of type ProductFormula. Got {type(terms[0])} instead."
            )

        if isinstance(terms[0], ProductFormula):
            self.recursive = True
        elif isinstance(terms[0], Hashable):
            if coeffs is None:
                raise ValueError("List of coefficients must be given.")
            if len(coeffs) != len(terms):
                raise ValueError("Number of coefficients must equal number of terms.")

            self.recursive = False

        self.fragments = set()
        for term in terms:
            if isinstance(term, ProductFormula):
                self.fragments = set.union(self.fragments, term.fragments)
            else:
                self.fragments.add(term)

        self.terms = terms
        self.coeffs = coeffs if coeffs else [1] * len(self.terms)
        self.exponent = exponent
        self.label = label

        self._ordered_terms = {}
        position = 0
        for term in self.terms:
            if term not in self._ordered_terms:
                self._ordered_terms[term] = position
                position += 1

    def __call__(self, t: float):
        ret = copy.copy(self)
        ret.label = f"{self.label}({t})"

        if ret.recursive:
            ret.terms = [term(t) for term in ret.terms]
            ret._ordered_terms = {
                term(t): position for term, position in ret._ordered_terms.items()
            }
        else:
            ret.coeffs = [t * coeff for coeff in ret.coeffs]

        return ret

    def __eq__(self, other: ProductFormula) -> bool:
        if self.recursive != other.recursive:
            return False

        if self.terms != other.terms:
            return False

        if self.coeffs != other.coeffs:
            return False

        return self.exponent == other.exponent

    def __hash__(self) -> int:
        terms = tuple(self.terms)
        coeffs = tuple(self.coeffs)
        return hash((terms, coeffs, self.exponent))

    def __matmul__(self, other: ProductFormula) -> ProductFormula:
        return ProductFormula([self, other], label=f"{self.label}@{other.label}")

    def __pow__(self, z: float) -> ProductFormula:
        ret = copy.copy(self)
        ret.exponent = z * self.exponent

        return ret

    def __repr__(self) -> str:
        if self.label:
            return f"{self.label}**{self.exponent}" if self.exponent != 1 else f"{self.label}"

        if self.recursive:
            return "@".join(term.__repr__() for term in self.terms)

        return "@".join([f"Exp({coeff}*H_{term})" for coeff, term in zip(self.coeffs, self.terms)])

    def to_matrix(self, fragments: dict[Hashable, np.ndarray]) -> np.ndarray:
        """Returns a numpy representation of the product formula.

        Args:
            fragments (Dict[Hashable, Fragment]): The matrix representations of the fragment labels.

        **Example**


        >>> import numpy as np
        >>> from pennylane.labs.trotter_error import ProductFormula
        >>>
        >>> frag_labels = ["A", "B", "C", "B", "A"]
        >>> frag_coeffs = [1/2, 1/2, 1, 1/2, 1/2]
        >>> second_order = ProductFormula(frag_labels, frag_coeffs)
        >>>
        >>> np.random.seed(42)
        >>> fragments = {
        >>>     "A": np.random.random(size=(3, 3)),
        >>>     "B": np.random.random(size=(3, 3)),
        >>>     "C": np.random.random(size=(3, 3)),
        >>> }
        >>>
        >>> second_order.to_matrix(fragments)
        [[20.53683969 24.33566914 25.4931284 ]
         [12.50207018 15.44505726 15.01069493]
         [13.52951601 17.64888648 18.04980336]]

        """
        accumulator = _MultiplicativeIdentity()
        for term, coeff in zip(self.terms, self.coeffs):
            if isinstance(term, ProductFormula):
                accumulator @= term.to_matrix(fragments)
            else:
                accumulator @= expm(coeff * fragments[term])

        return fractional_matrix_power(accumulator, self.exponent)

    @property
    def ordered_fragments(self) -> dict[Hashable, int]:
        """Return the fragment ordering used by the product formula.

        **Example**

        >>> from pennylane.labs.trotter_error import ProductFormula
        >>>
        >>> pf1 = ProductFormula(["A", "B", "C"], [1, 1, 1])
        >>> pf2 = ProductFormula(["X", "Y", "Z"], [1, 1, 1])
        >>>
        >>> pf = pf1 @ pf2
        >>>
        >>> pf.ordered_fragments
        {'A': 0, 'B': 1, 'C': 2, 'X': 3, 'Y': 4, 'Z': 5}
        """

        if not self.recursive:
            return self.ordered_terms

        ordered_fragments = {}
        position = 0
        for term in self.terms:
            for fragment in term.ordered_fragments:
                if fragment in ordered_fragments:
                    continue

                ordered_fragments[fragment] = position
                position += 1

        return ordered_fragments

    @property
    def ordered_terms(self) -> dict[Hashable, int]:
        """Return the term ordering used by the product formula.

        **Example**

        >>> from pennylane.labs.trotter_error import ProductFormula, bch_expansion
        >>>
        >>> frag_labels = ["A", "B", "C", "B", "A"]
        >>> frag_coeffs = [1/2, 1/2, 1, 1/2, 1/2]
        >>> second_order = ProductFormula(frag_labels, frag_coeffs, label="U")
        >>>
        >>> u = 1 / (4 - 4**(1/3))
        >>> v = 1 - 4*u
        >>>
        >>> fourth_order = second_order(u)**2 @ second_order(v) @ second_order(u)**2
        >>> fourth_order.ordered_terms
        {U(0.4144907717943757)@U(-0.6579630871775028): 0, U(0.4144907717943757)**2.0: 1}
        """

        ordered_terms = {}
        position = 0
        for term in self.terms:
            if term not in ordered_terms:
                ordered_terms[term] = position
                position += 1

        return ordered_terms


class _MultiplicativeIdentity:
    """A generic multiplicative identity that can be multiplied with any Python object."""

    def __matmul__(self, other: Any):
        return other

    def __rmatmul__(self, other: Any):
        return other

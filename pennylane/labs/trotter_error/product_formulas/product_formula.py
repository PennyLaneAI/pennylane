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
from typing import Dict, Sequence, Union

import numpy as np
from scipy.linalg import expm, fractional_matrix_power

from pennylane.labs.trotter_error.abstract import Fragment


class ProductFormula:
    """Class for representing product formulas"""

    def __init__(
        self,
        terms: Union[Sequence[Hashable], Sequence[ProductFormula]],
        coeffs: Sequence[float] = None,
        exponent: float = 1.0,
        label: str = None,
        include_i: bool = True,
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

        if coeffs and include_i:
            self.coeffs = [1j * coeff for coeff in coeffs]
        elif coeffs:
            self.coeffs = coeffs
        else:
            self.coeffs = [1] * len(self.terms)

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
            ret._ordered_terms = {term(t): position for term, position in ret._ordered_terms.items()}
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

    def to_matrix(self, fragments: Dict[Hashable, Fragment], accumulator: Fragment) -> np.ndarray:
        """Returns a numpy representation of the product formula"""
        acc = copy.copy(accumulator)
        for term, coeff in zip(self.terms, self.coeffs):
            if isinstance(term, ProductFormula):
                accumulator @= term.to_matrix(fragments, copy.copy(acc))
            else:
                accumulator @= expm(coeff * fragments[term])

        return fractional_matrix_power(accumulator, self.exponent)

    def ordered_fragments(self) -> Dict[Hashable, int]:
        """Return the fragment ordering used by the product formula"""
        if not self.recursive:
            return self._ordered_terms

        ordered_fragments = {}
        position = 0
        for term in self.terms:
            for fragment in term.ordered_fragments():
                if fragment in ordered_fragments:
                    continue

                ordered_fragments[fragment] = position
                position += 1

        return ordered_fragments

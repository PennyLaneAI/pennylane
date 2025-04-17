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

from typing import Hashable, Sequence, Union

from pennylane.labs.trotter_error.product_formulas.bch_tree import BCHTree, bch_approx


class ProductFormula:
    """Class for representing product formulas"""

    def __init__(
        self,
        coeffs: Sequence[float],
        fragments: Sequence[Union[Hashable, ProductFormula]],
        exponent: int = 1,
    ):

        if len(coeffs) != len(fragments):
            raise ValueError(f"Got {len(coeffs)} coefficients and {len(fragments)} fragments.")

        for fragment in fragments:
            if not isinstance(fragment, (Hashable, ProductFormula)):
                raise TypeError(
                    f"Fragments must have type Hashable or ProductFormula, got {type(fragment)} instead."
                )

        self.labels = set()
        for fragment in fragments:
            if isinstance(fragment, ProductFormula):
                self.labels = set.union(self.labels, fragment.labels)
            else:
                self.labels.add(fragment)

        self.coeffs = coeffs
        self.fragments = fragments
        self.exponent = exponent

    def bch_ast(self, max_order: int) -> BCHTree:
        """Return an AST representation of the BCH expansion"""
        bch_fragments = []

        for coeff, fragment in zip(self.coeffs, self.fragments):
            if isinstance(fragment, ProductFormula):
                bch_fragments.append(BCHTree.heff_node(fragment.bch_ast(max_order), coeff))
            else:
                bch_fragments.append(coeff * BCHTree.fragment_node(fragment))

        return _bch_ast(bch_fragments, max_order)

    def __repr__(self) -> str:
        reps = [
            f"Exp({coeff}*H_{fragment})" for coeff, fragment in zip(self.coeffs, self.fragments)
        ]

        return "*".join(reps)

    def _apply_coeff(self, x: float) -> ProductFormula:
        return ProductFormula(
            coeffs=[x * coeff for coeff in self.coeffs],
            fragments=self.fragments,
            exponent=self.exponent,
        )


def _bch_ast(fragments: Sequence[BCHTree], max_order: int) -> BCHTree:
    if len(fragments) == 2:
        return bch_approx(fragments[0], fragments[1], max_order)

    head, *tail = fragments
    return bch_approx(head, _bch_ast(tail, max_order), max_order).simplify(max_order)

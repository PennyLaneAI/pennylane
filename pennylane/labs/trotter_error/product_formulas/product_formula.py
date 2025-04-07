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

from typing import Sequence

from pennylane.labs.trotter_error.product_formulas.bch_tree import BCHTree, bch_approx


class ProductFormula:
    """Class for representing product formulas"""

    def __init__(self, coeffs: Sequence[float], fragments: Sequence[int]):

        if len(coeffs) != len(fragments):
            raise ValueError(f"Got {len(coeffs)} coefficients and {len(fragments)} fragments.")

        uniq_frags = set(fragments)
        n_frags = len(uniq_frags)

        for i in range(n_frags):
            if i not in uniq_frags:
                raise ValueError(
                    "Unique elements of `fragments` must be consecutive integers starting with 0."
                )

        self.coeffs = coeffs
        self.n_frags = n_frags
        self.fragments = fragments

    def bch_ast(self) -> BCHTree:
        """Return an AST representation of the BCH expansion"""
        bch_fragments = [
            coeff * BCHTree.fragment_node(fragment)
            for coeff, fragment in zip(self.coeffs, self.fragments)
        ]
        return _bch_ast(bch_fragments)

    def bch_ast_2(self) -> BCHTree:
        """Return an AST representation of the BCH expansion"""
        bch_fragments = [
            coeff * BCHTree.fragment_node(fragment)
            for coeff, fragment in zip(self.coeffs, self.fragments)
        ]
        return _bch_ast_2(bch_fragments)

    def __repr__(self) -> str:
        reps = [
            f"Exp({coeff}*H_{fragment})" for coeff, fragment in zip(self.coeffs, self.fragments)
        ]

        return "*".join(reps)


def _bch_ast_2(ops):
    return bch_approx(bch_approx(ops[0], ops[1]), bch_approx(ops[2], ops[3]))


def _bch_ast(ops: Sequence[BCHTree]) -> BCHTree:

    if len(ops) == 2:
        return bch_approx(ops[0], ops[1])

    head, *tail = ops
    return bch_approx(head, _bch_ast(tail))

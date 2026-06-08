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
        symbol_set = set()

        for symbol in self.symbols:
            if isinstance(symbol, ProductFormula):
                symbol_set = set.union(symbol_set, symbol.symbol_set)
            else:
                symbol_set.add(symbol)

        return symbol_set

    @property
    def is_recursive(self) -> bool:
        return self._recursive

    @property
    def is_symmetric(self) -> bool:
        return self.symbols == self.symbols[::-1] and self.coeffs == self.coeffs[::-1]

    @classmethod
    def prod(cls, product_formulas: Sequence[ProductFormula]) -> ProductFormula:
        if not all(isinstance(pf, ProductFormula) for pf in product_formulas):
            raise TypeError("Product formulas must be type `ProductFormula`.")

        product_formula = super().__new__(cls)
        product_formula.symbols = copy.deepcopy(product_formulas)
        product_formula.coeffs = [1] * len(product_formulas)
        product_formula.exponent = 1.0
        product_formula._recursive = True

        return product_formula

    def __call__(self, t: float) -> ProductFormula:
        pf = copy.deepcopy(self)

        if pf.is_recursive:
            pf.symbols = [symbol(t) for symbol in pf.symbols]
            return pf

        pf.coeffs = [t * coeff for coeff in pf.coeffs]

        return pf

    def __eq__(self, other: ProductFormula) -> bool:
        if self.symbols != other.symbols:
            return False

        if self.coeffs != other.coeffs:
            return False

        return self.exponent == other.exponent

    def __hash__(self) -> int:
        return hash((tuple(self.symbols), tuple(self.coeffs)))

    def __pow__(self, z: float) -> ProductFormula:
        if not isinstance(z, (int, float)):
            raise TypeError("Exponent must be int or float.")

        ret = copy.deepcopy(self)
        ret.exponent = z * self.exponent

        return ret

    def __repr__(self) -> str:
        return f"PF[{tuple(zip(self.symbols, self.coeffs))}]**{self.exponent}"

# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The RealspaceOperator class"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import product
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import scipy as sp

from pennylane.labs.trotter import Fragment
from pennylane.labs.trotter.realspace.matrix import (
    _zeros,
    op_norm,
    string_to_matrix,
    tensor_with_identity,
)

from .realspace_coefficients import RealspaceCoeffs


class RealspaceOperator:
    """The RealspaceOperator class"""

    def __init__(self, modes: int, ops: Tuple[str], coeffs: RealspaceCoeffs) -> RealspaceOperator:
        self.modes = modes
        self.ops = ops
        self.coeffs = coeffs

    def matrix(
        self, gridpoints: int, basis: str = "realspace", sparse: bool = False
    ) -> Union[np.ndarray, sp.sparse.csr_array]:
        """Return a matrix representation of the operator"""

        matrices = [string_to_matrix(op, gridpoints, basis=basis, sparse=sparse) for op in self.ops]
        final_matrix = _zeros(shape=(gridpoints**self.modes, gridpoints**self.modes), sparse=sparse)

        if sparse:
            indices = self.coeffs.nonzero()
        else:
            indices = product(range(self.modes), repeat=len(self.ops))

        for index in indices:
            matrix = self.coeffs[index] * tensor_with_identity(
                self.modes, gridpoints, index, matrices, sparse=sparse
            )
            final_matrix = final_matrix + matrix

        return final_matrix

    def __add__(self, other: RealspaceOperator) -> RealspaceOperator:
        if self.is_zero:
            return other

        if other.is_zero:
            return self

        if self.ops != other.ops:
            raise ValueError(f"Cannot add term {self.ops} with term {other.ops}.")

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot add RealspaceOperator on {self.modes} modes with RealspaceOperator on {other.modes} modes."
            )

        return RealspaceOperator(
            self.modes, self.ops, RealspaceCoeffs.sum_node(self.coeffs, other.coeffs)
        )

    def __sub__(self, other: RealspaceOperator) -> RealspaceOperator:
        if self.is_zero:
            return (-1) * other

        if other.is_zero:
            return self

        if self.ops != other.ops:
            raise ValueError(f"Cannot subtract term {self.ops} with term {other.ops}.")

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot subtract RealspaceOperator on {self.modes} modes with RealspaceOperator on {other.modes} modes."
            )

        return RealspaceOperator(
            self.modes,
            self.ops,
            RealspaceCoeffs.sum_node(self.coeffs, RealspaceCoeffs.scalar_node(-1, other.coeffs)),
        )

    def __mul__(self, scalar: float) -> RealspaceOperator:
        if np.isclose(scalar, 0):
            return RealspaceOperator.zero_term(self.modes)

        return RealspaceOperator(
            self.modes, self.ops, RealspaceCoeffs.scalar_node(scalar, self.coeffs)
        )

    __rmul__ = __mul__

    def __imul__(self, scalar: float) -> RealspaceOperator:
        if np.isclose(scalar, 0):
            return RealspaceOperator.zero_term(self.modes)

        self.coeffs = RealspaceCoeffs.scalar_node(scalar, self.coeffs)
        return self

    def __matmul__(self, other: RealspaceOperator) -> RealspaceOperator:
        if other.is_zero:
            return self

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot multiply RealspaceOperator on {self.modes} modes with RealspaceOperator on {other.modes} modes."
            )

        return RealspaceOperator(
            self.modes, self.ops + other.ops, RealspaceCoeffs.outer_node(self.coeffs, other.coeffs)
        )

    def __repr__(self) -> str:
        return f"({self.ops.__repr__()}, {self.coeffs.__repr__()})"

    def __eq__(self, other: RealspaceOperator) -> bool:
        if self.ops != other.ops:
            return False

        return self.coeffs == other.coeffs

    @property
    def is_zero(self) -> bool:
        """If is_zero returns true the term evaluates to zero, however there are false negatives"""
        return self.coeffs.is_zero

    @classmethod
    def zero_term(cls, modes) -> RealspaceOperator:
        """Returns a RealspaceOperator representing 0"""
        return RealspaceOperator(modes, tuple(), RealspaceCoeffs.tensor_node(np.array(0)))

    def get_coefficients(self, threshold: float = 0.0):
        """Return the coefficients in a dictionary"""
        return self.coeffs.get_coefficients(threshold)


class RealspaceSum(Fragment):
    """The RealspaceSum class"""

    def __init__(self, modes: int, ops: Sequence[RealspaceOperator]):
        # pylint: disable=unnecessary-lambda
        for op in ops:
            if op.modes != modes:
                raise ValueError(
                    f"RealspaceSum on {modes} modes can only contain RealspaceOperators on {modes}. Found a RealspaceOperator on {op.modes} modes."
                )

        ops = tuple(filter(lambda op: not op.is_zero, ops))
        self.is_zero = len(ops) == 0

        self.modes = modes
        self._lookup = defaultdict(lambda: RealspaceOperator.zero_term(self.modes))

        for op in ops:
            self._lookup[op.ops] += op

        self.ops = tuple(self._lookup.values())

    def __add__(self, other: RealspaceSum) -> RealspaceSum:
        if self.modes != other.modes:
            raise ValueError(
                f"Cannot add RealspaceSum on {self.modes} modes with RealspaceSum on {other.modes} modes."
            )

        l_ops = {term.ops for term in self.ops}
        r_ops = {term.ops for term in other.ops}

        new_ops = []

        for op in l_ops.intersection(r_ops):
            new_ops.append(self._lookup[op] + other._lookup[op])

        for op in l_ops.difference(r_ops):
            new_ops.append(self._lookup[op])

        for op in r_ops.difference(l_ops):
            new_ops.append(other._lookup[op])

        return RealspaceSum(self.modes, new_ops)

    def __sub__(self, other: RealspaceSum) -> RealspaceSum:
        if self.modes != other.modes:
            raise ValueError(
                f"Cannot subtract RealspaceSum on {self.modes} modes with RealspaceSum on {other.modes} modes."
            )

        l_ops = {term.ops for term in self.ops}
        r_ops = {term.ops for term in other.ops}

        new_terms = []

        for op in l_ops.intersection(r_ops):
            new_terms.append(self._lookup[op] - other._lookup[op])

        for op in l_ops.difference(r_ops):
            new_terms.append(self._lookup[op])

        for op in r_ops.difference(l_ops):
            new_terms.append((-1) * other._lookup[op])

        return RealspaceSum(self.modes, new_terms)

    def __mul__(self, scalar: float) -> RealspaceSum:
        return RealspaceSum(self.modes, [scalar * term for term in self.ops])

    __rmul__ = __mul__

    def __imul__(self, scalar: float) -> RealspaceSum:
        for term in self.ops:
            term *= scalar

        return self

    def __matmul__(self, other: RealspaceSum) -> RealspaceSum:
        return RealspaceSum(
            self.modes,
            [
                RealspaceOperator(
                    self.modes, l_term.ops + r_term.ops, l_term.coeffs @ r_term.coeffs
                )
                for l_term, r_term in product(self.ops, other.ops)
            ],
        )

    def __repr__(self) -> str:
        return self.ops.__repr__()

    def __eq__(self, other: RealspaceSum) -> bool:
        return self._lookup == other._lookup

    @classmethod
    def zero(cls, modes: int) -> RealspaceSum:
        """Return a RealspaceSum representing 0"""
        return RealspaceSum(modes, [RealspaceOperator.zero_term(modes)])

    def matrix(self, gridpoints: int, basis: str = "realspace", sparse: bool = False):
        """Return a matrix representation of the RealspaceSum"""

        final_matrix = _zeros(shape=(gridpoints**self.modes, gridpoints**self.modes), sparse=sparse)
        for op in self.ops:
            final_matrix = final_matrix + op.matrix(gridpoints, basis=basis, sparse=sparse)

        return final_matrix

    def norm(self, params: Dict) -> float:
        try:
            gridpoints = params["gridpoints"]
        except KeyError as e:
            raise KeyError("Need to specify the number of gridpoints") from e

        try:
            sparse = params["sparse"]
        except KeyError:
            sparse = False

        norm = 0

        for op in self.ops:
            term_op_norm = math.prod(map(lambda op: op_norm(gridpoints) ** len(op), op.ops))

            indices = (
                op.coeffs.nonzero() if sparse else product(range(self.modes), repeat=len(op.ops))
            )
            coeff_sum = sum(abs(op.coeffs.compute(index)) for index in indices)

            norm += coeff_sum * term_op_norm

        return norm

    def get_coefficients(self, threshold: float = 0.0):
        coeffs = {}
        for op in self.ops:
            coeffs[op.ops] = op.get_coefficients(threshold)

        return coeffs

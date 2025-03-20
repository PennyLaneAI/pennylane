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
    _op_norm,
    _string_to_matrix,
    _tensor_with_identity,
    _zeros,
)

from .realspace_coefficients import RealspaceCoeffs


class RealspaceOperator:
    r"""This class represents the summation of a product of position and momentum operators over the vibrational modes.
    For example,

    ..math:: \sum_{i,j} \phi_{i,j}Q_i Q_j

    Args:
        modes (int): the number of vibrational modes
        ops (Tuple[str]): a tuple representation of the position and momentum operators
        coeffs (``RealspaceCoeffs``): an expression tree which evaluates the entries of the coefficient tensor

    **Example**

    We build a ``RealspaceOperator`` in the following way.

    >>> from pennylane.labs.trotter import RealspaceOperator, RealspaceCoeffs
    >>> import numpy as np
    >>> n_modes = 5
    >>> ops = ("Q", "Q")
    >>> coeffs = RealspaceCoeffs.coeffs(np.random(shape=(n_modes, n_modes)))

    """

    def __init__(self, modes: int, ops: Tuple[str], coeffs: RealspaceCoeffs) -> RealspaceOperator:
        self.modes = modes
        self.ops = ops
        self.coeffs = coeffs

    def matrix(
        self, gridpoints: int, basis: str = "realspace", sparse: bool = False
    ) -> Union[np.ndarray, sp.sparse.csr_array]:
        """Return a matrix representation of the operator

        Args:
            gridpoints (int): the number of gridpoints used to discretize the position/momentum operators
            basis (str): the basis of the matrix, available options are ``realspace`` and ``harmonic``
            sparse (bool): if True returns a sparse matrix, otherwise a dense matrix

        Returns:
            Union[ndarray, csr_array]: the matrix representation of the ``RealspaceOperator``

        """

        matrices = [
            _string_to_matrix(op, gridpoints, basis=basis, sparse=sparse) for op in self.ops
        ]
        final_matrix = _zeros(shape=(gridpoints**self.modes, gridpoints**self.modes), sparse=sparse)

        if sparse:
            indices = list(self.coeffs.nonzero().keys())
        else:
            indices = product(range(self.modes), repeat=len(self.ops))

        for index in indices:
            matrix = self.coeffs[index] * _tensor_with_identity(
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
            return RealspaceOperator.zero(self.modes)

        return RealspaceOperator(
            self.modes, self.ops, RealspaceCoeffs.scalar_node(scalar, self.coeffs)
        )

    __rmul__ = __mul__

    def __imul__(self, scalar: float) -> RealspaceOperator:
        if np.isclose(scalar, 0):
            return RealspaceOperator.zero(self.modes)

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
        """Always returns true when the operator is zero, but with false positives

        Returns:
            bool: if False, the operator is guarnateed to be non-zero, if True the operator is likely non-zero, but with some edge cases
        """
        return self.coeffs.is_zero

    @classmethod
    def zero(cls, modes) -> RealspaceOperator:
        """Returns a RealspaceOperator representing the zero operator

        Args:
            modes (int): the number of vibrational modes (needed for consistency with arithmetic operations)

        Returns:
            RealspaceOperator: a representation of the zero operator

        """
        return RealspaceOperator(modes, tuple(), RealspaceCoeffs.tensor_node(np.array(0)))

    def get_coefficients(self, threshold: float = 0.0) -> Dict[Tuple[int], float]:
        """Return the coefficients in a dictionary

        Args:
            threshold (float): only return coefficients whose magnitude is greater than ``threshold``

        Returns:
            Dict[Tuple[int], float]: a dictionary whose keys are the nonzero indices, and values are the coefficients

        """
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
        self._lookup = defaultdict(lambda: RealspaceOperator.zero(self.modes))

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
        """Returns a RealspaceOperator representing the zero operator

        Args:
            modes (int): the number of vibrational modes (needed for consistency with arithmetic operations)

        Returns:
            RealspaceOperator: a representation of the zero operator

        """
        return RealspaceSum(modes, [RealspaceOperator.zero(modes)])

    def matrix(
        self, gridpoints: int, basis: str = "realspace", sparse: bool = False
    ) -> Union[np.ndarray, sp.sparse.cs_array]:
        """Return a matrix representation of the ``RealspaceSum``.

        Args:
            gridpoints (int): the number of gridpoints used to discretize the position/momentum operators
            basis (str): the basis of the matrix, available options are ``realspace`` and ``harmonic``
            sparse (bool): if True returns a sparse matrix, otherwise a dense matrix

        Returns:
            Union[ndarray, csr_array]: the matrix representation of the ``RealspaceOperator``

        """

        final_matrix = _zeros(shape=(gridpoints**self.modes, gridpoints**self.modes), sparse=sparse)
        for op in self.ops:
            final_matrix = final_matrix + op.matrix(gridpoints, basis=basis, sparse=sparse)

        return final_matrix

    def norm(self, params: Dict) -> float:
        """Returns an upper bound on the spectral norm of the operator.

        Args:
            params (Dict): The dictionary of parameters. The supported parameters are

                * ``gridpoints`` (int): the number of gridpoints used to discretize the operator
                * ``sparse`` (bool): If True, use optimizations for sparse operators. Defaults to False.

        Returns:
            float: an upper bound on the spectral norm of the operator

        """

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
            term_op_norm = math.prod(map(lambda op: _op_norm(gridpoints) ** len(op), op.ops))

            if sparse:
                coeffs = op.coeffs.nonzero()
                coeff_sum = sum(abs(val) for val in coeffs.values())
            else:
                indices = product(range(self.modes), repeat=len(op.ops))
                coeff_sum = sum(abs(op.coeffs.compute(index)) for index in indices)

            norm += coeff_sum * term_op_norm

        return norm

    def get_coefficients(self, threshold: float = 0.0) -> Dict[Tuple[str], Dict]:
        """Return a dictionary containing the coefficients of the ``RealspaceSum``

        Args:
            threshold (float): only return coefficients whose magnitude is greater than ``threshold``

        Returns:
            Dict: a dictionary whose keys correspond to the RealspaceOperators in the sum, and whose values are dictionaries obtained by ``RealspaceOperator.get_coefficients``
        """

        coeffs = {}
        for op in self.ops:
            coeffs[op.ops] = op.get_coefficients(threshold)

        return coeffs

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

from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.realspace.matrix import (
    _op_norm,
    _string_to_matrix,
    _tensor_with_identity,
    _zeros,
)

from .realspace_coefficients import RealspaceCoeffs


class RealspaceOperator:
    r"""Represents the summation of a product of position and momentum operators over the vibrational modes.
    For example,

    .. math:: \sum_{i,j} \phi_{i,j}Q_i Q_j

    Args:
        modes (int): the number of vibrational modes
        ops (Sequence[str]): a sequence representation of the position and momentum operators
        coeffs (``RealspaceCoeffs``): an expression tree which evaluates the entries of the coefficient tensor

    **Example**

    We build a :class:`~.pennylane.labs.trotter_error.RealspaceOperator` in the following way.

    >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs
    >>> import numpy as np
    >>> n_modes = 5
    >>> ops = ("Q", "Q")
    >>> coeffs = RealspaceCoeffs(np.random(shape=(n_modes, n_modes)), label="phi")
    >>> rs_op = RealspaceOperator(n_modes, ops, coeffs)

    """

    def __init__(
        self, modes: int, ops: Sequence[str], coeffs: Union[RealspaceCoeffs, np.ndarray, float]
    ) -> RealspaceOperator:
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
            Union[ndarray, csr_array]: the matrix representation of the :class:`~.pennylane.labs.trotter_error.RealspaceOperator`

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
        if self._is_zero:
            return other

        if other._is_zero:
            return self

        if self.ops != other.ops:
            raise ValueError(f"Cannot add term {self.ops} with term {other.ops}.")

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot add RealspaceOperator on {self.modes} modes with RealspaceOperator on {other.modes} modes."
            )

        return RealspaceOperator(self.modes, self.ops, self.coeffs + other.coeffs)

    def __sub__(self, other: RealspaceOperator) -> RealspaceOperator:
        if self._is_zero:
            return (-1) * other

        if other._is_zero:
            return self

        if self.ops != other.ops:
            raise ValueError(f"Cannot subtract term {self.ops} with term {other.ops}.")

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot subtract RealspaceOperator on {self.modes} modes with RealspaceOperator on {other.modes} modes."
            )

        return RealspaceOperator(self.modes, self.ops, self.coeffs - other.coeffs)

    def __mul__(self, scalar: float) -> RealspaceOperator:
        if np.isclose(scalar, 0):
            return RealspaceOperator.zero(self.modes)

        return RealspaceOperator(self.modes, self.ops, scalar * self.coeffs)

    __rmul__ = __mul__

    def __matmul__(self, other: RealspaceOperator) -> RealspaceOperator:
        if other._is_zero:
            return self

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot multiply RealspaceOperator on {self.modes} modes with RealspaceOperator on {other.modes} modes."
            )

        return RealspaceOperator(self.modes, self.ops + other.ops, self.coeffs @ other.coeffs)

    def __repr__(self) -> str:
        return f"({self.ops.__repr__()}, {self.coeffs.__repr__()})"

    def __eq__(self, other: RealspaceOperator) -> bool:
        if self.ops != other.ops:
            return False

        return self.coeffs == other.coeffs

    @property
    def _is_zero(self) -> bool:
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
        return RealspaceOperator(modes, tuple(), RealspaceCoeffs(np.array(0)))

    def get_coefficients(self, threshold: float = 0.0) -> Dict[Tuple[int], float]:
        """Return the coefficients in a dictionary

        Args:
            threshold (float): only return coefficients whose magnitude is greater than ``threshold``

        Returns:
            Dict[Tuple[int], float]: a dictionary whose keys are the nonzero indices, and values are the coefficients

        """
        return self.coeffs.get_coefficients(threshold)


class RealspaceSum(Fragment):
    r"""The :class:`~pennylane.labs.trotter_error.RealspaceSum` class is used to represent a Hamiltonian that is built from a sum of :class:`~.pennylane.labs.trotter_error.RealspaceOperator` objects
    For example, the vibrational hamiltonian

    .. math:: \sum_i \frac{\omega_i}{2} P_i^2 + \sum_i \frac{\omega_i}{2} Q_i^2 + \sum_i \phi^{(1)}_i Q_i + \sum_{i,j} \phi^{(2)}_{ij} Q_i Q_j + \dots,

    is a sum of sums, where each sum can be expressed by a :class:`~.pennylane.labs.trotter_error.RealspaceOperator`. A vibrational Hamiltonian can be represented as a :class:`~pennylane.labs.trotter_error.RealspaceSum` which
    contains a list of :class:`~.pennylane.labs.trotter_error.RealspaceOperator` objects representing these sums.

    Args:
        modes (int): the number of vibrational modes
        ops (Sequence[RealspaceOperator]): a sequence containing :class:`~.pennylane.labs.trotter_error.RealspaceOperator` objects representing the sums in the Hamiltonian

    **Example**

    We can build the harmonic part of the vibrational Hamiltonian with the following code.

    >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs, RealspaceSum
    >>> n_modes = 2
    >>> freqs = np.array([1.23, 3.45])
    >>> coeffs = RealspaceCoeffs.coeffs(freqs, label="omega")
    >>> rs_op1 = RealspaceOperator(n_modes, ("PP",), coeffs)
    >>> rs_op2 = RealspaceOperator(n_modes, ("QQ",), coeffs)
    >>> rs_sum = RealspaceSum(n_modes, [rs_op1, rs_op2])
    """

    def __init__(self, modes: int, ops: Sequence[RealspaceOperator]):
        # pylint: disable=unnecessary-lambda, protected-access
        for op in ops:
            if op.modes != modes:
                raise ValueError(
                    f"RealspaceSum on {modes} modes can only contain RealspaceOperators on {modes}. Found a RealspaceOperator on {op.modes} modes."
                )

        ops = tuple(filter(lambda op: not op._is_zero, ops))
        self._is_zero = len(ops) == 0

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

        for op in l_ops.union(r_ops):
            new_ops.append(self._lookup[op] + other._lookup[op])

        return RealspaceSum(self.modes, new_ops)

    def __sub__(self, other: RealspaceSum) -> RealspaceSum:
        if self.modes != other.modes:
            raise ValueError(
                f"Cannot subtract RealspaceSum on {self.modes} modes with RealspaceSum on {other.modes} modes."
            )

        l_ops = {term.ops for term in self.ops}
        r_ops = {term.ops for term in other.ops}

        new_terms = []

        for op in l_ops.union(r_ops):
            new_terms.append(self._lookup[op] - other._lookup[op])

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
        """Returns a :class:`~.pennylane.labs.trotter_error.RealspaceOperator` representing the zero operator

        Args:
            modes (int): the number of vibrational modes (needed for consistency with arithmetic operations)

        Returns:
            RealspaceOperator: a representation of the zero operator

        """
        return RealspaceSum(modes, [RealspaceOperator.zero(modes)])

    def matrix(
        self, gridpoints: int, basis: str = "realspace", sparse: bool = False
    ) -> Union[np.ndarray, sp.sparse.cs_array]:
        """Return a matrix representation of the :class:`~pennylane.labs.trotter_error.RealspaceSum`.

        Args:
            gridpoints (int): the number of gridpoints used to discretize the position/momentum operators
            basis (str): the basis of the matrix, available options are ``realspace`` and ``harmonic``
            sparse (bool): if True returns a sparse matrix, otherwise a dense matrix

        Returns:
            Union[ndarray, csr_array]: the matrix representation of the :class:`~.pennylane.labs.trotter_error.RealspaceOperator`

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

        sparse = params.get("sparse", False)
        gridpoints = params.get("gridpoints", None)
        if gridpoints is None:
            raise KeyError("Need to specify the number of gridpoints")

        norm = 0

        for op in self.ops:
            term_op_norm = math.prod(map(lambda op: _op_norm(gridpoints) ** len(op), op.ops))

            if sparse:
                coeffs = op.coeffs.nonzero()
                coeff_sum = sum(abs(val) for val in coeffs.values())
            else:
                indices = product(range(self.modes), repeat=len(op.ops))
                coeff_sum = sum(abs(op.coeffs[index]) for index in indices)

            norm += coeff_sum * term_op_norm

        return norm

    def get_coefficients(self, threshold: float = 0.0) -> Dict[Tuple[str], Dict]:
        """Return a dictionary containing the coefficients of the :class:`~pennylane.labs.trotter_error.RealspaceSum`.

        Args:
            threshold (float): only return coefficients whose magnitude is greater than ``threshold``

        Returns:
            Dict: a dictionary whose keys correspond to the RealspaceOperators in the sum, and whose values are dictionaries obtained by :func:`~.pennylane.labs.trotter_error.RealspaceOperator.get_coefficients`
        """

        coeffs = {}
        for op in self.ops:
            coeffs[op.ops] = op.get_coefficients(threshold)

        return coeffs

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
from collections.abc import Sequence
from itertools import product

import numpy as np
import scipy as sp

from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.realspace import HOState
from pennylane.labs.trotter_error.realspace.matrix import (
    _op_norm,
    _string_to_matrix,
    _tensor_with_identity,
    _zeros,
)

from .realspace_coefficients import RealspaceCoeffs, _RealspaceTree


class RealspaceOperator:
    r"""Represents a linear combination of a product of position and momentum operators.
    The ``RealspaceOperator`` class can be used to represent components of a vibrational
    Hamiltonian, e.g., the following sum over a product of two position operators :math:`Q`:

    .. math:: \sum_{i,j=1}^n \phi_{i,j}Q_i Q_j,

    where :math:`\phi_{i, j}` represents the coefficient and is a constant.

    Args:
        modes (int): the number of vibrational modes
        ops (Sequence[str]): a sequence representation of the position and momentum operators
        coeffs (``RealspaceCoeffs``): an expression tree which evaluates the entries of the coefficient tensor

    **Example**

    This example uses :class:`~.pennylane.labs.trotter_error.RealspaceOperator` to build the
    operator :math:`\sum_{i,j=1}^2 \phi_{i,j}Q_i Q_j`. The operator represents a sum over 2 modes
    for the position operators :math:`Q_iQ_j`.

    >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs
    >>> import numpy as np
    >>> n_modes = 2
    >>> ops = ("Q", "Q")
    >>> coeffs = RealspaceCoeffs(np.array([[1, 0], [0, 1]]), label="phi")
    >>> RealspaceOperator(n_modes, ops, coeffs)
    RealspaceOperator(5, ('Q', 'Q'), phi[idx0,idx1])
    """

    def __init__(
        self, modes: int, ops: Sequence[str], coeffs: RealspaceCoeffs | np.ndarray | float
    ) -> RealspaceOperator:

        if coeffs.shape != (modes,) * len(ops):
            raise ValueError(
                f"coeffs has shape {coeffs.shape}, but shape {(modes, ) * len(ops)} was expected."
            )

        self.modes = modes
        self.ops = ops
        self.coeffs = coeffs

    def matrix(
        self, gridpoints: int, basis: str = "realspace", sparse: bool = False
    ) -> np.ndarray | sp.sparse.csr_array:
        """Return a matrix representation of the operator.

        Args:
            gridpoints (int): the number of gridpoints used to discretize the position or momentum operators
            basis (str): the basis of the matrix, available options are ``realspace`` and ``harmonic``
            sparse (bool): if ``True`` returns a sparse matrix, otherwise returns a dense matrix

        Returns:
            Union[ndarray, scipy.sparse.csr_array]: the matrix representation of the :class:`~.pennylane.labs.trotter_error.RealspaceOperator`

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs
        >>> import numpy as np
        >>> n_modes = 2
        >>> ops = ("Q", "Q")
        >>> coeffs = RealspaceCoeffs(np.array([[1, 0], [0, 1]]), label="phi")
        >>> RealspaceOperator(n_modes, ops, coeffs).matrix(2)
        [[6.28318531 0.         0.         0.        ]
         [0.         3.14159265 0.         0.        ]
         [0.         0.         3.14159265 0.        ]
         [0.         0.         0.         0.        ]]
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

    def __imul__(self, scalar: float) -> RealspaceOperator:
        if np.isclose(scalar, 0):
            return RealspaceOperator.zero(self.modes)

        self.coeffs = _RealspaceTree.scalar_node(scalar, self.coeffs)
        return self

    def __matmul__(self, other: RealspaceOperator) -> RealspaceOperator:
        if self.modes != other.modes:
            raise ValueError(
                f"Cannot multiply RealspaceOperator on {self.modes} modes with RealspaceOperator on {other.modes} modes."
            )

        return RealspaceOperator(self.modes, self.ops + other.ops, self.coeffs @ other.coeffs)

    def __repr__(self) -> str:
        return f"RealspaceOperator({self.modes}, {self.ops.__repr__()}, {self.coeffs.__repr__()})"

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
        """Returns a ``RealspaceOperator`` representing the zero operator.

        Args:
            modes (int): the number of vibrational modes

        Returns:
            RealspaceOperator: a representation of the zero operator

        """
        return RealspaceOperator(modes, tuple(), RealspaceCoeffs(np.array(0)))

    def get_coefficients(self, threshold: float = 0.0) -> dict[tuple[int], float]:
        """Return the non-zero coefficients in a dictionary.

        Args:
            threshold (float): tolerance to return coefficients whose magnitude is greater than ``threshold``

        Returns:
            Dict[Tuple[int], float]: a dictionary whose keys are the nonzero indices, and values are the coefficients

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs
        >>> import numpy as np
        >>> n_modes = 2
        >>> ops = ("Q", "Q")
        >>> coeffs = RealspaceCoeffs(np.array([[1, 0], [0, 1]]), label="phi")
        >>> RealspaceOperator(n_modes, ops, coeffs).get_coefficients()
        {(0, 0): 1, (1, 1): 1}
        """
        return self.coeffs.nonzero(threshold)


class RealspaceSum(Fragment):
    r"""Represents a linear combination of :class:`~.pennylane.labs.trotter_error.RealspaceOperator` objects.

    The :class:`~pennylane.labs.trotter_error.RealspaceSum` class can be used to represent a
    Hamiltonian that is built from a sum of
    :class:`~.pennylane.labs.trotter_error.RealspaceOperator` objects. For example, the vibrational
    hamiltonian, adapted from Eq. 4 of `arXiv:1703.09313 <https://arxiv.org/abs/1703.09313>`_,

    .. math:: \sum_i \frac{\omega_i}{2} P_i^2 + \sum_i \frac{\omega_i}{2} Q_i^2 + \sum_i \phi^{(1)}_i Q_i + \sum_{i,j} \phi^{(2)}_{ij} Q_i Q_j + \dots,

    is a sum of terms where each term can be expressed by a :class:`~.pennylane.labs.trotter_error.RealspaceOperator`.

    Args:
        modes (int): the number of vibrational modes
        ops (Sequence[RealspaceOperator]): a sequence containing :class:`~.pennylane.labs.trotter_error.RealspaceOperator` objects

    **Example**

    We can build the harmonic part of a vibrational Hamiltonian,
    :math:`\sum_i \frac{\omega_i}{2} P_i^2 + \sum_i \frac{\omega_i}{2} Q_i^2`, with the following
    code.

    >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs, RealspaceSum
    >>> import numpy as np
    >>> n_modes = 2
    >>> freqs = np.array([1.23, 3.45]) / 2
    >>> coeffs = RealspaceCoeffs(freqs, label="omega")
    >>> rs_op1 = RealspaceOperator(n_modes, ("PP",), coeffs)
    >>> rs_op2 = RealspaceOperator(n_modes, ("QQ",), coeffs)
    >>> RealspaceSum(n_modes, [rs_op1, rs_op2])
    RealspaceSum((RealspaceOperator(2, ('PP',), omega[idx0]), RealspaceOperator(2, ('QQ',), omega[idx0])))
    """

    def __init__(self, modes: int, ops: Sequence[RealspaceOperator]):
        # pylint: disable=protected-access
        for op in ops:
            if op.modes != modes:
                raise ValueError(
                    f"RealspaceSum on {modes} modes can only contain RealspaceOperators on {modes}. Found a RealspaceOperator on {op.modes} modes."
                )

        for op in ops:
            assert op.coeffs is not None

        ops = tuple(filter(lambda op: not op._is_zero, ops))
        self._is_zero = len(ops) == 0

        self.modes = modes

        # Note defaultdict with custom types cannot be used with mp_pool or cf_procpool
        # https://stackoverflow.com/questions/9256687/using-defaultdict-with-multiprocessing
        self._lookup = {}

        for op in ops:
            if op.ops in self._lookup:
                self._lookup[op.ops] += op
            else:
                self._lookup[op.ops] = op

        for op in ops:
            assert self._get_op_lookup(op.ops).coeffs is not None

        self.ops = tuple(self._lookup.values()) if self._lookup else tuple()

    def _get_op_lookup(self, op):
        """Returns the operator lookup for a given operator.

        Args:
            op (str): the operator string to look up

        Returns:
            RealspaceOperator: the corresponding RealspaceOperator object
        """
        if op not in self._lookup:
            return RealspaceOperator.zero(self.modes)

        return self._lookup[op]

    def __add__(self, other: RealspaceSum) -> RealspaceSum:
        if self.modes != other.modes:
            raise ValueError(
                f"Cannot add RealspaceSum on {self.modes} modes with RealspaceSum on {other.modes} modes."
            )

        l_ops = {term.ops for term in self.ops}
        r_ops = {term.ops for term in other.ops}

        new_ops = []

        for op in l_ops.union(r_ops):
            new_ops.append(self._get_op_lookup(op) + other._get_op_lookup(op))

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
            new_terms.append(self._get_op_lookup(op) - other._get_op_lookup(op))

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
        return f"RealspaceSum({self.ops.__repr__()})"

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
    ) -> np.ndarray | sp.sparse.cs_array:
        """Return a matrix representation of the :class:`~pennylane.labs.trotter_error.RealspaceSum`.

        Args:
            gridpoints (int): the number of gridpoints used to discretize the position/momentum operators
            basis (str): the basis of the matrix, available options are ``realspace`` and ``harmonic``
            sparse (bool): if ``True`` returns a sparse matrix, otherwise a dense matrix

        Returns:
            Union[ndarray, scipy.sparse.csr_array]: the matrix representation of the :class:`~.pennylane.labs.trotter_error.RealspaceOperator`

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs, RealspaceSum
        >>> import numpy as np
        >>> n_modes = 2
        >>> freqs = np.array([1.23, 3.45])
        >>> coeffs = RealspaceCoeffs(freqs, label="omega")
        >>> rs_op1 = RealspaceOperator(n_modes, ("PP",), coeffs)
        >>> rs_op2 = RealspaceOperator(n_modes, ("QQ",), coeffs)
        >>> RealspaceSum(n_modes, [rs_op1, rs_op2]).matrix(2)
        [[22.05398043+0.00000000e+00j -5.41924733+6.63666389e-16j
          -1.93207948+2.36611495e-16j  0.        +0.00000000e+00j]
         [-5.41924733-6.63666389e-16j 11.21548577+0.00000000e+00j
           0.        +0.00000000e+00j -1.93207948+2.36611495e-16j]
         [-1.93207948-2.36611495e-16j  0.        +0.00000000e+00j
          18.18982146+0.00000000e+00j -5.41924733+6.63666389e-16j]
         [ 0.        +0.00000000e+00j -1.93207948-2.36611495e-16j
          -5.41924733-6.63666389e-16j  7.35132681+0.00000000e+00j]]
        """

        final_matrix = _zeros(shape=(gridpoints**self.modes, gridpoints**self.modes), sparse=sparse)
        for op in self.ops:
            final_matrix = final_matrix + op.matrix(gridpoints, basis=basis, sparse=sparse)

        return final_matrix

    def norm(self, params: dict) -> float:
        """Returns an upper bound on the spectral norm of the operator.

        Args:
            params (Dict): The dictionary of parameters. The supported parameters are

                * ``gridpoints`` (int): the number of gridpoints used to discretize the operator
                * ``sparse`` (bool): If ``True``, use optimizations for sparse operators. Defaults to ``False``.

        Returns:
            float: an upper bound on the spectral norm of the operator

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs, RealspaceSum
        >>> import numpy as np
        >>> n_modes = 2
        >>> freqs = np.array([1.23, 3.45])
        >>> coeffs = RealspaceCoeffs(freqs, label="omega")
        >>> rs_op1 = RealspaceOperator(n_modes, ("PP",), coeffs)
        >>> rs_op2 = RealspaceOperator(n_modes, ("QQ",), coeffs)
        >>> params = {"gridpoints": 2, "sparse": True}
        >>> RealspaceSum(n_modes, [rs_op1, rs_op2]).norm(params)
        29.405307237600457
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

    def apply(self, state: HOState) -> HOState:
        """Apply the :class:`~.pennylane.labs.trotter_error.RealspaceSum` to an input :class:`~.pennylane.labs.trotter_error.HOState` object."""
        if not isinstance(state, HOState):
            raise TypeError

        mat = self.matrix(state.gridpoints, basis="harmonic", sparse=True)

        return HOState(
            state.modes,
            state.gridpoints,
            mat @ state.vector,
        )

    def get_coefficients(self, threshold: float = 0.0) -> dict[tuple[str], dict]:
        """Return a dictionary containing the non-zero coefficients of the :class:`~pennylane.labs.trotter_error.RealspaceSum`.

        Args:
            threshold (float): tolerance to return coefficients whose magnitude is greater than ``threshold``

        Returns:
            Dict: a dictionary whose keys correspond to the RealspaceOperators in the sum, and whose
                values are dictionaries obtained by :func:`~.pennylane.labs.trotter_error.RealspaceOperator.get_coefficients`

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceCoeffs, RealspaceSum
        >>> import numpy as np
        >>> n_modes = 2
        >>> freqs = np.array([1.23, 3.45])
        >>> coeffs = RealspaceCoeffs(freqs, label="omega")
        >>> rs_op1 = RealspaceOperator(n_modes, ("PP",), coeffs)
        >>> rs_op2 = RealspaceOperator(n_modes, ("QQ",), coeffs)
        >>> RealspaceSum(n_modes, [rs_op1, rs_op2]).get_coefficients()
        {('PP',): {(0,): 1.23, (1,): 3.45}, ('QQ',): {(0,): 1.23, (1,): 3.45}}
        """

        coeffs = {}
        for op in self.ops:
            coeffs[op.ops] = op.get_coefficients(threshold)

        return coeffs

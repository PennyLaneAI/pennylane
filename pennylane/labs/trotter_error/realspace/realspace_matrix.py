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
"""The RealspaceMatrix class"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import scipy as sp

from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.realspace import HOState, RealspaceSum, VibronicHO
from pennylane.labs.trotter_error.realspace.matrix import _kron, _zeros

# pylint: disable=protected-access


class RealspaceMatrix(Fragment):
    r"""Implements a dictionary of :class:`~.pennylane.labs.trotter_error.RealspaceSum` objects.

    This can be used to represent the fragments of a vibronic Hamiltonian given by, Eq. 3
    of `arXiv:2411.13669 <https://arxiv.org/abs/2411.13669v1>`_,

    .. math:: V_{i,j} = \lambda_{i,j} + \sum_{r} \phi^{(1)}_{i,j,r} Q_r + \sum_{r,s} \phi^{(2)}_{i,j,r,s} Q_r Q_s + \sum_{r,s,t} \phi^{(3)}_{i,j,r,s,t} Q_r Q_s Q_t + \dots,

    where the dictionary is indexed by tuples :math:`(i, j)` and the values are
    :class:`~.RealspaceSum` objects representing the operator :math:`V_{i,j}`.

    Args:
        states (int): the number of electronic states
        modes (int): the number of vibrational modes
        blocks (Dict[Tuple[int, int], RealspaceSum): a dictionary representation of the block matrix

    **Example**

    >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceSum, RealspaceCoeffs, RealspaceMatrix
    >>> import numpy as np
    >>> n_states = 1
    >>> n_modes = 5
    >>> op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
    >>> op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
    >>> rs_sum = RealspaceSum(n_modes, [op1, op2])
    >>> RealspaceMatrix(n_states, n_modes, {(0, 0): rs_sum})
    RealspaceMatrix({(0, 0): RealspaceSum((RealspaceOperator(5, (), 1), RealspaceOperator(5, 'Q', phi[idx0])))})
    """

    def __init__(
        self,
        states: int,
        modes: int,
        blocks: dict[tuple[int, int], RealspaceSum] = None,
    ) -> RealspaceMatrix:

        if blocks is None:
            blocks = {}

        self._blocks = blocks
        self.states = states
        self.modes = modes

    def block(self, row: int, col: int) -> RealspaceSum:
        """Return the :class:`~.pennylane.labs.trotter_error.RealspaceSum` object located at the
        ``(row, col)`` entry of the :class:`~.pennylane.labs.trotter_error.RealspaceMatrix`.

        Args:
            row (int): the row of the index
            col (int): the column of the index

        Returns:
            RealspaceSum: the :class:`~.pennylane.labs.trotter_error.RealspaceSum` object indexed
                at ``(row, col)``

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceSum, RealspaceCoeffs, RealspaceMatrix
        >>> import numpy as np
        >>> n_states = 1
        >>> n_modes = 5
        >>> op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
        >>> op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
        >>> rs_sum = RealspaceSum(n_modes, [op1, op2])
        >>> RealspaceMatrix(n_states, n_modes, {(0, 0): rs_sum}).block(0, 0)
        RealspaceSum((RealspaceOperator(5, (), 1), RealspaceOperator(5, 'Q', phi[idx0])))
        """
        if row < 0 or col < 0:
            raise IndexError(f"Index cannot be negative, got {(row, col)}.")
        if row >= self.states or col >= self.states:
            raise IndexError(
                f"Index out of bounds. Got {(row, col)} but there are only {self.states} states."
            )

        return self._blocks.get((row, col), RealspaceSum.zero(self.modes))

    def set_block(self, row: int, col: int, rs_sum: RealspaceSum) -> None:
        """Set the value of the block indexed at ``(row, col)``.

        Args:
            row (int): the row of the index
            col (int): the column of the index
            rs_sum (RealspaceSum): the :class:`~.pennylane.labs.trotter_error.RealspaceSum` object to stored in index ``(row, col)``

        Returns:
            None

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceSum, RealspaceCoeffs, RealspaceMatrix
        >>> import numpy as np
        >>> n_states = 2
        >>> n_modes = 5
        >>> op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
        >>> op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
        >>> rs_sum = RealspaceSum(n_modes, [op1, op2])
        >>> vib = RealspaceMatrix(n_states, n_modes, {(0, 0): rs_sum})
        >>> vib
        RealspaceMatrix({(0, 0): RealspaceSum((RealspaceOperator(5, (), 1), RealspaceOperator(5, 'Q', phi[idx0])))})
        >>> vib.set_block(1, 1, rs_sum)
        >>> vib
        RealspaceMatrix({(0, 0): RealspaceSum((RealspaceOperator(5, (), 1), RealspaceOperator(5, 'Q', phi[idx0]))), (1, 1): RealspaceSum((RealspaceOperator(5, (), 1), RealspaceOperator(5, 'Q', phi[idx0])))})
        """

        if not isinstance(rs_sum, RealspaceSum):
            raise TypeError(f"Block value must be RealspaceSum. Got {type(rs_sum)}.")
        if row < 0 or col < 0:
            raise IndexError(f"Index cannot be negative, got {(row, col)}.")
        if row >= self.states or col >= self.states:
            raise IndexError(
                f"Index out of bounds. Got {(row, col)} but there are only {self.states} states."
            )

        if rs_sum._is_zero:
            return

        self._blocks[(row, col)] = rs_sum

    def matrix(
        self, gridpoints: int, sparse: bool = False, basis: str = "realspace"
    ) -> np.ndarray | sp.sparse.csr_matrix:
        """Return a matrix representation of the operator.

        Args:
            gridpoints (int): the number of gridpoints used to discretize the position or momentum operators
            basis (str): the basis of the matrix, available options are ``realspace`` and ``harmonic``
            sparse (bool): if ``True`` returns a sparse matrix, otherwise returns a dense matrix

        Returns:
            Union[ndarray, scipy.sparse.csr_array]: the matrix representation of the :class:`~.pennylane.labs.trotter_error.RealspaceOperator`

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceSum, RealspaceCoeffs, RealspaceMatrix
        >>> import numpy as np
        >>> n_states = 1
        >>> n_modes = 5
        >>> op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
        >>> op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
        >>> rs_sum = RealspaceSum(n_modes, [op1, op2])
        >>> RealspaceMatrix(n_states, n_modes, {(0, 0): rs_sum}).matrix(2)
        [[-25.58680776   0.           0.         ...   0.           0.
            0.        ]
         [  0.         -16.72453851   0.         ...   0.           0.
            0.        ]
         [  0.           0.         -18.49699236 ...   0.           0.
            0.        ]
         ...
         [  0.           0.           0.         ...  -6.0898154    0.
            0.        ]
         [  0.           0.           0.         ...   0.          -7.86226925
            0.        ]
         [  0.           0.           0.         ...   0.           0.
            1.        ]]
        """
        pow2 = _next_pow_2(self.states)
        dim = pow2 * gridpoints**self.modes
        shape = (pow2, pow2)
        matrix = _zeros((dim, dim), sparse=sparse)

        for index, rs_sum in self._blocks.items():
            if sparse:
                data = np.array([1])
                indices = (np.array([index[0]]), np.array([index[1]]))
                indicator = sp.sparse.csr_array((data, indices), shape=shape)
            else:
                indicator = np.zeros(shape=shape)
                indicator[index] = 1

            block = rs_sum.matrix(gridpoints, basis=basis, sparse=sparse)
            matrix = matrix + _kron(indicator, block)

        return matrix

    def norm(self, params: dict) -> float:
        """Returns an upper bound on the spectral norm of the operator.

        Args:
            params (dict[str, Union[int, bool]]): The dictionary of parameters. The supported parameters are

                * ``gridpoints`` (int): the number of gridpoints used to discretize the operator
                * ``sparse`` (bool): If ``True``, use optimizations for sparse operators. Defaults to ``False``.

        Returns:
            float: an upper bound on the spectral norm of the operator

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceSum, RealspaceCoeffs, RealspaceMatrix
        >>> import numpy as np
        >>> n_states = 1
        >>> n_modes = 5
        >>> op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
        >>> op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
        >>> rs_sum = RealspaceSum(n_modes, [op1, op2])
        >>> params = {"gridpoints": 2, "sparse": True}
        >>> RealspaceMatrix(n_states, n_modes, {(0, 0): rs_sum}).norm(params)
        27.586807763582737
        """
        try:
            gridpoints = params["gridpoints"]
        except KeyError as e:
            raise KeyError("Need to specify the number of gridpoints") from e

        if not _is_pow_2(gridpoints) or gridpoints <= 0:
            raise ValueError(
                f"Number of gridpoints must be a positive power of 2, got {gridpoints}."
            )

        padded = RealspaceMatrix(_next_pow_2(self.states), self.modes, self._blocks)

        return padded._norm(params)

    def _norm(self, params: dict) -> float:
        """Returns an upper bound on the spectral norm. This method assumes ``self.states`` is a power of two."""
        if self.states == 1:
            return self.block(0, 0).norm(params)

        top_left, top_right, bottom_left, bottom_right = self._partition_into_quadrants()

        norm1 = max(top_left._norm(params), bottom_right._norm(params))
        norm2 = math.sqrt(top_right._norm(params) * bottom_left._norm(params))

        return norm1 + norm2

    def __add__(self, other: RealspaceMatrix) -> RealspaceMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot add RealspaceMatrix on {self.states} states with RealspaceMatrix on {other.states} states."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot add RealspaceMatrix on {self.modes} states with RealspaceMatrix on {other.modes} states."
            )

        new_blocks = {}
        l_keys = set(self._blocks.keys())
        r_keys = set(other._blocks.keys())

        for key in l_keys.intersection(r_keys):
            new_blocks[key] = self._blocks[key] + other._blocks[key]

        for key in l_keys.difference(r_keys):
            new_blocks[key] = self._blocks[key]

        for key in r_keys.difference(l_keys):
            new_blocks[key] = other._blocks[key]

        return RealspaceMatrix(
            self.states,
            self.modes,
            new_blocks,
        )

    def __sub__(self, other: RealspaceMatrix) -> RealspaceMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot subtract RealspaceMatrix on {self.states} states with RealspaceMatrix on {other.states} states."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot subtract RealspaceMatrix on {self.modes} states with RealspaceMatrix on {other.modes} states."
            )

        new_blocks = {}

        l_keys = set(self._blocks.keys())
        r_keys = set(other._blocks.keys())

        for key in l_keys.intersection(r_keys):
            new_blocks[key] = self._blocks[key] - other._blocks[key]

        for key in l_keys.difference(r_keys):
            new_blocks[key] = self._blocks[key]

        for key in r_keys.difference(l_keys):
            new_blocks[key] = (-1) * other._blocks[key]

        return RealspaceMatrix(
            self.states,
            self.modes,
            new_blocks,
        )

    def __mul__(self, scalar: float) -> RealspaceMatrix:
        new_blocks = {}
        for key in self._blocks.keys():
            new_blocks[key] = scalar * self._blocks[key]

        return RealspaceMatrix(self.states, self.modes, new_blocks)

    __rmul__ = __mul__

    def __imul__(self, scalar: float) -> RealspaceMatrix:
        for key in self._blocks.keys():
            self._blocks[key] *= scalar

        return self

    def __matmul__(self, other: RealspaceMatrix) -> RealspaceMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot multiply RealspaceMatrix on {self.states} states with RealspaceMatrix on {other.states} states."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot multiply RealspaceMatrix on {self.states} states with RealspaceMatrix on {other.states} states."
            )

        product_matrix = RealspaceMatrix(self.states, self.modes)

        for i, j in product(range(self.states), repeat=2):
            for op in self.block(i, j).ops:
                assert op.coeffs is not None
            for op in other.block(i, j).ops:
                assert op.coeffs is not None
            block_products = [self.block(i, k) @ other.block(k, j) for k in range(self.states)]
            block_sum = sum(block_products, RealspaceSum.zero(self.modes))
            product_matrix.set_block(i, j, block_sum)

        return product_matrix

    def __eq__(self, other: RealspaceMatrix):
        if self.states != other.states:
            return False

        if self.modes != other.modes:
            return False

        if self._blocks != other._blocks:
            return False

        return True

    def _partition_into_quadrants(self) -> tuple[RealspaceMatrix]:
        """Partitions the :class:`~.pennylane.labs.trotter_error.RealspaceMatrix` into four :class:`~.pennylane.labs.trotter_error.RealspaceMatrix` objects on ``self.states // 2`` states. This method assumes ``self.states`` is a power of two."""
        # pylint: disable=chained-comparison
        half = self.states // 2

        top_left = RealspaceMatrix(half, self.modes, {})
        top_right = RealspaceMatrix(half, self.modes, {})
        bottom_left = RealspaceMatrix(half, self.modes, {})
        bottom_right = RealspaceMatrix(half, self.modes, {})

        for index, word in self._blocks.items():
            x, y = index

            if x < half and y < half:
                top_left.set_block(x, y, word)
            if x < half and y >= half:
                top_right.set_block(x, y - half, word)
            if x >= half and y < half:
                bottom_left.set_block(x - half, y, word)
            if x >= half and y >= half:
                bottom_right.set_block(x - half, y - half, word)

        return top_left, top_right, bottom_left, bottom_right

    def apply(self, state: VibronicHO) -> VibronicHO:
        """Apply the :class:`~.pennylane.labs.trotter_error.RealspaceMatrix` to an input :class:`~.pennylane.labs.trotter_error.VibronicHO` on the right.

        Args:
            state (VibronicHO): a vibronic wavefunction

        Returns:
            VibronicHO: the result of applying the :class:`~.pennylane.labs.trotter_error.RealspaceMatrix` to ``state``

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceSum, RealspaceCoeffs, RealspaceMatrix
        >>> from pennylane.labs.trotter_error import HOState, VibronicHO
        >>> import numpy as np
        >>> n_states = 1
        >>> n_modes = 3
        >>> gridpoints = 2
        >>> op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs.coeffs(np.array(1), label="lambda"))
        >>> op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs.coeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
        >>> rs_sum = RealspaceSum(n_modes, [op1, op2])
        >>> vib_matrix = RealspaceMatrix(n_states, n_modes, {(0, 0): rs_sum})
        >>> state_dict = {(1, 0, 0): 1, (0, 1, 1): 1}
        >>> state = HOState.from_dict(n_modes, gridpoints, state_dict)
        >>> VibronicHO(n_states, n_modes, gridpoints, [state])
        VibronicHO([HOState(modes=3, gridpoints=2, <Compressed Sparse Row sparse array of dtype 'int64'
            with 2 stored elements and shape (8, 1)>
          Coords	Values
          (3, 0)	1
          (4, 0)	1)])
        """

        if self.states != len(state.ho_states):
            raise ValueError(
                f"Cannot apply RealspaceMatrix on {self.states} states to VibronicHO on {len(state.ho_states)} states."
            )

        if self.modes != state.modes:
            raise ValueError(
                f"Cannot apply RealspaceMatrix on {self.modes} modes to VibronicHO on {state.modes} modes."
            )

        ho_states = []
        for i in range(self.states):
            ho = sum(
                (self.block(i, j).apply(ho_state) for j, ho_state in enumerate(state.ho_states)),
                HOState.zero_state(state.modes, state.gridpoints),
            )
            ho_states.append(ho)

        return VibronicHO(
            states=state.states,
            modes=state.modes,
            gridpoints=state.gridpoints,
            ho_states=ho_states,
        )

    def get_coefficients(self, threshold: float = 0.0) -> dict[tuple[int, int], dict]:
        """Return a dictionary containing the coefficients of the :class:`~.pennylane.labs.trotter_error.RealspaceSum`

        Args:
            threshold (float): tolerance to return coefficients whose magnitude is greater than ``threshold``

        Returns:
            Dict: a dictionary whose keys are the indices of the :class:`~.pennylane.labs.trotter_error.RealspaceMatrix` and whose values are dictionaries obtained by :func:`~pennylane.labs.trotter_error.RealspaceSum.get_coefficients`

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceSum, RealspaceCoeffs, RealspaceMatrix
        >>> import numpy as np
        >>> n_states = 1
        >>> n_modes = 5
        >>> op1 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
        >>> op2 = RealspaceOperator(n_modes, ("P"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="chi"))
        >>> rs_sum = RealspaceSum(n_modes, [op1, op2])
        >>> RealspaceMatrix(n_states, n_modes, {(0, 0): rs_sum}).get_coefficients()
        {(0, 0): {'Q': {(0,): 1.0, (1,): 2.0, (2,): 3.0, (3,): 4.0, (4,): 5.0},
        'P': {(0,): 1.0, (1,): 2.0, (2,): 3.0, (3,): 4.0, (4,): 5.0}}}
        """
        d = {}
        for i, j in product(range(self.states), repeat=2):
            d[(i, j)] = self.block(i, j).get_coefficients(threshold)

        return d

    @classmethod
    def zero(cls, states: int, modes: int) -> RealspaceMatrix:
        """Return a :class:`~.pennylane.labs.trotter_error.RealspaceMatrix` representation of the zero operator.

        Args:
            states (int): the number of electronic states
            modes (int): the number of vibrational modes

        Returns:
            RealspaceMatrix: a :class:`~.pennylane.labs.trotter_error.RealspaceMatrix` on ``states`` electronic states and ``modes`` vibrational modes such that all coefficients are zero
        """
        return cls(states, modes, {})

    def __repr__(self):
        return f"RealspaceMatrix({self._blocks})"


def _is_pow_2(k: int) -> bool:
    """Test if k is a power of two"""
    return k & (k - 1) == 0


def _next_pow_2(k: int) -> int:
    """Return the smallest power of 2 greater than or equal to k"""
    return 2 ** (k - 1).bit_length()

"""The VibronicMatrix class"""

from __future__ import annotations

import math
from itertools import product
from typing import Dict, Tuple, Union

import numpy as np
import scipy as sp

from pennylane.labs.trotter import Fragment
from pennylane.labs.trotter.realspace import RealspaceSum
from pennylane.labs.trotter.utils import _kron, _zeros, is_pow_2, next_pow_2


class VibronicMatrix(Fragment):
    """The VibronicMatrix class"""

    def __init__(
        self,
        states: int,
        blocks: Dict[Tuple[int, int], RealspaceSum] = None,
        sparse: bool = False,
    ) -> VibronicMatrix:

        if blocks is None:
            blocks = {}

        self._blocks = blocks
        self.states = states
        self.sparse = sparse

    def block(self, row: int, col: int) -> RealspaceSum:
        """Return the block indexed at (row, col)"""
        if row < 0 or col < 0:
            raise IndexError(f"Index cannot be negative, got {(row, col)}.")
        if row >= self.states or col >= self.states:
            raise IndexError(
                f"Index out of bounds. Got {(row, col)} but there are only {self.states} states."
            )

        return self._blocks.get((row, col), RealspaceSum.zero())

    def set_block(self, row: int, col: int, word: RealspaceSum) -> None:
        """Set the value of the block indexed at (row, col)"""
        if not isinstance(word, RealspaceSum):
            raise TypeError(f"Block value must be RealspaceSum. Got {type(word)}.")
        if row < 0 or col < 0:
            raise IndexError(f"Index cannot be negative, got {(row, col)}.")
        if row >= self.states or col >= self.states:
            raise IndexError(
                f"Index out of bounds. Got {(row, col)} but there are only {self.states} states."
            )

        if word.is_zero:
            return

        self._blocks[(row, col)] = word

    def matrix(
        self, modes: int, gridpoints: int, sparse: bool = False, basis: str = "realspace"
    ) -> Union[np.ndarray, sp.sparse.csr_matrix]:
        """Returns a sparse matrix representing the operator discretized on the given number of gridpoints"""
        pow2 = next_pow_2(self.states)
        dim = pow2 * gridpoints**modes
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

            block = rs_sum.matrix(gridpoints, modes, basis=basis, sparse=sparse)
            matrix = matrix + _kron(indicator, block)

        return matrix

    def norm(self, modes: int, gridpoints: int) -> float:
        """Compute the spectral norm"""

        if not is_pow_2(gridpoints) or gridpoints <= 0:
            raise ValueError(
                f"Number of gridpoints must be a positive power of 2, got {gridpoints}."
            )

        return self._norm(modes, gridpoints)

    def _norm(self, modes: int, gridpoints: int) -> float:
        # pylint: disable=protected-access
        if self.states == 1:
            return self.block(0, 0).norm(gridpoints, modes, sparse=self.sparse)

        top_left, top_right, bottom_left, bottom_right = self._partition_into_quadrants()

        norm1 = max(top_left._norm(modes, gridpoints), bottom_right._norm(modes, gridpoints))
        norm2 = math.sqrt(top_right._norm(modes, gridpoints) * bottom_left._norm(modes, gridpoints))

        return norm1 + norm2

    def __add__(self, other: VibronicMatrix) -> VibronicMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot add VibronicMatrix on {self.states} states with VibronicMatrix on {other.states} states."
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

        return VibronicMatrix(
            self.states,
            new_blocks,
            sparse=(self.sparse and other.sparse),
        )

    def __sub__(self, other: VibronicMatrix) -> VibronicMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot subtract VibronicMatrix on {self.states} states with VibronicMatrix on {other.states} states."
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

        return VibronicMatrix(
            self.states,
            new_blocks,
            sparse=(self.sparse and other.sparse),
        )

    def __mul__(self, scalar: float) -> VibronicMatrix:
        new_blocks = {}
        for key in self._blocks.keys():
            new_blocks[key] = scalar * self._blocks[key]

        return VibronicMatrix(self.states, new_blocks, sparse=self.sparse)

    __rmul__ = __mul__

    def __imul__(self, scalar: float) -> VibronicMatrix:
        for key in self._blocks.keys():
            self._blocks[key] *= scalar

        return self

    def __matmul__(self, other: VibronicMatrix) -> VibronicMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot multiply VibronicMatrix on {self.states} states with VibronicMatrix on {other.states} states."
            )

        product_matrix = VibronicMatrix(self.states, sparse=(self.sparse and other.sparse))

        for i, j in product(range(self.states), repeat=2):
            block_products = [self.block(i, k) @ other.block(k, j) for k in range(self.states)]
            block_sum = sum(block_products, RealspaceSum.zero())
            product_matrix.set_block(i, j, block_sum)

        return product_matrix

    def __eq__(self, other: VibronicMatrix):
        if self.states != other.states:
            return False

        if self._blocks != other._blocks:
            return False

        return True

    def _partition_into_quadrants(self) -> Tuple[VibronicMatrix]:
        # pylint: disable=chained-comparison
        half = self.states // 2

        top_left = VibronicMatrix(half, {}, sparse=self.sparse)
        top_right = VibronicMatrix(half, {}, sparse=self.sparse)
        bottom_left = VibronicMatrix(half, {}, sparse=self.sparse)
        bottom_right = VibronicMatrix(half, {}, sparse=self.sparse)

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

"""The VibronicMatrix class"""

from __future__ import annotations

import math
from itertools import product
from typing import Dict, Tuple, Union

import numpy as np
import scipy as sp

from pennylane.labs.trotter import Fragment
from pennylane.labs.trotter.realspace import RealspaceSum
from pennylane.labs.trotter.realspace.matrix import _kron, _zeros

# pylint: disable=protected-access

class VibronicMatrix(Fragment):
    """The VibronicMatrix class"""

    def __init__(
        self,
        states: int,
        modes: int,
        blocks: Dict[Tuple[int, int], RealspaceSum] = None,
    ) -> VibronicMatrix:

        if blocks is None:
            blocks = {}

        self._blocks = blocks
        self.states = states
        self.modes = modes

    def block(self, row: int, col: int) -> RealspaceSum:
        """Return the block indexed at (row, col)"""
        if row < 0 or col < 0:
            raise IndexError(f"Index cannot be negative, got {(row, col)}.")
        if row >= self.states or col >= self.states:
            raise IndexError(
                f"Index out of bounds. Got {(row, col)} but there are only {self.states} states."
            )

        return self._blocks.get((row, col), RealspaceSum.zero(self.modes))

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
        self, gridpoints: int, sparse: bool = False, basis: str = "realspace"
    ) -> Union[np.ndarray, sp.sparse.csr_matrix]:
        """Returns a sparse matrix representing the operator discretized on the given number of gridpoints"""
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

    def norm(self, params: Dict) -> float:
        """Compute the spectral norm"""
        try:
            gridpoints = params["gridpoints"]
        except KeyError as e:
            raise KeyError("Need to specify the number of gridpoints") from e

        if not _is_pow_2(gridpoints) or gridpoints <= 0:
            raise ValueError(
                f"Number of gridpoints must be a positive power of 2, got {gridpoints}."
            )

        padded = VibronicMatrix(_next_pow_2(self.states), self.modes, self.blocks_)

        return padded._norm(params)

    def _norm(self, params: Dict) -> float:
        if self.states == 1:
            return self.block(0, 0).norm(params)

        top_left, top_right, bottom_left, bottom_right = self._partition_into_quadrants()

        norm1 = max(top_left._norm(params), bottom_right._norm(params))
        norm2 = math.sqrt(top_right._norm(params) * bottom_left._norm(params))

        return norm1 + norm2

    def __add__(self, other: VibronicMatrix) -> VibronicMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot add VibronicMatrix on {self.states} states with VibronicMatrix on {other.states} states."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot add VibronicMatrix on {self.modes} states with VibronicMatrix on {other.modes} states."
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
            self.modes,
            new_blocks,
        )

    def __sub__(self, other: VibronicMatrix) -> VibronicMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot subtract VibronicMatrix on {self.states} states with VibronicMatrix on {other.states} states."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot subtract VibronicMatrix on {self.modes} states with VibronicMatrix on {other.modes} states."
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
            self.modes,
            new_blocks,
        )

    def __mul__(self, scalar: float) -> VibronicMatrix:
        new_blocks = {}
        for key in self._blocks.keys():
            new_blocks[key] = scalar * self._blocks[key]

        return VibronicMatrix(self.states, self.modes, new_blocks)

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

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot multiply VibronicMatrix on {self.states} states with VibronicMatrix on {other.states} states."
            )

        product_matrix = VibronicMatrix(self.states, self.modes)

        for i, j in product(range(self.states), repeat=2):
            block_products = [self.block(i, k) @ other.block(k, j) for k in range(self.states)]
            block_sum = sum(block_products, RealspaceSum.zero(self.modes))
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

        top_left = VibronicMatrix(half, self.modes, {})
        top_right = VibronicMatrix(half, self.modes, {})
        bottom_left = VibronicMatrix(half, self.modes, {})
        bottom_right = VibronicMatrix(half, self.modes, {})

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

    def get_coefficients(self, threshold: float = 0.0):
        """Return the coefficients in a dictionary"""
        d = {}
        for i, j in product(range(self.states), repeat=2):
            d[(i, j)] = self.block(i, j).get_coefficients(threshold)

        return d


def _is_pow_2(k: int) -> bool:
    """Test if k is a power of two"""
    return (k & (k - 1) == 0) or k == 0


def _next_pow_2(k: int) -> int:
    """Return the smallest power of 2 greater than or equal to k"""
    return 2 ** (k - 1).bit_length()

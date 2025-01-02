"""The VibronicMatrix class"""

from __future__ import annotations

from itertools import product
from typing import Dict, Tuple

from scipy.sparse import csr_matrix
from vibronic_term import VibronicWord


class VibronicMatrix:
    """The VibronicMatrix class"""

    def __init__(
        self, states: int, modes: int, blocks: Dict[Tuple[int, int], VibronicWord] = None
    ) -> VibronicMatrix:
        if blocks is None:
            blocks = {}

        self._blocks = blocks
        self.states = states
        self.modes = modes

    def block(self, row: int, col: int) -> VibronicWord:
        """Return the block indexed at (row, col)"""
        return self._blocks.get((row, col), VibronicWord(tuple()))

    def set_block(self, row: int, col: int, word: VibronicWord) -> None:
        """Set the value of the block indexed at (row, col)"""
        if not isinstance(word, VibronicWord):
            raise TypeError(f"Block value must be VibronicWord. Got {type(word)}.")

        self._blocks[(row, col)] = word

    def matrix(self, gridpoints: int) -> csr_matrix:
        """Returns a sparse matrix representing the operator discretized on the given number of gridpoints"""
        raise NotImplementedError

    def __add__(self, other: VibronicMatrix) -> VibronicMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot add VibronicMatrix on {self.states} states with VibronicMatrix on {other.states} states."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot add VibronicMatrix on {self.modes} modes with VibronicMatrix on {other.modes} modes."
            )

        new_blocks = {}
        l_keys = set(self._blocks.keys())
        r_keys = set(self._blocks.keys())

        for key in l_keys.intersection(r_keys):
            new_blocks[key] = self._blocks[key] + other._blocks[key]

        for key in l_keys.difference(r_keys):
            new_blocks[key] = self._blocks[key]

        for key in r_keys.difference(l_keys):
            new_blocks[key] = other.blocks[key]

        return VibronicMatrix(self.states, self.modes, new_blocks)

    def __sub__(self, other: VibronicMatrix) -> VibronicMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot subtract VibronicMatrix on {self.states} states with VibronicMatrix on {other.states} states."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot subtract VibronicMatrix on {self.modes} modes with VibronicMatrix on {other.modes} modes."
            )

        new_blocks = {}
        l_keys = set(self._blocks.keys())
        r_keys = set(self._blocks.keys())

        for key in l_keys.intersection(r_keys):
            new_blocks[key] = self._blocks[key] - other._blocks[key]

        for key in l_keys.difference(r_keys):
            new_blocks[key] = self._blocks[key]

        for key in r_keys.difference(l_keys):
            new_blocks[key] = (-1) * other.blocks[key]

        return VibronicMatrix(self.states, self.modes, new_blocks)

    def __mul__(self, scalar: float) -> VibronicMatrix:
        new_blocks = {}
        for key in self._blocks.keys():
            new_blocks[key] = scalar * self._blocks[key]

        return VibronicMatrix(self.states, self.modes, new_blocks)

    __rmul__ = __mul__

    def __matmul__(self, other: VibronicMatrix) -> VibronicMatrix:
        if self.states != other.states:
            raise ValueError(
                f"Cannot multiply VibronicMatrix on {self.states} states with VibronicMatrix on {other.states} states."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot multiply VibronicMatrix on {self.modes} states with VibronicMatrix on {other.modes} states."
            )

        product_matrix = VibronicMatrix(self.states, self.modes)

        for i, j in product(range(self.states), repeat=2):
            block_products = [self.block(i, k) @ other.block(k, j) for k in range(self.states)]
            block_sum = sum(block_products, VibronicWord(tuple()))
            product_matrix.set_block(i, j, block_sum)

        return product_matrix

    def __eq__(self, other: VibronicMatrix):
        if self.states != other.states:
            return False

        if self.modes != other.modes:
            return False

        if self._blocks != other._blocks:
            return False

        return True


def commutator(a: VibronicMatrix, b: VibronicMatrix):
    """Return the commutator [a, b] = ab - ba"""
    return a @ b - b @ a

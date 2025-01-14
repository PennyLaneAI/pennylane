"""The VibronicMatrix class"""

from __future__ import annotations

import math
from itertools import product
from typing import Dict, Tuple, Union

import numpy as np
import scipy as sp

from pennylane.labs.vibronic.utils import _kron, _zeros, is_pow_2, op_norm, word_to_matrix

from .vibronic_term import VibronicWord


class VibronicMatrix:
    """The VibronicMatrix class"""

    def __init__(
        self, states: int, modes: int, blocks: Dict[Tuple[int, int], VibronicWord] = None
    ) -> VibronicMatrix:

        if not is_pow_2(states) or states == 0:
            raise ValueError(f"The number of states must be a power of 2, got {states}.")

        if blocks is None:
            blocks = {}

        self._blocks = blocks
        self.states = states
        self.modes = modes

    def block(self, row: int, col: int) -> VibronicWord:
        """Return the block indexed at (row, col)"""
        if row < 0 or col < 0:
            raise IndexError(f"Index cannot be negative, got {(row, col)}.")
        if row >= self.states or col >= self.states:
            raise IndexError(
                f"Index out of bounds. Got {(row, col)} but there are only {self.states} states."
            )

        return self._blocks.get((row, col), VibronicWord.zero_word())

    def set_block(self, row: int, col: int, word: VibronicWord) -> None:
        """Set the value of the block indexed at (row, col)"""
        if not isinstance(word, VibronicWord):
            raise TypeError(f"Block value must be VibronicWord. Got {type(word)}.")
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
        self, gridpoints: int, sparse: bool = False
    ) -> Union[np.ndarray, sp.sparse.csr_matrix]:
        """Returns a sparse matrix representing the operator discretized on the given number of gridpoints"""
        dim = self.states * gridpoints**self.modes
        matrix = _zeros((dim, dim), sparse=sparse)

        for index, word in self._blocks.items():
            if sparse:
                data = np.array([1])
                indices = (np.array([index[0]]), np.array([index[1]]))
                shape = (self.states, self.states)
                indicator = sp.sparse.csr_matrix((data, indices), shape=shape)
            else:
                indicator = np.zeros(shape=(self.states, self.states))
                indicator[index] = 1

            block = word_to_matrix(word, self.modes, gridpoints, sparse=sparse)
            matrix += _kron(indicator, block)

        return matrix

    def norm(self, gridpoints: int) -> float:
        """Compute the spectral norm"""

        if not is_pow_2(gridpoints) or gridpoints <= 0:
            raise ValueError(
                f"Number of gridpoints must be a positive power of 2, got {gridpoints}."
            )

        return self._norm(gridpoints)

    def _norm(self, gridpoints: int) -> float:
        # pylint: disable=protected-access
        if self.states == 1:
            return self._norm_base_case(gridpoints)

        top_left, top_right, bottom_left, bottom_right = self._partition_into_quadrants()

        norm1 = max(top_left._norm(gridpoints), bottom_right._norm(gridpoints))
        norm2 = math.sqrt(top_right._norm(gridpoints) * bottom_left._norm(gridpoints))

        return norm1 + norm2

    def _norm_base_case(self, gridpoints: int) -> float:
        if self.states != 1:
            raise RuntimeError("Base case called on VibronicMatrix with >1 state")

        word = self.block(0, 0)
        norm = 0

        for term in word.terms:
            term_op_norm = math.prod(map(lambda op: op_norm(gridpoints) ** len(op), term.ops))
            coeff_sum = sum(
                abs(term.coeffs.compute(index))
                for index in product(range(self.modes), repeat=len(term.ops))
            )
            norm += coeff_sum * term_op_norm

        return norm

    def _norm_base_case_bad(self, gridpoints: int) -> float:
        if self.states != 1:
            raise RuntimeError("Base case called on VibronicMatrix with >1 state")

        word = self.block(0, 0)
        norm = 0

        for term in word.terms:
            term_op_norm = math.prod(map(lambda op: op_norm(gridpoints) ** len(op), term.ops))
            scalar = term.coeffs.compute_average() * (self.modes ** len(term.ops))
            norm += scalar * term_op_norm

        return norm

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
        r_keys = set(other._blocks.keys())

        for key in l_keys.intersection(r_keys):
            new_blocks[key] = self._blocks[key] + other._blocks[key]

        for key in l_keys.difference(r_keys):
            new_blocks[key] = self._blocks[key]

        for key in r_keys.difference(l_keys):
            new_blocks[key] = other._blocks[key]

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
        r_keys = set(other._blocks.keys())

        for key in l_keys.intersection(r_keys):
            new_blocks[key] = self._blocks[key] - other._blocks[key]

        for key in l_keys.difference(r_keys):
            new_blocks[key] = self._blocks[key]

        for key in r_keys.difference(l_keys):
            new_blocks[key] = (-1) * other._blocks[key]

        return VibronicMatrix(self.states, self.modes, new_blocks)

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
                f"Cannot multiply VibronicMatrix on {self.modes} states with VibronicMatrix on {other.modes} states."
            )

        product_matrix = VibronicMatrix(self.states, self.modes)

        for i, j in product(range(self.states), repeat=2):
            block_products = [self.block(i, k) @ other.block(k, j) for k in range(self.states)]
            block_sum = sum(block_products, VibronicWord.zero_word())
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

    def num_coeffs(self) -> int:
        """Find number of coefficients"""

        term_count = 0
        zero_count = 0
        coeffs = 0
        for _, word in self._blocks.items():
            print(word.is_zero, len(word.terms))
            for term in word.terms:
                coeffs += self.modes ** len(term.ops)
                term_count += 1
                if term.is_zero:
                    zero_count += 1

        print(f"{zero_count}/{term_count}", coeffs)


def commutator(a: VibronicMatrix, b: VibronicMatrix):
    """Return the commutator [a, b] = ab - ba"""
    return a @ b - b @ a

"""The RealspaceOperator class"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import product
from typing import Sequence, Tuple

import numpy as np

from pennylane.labs.pf.abstract import Fragment
from pennylane.labs.pf.utils import op_norm
from .tree import Node


class RealspaceOperator:
    """The RealspaceOperator class"""

    def __init__(self, ops: Tuple[str], coeffs: Node) -> RealspaceOperator:
        self.ops = ops
        self.coeffs = coeffs

    def __add__(self, other: RealspaceOperator) -> RealspaceOperator:
        if self.is_zero:
            return other

        if other.is_zero:
            return self

        if self.ops != other.ops:
            raise ValueError(f"Cannot add term {self.ops} with term {other.ops}.")

        return RealspaceOperator(self.ops, Node.sum_node(self.coeffs, other.coeffs))

    def __sub__(self, other: RealspaceOperator) -> RealspaceOperator:
        if self.is_zero:
            return (-1) * other

        if other.is_zero:
            return self

        if self.ops != other.ops:
            raise ValueError(f"Cannot subtract term {self.ops} with term {other.ops}.")

        return RealspaceOperator(
            self.ops, Node.sum_node(self.coeffs, Node.scalar_node(-1, other.coeffs))
        )

    def __mul__(self, scalar: float) -> RealspaceOperator:
        if np.isclose(scalar, 0):
            return RealspaceOperator.zero_term()

        return RealspaceOperator(self.ops, Node.scalar_node(scalar, self.coeffs))

    __rmul__ = __mul__

    def __imul__(self, scalar: float) -> RealspaceOperator:
        if np.isclose(scalar, 0):
            return RealspaceOperator.zero_term()

        self.coeffs = Node.scalar_node(scalar, self.coeffs)
        return self

    def __matmul__(self, other: RealspaceOperator) -> RealspaceOperator:
        if other.is_zero:
            return self

        return RealspaceOperator(self.ops + other.ops, Node.outer_node(self.coeffs, other.coeffs))

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
    def zero_term(cls) -> RealspaceOperator:
        """Returns a RealspaceOperator representing 0"""
        return RealspaceOperator(tuple(), Node.tensor_node(np.array(0)))


class RealspaceSum(Fragment):
    """The RealspaceSum class"""

    def __init__(self, ops: Sequence[RealspaceOperator]) -> RealspaceSum:
        ops = tuple(filter(lambda op: not op.is_zero, ops))
        self.is_zero = len(ops) == 0

        self._lookup = defaultdict(
            lambda: RealspaceOperator.zero_term()
        )  # pylint: disable=unnecessary-lambda
        for op in ops:
            self._lookup[op.ops] += op

        self.ops = tuple(self._lookup.values())

    def __add__(self, other: RealspaceSum) -> RealspaceSum:
        l_ops = {term.ops for term in self.ops}
        r_ops = {term.ops for term in other.ops}

        new_ops = []

        for op in l_ops.intersection(r_ops):
            new_ops.append(self._lookup[op] + other._lookup[op])

        for op in l_ops.difference(r_ops):
            new_ops.append(self._lookup[op])

        for op in r_ops.difference(l_ops):
            new_ops.append(other._lookup[op])

        return RealspaceSum(new_ops)

    def __sub__(self, other: RealspaceSum) -> RealspaceSum:
        l_ops = {term.ops for term in self.ops}
        r_ops = {term.ops for term in other.ops}

        new_terms = []

        for op in l_ops.intersection(r_ops):
            new_terms.append(self._lookup[op] - other._lookup[op])

        for op in l_ops.difference(r_ops):
            new_terms.append(self._lookup[op])

        for op in r_ops.difference(l_ops):
            new_terms.append((-1) * other._lookup[op])

        return RealspaceSum(new_terms)

    def __mul__(self, scalar: float) -> RealspaceSum:
        return RealspaceSum([scalar * term for term in self.ops])

    __rmul__ = __mul__

    def __imul__(self, scalar: float) -> RealspaceSum:
        for term in self.ops:
            term *= scalar

        return self

    def __matmul__(self, other: RealspaceSum) -> RealspaceSum:
        return RealspaceSum(
            [
                RealspaceOperator(l_term.ops + r_term.ops, l_term.coeffs @ r_term.coeffs)
                for l_term, r_term in product(self.ops, other.ops)
            ]
        )

    def __repr__(self) -> str:
        return self.ops.__repr__()

    def __eq__(self, other: RealspaceSum) -> bool:
        return self._lookup == other._lookup

    @classmethod
    def zero_word(cls) -> RealspaceSum:
        """Return a RealspaceSum representing 0"""
        return RealspaceSum([RealspaceOperator.zero_term()])

    def norm(self, gridpoints: int, modes: int, sparse: bool = False) -> float:
        # pylint: disable=eval-used

        norm = 0
        if sparse:
            for op in self.ops:
                term_op_norm = math.prod(map(lambda op: op_norm(gridpoints) ** len(op), op.ops))
                compiled, local_vars = op.coeffs.compile(to_numpy=True)
                coeff_sum = eval(compiled, {}, local_vars)
                norm += coeff_sum * term_op_norm

            return norm

        for op in self.ops:
            term_op_norm = math.prod(map(lambda op: op_norm(gridpoints) ** len(op), op.ops))
            compiled, local_vars = op.coeffs.compile()

            coeff_sum = 0
            for index in product(range(modes), repeat=len(op.ops)):
                for i, j in enumerate(index):
                    local_vars[f"idx{i}"] = j

                coeff_sum += eval(compiled, {}, local_vars)

            norm += coeff_sum * term_op_norm

        return norm

    def mul_state(self, state):
        pass

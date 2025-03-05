"""The RealspaceOperator class"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import product
from typing import Sequence, Tuple, Union

import numpy as np
import scipy as sp

from pennylane.labs.trotter import Fragment
from pennylane.labs.trotter.realspace.ho_state import HOState
from pennylane.labs.trotter.utils import _zeros, op_norm, string_to_matrix, tensor_with_identity

from .tree import Node


class RealspaceOperator:
    """The RealspaceOperator class"""

    def __init__(self, modes: int, ops: Tuple[str], coeffs: Node) -> RealspaceOperator:
        self.modes = modes
        self.ops = ops
        self.coeffs = coeffs

    def matrix(
        self, gridpoints: int, modes: int, basis: str = "realspace", sparse: bool = False
    ) -> Union[np.ndarray, sp.sparse.csr_array]:
        """Return a matrix representation of the operator"""

        matrices = [string_to_matrix(op, gridpoints, basis=basis, sparse=sparse) for op in self.ops]
        final_matrix = _zeros(shape=(gridpoints**modes, gridpoints**modes), sparse=sparse)

        if sparse:
            indices = self.coeffs.nonzero()
        else:
            indices = product(range(modes), repeat=len(self.ops))

        compiled, local_vars = self.coeffs.compile()
        for index in indices:
            var_dict = {f"idx{i}": j for i, j in enumerate(index)}
            coeff = eval(compiled, var_dict, local_vars)
            matrix = coeff * tensor_with_identity(modes, gridpoints, index, matrices, sparse=sparse)
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

        return RealspaceOperator(self.modes, self.ops, Node.sum_node(self.coeffs, other.coeffs))

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
            self.modes, self.ops, Node.sum_node(self.coeffs, Node.scalar_node(-1, other.coeffs))
        )

    def __mul__(self, scalar: float) -> RealspaceOperator:
        if np.isclose(scalar, 0):
            return RealspaceOperator.zero_term(self.modes)

        return RealspaceOperator(self.modes, self.ops, Node.scalar_node(scalar, self.coeffs))

    __rmul__ = __mul__

    def __imul__(self, scalar: float) -> RealspaceOperator:
        if np.isclose(scalar, 0):
            return RealspaceOperator.zero_term(self.modes)

        self.coeffs = Node.scalar_node(scalar, self.coeffs)
        return self

    def __matmul__(self, other: RealspaceOperator) -> RealspaceOperator:
        if other.is_zero:
            return self

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot multiply RealspaceOperator on {self.modes} modes with RealspaceOperator on {other.modes} modes."
            )

        return RealspaceOperator(
            self.modes, self.ops + other.ops, Node.outer_node(self.coeffs, other.coeffs)
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
        return RealspaceOperator(modes, tuple(), Node.tensor_node(np.array(0)))


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

    def matrix(self, gridpoints: int, modes: int, basis: str = "realspace", sparse: bool = False):
        """Return a matrix representation of the RealspaceSum"""

        final_matrix = _zeros(shape=(gridpoints**modes, gridpoints**modes), sparse=sparse)
        for op in self.ops:
            final_matrix = final_matrix + op.matrix(gridpoints, modes, basis=basis, sparse=sparse)

        return final_matrix

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

    def apply(self, state: HOState) -> HOState:
        if not isinstance(state, HOState):
            raise TypeError

        mat = self.matrix(state.gridpoints, state.modes, basis="harmonic", sparse=True)

        return HOState.from_scipy(
            state.modes,
            state.gridpoints,
            mat @ state.vector,
        )

    def expectation(self, state_left: HOState, state_right: HOState) -> float:
        """Compute expectation value"""
        return state_left.dot(self.apply(state_right))

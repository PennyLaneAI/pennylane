"""Base class for Vibronic Hamiltonian"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from pennylane.labs.pf.realspace import Node, RealspaceOperator, RealspaceSum
from pennylane.labs.vibronic.utils import is_pow_2

from .vibronic_matrix import VibronicMatrix, commutator


class VibronicHamiltonian:
    """Base class for Vibronic Hamiltonians"""

    def __init__(
        self,
        states: int,
        modes: int,
        omegas: np.ndarray,
        phis: Sequence[np.ndarray],
        sparse: bool = False,
    ):
        if not is_pow_2(states) or states == 0:
            raise ValueError(f"States must be a positive power of 2, got {states} states.")

        for i, phi in enumerate(phis):
            shape = (states, states) + (modes,) * i

            if phi.shape != shape:
                raise ValueError(
                    f"{i}th order coefficient tensor must have shape {shape}, got shape {phi.shape}"
                )

        if omegas.shape != (modes,):
            raise TypeError(f"Omegas must have shape {(modes,)}, got shape {omegas.shape}.")

        self.states = states
        self.modes = modes
        self.phis = phis
        self.omegas = omegas
        self.sparse = sparse
        self.order = len(phis) - 1

    def fragment(self, index: int) -> VibronicMatrix:
        """Return the fragment at the given index"""

        if index not in range(self.states + 1):
            raise ValueError("Index out of range")

        if index == self.states:
            return self._p_fragment()

        return self._fragment(index)

    def v_word(self, i: int, j: int) -> RealspaceSum:
        """Get V_ij"""
        if i > self.states or j > self.states:
            raise ValueError(
                f"Dimension out of bounds. Got ({i}, {j}) but V is dimension ({self.states}, {self.states})."
            )

        realspace_ops = []
        for k, phi in enumerate(self.phis):
            op = ("Q",) * k
            realspace_op = RealspaceOperator(op, Node.tensor_node(phi[i, j], label=(f"phis[{k}][{i}, {j}]", self.phis)))
            realspace_ops.append(realspace_op)

        return RealspaceSum(realspace_ops)

    def _p_fragment(self) -> VibronicMatrix:
        term = RealspaceOperator(
            ("P", "P"),
            Node.tensor_node(np.diag(self.omegas) / 2, label=("omegas", np.diag(self.omegas) / 2)),
        )
        word = RealspaceSum((term,))
        blocks = {(i, i): word for i in range(self.states)}
        return VibronicMatrix(self.states, self.modes, blocks, sparse=self.sparse)

    def _fragment(self, i: int) -> VibronicMatrix:
        blocks = {(j, i ^ j): self.v_word(j, i ^ j) for j in range(self.states)}
        return VibronicMatrix(self.states, self.modes, blocks, sparse=self.sparse)

    def block_operator(self) -> VibronicMatrix:
        """Return the block representation of the Hamiltonian"""

        operator = self._p_fragment()
        for i in range(self.states):
            operator += self._fragment(i)

        return operator

    def matrix(self, gridpoints: int) -> csr_matrix:
        """Return a csr matrix representation of the Hamiltonian"""
        return self.block_operator().matrix(gridpoints)

    def epsilon(self, delta: float) -> VibronicMatrix:
        # pylint: disable=arguments-out-of-order
        """Compute the error matrix"""
        scalar = -(delta**2) / 24
        epsilon = VibronicMatrix(self.states, self.modes, sparse=self.sparse)

        for i in range(self.states):
            for j in range(i + 1, self.states + 1):
                epsilon += self._commute_fragments(i, i, j)
                for k in range(i + 1, self.states + 1):
                    epsilon += 2 * self._commute_fragments(k, i, j)

        epsilon *= scalar
        return epsilon

    def _commute_fragments(self, i: int, j: int, k: int) -> VibronicMatrix:
        # if i == self.states and j < self.states and k == self.states:
        #    return self._commute_hN_hm_hN(j)

        return commutator(self.fragment(i), commutator(self.fragment(j), self.fragment(k)))

    def _commute_hN_hm_hN(self, m: int) -> VibronicMatrix:
        blocks = {}
        for j in range(self.states):
            node = Node.scalar_node(
                2,
                Node.hadamard_node(
                    Node.tensor_node(
                        self.betas[j, m ^ j], label=(f"betas[{j},{m ^ j}]", self.betas)
                    ),
                    Node.outer_node(
                        Node.tensor_node(self.omegas, label=("omegas", np.diag(self.omegas) / 2)),
                        Node.tensor_node(self.omegas, label=("omegas", np.diag(self.omegas) / 2)),
                    ),
                ),
            )

            term = RealspaceOperator(("P", "P"), node)
            blocks[(j, m ^ j)] = RealspaceSum((term,))

        return VibronicMatrix(self.states, self.modes, blocks, sparse=self.sparse)

    def __add__(self, other: VibronicHamiltonian) -> VibronicHamiltonian:
        if not isinstance(other, VibronicHamiltonian):
            raise TypeError(f"Cannot add VibronicHamiltonian with type {type(other)}.")

        if self.states != other.states:
            raise ValueError(
                f"Cannot add VibronicHamiltonian on {self.states} with VibronicHamiltonian on {other.states}."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot add VibronicHamiltonian on {self.modes} with VibronicHamiltonian on {other.modes}."
            )

        return VibronicHamiltonian(
            self.states,
            self.modes,
            self.omegas + other.omegas,
            [x + y for x, y in zip(self.phis, other.phis)],
        )

    def __mul__(self, scalar) -> VibronicHamiltonian:
        return VibronicHamiltonian(
            self.states,
            self.modes,
            scalar * self.omegas,
            [scalar * phi for phi in self.phis],
        )

    __rmul__ = __mul__

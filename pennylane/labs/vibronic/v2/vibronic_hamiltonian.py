"""Base class for Vibronic Hamiltonian"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from vibronic_matrix import VibronicMatrix, commutator
from vibronic_term import VibronicTerm, VibronicWord
from vibronic_tree import Node


class VibronicHamiltonian:
    """Base class for Vibronic Hamiltonians"""

    def __init__(
        self,
        states: int,
        modes: int,
        alphas: np.ndarray,
        betas: np.ndarray,
        lambdas: np.ndarray,
        omegas: np.ndarray,
    ):
        if alphas.shape != (states, states, modes):
            raise TypeError
        if betas.shape != (states, states, modes, modes):
            raise TypeError
        if lambdas.shape != (states, states):
            raise TypeError
        if omegas.shape != (modes,):
            raise TypeError

        self.states = states
        self.modes = modes
        self.alphas = alphas
        self.betas = betas
        self.lambdas = lambdas
        self.omegas = omegas

    def fragment(self, index: int) -> VibronicMatrix:
        """Return the fragment at the given index"""
        if index not in range(self.states + 1):
            raise ValueError("Index out of range")

        if index == self.states:
            return self._p_fragment()

        return self._fragment(index)

    def v_word(self, i: int, j: int) -> VibronicWord:
        """Get V_ij"""
        if i > self.states or j > self.states:
            raise ValueError(
                f"Dimension out of bounds. Got ({i}, {j}) but V is dimension ({self.states}, {self.states})."
            )

        return VibronicWord(
            (
                VibronicTerm(tuple(), Node.tensor_node(self.lambdas[i, j])),
                VibronicTerm(("Q",), Node.tensor_node(self.alphas[i, j])),
                VibronicTerm(("Q", "Q"), Node.tensor_node(self.betas[i, j])),
            )
        )

    def _p_fragment(self) -> VibronicMatrix:
        term = VibronicTerm(("P", "P"), Node.tensor_node(np.diag(self.omegas) / 2))
        word = VibronicWord((term,))
        blocks = {(i, i): word for i in range(self.states)}
        return VibronicMatrix(self.states, self.modes, blocks)

    def _fragment(self, i: int) -> VibronicMatrix:
        blocks = {(j, i ^ j): self.v_word(j, i ^ j) for j in range(self.states)}
        return VibronicMatrix(self.states, self.modes, blocks)

    def block_operator(self) -> VibronicMatrix:
        """Return the block representation of the Hamiltonian"""

        operator = self._p_fragment()
        for i in range(self.states):
            operator += self._fragment(i)

        return operator

    def matrix(self, gridpoints: int) -> csr_matrix:
        """Return a csr matrix representation of the Hamiltonian"""
        return self.block_operator().matrix(gridpoints)

    def epsilon(self, delta) -> VibronicMatrix:
        # pylint: disable=arguments-out-of-order
        """Compute the error matrix"""
        scalar = -(delta**2) / 24
        epsilon = VibronicMatrix(self.states, self.modes)

        for i in range(self.states):
            for j in range(i + 1, self.states + 1):
                epsilon += self._commute_fragments(i, i, j)
                for k in range(i + 1, self.states + 1):
                    epsilon += 2 * self._commute_fragments(k, i, j)

        epsilon *= scalar
        return epsilon

    def _commute_fragments(self, i: int, j: int, k: int) -> VibronicMatrix:
        if i == self.states and j < self.states and k == self.states:
            return self._commute_hN_hm_hN(j)

        return commutator(self.fragment(i), commutator(self.fragment(j), self.fragment(k)))

    def _commute_hN_hm_hN(self, m: int) -> VibronicMatrix:
        blocks = {}
        for j in range(self.states):
            node = Node.scalar_node(
                2,
                Node.hadamard_node(
                    Node.tensor_node(self.betas[j, m ^ j]),
                    Node.outer_node(Node.tensor_node(self.omegas), Node.tensor_node(self.omegas)),
                ),
            )

            term = VibronicTerm(("P", "P"), node)
            blocks[(j, m ^ j)] = VibronicWord((term,))

        return VibronicMatrix(self.states, self.modes, blocks)

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
            self.alphas + other.alphas,
            self.betas + other.betas,
            self.lambdas + other.lambdas,
            self.omegas + other.omegas,
        )

    def __mul__(self, scalar) -> VibronicHamiltonian:
        return VibronicHamiltonian(
            self.states,
            self.modes,
            scalar * self.alphas,
            scalar * self.betas,
            scalar * self.lambdas,
            scalar * self.omegas,
        )

    __rmul__ = __mul__

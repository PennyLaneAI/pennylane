"""Base classes for Vibronic Hamiltonians"""

from __future__ import annotations

import itertools
from typing import Dict, Tuple

import numpy as np
import scipy as sp


class VibronicWord:
    """Representation of the operators inside the matrix representation of the Vibronic
    Hamiltonian"""

    def __init__(self, operator: dict = None):
        if operator is None:
            operator = {}

        self.operator = operator

    def __bool__(self):
        return bool(self.operator)

    def __eq__(self, other: VibronicWord) -> bool:
        if set(self.operator.keys()) != set(other.operator.keys()):
            return False

        for key in self.operator.keys():
            if not np.allclose(self.operator[key], other.operator[key]):
                return False

        return True

    def __repr__(self) -> str:
        return self.operator.__repr__()

    def __add__(self, other: VibronicWord) -> VibronicWord:
        new_operator = _add_dicts(self.operator, other.operator)
        return VibronicWord(operator=new_operator)

    def __sub__(self, other: VibronicWord) -> VibronicWord:
        new_operator = _sub_dicts(self.operator, other.operator)
        return VibronicWord(operator=new_operator)
        return self + (-1) * other

    def __mul__(self, scalar) -> VibronicWord:
        if not isinstance(scalar, (complex, int, float)):
            raise TypeError(f"Cannot multiply VibronicWord by type {type(scalar)}.")

        new_operator = {}
        for key, value in self.operator.items():
            new_operator[key] = value * scalar

        return VibronicWord(operator=new_operator)

    def get_op(self, op: Tuple[str]):
        """Return the operator indexed by op"""
        return self.operator[op]

    __rmul__ = __mul__

    def __matmul__(self, other: VibronicWord) -> VibronicWord:
        if not isinstance(other, VibronicWord):
            raise TypeError(f"Cannot multiply VibronicWord by type {type(other)}.")

        new_operator = {}
        for l_key, l_value in self.operator.items():
            for r_key, r_value in other.operator.items():
                new_value = np.multiply.outer(l_value, r_value)
                new_key = l_key + r_key

                _add_coeffs(new_operator, new_key, new_value)

        return VibronicWord(operator=new_operator)

    def add_op(self, operator, coeffs) -> None:
        """Add an operator to the word"""
        self.operator[operator] = coeffs


class VibronicBlockMatrix:
    """Representation of a block matrix made of position/momentum operators"""

    def __init__(self, dim: int, blocks: Dict[Tuple[int, int], VibronicWord] = None):
        if blocks is None:
            blocks = {}

        self.blocks = blocks
        self.dim = dim

    def get_block(self, row: int, col: int) -> VibronicWord:
        """Get the word inside the (row, col) block"""
        return self.blocks.get((row, col), VibronicWord({}))

    def set_block(self, row: int, col: int, word: VibronicWord) -> None:
        """Set the word inside the (row, col) block"""
        self.blocks[(row, col)] = word

    def __repr__(self) -> str:
        return self.blocks.__repr__()

    def __add__(self, other: VibronicBlockMatrix) -> VibronicBlockMatrix:
        if self.dim != other.dim:
            raise ValueError(
                f"Cannot add VibronicBlockMatrix of shape ({self.dim}, {self.dim}) with shape ({other.dim}, {other.dim})."
            )

        new_blocks = _add_dicts(self.blocks, other.blocks)

        return VibronicBlockMatrix(dim=self.dim, blocks=new_blocks)

    def __sub__(self, other: VibronicBlockMatrix) -> VibronicBlockMatrix:
        return self + (-1) * other

    def __mul__(self, scalar) -> VibronicBlockMatrix:
        if not isinstance(scalar, (complex, int, float)):
            raise TypeError(f"Cannot multiply VibronicBlockMatrix by type {type(scalar)}.")

        new_blocks = {}
        for key, value in self.blocks.items():
            new_blocks[key] = value * scalar

        return VibronicBlockMatrix(dim=self.dim, blocks=new_blocks)

    __rmul__ = __mul__

    def __matmul__(self, other: VibronicBlockMatrix) -> VibronicBlockMatrix:
        if self.dim != other.dim:
            raise ValueError(
                f"Cannot multiply VibronicBlockMatrix of shape ({self.dim}, {self.dim}) with shape ({other.dim}, {other.dim})."
            )

        new_blocks = {}
        for i in range(self.dim):
            for j in range(self.dim):
                block_products = [
                    self.get_block(i, k) @ other.get_block(k, j) for k in range(self.dim)
                ]
                block_sum = sum(block_products, VibronicWord({}))
                if block_sum:
                    new_blocks[(i, j)] = block_sum

        return VibronicBlockMatrix(dim=self.dim, blocks=new_blocks)

    def __eq__(self, other: VibronicBlockMatrix) -> VibronicBlockMatrix:
        if self.dim != other.dim:
            return False

        for i in range(self.dim):
            for j in range(self.dim):
                if self.get_block(i, j) != other.get_block(i, j):
                    return False

        return True

    def matrix(self, gridpoints: int, block_dim: int):
        """Matrix representation of the VibronicBlockMatrix"""

        dim = self.dim * (gridpoints**block_dim)
        final_matrix = sp.sparse.csr_matrix((dim, dim))
        for key, value in self.blocks.items():
            data = np.array([1])
            indices = (np.array([key[0]]), np.array([key[1]]))
            shape = (self.dim, self.dim)
            indicator = sp.sparse.csr_matrix((data, indices), shape=shape)
            for word, coeffs in value.operator.items():
                block = _matrix_from_op(word, coeffs, block_dim, gridpoints)
                final_matrix += sp.sparse.kron(indicator, block)

        return final_matrix


class VibronicHamiltonian:
    """Class representation of the Vibronic Hamiltonian"""

    def __init__(self, states, modes, alphas, betas, lambdas, omegas):
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

    def __add__(self, other: VibronicHamiltonian):
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

    def __mul__(self, scalar):
        return VibronicHamiltonian(
            self.states,
            self.modes,
            scalar * self.alphas,
            scalar * self.betas,
            scalar * self.lambdas,
            scalar * self.omegas,
        )

    __rmul__ = __mul__

    def fragment(self, index) -> VibronicBlockMatrix:
        """Get the fragment at the specified index"""

        if index not in range(self.states + 1):
            raise ValueError("Index out of range")

        if index == self.states:
            return self._p_fragment()

        return self._fragment(index)

    def v_matrix(self, i: int, j: int) -> VibronicWord:
        """Get V_ij"""
        if i > self.states or j > self.states:
            raise ValueError(
                f"Dimension out of bounds. Got ({i}, {j}) but V is dimension ({self.states}, {self.states})."
            )
        word = VibronicWord()
        word.add_op((), self.lambdas[i, j])
        word.add_op(("Q",), self.alphas[i, j])
        word.add_op(("Q", "Q"), self.betas[i, j])
        return word

    def _fragment(self, i) -> VibronicBlockMatrix:
        fragment = VibronicBlockMatrix(dim=self.states)
        for j in range(self.states):
            word = self.v_matrix(j, i ^ j)
            fragment.set_block(j, i ^ j, word)

        return fragment

    def _p_fragment(self) -> VibronicBlockMatrix:
        word = VibronicWord({("PP",): self.omegas / 2})
        blocks = {(i, i): word for i in range(self.states)}
        return VibronicBlockMatrix(dim=self.states, blocks=blocks)

    def block_operator(self) -> VibronicBlockMatrix:
        """Return the block representation of the Hamiltonian"""

        operator = self._p_fragment()
        for i in range(self.states):
            operator += self._fragment(i)

        return operator

    def epsilon(self, delta: float) -> VibronicBlockMatrix:
        """Return the error matrix"""
        # pylint: disable=arguments-out-of-order

        scalar = -(delta**2) / 24
        epsilon = VibronicBlockMatrix(dim=self.states)

        for i in range(self.states):
            for j in range(i + 1, self.states + 1):
                epsilon += self.commute_fragments(i, i, j)
                for k in range(i + 1, self.states + 1):
                    epsilon += 2 * self.commute_fragments(k, i, j)

        return scalar * epsilon

    def commute_fragments(self, i, j, k):
        """Returns [H_i, [H_j, H_k]]"""
        if i == self.states and j < self.states and k == self.states:
            return self._commute_hn_hm_hn(j)

        return commutator(self.fragment(i), commutator(self.fragment(j), self.fragment(k)))

    def _commute_hn_hm_hn(self, m):
        """Special case for [H_N, [H_m, H_N]] where N = self.states and m < self.states"""
        block_matrix = VibronicBlockMatrix(dim=self.states)
        for j in range(self.states):
            coeffs = self.betas[j, m ^ j] * np.multiply.outer(self.omegas, self.omegas)
            operator = VibronicWord({("P", "P"): coeffs})
            block_matrix.set_block(j, m ^ j, operator)

        return 2 * block_matrix

    def matrix(self, gridpoints: int) -> sp.sparse.csr_matrix:
        """Return a matrix representation of the Hamiltonian"""

        return self.block_operator().matrix(gridpoints, self.modes)


def commutator(a: VibronicBlockMatrix, b: VibronicBlockMatrix) -> VibronicBlockMatrix:
    """Return the commutator [a, b]"""
    return a @ b - b @ a


def _add_dicts(d1: dict, d2: dict):
    new_dict = {}
    for key in d1.keys():
        if key in d2.keys():
            new_dict[key] = d1[key] + d2[key]
        else:
            new_dict[key] = d1[key]

    for key in d2.keys():
        if key not in d1.keys():
            new_dict[key] = d2[key]

    return new_dict


def _sub_dicts(d1: dict, d2: dict):
    new_dict = {}
    for key in d1.keys():
        if key in d2.keys():
            new_dict[key] = d1[key] - d2[key]
        else:
            new_dict[key] = d1[key]

    for key in d2.keys():
        if key not in d1.keys():
            new_dict[key] = d2[key]

    return new_dict


def _add_coeffs(d: dict, key: Tuple[str], coeffs: np.ndarray) -> None:
    try:
        d[key] += coeffs
    except KeyError:
        d[key] = coeffs


def position_operator(gridpoints: int, power: int) -> np.ndarray:
    """Returns a discretization of the position operator"""

    values = ((np.arange(gridpoints) - gridpoints / 2) * (np.sqrt(2 * np.pi / gridpoints))) ** power
    return sp.sparse.diags(values, 0, format="csr")


def momentum_operator(gridpoints: int, power: int) -> np.ndarray:
    """Returns a discretization of the momentum operator"""

    values = np.arange(gridpoints)
    values[gridpoints // 2 :] -= gridpoints
    values = (values * (np.sqrt(2 * np.pi / gridpoints))) ** power
    dft = sp.linalg.dft(gridpoints, scale="sqrtn")
    matrix = dft @ np.diag(values) @ dft.conj().T

    return sp.sparse.csr_matrix(matrix)


def _tensor_with_id(op: np.ndarray, base_dimension: int, length: int, position: int):
    if length < 1:
        raise ValueError("Length must be greater than 0")

    if position > length:
        raise ValueError("Position must be less than or equal to length")

    if length == 1:
        return op

    id = sp.sparse.identity(base_dimension, format="csr")

    if position == 1:
        id = sp.sparse.identity(base_dimension ** (length - 1))
        return sp.sparse.kron(op, id)

    if position == length:
        id = sp.sparse.identity(base_dimension ** (length - 1))
        return sp.sparse.kron(id, op)

    id_left = sp.sparse.identity(base_dimension ** (position - 1))
    id_right = sp.sparse.identity(base_dimension ** (length - position))

    return sp.sparse.kron(id_left, sp.sparse.kron(op, id_right))


def _matrix_from_op(op: Tuple[str], coeffs: np.ndarray, modes: int, gridpoints: int):
    if op == ():
        return sp.sparse.identity(gridpoints**modes, format="csr") * coeffs

    matrices = [_term_to_matrix(term, gridpoints) for term in op]

    final_matrix = sp.sparse.csr_matrix((gridpoints**modes, gridpoints**modes))
    for index in itertools.product(range(modes), repeat=len(op)):
        term = sp.sparse.identity(gridpoints**modes, format="csr")
        for count, i in enumerate(index):
            term = term @ _tensor_with_id(matrices[count], gridpoints, modes, i + 1)

        final_matrix += coeffs[index] * term

    return final_matrix


def _term_to_matrix(term: str, gridpoints: int) -> np.ndarray:
    mat = sp.sparse.identity(gridpoints, format="csr")
    p = momentum_operator(gridpoints, 1)
    q = position_operator(gridpoints, 1)

    for char in term:
        if char == "P":
            mat = mat @ p
            continue

        if char == "Q":
            mat = mat @ q
            continue

        raise ValueError(f"Operator terms must only contain P and Q. Got {char}.")

    return mat

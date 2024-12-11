"""Base classes for Vibronic Hamiltonians"""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np


class VibronicWord():
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
        return self + (-1)*other

    def __mul__(self, scalar) -> VibronicWord:
        if not isinstance(scalar, (complex, int, float)):
            raise TypeError(f"Cannot multiply VibronicWord by type {type(scalar)}.")

        new_operator = {}
        for key, value in self.operator.items():
            new_operator[key] = value * scalar

        return VibronicWord(operator=new_operator)

    def get_op(self, op: Tuple[str]):
        return self.operator[op]

    __rmul__ = __mul__

    def __matmul__(self, other: VibronicWord) -> VibronicWord:
        if not isinstance(other, VibronicWord):
            raise TypeError(f"Cannot multiply VibronicWord by type {type(other)}.")

        new_operator = {}
        for l_key, l_value in self.operator.items():
            for r_key, r_value in other.operator.items():
                new_value = np.multiply.outer(l_value, r_value)

                if l_key == ("I",):
                    new_key = r_key
                elif r_key == ("I",):
                    new_key = l_key
                else:
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

    def set_block(self, row: int, col:int, word: VibronicWord) -> None:
        """Set the word inside the (row, col) block"""
        self.blocks[(row, col)] = word

    def __repr__(self) -> str:
        return self.blocks.__repr__()

    def __add__(self, other: VibronicBlockMatrix) -> VibronicBlockMatrix:
        if self.dim != other.dim:
            raise ValueError(f"Cannot add VibronicBlockMatrix of shape ({self.dim}, {self.dim}) with shape ({other.dim}, {other.dim}).")

        new_blocks = _add_dicts(self.blocks, other.blocks)

        return VibronicBlockMatrix(dim=self.dim, blocks=new_blocks)

    def __sub__(self, other: VibronicBlockMatrix) -> VibronicBlockMatrix:
        return self + (-1)*other

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
            raise ValueError(f"Cannot multiply VibronicBlockMatrix of shape ({self.dim}, {self.dim}) with shape ({other.dim}, {other.dim}).")

        new_blocks = {}
        for i in range(self.dim):
            for j in range(self.dim):
                block_products = [self.get_block(i,k)@other.get_block(k,j) for k in range(self.dim)]
                block_sum = sum(block_products, VibronicWord({}))
                if block_sum:
                    new_blocks[(i,j)] = block_sum

        return VibronicBlockMatrix(dim=self.dim, blocks=new_blocks)

    def __eq__(self, other: VibronicBlockMatrix) -> VibronicBlockMatrix:
        if self.dim != other.dim:
            return False

        for i in range(self.dim):
            for j in range(self.dim):
                if self.get_block(i, j) != other.get_block(i, j):
                    return False

        return True


class VibronicHamiltonian:
    """Class representation of the Vibronic Hamiltonian"""

    def __init__(self, states, modes, alphas, betas, lambdas, omegas):
        if alphas.shape != (states,states,modes):
            raise TypeError
        if betas.shape != (states,states,modes,modes):
            raise TypeError
        if lambdas.shape != (states,states):
            raise TypeError
        if omegas.shape != (modes,):
            raise TypeError

        self.states = states
        self.modes = modes
        self.alphas = alphas
        self.betas = betas
        self.lambdas = lambdas
        self.omegas = omegas

    def fragment(self, index) -> VibronicBlockMatrix:
        """Get the fragment at the specified index"""

        if index not in range(self.states+1):
            raise ValueError("Index out of range")

        if index == self.states:
            return self._p_fragment()

        return self._fragment(index)

    def v_matrix(self, i: int, j: int) -> VibronicWord:
        """Get V_ij"""
        word = VibronicWord()
        word.add_op(("I",), self.lambdas[i, j])
        word.add_op(("Q",), self.alphas[i, j])
        word.add_op(("Q", "Q"), self.betas[i, j])
        return word

    def _fragment(self, i) -> VibronicBlockMatrix:
        fragment = VibronicBlockMatrix(dim=self.states)
        for j in range(self.states):
            word = self.v_matrix(j, i^j)
            fragment.set_block(j, i^j, word)

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

        scalar = -(delta**2)/24
        epsilon = VibronicBlockMatrix(dim=self.states)

        for i in range(self.states):
            for j in range(i+1, self.states+1):
                sum_k = sum([self.fragment(k) for k in range(i+1, self.states+1)], VibronicBlockMatrix(dim=self.states))
                com_ij = commutator(self.fragment(i), self.fragment(j))
                epsilon += commutator(self.fragment(i) + 2*sum_k, com_ij)

        return scalar*epsilon


def commutator(a: VibronicBlockMatrix, b: VibronicBlockMatrix) -> VibronicBlockMatrix:
    return a@b - b@a

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

def _add_coeffs(d: dict, key: Tuple[str], coeffs: np.ndarray) -> None:
    try:
        d[key] += coeffs
    except KeyError:
        d[key] = coeffs

def _simplify_word(word: str) -> str:
    if set(word) == {"I"}:
        return "I"

    return word.replace("I", "")

if __name__ == "__main__":
    n = 3
    m = 4
    alphas = np.random.random((n, n, m))
    betas = np.random.random((n, n, m, m))
    lambdas = np.random.random((n, n))
    omegas = np.random.random((m,))

    vham = VibronicHamiltonian(n, m, alphas, betas, lambdas, omegas)
    frag1 = vham.fragment(2)
    frag2 = vham.fragment(3)
    print(commutator(frag1, frag2))

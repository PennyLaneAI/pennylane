"""Functions for transforming a VibronicMatrix into a scipy csr matrix"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import scipy as sp

if TYPE_CHECKING:
    from pennylane.labs.vibronic.vibronic import VibronicTerm, VibronicWord


def position_operator(
    gridpoints: int, power: int, sparse: bool = False
) -> Union[np.ndarray, sp.sparse.csr_matrix]:
    """Returns a discretization of the position operator"""

    values = ((np.arange(gridpoints) - gridpoints / 2) * (np.sqrt(2 * np.pi / gridpoints))) ** power

    if sparse:
        return sp.sparse.diags(values, 0, format="csr")

    return np.diag(values)


def momentum_operator(
    gridpoints: int, power: int, sparse: bool = False
) -> Union[np.ndarray, sp.sparse.csr_matrix]:
    """Returns a discretization of the momentum operator"""

    values = np.arange(gridpoints)
    values[gridpoints // 2 :] -= gridpoints
    values = (values * (np.sqrt(2 * np.pi / gridpoints))) ** power
    dft = sp.linalg.dft(gridpoints, scale="sqrtn")
    matrix = dft @ np.diag(values) @ dft.conj().T

    if sparse:
        return sp.sparse.csr_matrix(matrix)

    return matrix


def op_to_matrix(
    op: str, gridpoints: int, sparse: bool = False
) -> Union[np.ndarray, sp.sparse.csr_matrix]:
    """Return a csr matrix representation of a Vibronic op"""

    matrix = _identity(gridpoints, sparse=sparse)
    p = momentum_operator(gridpoints, 1, sparse=sparse)
    q = position_operator(gridpoints, 1, sparse=sparse)

    for char in op:
        if char == "P":
            matrix @= p
            continue

        if char == "Q":
            matrix @= q
            continue

        raise ValueError(f"Operator terms must only contain P and Q. Got {char}.")

    return matrix


def term_to_matrix(
    term: VibronicTerm, modes: int, gridpoints: int, sparse: bool = False
) -> Union[np.ndarray, sp.sparse.csr_matrix]:
    """Return a matrix representation of a VibronicTerm"""

    matrices = [op_to_matrix(op, gridpoints, sparse=sparse) for op in term.ops]

    final_matrix = _zeros((gridpoints**modes, gridpoints**modes), sparse=sparse)
    for index in product(range(modes), repeat=len(term.ops)):
        matrix = _single_term_matrix(modes, gridpoints, index, matrices, sparse=sparse)
        matrix *= term.coeffs.compute(index)
        final_matrix += matrix

    return final_matrix


def _single_term_matrix(
    modes: int,
    gridpoints: int,
    index: Tuple[int],
    ops: Tuple[Union[np.ndarray, sp.sparse.csr_matrix]],
    sparse: bool = False,
) -> Union[np.ndarray, sp.sparse.csr_matrix]:
    lookup = {}

    for mode in range(modes):
        lookup[mode] = _identity(gridpoints, sparse=sparse)
        for count, i in enumerate(index):
            if i == mode:
                lookup[mode] @= ops[count]

    matrix = lookup[modes - 1]
    for mode in range(modes - 2, -1, -1):
        if mode in index:
            matrix = _kron(lookup[mode], matrix)
        else:
            matrix = sp.linalg.block_diag(*[matrix] * gridpoints)

    return matrix


def word_to_matrix(
    word: VibronicWord, modes: int, gridpoints: int, sparse: bool = False
) -> sp.sparse.csr_matrix:
    """Return a csr matrix representation of a VibronicWord"""

    final_matrix = _zeros((gridpoints**modes, gridpoints**modes), sparse=sparse)
    for term in word.terms:
        final_matrix += term_to_matrix(term, modes, gridpoints, sparse=sparse)

    return final_matrix


def _identity(dim: int, sparse: bool) -> Union[np.ndarray, sp.sparse.csr_matrix]:
    if sparse:
        return sp.sparse.identity(dim, format="csr")

    return np.eye(dim, dtype=np.complex128)


def _kron(a: Union[np.ndarray, sp.sparse.csr_matrix], b: Union[np.ndarray, sp.sparse.csr_matrix]):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.kron(a, b)

    if isinstance(a, sp.sparse.csr_matrix) and isinstance(b, sp.sparse.csr_matrix):
        return sp.sparse.kron(a, b)

    raise TypeError


def _zeros(shape: Tuple[int], sparse: bool = False):
    if sparse:
        return sp.sparse.csr_matrix(shape)

    return np.zeros(shape, dtype=np.complex128)


def op_norm(gridpoints: int) -> float:
    """The norm of P and Q"""
    return np.sqrt(gridpoints * np.pi / 2)

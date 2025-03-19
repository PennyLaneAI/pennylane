# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Private helper functions for converting RealspaceOperator objects to matrices"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import scipy as sp

if TYPE_CHECKING:
    from pennylane.labs.trotter.realspace import RealspaceOperator, RealspaceSum


def _position_operator(
    gridpoints: int, sparse: bool = False, basis: str = "realspace"
) -> Union[np.ndarray, sp.sparse.csr_array]:
    """Returns a discretization of the position operator"""

    if basis == "realspace":
        matrix = _realspace_position(gridpoints)
        return sp.sparse.csr_array(matrix) if sparse else matrix

    if basis == "harmonic":
        matrix = _harmonic_position(gridpoints)
        return sp.sparse.csr_array(matrix) if sparse else matrix

    raise ValueError(f'"{basis}" is not a valid basis')


def _realspace_position(gridpoints: int) -> np.ndarray:
    values = (np.arange(gridpoints) - gridpoints / 2) * (np.sqrt(2 * np.pi / gridpoints))
    return np.diag(values)


def _harmonic_position(gridpoints: int) -> np.ndarray:
    rows = np.array(list(range(1, gridpoints)) + list(range(0, gridpoints - 1)))
    cols = np.array(list(range(0, gridpoints - 1)) + list(range(1, gridpoints)))
    vals = np.array([np.sqrt(i) for i in range(1, gridpoints)] * 2)

    matrix = np.zeros(shape=(gridpoints, gridpoints))
    matrix[rows, cols] = vals

    return 1 / np.sqrt(2) * matrix


def _momentum_operator(
    gridpoints: int, sparse: bool = False, basis: str = "realspace"
) -> Union[np.ndarray, sp.sparse.csr_array]:
    """Returns a discretization of the momentum operator"""

    if basis == "realspace":
        matrix = _realspace_momentum(gridpoints)
        return sp.sparse.csr_array(matrix) if sparse else matrix

    if basis == "harmonic":
        matrix = _harmonic_momentum(gridpoints)
        return sp.sparse.csr_array(matrix) if sparse else matrix

    raise ValueError(f'"{basis}" is not a valid basis')


def _realspace_momentum(gridpoints: int) -> np.ndarray:
    values = np.arange(gridpoints)
    values[gridpoints // 2 :] -= gridpoints
    values = values * (np.sqrt(2 * np.pi / gridpoints))
    dft = sp.linalg.dft(gridpoints, scale="sqrtn")
    matrix = dft @ np.diag(values) @ dft.conj().T

    return matrix


def _harmonic_momentum(gridpoints: int) -> np.ndarray:
    rows = np.array(list(range(1, gridpoints)) + list(range(0, gridpoints - 1)))
    cols = np.array(list(range(0, gridpoints - 1)) + list(range(1, gridpoints)))
    vals = np.array(
        [np.sqrt(i) for i in range(1, gridpoints)] + [-np.sqrt(i) for i in range(1, gridpoints)]
    )

    matrix = np.zeros(shape=(gridpoints, gridpoints))
    matrix[rows, cols] = vals

    return (1j / np.sqrt(2)) * matrix


def _creation_operator(gridpoints: int, sparse: bool = False) -> Union[np.ndarray, sp.sparse.array]:
    """Return a matrix representation of the creation operator"""
    rows = np.array(range(0, gridpoints - 1))
    cols = np.array(range(1, gridpoints))
    vals = np.array([np.sqrt(i) for i in range(1, gridpoints)])

    matrix = np.zeros(shape=(gridpoints, gridpoints))
    matrix[rows, cols] = vals

    return sp.sparse.csr_array(matrix) if sparse else matrix


def _annihilation_operator(
    gridpoints: int, sparse: bool = False
) -> Union[np.ndarray, sp.sparse.array]:
    """Return a matrix representation of the annihilation operator"""
    rows = np.array(range(1, gridpoints))
    cols = np.array(range(0, gridpoints - 1))
    vals = np.array([np.sqrt(i) for i in range(1, gridpoints)])

    matrix = np.zeros(shape=(gridpoints, gridpoints))
    matrix[rows, cols] = vals

    return sp.sparse.csr_array(matrix) if sparse else matrix


def _string_to_matrix(
    op: str, gridpoints: int, sparse: bool = False, basis: str = "realspace"
) -> Union[np.ndarray, sp.sparse.csr_array]:
    """Return a csr matrix representation of a Vibronic op"""

    matrix = _identity(gridpoints, sparse=sparse)
    p = _momentum_operator(gridpoints, basis=basis, sparse=sparse)
    q = _position_operator(gridpoints, basis=basis, sparse=sparse)

    for char in op:
        if char == "P":
            matrix = matrix @ p
            continue

        if char == "Q":
            matrix = matrix @ q
            continue

        raise ValueError(f"Operator terms must only contain P and Q. Got {char}.")

    return matrix


def _tensor_with_identity(
    modes: int,
    gridpoints: int,
    index: Tuple[int],
    ops: Tuple[Union[np.ndarray, sp.sparse.csr_array]],
    sparse: bool = False,
) -> Union[np.ndarray, sp.sparse.csr_array]:
    """Tensor the input matrices with the identity"""
    lookup = {}

    for mode in range(modes):
        lookup[mode] = _identity(gridpoints, sparse=sparse)
        for count, i in enumerate(index):
            if i == mode:
                lookup[mode] = lookup[mode] @ ops[count]

    matrix = lookup[0]
    for mode in range(1, modes):
        matrix = _kron(matrix, lookup[mode])

    return matrix


def _identity(dim: int, sparse: bool) -> Union[np.ndarray, sp.sparse.csr_array]:
    if sparse:
        return sp.sparse.eye_array(dim, format="csr")

    return np.eye(dim)


def _kron(a: Union[np.ndarray, sp.sparse.csr_array], b: Union[np.ndarray, sp.sparse.csr_array]):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.kron(a, b)

    if isinstance(a, sp.sparse.csr_array) and isinstance(b, sp.sparse.csr_array):
        return sp.sparse.kron(a, b, format="csr")

    raise TypeError(f"Matrices must be ndarray or csr_array. Got {type(a)} and {type(b)}.")


def _zeros(shape: Tuple[int], sparse: bool = False) -> Union[np.ndarray, sp.sparse.csr_array]:
    if sparse:
        return sp.sparse.csr_array(shape)

    return np.zeros(shape)


def _op_norm(gridpoints: int) -> float:
    """The norm of P and Q"""
    return np.sqrt(gridpoints * np.pi / 2)

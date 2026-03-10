# Copyright 2025 Xanadu Quantum Technologies Inc.

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


from functools import reduce

import numpy as np
import scipy as sp


def _position_operator(
    gridpoints: int, sparse: bool = False, basis: str = "realspace"
) -> np.ndarray | sp.sparse.csr_array:
    """Returns a matrix representation of the position operator"""

    if basis == "realspace":
        matrix = _realspace_position(gridpoints)
        return sp.sparse.csr_array(matrix) if sparse else matrix

    if basis == "harmonic":
        matrix = _harmonic_position(gridpoints)
        return sp.sparse.csr_array(matrix) if sparse else matrix

    raise ValueError(f'"{basis}" is not a valid basis')


def _realspace_position(gridpoints: int) -> np.ndarray:
    """Return a matrix representation of the position operator in the realspace basis"""
    values = (np.arange(gridpoints) - gridpoints / 2) * (np.sqrt(2 * np.pi / gridpoints))
    return np.diag(values)


def _harmonic_position(gridpoints: int) -> np.ndarray:
    """Return a matrix representation of the position operator in the harmonic basis"""
    rows = np.array(list(range(1, gridpoints)) + list(range(0, gridpoints - 1)))
    cols = np.array(list(range(0, gridpoints - 1)) + list(range(1, gridpoints)))
    vals = np.array([np.sqrt(i) for i in range(1, gridpoints)] * 2)

    matrix = np.zeros(shape=(gridpoints, gridpoints))
    matrix[rows, cols] = vals

    return 1 / np.sqrt(2) * matrix


def _momentum_operator(
    gridpoints: int, sparse: bool = False, basis: str = "realspace"
) -> np.ndarray | sp.sparse.csr_array:
    """Returns a matrix representation of the momentum operator"""

    if basis == "realspace":
        matrix = _realspace_momentum(gridpoints)
        return sp.sparse.csr_array(matrix) if sparse else matrix

    if basis == "harmonic":
        matrix = _harmonic_momentum(gridpoints)
        return sp.sparse.csr_array(matrix) if sparse else matrix

    raise ValueError(f'"{basis}" is not a valid basis')


def _realspace_momentum(gridpoints: int) -> np.ndarray:
    """Returns a matrix representation of the momenumt operator in the realspace basis"""
    values = np.arange(gridpoints)
    values[gridpoints // 2 :] -= gridpoints
    values = values * (np.sqrt(2 * np.pi / gridpoints))
    dft = sp.linalg.dft(gridpoints, scale="sqrtn")
    matrix = dft @ np.diag(values) @ dft.conj().T

    return matrix


def _harmonic_momentum(gridpoints: int) -> np.ndarray:
    """Returns a matrix representation of the momenumt operator in the harmonic basis"""
    rows = np.array(list(range(1, gridpoints)) + list(range(0, gridpoints - 1)))
    cols = np.array(list(range(0, gridpoints - 1)) + list(range(1, gridpoints)))
    vals = np.array(
        [np.sqrt(i) for i in range(1, gridpoints)] + [-np.sqrt(i) for i in range(1, gridpoints)]
    )

    matrix = np.zeros(shape=(gridpoints, gridpoints))
    matrix[rows, cols] = vals

    return (1j / np.sqrt(2)) * matrix


def _string_to_matrix(
    op: str, gridpoints: int, sparse: bool = False, basis: str = "realspace"
) -> np.ndarray | sp.sparse.csr_array:
    """Transforms a string of P's and Q's into a matrix representing the product of position and momenutm operators"""

    matrix = _identity(gridpoints, sparse=sparse)
    p = _momentum_operator(gridpoints, basis=basis, sparse=sparse)
    q = _position_operator(gridpoints, basis=basis, sparse=sparse)
    mat_dict = {"P": p, "Q": q}

    return reduce(lambda x, y: x @ mat_dict[y], op, matrix)


def _tensor_with_identity(
    modes: int,
    gridpoints: int,
    index: tuple[int],
    ops: tuple[np.ndarray | sp.sparse.csr_array],
    sparse: bool = False,
) -> np.ndarray | sp.sparse.csr_array:
    """Tensor the input matrices with the identity"""
    lookup = [_identity(gridpoints, sparse=sparse)] * modes

    for mode in range(modes):
        for count, i in enumerate(index):
            if i == mode:
                lookup[mode] = lookup[mode] @ ops[count]

    return reduce(_kron, lookup[1:], lookup[0])


def _identity(dim: int, sparse: bool) -> np.ndarray | sp.sparse.csr_array:
    """Return a matrix representation of the identity operator"""
    if sparse:
        return sp.sparse.eye_array(dim, format="csr")

    return np.eye(dim)


def _kron(a: np.ndarray | sp.sparse.csr_array, b: np.ndarray | sp.sparse.csr_array):
    """Return the Kronecker product of two matrices"""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.kron(a, b)

    if isinstance(a, sp.sparse.csr_array) and isinstance(b, sp.sparse.csr_array):
        return sp.sparse.kron(a, b, format="csr")

    raise TypeError(f"Matrices must both be ndarray or csr_array. Got {type(a)} and {type(b)}.")


def _zeros(shape: tuple[int], sparse: bool = False) -> np.ndarray | sp.sparse.csr_array:
    """Return a matrix representation of the zero operator"""
    if sparse:
        return sp.sparse.csr_array(shape)

    return np.zeros(shape)


def _op_norm(gridpoints: int) -> float:
    """The norm of P and Q"""
    return np.sqrt(gridpoints * np.pi / 2)

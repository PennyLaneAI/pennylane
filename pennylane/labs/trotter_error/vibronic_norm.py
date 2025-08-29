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
"""This module contains functions for computing vibronic norm."""

import math
from functools import cache
from itertools import product

import numpy as np
import scipy as sp

from pennylane.labs.trotter_error import RealspaceMatrix, RealspaceSum
from pennylane.labs.trotter_error.realspace.matrix import _momentum_operator, _position_operator


def vibronic_norm(hamiltonian: RealspaceMatrix, gridpoints: int):
    """Returns the norm of a vibronic Hamiltonian.

    Args:
        hamiltonian (RealspaceMatrix): The vibronic Hamiltonian.
        gridpoints (int):

    Returns:
        ndarray: matrix of the norms of the vibronic Hamiltonian blocks

    **Example**

    >>> from pennylane.labs.trotter_error import vibronic_norm
    >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceSum, RealspaceCoeffs, RealspaceMatrix
    >>> import numpy as np
    >>> n_states = 1
    >>> n_modes = 5
    >>> op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
    >>> op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
    >>> rs_sum = RealspaceSum(n_modes, [op1, op2])
    >>> rs_mat = RealspaceMatrix(n_states, n_modes, {(0, 0): rs_sum})
    >>> vibronic_norm(rs_mat, 2)
    array([[11.60660172]])
    """

    if not gridpoints & (gridpoints - 1) == 0 or gridpoints <= 0:
        raise ValueError(f"Number of gridpoints must be a positive power of 2, got {gridpoints}.")

    padded = RealspaceMatrix(
        2 ** (hamiltonian.states - 1).bit_length(), hamiltonian.modes, hamiltonian._blocks
    )

    norms = np.zeros(shape=(padded.states, padded.states))

    for i, j in [k for k in product(range(hamiltonian.states), repeat=2)]:
        norms[i, j] = _block_norm(padded.block(i, j), gridpoints)

    return norms


def _block_norm(block: RealspaceSum, gridpoints: int):
    """Returns the norm of a vibronic Hamiltonian block.

    Args:
        block (RealspaceSum): The vibronic Hamiltonian block.
        gridpoints (int):

    Returns:
        float: the norms of the vibronic Hamiltonian blocks

    **Example**

    >>> from pennylane.labs.trotter_error import RealspaceOperator, RealspaceSum, RealspaceCoeffs
    >>> import numpy as np
    >>> n_states = 1
    >>> n_modes = 5
    >>> op1 = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
    >>> op2 = RealspaceOperator(n_modes, ("Q"), RealspaceCoeffs(np.array([1, 2, 3, 4, 5]), label="phi"))
    >>> rs_sum = RealspaceSum(n_modes, [op1, op2])
    >>> _block_norm(rs_sum, 2)
    11.606601717798213
    """
    mode_groups = {}

    for op in block.ops:
        for index, coeff in op.coeffs.nonzero().items():
            group = frozenset(index)

            mode_strs = {i: "" for i in group}
            for i, mat in zip(index, op.ops):
                if mat == "P":
                    mode_strs[i] += "P"
                elif mat == "Q":
                    mode_strs[i] += "Q"

            sorted_ops = tuple(mode_strs[i] for i in sorted(group))

            try:
                mode_groups[group]["ops"].append(sorted_ops)
                mode_groups[group]["coeffs"].append(coeff)
            except KeyError:
                mode_groups[group] = {"ops": [sorted_ops], "coeffs": [coeff]}

    return sum(_get_eigenvalue(group_ops, gridpoints) for group_ops in mode_groups.values())


def _get_eigenvalue(group_ops, gridpoints):
    """Returns the largest eigenvalue of a linear combination of position and momentum products.

    Args:
        group_ops (dict): A dictionary of operators and coefficients as key and values, respectively.
        gridpoints (int):

    Returns:
        float: largest eigenvalue of the input operator

    **Example**

    >>> group_ops = {'ops': [('QPP',), ('QQ',), ('Q',), ('PPQ',)],
    ...              'coeffs': [(0.01), (0.02j), (0.03), (0.04j)]}
    >>> _get_eigenvalue(group_ops, 4)
    0.15442961739479524
    """
    ops, coeffs = group_ops["ops"], group_ops["coeffs"]
    mat = coeffs[0] * build_mat(ops[0], gridpoints)

    for op, coeff in zip(ops[1:], coeffs[1:]):
        mat += coeff * build_mat(op, gridpoints)

    _, _, values = sp.sparse.find(mat)
    if np.allclose(values, 0):
        return 0

    try:
        eigvals, _ = sp.sparse.linalg.eigs(mat, k=1)
        return np.abs(eigvals[0])
    except Exception:
        pass

    try:
        return sp.sparse.linalg.norm(mat, ord=2)
    except Exception:
        pass

    return 0


@cache
def build_mat(ops, gridpoints):
    """Returns the sparse matrix form of position and momentum products.

    For a tuple of position and momentum products, e.g., ('QQ', 'QQPP'), this function first
    computes the matrix form of each poduct, i.e., mat(QQ), mat(QQPP) and then returns the Kronecker
    product of the matricesas.

    Args:
        ops (typle(string): A tuple containing position and momentum products.
        gridpoints (int):

    Returns:
        csr_array: the sparse matrix form of position and momentum products

    **Example**

    >>> mat = build_mat(('QQ', 'QQPP'), 4)
    >>> mat.shape
    (16, 16)
    """
    if len(ops) == 0:
        return sp.sparse.eye(gridpoints)

    mats = [sp.sparse.eye(gridpoints, dtype=np.complex128)] * len(ops)
    for i, op in enumerate(ops):
        for ch in op:
            if ch == "P":
                mats[i] @= _momentum_operator(gridpoints, basis="harmonic", sparse=True)
            elif ch == "Q":
                mats[i] @= _position_operator(gridpoints, basis="harmonic", sparse=True)

    ret = mats[0]

    for mat in mats[1:]:
        ret = sp.sparse.kron(ret, mat, format="csr")

    return ret


def _compute_norm(norm_mat: np.ndarray):
    if norm_mat.shape == (1, 1):
        return norm_mat[0, 0]

    half = norm_mat.shape[0] // 2

    top_left = norm_mat[0:half, 0:half]
    top_right = norm_mat[0:half, half:]
    bottom_left = norm_mat[half:, 0:half]
    bottom_right = norm_mat[half:, half:]

    norm1 = max(_compute_norm(top_left), _compute_norm(bottom_right))
    norm2 = math.sqrt(_compute_norm(top_right) * _compute_norm(bottom_left))

    return norm1 + norm2

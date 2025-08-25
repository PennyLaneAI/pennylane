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

import numpy as np
import scipy as sp

from pennylane.labs.trotter_error import (
    ProductFormula,
    RealspaceMatrix,
    RealspaceSum,
    effective_hamiltonian,
    vibronic_fragments,
)

from pennylane.labs.trotter_error.realspace.matrix import _momentum_operator, _position_operator


def vibronic_norm(rs_mat: RealspaceMatrix, gridpoints: int, numbers: list):
    if not _is_pow_2(gridpoints) or gridpoints <= 0:
        raise ValueError(f"Number of gridpoints must be a positive power of 2, got {gridpoints}.")

    padded = RealspaceMatrix(_next_pow_2(rs_mat.states), rs_mat.modes, rs_mat._blocks)

    norms = np.zeros(shape=(padded.states, padded.states))

    for batch in numbers:
        i, j = batch
        norms[i, j] = _block_norm(padded.block(i, j), gridpoints)

    return norms


def _block_norm(rs_sum: RealspaceSum, gridpoints: int):
    mode_groups = {}

    for op in rs_sum.ops:
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


@cache
def build_mat(gridpoints, ops):
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


def _get_eigenvalue(group_ops, gridpoints):
    ops, coeffs = group_ops["ops"], group_ops["coeffs"]
    mat = coeffs[0] * build_mat(gridpoints, ops[0])

    for op, coeff in zip(ops[1:], coeffs[1:]):
        mat += coeff * build_mat(gridpoints, op)

    _, _, values = sp.sparse.find(mat)
    if np.allclose(values, 0):
        return 0

    try:
        eigvals, _ = sp.sparse.linalg.eigs(mat, k=1, tol=1e-5)
        return np.abs(eigvals[0])
    except Exception:
        pass

    try:
        return sp.sparse.linalg.norm(mat, ord=2)
    except Exception:
        pass

    return 0


def _is_pow_2(k: int) -> bool:
    """Test if k is a power of two"""
    return k & (k - 1) == 0


def _next_pow_2(k: int) -> int:
    """Return the smallest power of 2 greater than or equal to k"""
    return 2 ** (k - 1).bit_length()


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

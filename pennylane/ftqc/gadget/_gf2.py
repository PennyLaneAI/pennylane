# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal GF(2) linear algebra used by the gadget verifier.

Kept dependency-free (numpy only) so that authoring and verification never pull in a
compiler, a simulator, or a decoder. Matrices are dense uint8 arrays with entries in
{0, 1}; gadget-scale matrices (thousands of columns) are fine dense, and the verifier
is not on any hot path.
"""

from __future__ import annotations

import numpy as np

Matrix = np.ndarray


def as_bits(matrix, n_cols: int | None = None) -> Matrix:
    """Coerce to a 2-D uint8 GF(2) matrix, allowing an empty matrix with n_cols columns."""
    arr = np.asarray(matrix, dtype=np.uint8)
    if arr.size == 0:
        if n_cols is None:
            raise ValueError("an empty matrix needs an explicit column count")
        return np.zeros((0, n_cols), dtype=np.uint8)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {arr.shape}")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("matrix entries must be 0 or 1")
    return arr % 2


def row_reduce(matrix: Matrix) -> tuple[Matrix, list[int]]:
    """Return (reduced row echelon form, pivot column indices) over GF(2)."""
    work = as_bits(matrix, 0 if matrix is None else None).copy()
    rows, cols = work.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(work[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            work[[r, i]] = work[[i, r]]
        hit = np.nonzero(work[:, c])[0]
        hit = hit[hit != r]
        if hit.size:
            work[hit] ^= work[r]
        pivots.append(c)
        r += 1
    return work[:r], pivots


def rank(matrix: Matrix) -> int:
    """GF(2) rank."""
    if matrix.size == 0:
        return 0
    return row_reduce(matrix)[0].shape[0]


def in_row_space(vector: Matrix, matrix: Matrix) -> bool:
    """True if `vector` is a GF(2) combination of the rows of `matrix`."""
    vec = as_bits(vector, matrix.shape[1]).reshape(1, -1)
    if np.all(vec == 0):
        return True
    if matrix.shape[0] == 0:
        return False
    return rank(matrix) == rank(np.vstack([matrix, vec]))


def null_space(matrix: Matrix) -> Matrix:
    """Rows spanning ``{x : matrix @ x = 0 mod 2}``."""
    if matrix.shape[0] == 0:
        return np.eye(matrix.shape[1], dtype=np.uint8)
    rref, pivots = row_reduce(matrix)
    n = matrix.shape[1]
    free = [c for c in range(n) if c not in pivots]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, p in enumerate(pivots):
            basis[i, p] = rref[r, f]
    return basis


def solve(rows: Matrix, target: Matrix) -> np.ndarray | None:
    """Coefficients ``c`` with ``c @ rows == target mod 2``, or None if unsolvable."""
    tgt = as_bits(target, rows.shape[1]).reshape(-1)
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.uint8) if not tgt.any() else None
    a = np.hstack([rows.T.astype(np.uint8), tgt.reshape(-1, 1)])
    rref, pivots = row_reduce(a)
    m = rows.shape[0]
    if m in pivots:
        return None
    coeffs = np.zeros(m, dtype=np.uint8)
    for r, p in enumerate(pivots):
        coeffs[p] = rref[r, m]
    return coeffs


def supported_combinations(rows: Matrix, mask: np.ndarray) -> Matrix:
    """Combinations of `rows` whose support lies entirely inside `mask`.

    Used to find which products of known stabilizers become deterministic when only the
    qubits in `mask` are measured out.
    """
    if rows.shape[0] == 0:
        return np.zeros((0, rows.shape[1]), dtype=np.uint8)
    outside = rows[:, ~mask]
    if outside.shape[1] == 0:
        coeffs = np.eye(rows.shape[0], dtype=np.uint8)
    else:
        coeffs = null_space(outside.T.astype(np.uint8))
    if coeffs.shape[0] == 0:
        return np.zeros((0, rows.shape[1]), dtype=np.uint8)
    prod = (coeffs.astype(np.uint16) @ rows.astype(np.uint16) % 2).astype(np.uint8)
    keep = prod.any(axis=1)
    reduced, _ = row_reduce(prod[keep]) if keep.any() else (prod[keep], [])
    return reduced


def commutes(hx: Matrix, hz: Matrix) -> Matrix:
    """Return the GF(2) product Hx Hz^T; all-zero means every X check commutes with
    every Z check (the CSS condition)."""
    if hx.shape[0] == 0 or hz.shape[0] == 0:
        return np.zeros((hx.shape[0], hz.shape[0]), dtype=np.uint8)
    return (hx.astype(np.uint16) @ hz.T.astype(np.uint16) % 2).astype(np.uint8)


def row_weights(matrix: Matrix) -> np.ndarray:
    """Weight of every row."""
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return matrix.sum(axis=1, dtype=np.int64)


def col_weights(matrix: Matrix) -> np.ndarray:
    """Weight of every column."""
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.int64)
    return matrix.sum(axis=0, dtype=np.int64)

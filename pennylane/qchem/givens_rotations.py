# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the functions needed for performing basis transformation defined by a set of fermionic ladder operators.
"""

from pennylane import numpy as np


def givens_matrix(a, b, left=True, tol=1e-8):
    """Build a Givens rotation matrix with a phase."""
    abs_a, abs_b = np.abs(a), np.abs(b)
    if abs_a < tol:
        cosine, sine, phase = 1.0, 0.0, 1.0
    elif abs_b < tol:
        cosine, sine, phase = 0.0, 1.0, 1.0
    else:
        hypot = np.hypot(abs_a, abs_b)
        cosine = abs_b / hypot
        sine = abs_a / hypot
        phase = 1.0 * b / abs_b * a.conjugate() / abs_a

    if not left:
        givens_mat = np.array([[phase * sine, cosine], [-phase * cosine, sine]])
    else:
        givens_mat = np.array([[phase * cosine, -sine], [phase * sine, cosine]])

    return givens_mat


def givens_rotate(unitary, grot_mat, indices, row=True):
    """Apply in-place Givens roation on the unitary on the given indices."""

    if len(indices) != 2:
        raise ValueError(f"Indices must have length 2, got {len(indices)}")

    if row:
        unitary[indices] = grot_mat @ unitary[indices]
    else:
        unitary[:, indices] = (grot_mat @ unitary[:, indices].T).T


def givens_decomposition(unitary):
    """Decomposes a unitary into sequence of Givens Rotation gates and a diagonal Phase matrix.
    Based on the optimal construction given by Clements in Optica Vol. 3, Issue 12, pp. 1460-1465 (2016)."""

    unitary, (M, N) = unitary.copy(), unitary.shape
    if M != N:
        raise ValueError(f"The unitary matrix should be of shape NxN, got {unitary.shape}")

    left_givens, right_givens = [], []
    for i in range(1, N):
        if i % 2:
            for j in range(0, i):
                indices = [i - j - 1, i - j]
                grot_mat = givens_matrix(*unitary[N - j - 1, indices].T, left=True)
                givens_rotate(unitary, grot_mat, indices, row=False)
                right_givens.append((grot_mat.conj(), indices))
        else:
            for j in range(1, i + 1):
                indices = [N + j - i - 2, N + j - i - 1]
                grot_mat = givens_matrix(*unitary[indices, j - 1], left=False)
                givens_rotate(unitary, grot_mat, indices, row=True)
                left_givens.append((grot_mat, indices))

    nleft_givens = []
    for (grot_mat, (i, j)) in reversed(left_givens):
        sphase_mat = np.diag(np.diag(unitary)[[i, j]])
        decomp_mat = grot_mat.conj().T @ sphase_mat
        givens_mat = givens_matrix(*decomp_mat[1, :].T)
        nphase_mat = decomp_mat @ givens_mat.T

        # check for T_{m,n}^{-1} x D = D x T.
        if not np.allclose(nphase_mat @ givens_mat.conj(), decomp_mat):
            raise ValueError("Failed to shift phase transposition.")

        unitary[i, i], unitary[j, j] = np.diag(nphase_mat)
        nleft_givens.append((givens_mat.conj(), (i, j)))

    phases, ordered_rotations = np.diag(unitary), []
    for (grot_mat, (i, j)) in list(reversed(nleft_givens)) + list(reversed(right_givens)):
        if not np.all(np.isreal(grot_mat[0, 1]) and np.isreal(grot_mat[1, 1])):
            raise ValueError(f"Incorrect Givens Rotation encountered, {grot_mat}")
        ordered_rotations.append((grot_mat, (i, j)))

    return phases, ordered_rotations

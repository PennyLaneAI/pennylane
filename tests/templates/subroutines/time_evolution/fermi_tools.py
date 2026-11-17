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
"""Independent Jordan-Wigner / Gaussian fermionic toolkit for the TrotterCDF tests, built from
first principles (occupation basis / minors) with no reference to :class:`~.BasisRotation`.

Mode ordering convention: mode ``m`` sits at bit position ``m`` (mode 0 is the most significant
bit, matching PennyLane's wire-0-most-significant convention). This lets a test build the exact
matrix of a fermionic Hamiltonian and compare it to a circuit without reusing any template
internals.
"""

import numpy as np


def occupations(nmodes):
    """All occupation tuples, indexed by the computational basis index (mode 0 = MSB)."""
    return [
        tuple(m for m in range(nmodes) if (idx >> (nmodes - 1 - m)) & 1) for idx in range(2**nmodes)
    ]


def one_body_matrix(mat):
    """Matrix of :math:`\\sum_{ij} \\text{mat}_{ij}\\, c^\\dagger_i c_j` in the occupation basis."""
    nmodes = mat.shape[0]
    occ = occupations(nmodes)
    dim = 2**nmodes
    out = np.zeros((dim, dim), dtype=complex)
    for ix, x in enumerate(occ):
        for i in range(nmodes):
            for j in range(nmodes):
                if mat[i, j] == 0 or j not in x or (i != j and i in x):
                    continue
                rem = [m for m in x if m != j]
                sign = (-1) ** sum(1 for m in x if m < j)
                new = sorted(rem + [i])
                sign *= (-1) ** sum(1 for m in rem if m < i)
                iy = sum(1 << (nmodes - 1 - m) for m in new)
                out[iy, ix] += mat[i, j] * sign
    return out


def permute_qubits(mat, perm):
    """Reorder tensor factors: output factor ``k`` holds the input factor at ``perm[k]``."""
    n = int(np.log2(mat.shape[0]))
    tensor = mat.reshape([2] * (2 * n))
    axes = list(perm) + [n + p for p in perm]
    return tensor.transpose(axes).reshape(mat.shape)

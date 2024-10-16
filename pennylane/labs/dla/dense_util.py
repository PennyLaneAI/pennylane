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
"""Utility tools for dense Lie algebra representations"""

from itertools import product

import numpy as np

import pennylane as qml
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform


def _make_phase_mat(n):
    phase_mat = qml.math.ones((2,) * (2 * n), dtype=complex)
    for idx in range(n):
        index = [slice(None)] * (2 * n)
        index[idx] = index[idx + n] = 1
        phase_mat[tuple(index)] *= 1j
    phase_mat = qml.math.reshape(phase_mat, (2**n, 2**n))
    return phase_mat


def _make_permutation_indices(dim):
    indices = [qml.math.arange(dim)]
    for idx in range(dim - 1):
        indices.append(qml.math.bitwise_xor(indices[-1], (idx + 1) ^ (idx)))
    return indices


def _make_extraction_indices(n):
    indices = []
    for pauli_rep in product("IXYZ", repeat=n):
        bit_array = qml.math.array(
            [[(rep in "YZ"), (rep in "XY")] for rep in pauli_rep], dtype=int
        ).T
        indices.append(tuple(int("".join(map(str, x)), 2) for x in bit_array))
    return tuple(zip(*indices))


def pauli_decompose(H):
    r"""Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Args:
        H (tensor_like[complex]): a Hermitian matrix of dimension ``(2**n, 2**n)`` or a collection of Hermitian matrices of dimension ``(batch, 2**n, 2**n)``.

    Returns:
        Union[~.Hamiltonian, ~.PauliSentence]: the matrix decomposed as a linear combination
        of Pauli operators, returned either as a :class:`~.Hamiltonian` or :class:`~.PauliSentence`
        instance.

    """
    # Preparations
    shape = H.shape
    dim = shape[-1]
    n = int(np.round(np.log2(dim)))
    assert dim == 2**n

    # Permutation
    indices = _make_permutation_indices(dim)
    # Apply the permutation by slicing and stacking again
    sliced_H = [
        qml.math.take(H[..., idx, :], _indices, axis=-1) for idx, _indices in enumerate(indices)
    ]
    sliced_H = qml.math.cast(qml.math.stack(sliced_H), complex)
    # Move leading axis (different permutation slices) to last position and combine broadcasting axis
    # and slicing axis into one leading axis (because `_walsh_hadamard_transform` only takes one batch axis)
    term_mat = qml.math.reshape(qml.math.moveaxis(sliced_H, 0, -1), (-1, dim))
    # Apply Walsh-Hadamard transform
    hadamard_transform_mat = _walsh_hadamard_transform(term_mat)
    # Reshape again to separate actual broadcasting axis and previous slicing axis
    hadamard_transform_mat = qml.math.reshape(hadamard_transform_mat, shape)
    # _make phase matrix that allows us to figure out phase contributions from Pauli Y terms.
    phase_mat = qml.math.convert_like(_make_phase_mat(n), H)
    # Multiply phase matrix to Hadamard transformed matrix and transpose the two Hilbert-space-dim axes
    coefficients = qml.math.moveaxis(qml.math.multiply(hadamard_transform_mat, phase_mat), -2, -1)
    # Extract the coefficients by reordering them according to the encoding in `qml.pauli.pauli_decompose`
    indices = _make_extraction_indices(n)
    coefficients = coefficients[..., indices[0], indices[1]].reshape((-1, dim**2))[..., 1:]
    return coefficients

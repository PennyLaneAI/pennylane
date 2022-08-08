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

"""Utility functions common to classical shadows"""

import pennylane as qml
from pennylane import numpy as np


def median_of_means(arr, num_batches):
    """
    The median of means of the given array.

    The array is split into the specified number of batches. The mean value
    of each batch is taken, then the median of the mean values is returned.

    Args:
        arr (tensor-like[float]): The 1-D array for which the median of means
            is determined
        num_batches (int): The number of batches to split the array into

    Returns:
        float: The median of means
    """
    means = []
    batch_size = int(np.ceil(arr.shape[0] / num_batches))

    for i in range(num_batches):
        means.append(np.mean(arr[i * batch_size : (i + 1) * batch_size]))

    return np.median(means)


def pauli_expval(bits, recipes, word):
    """
    The approximate expectation value of a Pauli word given the bits and recipes
    from a classical shadow measurement.

    Args:
        bits (tensor-like[int]): An array with shape ``(T, n)``, where ``T`` is the
            number of snapshots and ``n`` is the number of measured qubits. Each
            entry must be either ``0`` or ``1`` depending on the sample for the
            corresponding snapshot and qubit.
        recipes (tensor-like[int]): An array with shape ``(T, n)``. Each entry
            must be either ``0``, ``1``, or ``2`` depending on the selected Pauli
            measurement for the corresponding snapshot and qubit. ``0`` corresponds
            to PauliX, ``1`` to PauliY, and ``2`` to PauliZ.
        word (tensor-like[int]): An array with shape ``(n,)``. Each entry must be
            either ``0``, ``1``, ``2``, or ``-1`` depending on the Pauli observable
            on each qubit. For example, when ``n=3``, the observable ``PauliY(0) @ PauliX(2)``
            corresponds to the word ``np.array([1 -1 0])``.

    Returns:
        tensor-like[float]: An array with shape ``(T,)`` containing the value
            of the Pauli observable for each snapshot. The expectation can be
            found by averaging across the snapshots.
    """
    T = recipes.shape[0]

    # -1 in the word indicates an identity observable on that qubit
    id_mask = word == -1

    # determine snapshots and qubits that match the word
    indices = recipes == word
    indices = np.logical_or(indices, np.tile(id_mask, (T, 1)))
    indices = np.all(indices, axis=1)

    bits = bits[:, np.logical_not(id_mask)]
    bits = np.sum(bits, axis=1) % 2

    return np.where(indices, 1 - 2 * bits, 0) * 3 ** np.count_nonzero(np.logical_not(id_mask))

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
"""Arithmetic of Pauli words and sentences with a dense representation."""

import numpy as np

from pennylane.pauli import PauliSentence, PauliWord

str_mapping = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def ps_to_tensor(pauli_sentence: PauliSentence, n: int, dtype=np.float64) -> np.ndarray:
    """Convert a ``PauliSentence`` on at most ``n`` qubits into an ``n``-tensor
    of shape :math:`(4, 4,\cdots 4)` with the entries corresponding to the Pauli
    words in the input set to the corresponding coefficients, and all other entries
    being zero.

    Args:
        pauli_sentence (PauliSentence): ``PauliSentence`` to convert into a tensor.
        n (int): Total number of wires on which the tensor should be represented.
        dtype (type): Data type to be used for the tensor.

    Returns:
        np.ndarray: The tensor encoding the input ``pauli_sentence`` as a tensor
        on ``n`` qubits.

    **Example**

    """
    tensor = np.zeros([4] * n, dtype=dtype)
    # Iterate over Pauli sentence and initialize tensor entries corresponding
    # to Pauli words in the sentence.
    for pw, coeff in pauli_sentence.items():
        # Perform the mapping from string to tensor index
        # For each wire, determine the index from 0 to 3 that corresponds to
        # the Pauli word component on that wire, according to ``str_mapping`` above.
        idx = tuple(str_mapping[pw.get(i, "I")] for i in range(n))
        tensor[idx] = coeff
    return tensor


contractor = np.array(
    [  # axes ordering: (in1, in2, out)
        [
            [1, 0, 0, 0],  # I * I = I
            [0, 1, 0, 0],  # I * X = X
            [0, 0, 1, 0],  # I * Y = Y
            [0, 0, 0, 1],  # I * Z = Z
        ],
        [
            [0, 1, 0, 0],  #   X * I = X
            [1, 0, 0, 0],  #   X * X = I
            [0, 0, 0, 1j],  #  X * Y = iZ
            [0, 0, -1j, 0],  # X * Z = -iY
        ],
        [
            [0, 0, 1, 0],  #   Y * I = Y
            [0, 0, 0, -1j],  # Y * X = -iZ
            [1, 0, 0, 0],  #   Y * Y = I
            [0, 1j, 0, 0],  #  Y * Z = iX
        ],
        [
            [0, 0, 0, 1],  #   Z * I = Z
            [0, 0, 1j, 0],  #  Z * X = iY
            [0, -1j, 0, 0],  # Z * Y = -iX
            [1, 0, 0, 0],  #   Z * Z = I
        ],
    ]
)


def product(tensor1, tensor2, broadcasted=False):
    """Product of two tensors representing Pauli sentences on the same number of wires.

    Args:
        tensor1
        tensor2

    Returns:
        np.ndarray

    Supports broadcasting in leading axes for one or both inputs.
    If both inputs are broadcasted, the broadcasting axes are not merged.
    """
    # TODO: Replace einsum by optimized-order tensordots
    if broadcasted == "merge":
        broadcasted = [True, True]
        broadcasting_strs = ["m", "m", "m"]
    else:
        if broadcasted is False:
            broadcasted = [False, False]
        elif broadcasted is True:
            broadcasted = [True, True]
        broadcasting_strs = ["m" * int(broadcasted[0]), "n" * int(broadcasted[1])]
        broadcasting_strs.append(broadcasting_strs[0] + broadcasting_strs[1])

    n = tensor1.ndim - int(broadcasted[0])
    if n > 6 or (tensor2.ndim - int(broadcasted[1])) != n:
        raise NotImplementedError
    idx_groups_inputs = [
        broadcasting_strs[0] + "abcdef"[:n],
        broadcasting_strs[1] + "ghijkl"[:n],
    ] + ["agz", "bhy", "cix", "djw", "ekv", "flu"][:n]
    idx_output = broadcasting_strs[2] + "zyxwvu"[:n]
    einsum_str = ",".join(idx_groups_inputs) + f"->{idx_output}"
    return np.einsum(einsum_str, tensor1, tensor2, *([contractor] * n))


def commutator(tensor1, tensor2):
    """Commutator between two tensors representing Pauli sentences on the same number of wires.

    Args:
        tensor1
        tensor2

    Returns:
        np.ndarray
    """
    return product(tensor1, tensor2) - product(tensor2, tensor1)

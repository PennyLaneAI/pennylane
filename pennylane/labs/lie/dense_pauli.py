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


def _broadcast_validation(tensor1, tensor2, broadcasted):
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
    return n, broadcasting_strs


def product(tensor1, tensor2, broadcasted=False):
    """Product of two tensors representing Pauli sentences on the same number of wires.

    Args:
        tensor1 (np.ndarray): First factor for the product.
        tensor2 (np.ndarray): Second factor for the product, with same shape as first
            factor up to broadcasting
        broadcasted (Union[bool, list[bool]]): Whether, and which, inputs are broadcasted.
            See below for details.

    Returns:
        np.ndarray: Pauli product of the two input tensors.

    **Broadcasting**

    This function supports broadcasting in up to one leading axis for one or both inputs.
    For this, the ``broadcasted`` input has to be specified:

    - ``False`` or ``[False, False]``: Neither input is broadcasted
    - ``True`` or ``[True, True]``: Both inputs are broadcasted, the output has two
      broadcasting axes.
    - ``[True, False]`` or ``[False, True]``: Only the indicated input is broadcasted
    - ``"merge"``: Both inputs are broadcasted, the output only has one broadcasting axis.

    """
    n, broadcasting_strs = _broadcast_validation(tensor1, tensor2, broadcasted)
    idx_groups_inputs = [
        broadcasting_strs[0] + "abcdef"[:n],
        broadcasting_strs[1] + "ghijkl"[:n],
    ] + ["agz", "bhy", "cix", "djw", "ekv", "flu"][:n]
    idx_output = broadcasting_strs[2] + "zyxwvu"[:n]
    # TODO: Replace einsum by optimized-order tensordots
    einsum_str = ",".join(idx_groups_inputs) + f"->{idx_output}"
    return np.einsum(einsum_str, tensor1, tensor2, *([contractor] * n))


def commutator(tensor1, tensor2, broadcasted=False):
    """Commutator between two tensors representing Pauli sentences on the same number of wires.

    Args:
        tensor1 (np.ndarray): First term in the commutator.
        tensor2 (np.ndarray): Second term in the commutator, with same shape as first
            term up to broadcasting
        broadcasted (Union[bool, list[bool]]): Whether, and which, inputs are broadcasted.
            See below for details.

    Returns:
        np.ndarray: Pauli product of the two input tensors.

    **Broadcasting**

    This function supports broadcasting in up to one leading axis for one or both inputs.
    For this, the ``broadcasted`` input has to be specified:

    - ``False`` or ``[False, False]``: Neither input is broadcasted
    - ``True`` or ``[True, True]``: Both inputs are broadcasted, the output has two
      broadcasting axes.
    - ``[True, False]`` or ``[False, True]``: Only the indicated input is broadcasted
    - ``"merge"``: Both inputs are broadcasted, the output only has one broadcasting axis.

    """
    term1 = product(tensor1, tensor2, broadcasted)
    term2 = np.moveaxis(product(tensor2, tensor1, broadcasted), 1, 0)
    return term1 - term2


def inner_product(tensor1, tensor2, broadcasted=False):
    """ """
    n, broadcasting_strs = _broadcast_validation(tensor1, tensor2, broadcasted)
    idx_groups_inputs = [broadcasting_strs[0] + "abcdef"[:n], broadcasting_strs[1] + "abcdef"[:n]]
    idx_output = broadcasting_strs[2]
    # TODO: Replace einsum by optimized-order tensordots
    einsum_str = ",".join(idx_groups_inputs) + f"->{idx_output}"
    return np.einsum(einsum_str, tensor1.conj(), tensor2)


class VSpace:
    """A class representing a vector space of ``PauliSentence``\ s, encoded as tensors."""

    _tensors: np.ndarray
    num_wires: int
    is_lie_closed: bool

    def __init__(self, tensors, tol=1e-10):
        tensors, tensor_shape, num_wires = self._validate_tensors(tensors)
        self._tensors = tensors
        self.tensor_shape = tensor_shape
        self.num_wires = num_wires
        self.is_lie_closed = False
        self.tol = tol

    @staticmethod
    def _validate_tensors(tensors):
        if isinstance(tensors, (list, tuple)):
            tensors = np.stack(tensors)
        exp_shape = (4,) * (tensors.ndim - 1)
        assert tensors.shape[1:] == exp_shape
        num_wires = tensors.ndim - 1
        return tensors, exp_shape, num_wires

    @property
    def tensors(self):
        return self._tensors

    def __len__(self):
        return len(self._tensors)

    def lie_closure(self, max_iterations: int = 10000):
        if self.is_lie_closed:
            return self.tensors
        old_length = 0
        iteration = 0

        while (len(self) > old_length) and (iteration < max_iterations):
            coms = commutator(self.tensors, self.tensors[old_length:], broadcasted=True)
            old_length = len(self)
            coms = coms.reshape((-1, *self.tensor_shape))
            self.add(coms)

        self.is_lie_closed = True
        return self.tensors

    def project(self, other):
        if isinstance(other, (tuple, list)):
            other = np.stack(other)
        overlaps = inner_product(self.tensors, other, broadcasted=True)
        gram_mat = inner_product(self.tensors, self.tensors, broadcasted=True)
        coeffs = np.linalg.pinv(gram_mat) @ overlaps
        return np.moveaxis(np.tensordot(self.tensors, coeffs, axes=[[0], [0]]), -1, 0)

    def add(self, new_tensors):
        new_tensors -= self.project(new_tensors)
        U, S, _ = np.linalg.svd(new_tensors.reshape((len(new_tensors), -1)).T)
        rank = np.sum(S > self.tol)
        new_tensors = U[:, :rank].T.reshape((rank, *self.tensor_shape))
        self._tensors = np.concatenate([self._tensors, new_tensors], axis=0)

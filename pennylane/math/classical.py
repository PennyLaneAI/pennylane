# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differentiable classical functions"""
from autoray import numpy as np


def frobenius_inner_product(A, B, normalize=False):
    r"""Frobenius inner product between two matrices.

    .. math::

        \langle A, B \rangle_F = \sum_{i,j=1}^n A_{ij} B_{ij} = \operatorname{tr} (A^T B)

    The Frobenius inner product is equivalent to the Hilbert-Schmidt inner product for
    matrices with real-valued entries.

    Args:
        A (array[float]): First matrix, assumed to be a square array.
        B (array[float]): Second matrix, assumed to be a square array.
        normalize (bool): If True, divide the inner_product by the Frobenius norms of A and B.
            Defaults to False.

    Returns:
        float: Frobenius inner product of A and B

    **Example**

    >>> A = np.random.random((3,3))
    >>> B = np.random.random((3,3))
    >>> qml.math.frobenius_inner_product(A, B)
    3.091948202943376
    """
    inner_product = np.sum(A * B)

    if normalize:
        inner_product /= np.linalg.norm(A, "fro") * np.linalg.norm(B, "fro")

    return inner_product

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
"""
This file contains functionalities that simplify working with kernels.
"""
from pennylane import numpy as np


def square_kernel_matrix(X, kernel, assume_normalized_kernel=False):
    r"""Computes the square matrix of pairwise kernel values for a given dataset.

    Args:
        X (list[datapoint]): List of datapoints
        kernel ((datapoint, datapoint) -> float): Kernel function that maps
            datapoints to kernel value.
        assume_normalized_kernel (bool, optional): Assume that the kernel is normalized, in
            which case the diagonal of the kernel matrix is set to 1, avoiding unnecessary
            computations.

    Returns:
        array[float]: The square matrix of kernel values.

    **Example:**

    Consider a simple kernel function based on :class:`~.templates.embeddings.AngleEmbedding`:

    .. code-block :: python

        dev = qml.device('default.qubit', wires=2, shots=None)
        @qml.qnode(dev)
        def circuit(x1, x2):
            qml.templates.AngleEmbedding(x1, wires=dev.wires)
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=dev.wires)
            return qml.probs(wires=dev.wires)

        kernel = lambda x1, x2: circuit(x1, x2)[0]

    We can then compute the kernel matrix on a set of 4 (random) feature
    vectors ``X`` via

    >>> X = np.random.random((4, 2))
    >>> qml.kernels.square_kernel_matrix(X, kernel)
    tensor([[1.        , 0.9532702 , 0.96864001, 0.90932897],
            [0.9532702 , 1.        , 0.99727485, 0.95685561],
            [0.96864001, 0.99727485, 1.        , 0.96605621],
            [0.90932897, 0.95685561, 0.96605621, 1.        ]], requires_grad=True)
    """
    N = len(X)
    matrix = [0] * N ** 2

    for i in range(N):
        for j in range(i, N):
            if assume_normalized_kernel and i == j:
                matrix[N * i + j] = 1.0
            else:
                matrix[N * i + j] = kernel(X[i], X[j])
                matrix[N * j + i] = matrix[N * i + j]

    return np.array(matrix).reshape((N, N))


def kernel_matrix(X1, X2, kernel):
    r"""Computes the matrix of pairwise kernel values for two given datasets.

    Args:
        X1 (list[datapoint]): List of datapoints (first argument)
        X2 (list[datapoint]): List of datapoints (second argument)
        kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.

    Returns:
        array[float]: The square matrix of kernel values.

    **Example:**

    Consider a simple kernel function based on :class:`~.templates.embeddings.AngleEmbedding`:

    .. code-block :: python

        dev = qml.device('default.qubit', wires=2, shots=None)
        @qml.qnode(dev)
        def circuit(x1, x2):
            qml.templates.AngleEmbedding(x1, wires=dev.wires)
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=dev.wires)
            return qml.probs(wires=dev.wires)

        kernel = lambda x1, x2: circuit(x1, x2)[0]

    With this method we can systematically evaluate the kernel function ``kernel`` on
    pairs of datapoints, where the points stem from different datasets, like a training
    and a test dataset.

    >>> X_train = np.random.random((4,2))
    >>> X_test = np.random.random((3,2))
    >>> qml.kernels.kernel_matrix(X_train, X_test, kernel)
    tensor([[0.88875298, 0.90655175, 0.89926447],
            [0.93762197, 0.98163781, 0.93076383],
            [0.91977339, 0.9799841 , 0.91582698],
            [0.80376818, 0.98720925, 0.79349212]], requires_grad=True)

    As we can see, for :math:`n` and :math:`m` datapoints in the first and second
    dataset respectively, the output matrix has the shape :math:`n\times m`.
    """
    N = len(X1)
    M = len(X2)

    matrix = [0] * N * M
    for i in range(N):
        for j in range(M):
            matrix[M * i + j] = kernel(X1[i], X2[j])

    return np.array(matrix).reshape((N, M))

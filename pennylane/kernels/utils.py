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
from itertools import product

from pennylane import math


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

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev)
        def circuit(x1, x2):
            qml.templates.AngleEmbedding(x1, wires=dev.wires)
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=dev.wires)
            return qml.probs(wires=dev.wires)

        kernel = lambda x1, x2: circuit(x1, x2)[0]

    We can then compute the kernel matrix on a set of 4 (random) feature
    vectors ``X`` via

    >>> rng = np.random.default_rng(seed=1234)
    >>> X = rng.random((4, 2))
    >>> qml.kernels.square_kernel_matrix(X, kernel)
    array([[1.        , 0.9957817 , 0.88043387, 0.87011008],
           [0.9957817 , 1.        , 0.90680189, 0.88760331],
           [0.88043387, 0.90680189, 1.        , 0.98850996],
           [0.87011008, 0.88760331, 0.98850996, 1.        ]])
    """
    N = math.shape(X)[0]
    if assume_normalized_kernel and N == 1:
        return math.eye(1, like=math.get_interface(X))

    matrix = [None] * N**2

    # Compute all off-diagonal kernel values, using symmetry of the kernel matrix
    for i in range(N):
        for j in range(i + 1, N):
            matrix[N * i + j] = (kernel_value := kernel(X[i], X[j]))
            matrix[N * j + i] = kernel_value

    if assume_normalized_kernel:
        # Create a one-like entry that has the same interface and batching as the kernel output
        # As we excluded the case N=1 together with assume_normalized_kernel above, matrix[1] exists
        one = math.ones_like(matrix[1])
        for i in range(N):
            matrix[N * i + i] = one
    else:
        # Fill the diagonal by computing the corresponding kernel values
        for i in range(N):
            matrix[N * i + i] = kernel(X[i], X[i])

    shape = (N, N) if math.ndim(matrix[0]) == 0 else (N, N, math.size(matrix[0]))

    return math.moveaxis(math.reshape(math.stack(matrix), shape), -1, 0)


def kernel_matrix(X1, X2, kernel):
    r"""Computes the matrix of pairwise kernel values for two given datasets.

    Args:
        X1 (list[datapoint]): List of datapoints (first argument)
        X2 (list[datapoint]): List of datapoints (second argument)
        kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.

    Returns:
        array[float]: The matrix of kernel values.

    **Example:**

    Consider a simple kernel function based on :class:`~.templates.embeddings.AngleEmbedding`:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev)
        def circuit(x1, x2):
            qml.templates.AngleEmbedding(x1, wires=dev.wires)
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=dev.wires)
            return qml.probs(wires=dev.wires)

        kernel = lambda x1, x2: circuit(x1, x2)[0]

    With this method we can systematically evaluate the kernel function ``kernel`` on
    pairs of datapoints, where the points stem from different datasets, like a training
    and a test dataset.

    >>> rng = np.random.default_rng(seed=1234)
    >>> X_train = rng.random((4,2))
    >>> X_test = rng.random((3,2))
    >>> qml.kernels.kernel_matrix(X_train, X_test, kernel)
    array([[0.99656842, 0.91774724, 0.93966202],
           [0.99958227, 0.91468777, 0.91127346],
           [0.89479886, 0.937256  , 0.80459952],
           [0.87448042, 0.96924743, 0.84069076]])

    As we can see, for :math:`n` and :math:`m` datapoints in the first and second
    dataset respectively, the output matrix has the shape :math:`n\times m`.
    """
    N = math.shape(X1)[0]
    M = math.shape(X2)[0]

    matrix = math.stack([kernel(x, y) for x, y in product(X1, X2)])

    if math.ndim(matrix[0]) == 0:
        return math.reshape(matrix, (N, M))

    return math.moveaxis(math.reshape(matrix, (N, M, math.size(matrix[0]))), -1, 0)

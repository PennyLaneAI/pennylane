# Copyright

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
This file contains functionalities for kernel related costs.
See `here <https://www.doi.org/10.1007/s10462-012-9369-4>`_ for a review.
"""
from pennylane import numpy as np
from ..utils import frobenius_inner_product


def square_kernel_matrix(X, kernel, assume_normalized_kernel=False):
    """Computes the square matrix of pairwise kernel values for a given dataset.

    Args:
        X (list[datapoint]): List of datapoints
        kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.
        assume_normalized_kernel (bool, optional): Assume that the kernel is normalized, i.e.
            that when both arguments are the same datapoint the kernel evaluates to 1. Defaults to False.

    Returns:
        array[float]: The square matrix of kernel values.
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
    """Kernel polarization of a given kernel function.

    Args:
        X1 (list[datapoint]): List of datapoints (first argument)
        X2 (list[datapoint]): List of datapoints (second argument)
        kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.

    Returns:
        array[float]: The square matrix of kernel values.
    """
    N = len(X1)
    M = len(X2)

    matrix = [0] * N * M
    for i in range(N):
        for j in range(M):
            matrix[M * i + j] = kernel(X1[i], X2[j])

    return np.array(matrix).reshape((N, M))


def kernel_polarization(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
    normalize=False,
):
    """Kernel polarization of a given kernel function.

    Args:
        X (list[datapoint]): List of datapoints
        Y (list[float]): List of class labels of datapoints, assumed to be either -1 or 1.
        kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.
        assume_normalized_kernel (bool, optional): Assume that the kernel is normalized, i.e.
            that when both arguments are the same datapoint the kernel evaluates to 1. Defaults to False.
        rescale_class_labels (bool, optional): Rescale the class labels. This is important to take
            care of unbalanced datasets. Defaults to True.
        normalize (bool): If True, rescale the polarization to the kernel_target_alignment. Defaults to False.

    Returns:
        float: The kernel polarization.
    """
    K = square_kernel_matrix(X, kernel, assume_normalized_kernel=assume_normalized_kernel)

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)

    return frobenius_inner_product(K, T, normalize=normalize)


def kernel_target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel target alignment of a given kernel function.

    Args:
        X (list[datapoint]): List of datapoints
        Y (list[float]): List of class labels of datapoints, assumed to be either -1 or 1.
        kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.
        assume_normalized_kernel (bool, optional): Assume that the kernel is normalized, i.e.
            that when both arguments are the same datapoint the kernel evaluates to 1. Defaults to False.
        rescale_class_labels (bool, optional): Rescale the class labels. This is important to take
            care of unbalanced datasets. Defaults to True.

    Returns:
        float: The kernel-target alignment.
    """
    return kernel_polarization(
        X,
        Y,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
        rescale_class_labels=rescale_class_labels,
        normalize=True,
    )

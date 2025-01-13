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
"""Utility tools for dynamical Lie algebra functionality"""
from typing import Union

import numpy as np

from pennylane.operation import Operator
from pennylane.pauli import PauliSentence


def trace_inner_product(
    A: Union[PauliSentence, Operator, np.ndarray], B: Union[PauliSentence, Operator, np.ndarray]
):
    r"""Implementation of the trace inner product :math:`\langle A, B \rangle = \text{tr}\left(A B\right)/\text{dim}(A)` between two Hermitian operators :math:`A` and :math:`B`.

    If the inputs are ``np.ndarray``, leading broadcasting axes are supported for either or both
    inputs.

    Args:
        A (Union[PauliSentence, Operator, np.ndarray]): First operator
        B (Union[PauliSentence, Operator, np.ndarray]): Second operator

    Returns:
        Union[float, np.ndarray]: Result is either a single float or a batch of floats.

    **Example**

    >>> from pennylane.labs.dla import trace_inner_product
    >>> trace_inner_product(qml.X(0) + qml.Y(0), qml.Y(0) + qml.Z(0))
    1.0

    If both operators are dense arrays, a leading batch dimension is broadcasted.

    >>> batch = 10
    >>> ops1 = np.random.rand(batch, 16, 16)
    >>> op2 = np.random.rand(16, 16)
    >>> trace_inner_product(ops1, op2).shape
    (10,)
    >>> trace_inner_product(op2, ops1).shape
    (10,)

    We can also have both arguments broadcasted.

    >>> trace_inner_product(ops1, ops1).shape
    (10, 10)

    """
    if getattr(A, "pauli_rep", None) is not None and getattr(B, "pauli_rep", None) is not None:
        return (A.pauli_rep @ B.pauli_rep).trace()

    if all(isinstance(op, np.ndarray) for op in A) and all(isinstance(op, np.ndarray) for op in B):
        A = np.array(A)
        B = np.array(B)

    if isinstance(A, np.ndarray):
        assert A.shape[-2:] == B.shape[-2:]
        # The axes of the first input are switched, compared to tr[A@B], because we need to
        # transpose A.
        return np.tensordot(A, B, axes=[[-1, -2], [-2, -1]]) / A.shape[-1]

    return NotImplemented

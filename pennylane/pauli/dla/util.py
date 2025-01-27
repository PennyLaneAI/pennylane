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
from typing import Iterable, Union

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence
from pennylane.typing import TensorLike


def trace_inner_product(
    A: Union[PauliSentence, Operator, np.ndarray], B: Union[PauliSentence, Operator, np.ndarray]
):
    r"""Trace inner product :math:`\langle A, B \rangle = \text{tr}\left(A^\dagger B\right)/\text{dim}(A)` between two operators :math:`A` and :math:`B`.

    If the inputs are ``np.ndarray``, leading broadcasting axes are supported for either or both
    inputs.

    .. warning::

        Operator inputs are assumed to be Hermitian. In particular,
        sums of Pauli operators are assumed to have real-valued coefficients.
        We recommend to use matrix representations for non-Hermitian inputs.
        In case of non-Hermitian :class:`~PauliSentence` or :class:`Operator` inputs,
        the Hermitian conjugation needs to be done manually by inputting :math:`A = A^\dagger`.

    Args:
        A (Union[PauliSentence, Operator, np.ndarray]): First operator
        B (Union[PauliSentence, Operator, np.ndarray]): Second operator

    Returns:
        Union[float, np.ndarray]: Result is either a single float or a batch of floats.

    **Example**

    >>> from pennylane.pauli import trace_inner_product
    >>> trace_inner_product(qml.X(0) + qml.Y(0), qml.Y(0) + qml.Z(0))
    1.0

    If both operators are arrays, a leading batch dimension is broadcasted.

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

    .. details::
        :title: Usage Details

        :class:`~PauliSentence` and :class:`~Operator` inputs are assumed to be Hermitian. In particular,
        the input ``A`` is not conjugated when operators are used. To get correct results, we can either use
        the matrix representation or manually conjugate the operator.

        >>> A = qml.X(0) - 1j * qml.Y(0)
        >>> Ad = qml.X(0) + 1j * qml.Y(0)
        >>> B = qml.X(0) + 1j * qml.Y(0)
        >>> trace_inner_product(Ad, B) == trace_inner_product(qml.matrix(A), qml.matrix(B))
        True

    """
    if getattr(A, "pauli_rep", None) is not None and getattr(B, "pauli_rep", None) is not None:
        # No dagger needed as paulis are Hermitian
        return (A.pauli_rep @ B.pauli_rep).trace()

    if isinstance(A, Iterable) and isinstance(B, Iterable):

        if not isinstance(A, TensorLike) or isinstance(A, (list, tuple)):
            interface_A = qml.math.get_interface(A)
            A = qml.math.array(A, like=interface_A)

        if not isinstance(B, TensorLike) or isinstance(B, (list, tuple)):
            interface_B = qml.math.get_interface(B)
            B = qml.math.array(B, like=interface_B)

        assert A.shape[-2:] == B.shape[-2:]
        # The axes of the first input are switched, compared to tr[A@B], because we need to
        # transpose A.
        return qml.math.tensordot(qml.math.conj(A), B, axes=[[-2, -1], [-2, -1]]) / A.shape[-1]

    raise NotImplementedError(
        "Inputs to pennylane.pauli.trace_inner_product need to be iterables of matrices or operators with a pauli_rep"
    )

# Copyright 2025 Xanadu Quantum Technologies Inc.

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
from collections.abc import Iterable

import pennylane as qp
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence
from pennylane.typing import TensorLike


def trace_inner_product(
    A: PauliSentence | Operator | TensorLike, B: PauliSentence | Operator | TensorLike
):
    r"""Trace inner product :math:`\langle A, B \rangle = \text{tr}\left(A^\dagger B\right)/\text{dim}(A)` between two operators :math:`A` and :math:`B`.

    If the inputs are ``np.ndarray``, leading [broadcasting](https://docs.pennylane.ai/en/stable/introduction/circuits.html#parameter-broadcasting-in-qnodes) axes are supported for either or both
    inputs.

    .. warning::

        Operator inputs are assumed to be Hermitian. In particular,
        sums of Pauli operators are assumed to have real-valued coefficients.
        We recommend to use matrix representations for non-Hermitian inputs.
        In case of non-Hermitian :class:`~PauliSentence` or :class:`Operator` inputs,
        the Hermitian conjugation needs to be done manually by inputting :math:`A^\dagger`.

    Args:
        A (Union[PauliSentence, Operator, TensorLike]): First operator
        B (Union[PauliSentence, Operator, TensorLike]): Second operator of the same type as ``A``

    Returns:
        Union[float, TensorLike]: Result is either a single float or an array of floats (in batches of the broadcasting dimension).

    **Example**

    >>> from pennylane.pauli import trace_inner_product
    >>> trace_inner_product(qp.X(0) + qp.Y(0), qp.Y(0) + qp.Z(0))
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

        >>> A = qp.X(0) - 1j * qp.Y(0)
        >>> Ad = qp.X(0) + 1j * qp.Y(0)
        >>> B = qp.X(0) + 1j * qp.Y(0)
        >>> print(trace_inner_product(Ad, B) == trace_inner_product(qp.matrix(A), qp.matrix(B)))
        True

    """
    if getattr(A, "pauli_rep", None) is not None and getattr(B, "pauli_rep", None) is not None:
        # No dagger needed as paulis are Hermitian
        return (A.pauli_rep @ B.pauli_rep).trace()

    if isinstance(A, Iterable) and isinstance(B, Iterable):

        if isinstance(A, (list, tuple)):
            interface_A = qp.math.get_interface(A[0])
            A = qp.math.stack(A, like=interface_A)

        if isinstance(B, (list, tuple)):
            interface_B = qp.math.get_interface(B[0])
            B = qp.math.stack(B, like=interface_B)

        # tr(A^dagger @ B) = (A^dagger)_ij B_ji = A^*_ji B_ji
        return (
            qp.math.tensordot(qp.math.conj(A), B, axes=[[-2, -1], [-2, -1]])
            / qp.math.shape(A)[-1]
        )

    raise NotImplementedError(
        "Inputs to pennylane.pauli.trace_inner_product need to be of the same type and iterables of matrices or operators with a pauli_rep"
    )

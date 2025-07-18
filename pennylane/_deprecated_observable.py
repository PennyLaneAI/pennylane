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
"""
This module contains the deprecated Observable class to allow for deprecation warnings on access to the class.
"""
import warnings
from typing import Union

import pennylane as qml

from .operation import Operation, Operator


class Observable(Operator):
    """Base class representing observables.

    .. warning::

        ``qml.operation.Observable`` is now deprecated. A generic operator can be used anywhere an ``Observable``
        can, and is less restrictive.  To preserve prior ``Observable`` default behavior, an operator can override
        ``Operator.queue()`` with empty behavior, and set ``is_hermitian = True`` manually:

        .. code-block:: python

            class MyObs(Operator):

                is_hermitian = True

                def queue(self, context=qml.QueuingManager):
                    return self



    Args:
        params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wi're label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    @property
    def _queue_category(self):
        return "_ops" if isinstance(self, Operation) else None

    @property
    def is_hermitian(self) -> bool:
        """All observables must be hermitian"""
        return True

    def compare(self, other: Union["Observable", "qml.ops.LinearCombination"]) -> bool:
        r"""Compares with another :class:`~Observable`, to determine if they are equivalent.

        .. warning::

            This method is deprecated. ``qml.equal`` or ``op1 == op2`` should be used instead.

        Observables are equivalent if they represent the same operator
        (their matrix representations are equal), and they are defined on the same wires.

        .. Warning::

            The compare method does **not** check if the matrix representation
            of a :class:`~.Hermitian` observable is equal to an equivalent
            observable expressed in terms of Pauli matrices.
            To do so would require the matrix form to be calculated, which would
            drastically increase runtime.

        Returns:
            (bool): True if equivalent.

        **Examples**

        >>> ob1 = qml.X(0) @ qml.Identity(1)
        >>> ob2 = qml.Hamiltonian([1], [qml.X(0)])
        >>> ob1.compare(ob2)
        True
        >>> ob1 = qml.X(0)
        >>> ob2 = qml.Hermitian(np.array([[0, 1], [1, 0]]), 0)
        >>> ob1.compare(ob2)
        False
        """
        warnings.warn(
            "The compare method is deprecated and will be removed in v0.43."
            " op1 == op2 or qml.equal should be used instead.",
            qml.exceptions.PennyLaneDeprecationWarning,
        )
        return qml.equal(self, other)

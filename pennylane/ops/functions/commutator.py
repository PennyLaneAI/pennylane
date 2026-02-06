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
"""
This file contains the implementation of the commutator function in PennyLane
"""
import pennylane as qp
from pennylane.pauli import PauliSentence, PauliWord


def commutator(op1, op2, pauli=False):
    r"""Compute commutator between two operators in PennyLane

    .. math:: [O_1, O_2] = O_1 O_2 - O_2 O_1

    Args:
        op1 (Union[Operator, PauliWord, PauliSentence]): First operator
        op2 (Union[Operator, PauliWord, PauliSentence]): Second operator
        pauli (bool): When ``True``, all results are passed as a ``PauliSentence`` instance. Else, results are always returned as ``Operator`` instances.

    Returns:
        ~Operator or ~PauliSentence: The commutator

    **Examples**

    You can compute commutators between operators in PennyLane.

    >>> qp.commutator(qp.X(0), qp.Y(0))
    2j * Z(0)

    >>> op1 = qp.X(0) @ qp.X(1)
    >>> op2 = qp.Y(0) @ qp.Y(1)
    >>> qp.commutator(op1, op2)
    0 * I()

    We can return a :class:`~PauliSentence` instance by setting ``pauli=True``.

    >>> op1 = qp.X(0) @ qp.X(1)
    >>> op2 = qp.Y(0) + qp.Y(1)
    >>> res = qp.commutator(op1, op2, pauli=True)
    >>> res
    2j * Z(0) @ X(1)
    + 2j * X(0) @ Z(1)
    >>> isinstance(res, PauliSentence)
    True

    We can also input :class:`~PauliWord` and :class:`~PauliSentence` instances.

    >>> op1 = PauliWord({0:"X", 1:"X"})
    >>> op2 = PauliWord({0:"Y"}) + PauliWord({1:"Y"})
    >>> res = qp.commutator(op1, op2, pauli=True)
    >>> res
    2j * Z(0) @ X(1)
    + 2j * X(0) @ Z(1)
    >>> isinstance(res, PauliSentence)
    True

    Note that when ``pauli=False``, even if Pauli operators are used
    as inputs, ``qp.commutator`` returns Operators.

    >>> res = qp.commutator(op1, op2, pauli=False)
    >>> res
    2j * (Z(0) @ X(1)) + 2j * (X(0) @ Z(1))
    >>> isinstance(res, qp.operation.Operator)
    True

    It is worth noting that computing commutators with Paulis is typically faster.
    Internally, ``qp.commutator`` uses the ``op.pauli_rep`` whenever that is available for both operators.

    .. details::
        :title: Usage Details

        The input and result of ``qp.commutator`` is not recorded in a tape context (and inside a :class:`~QNode`).

        .. code-block:: python3

            with qp.tape.QuantumTape() as tape:
                a = qp.X(0)      # gets recorded
                b = PauliWord({0:"Y"}) # does not get recorded
                comm = qp.commutator(a, b) # does not get recorded

        In this example, we obtain ``tape.operations = [qp.X(0)]``. When desired, we can still record the result of
        the commutator by using :func:`~apply`, i.e. ``qp.apply(comm)`` inside the recording context.

        A peculiarity worth repeating is how in a recording context every created operator is recorded.

        .. code-block:: python3

            with qp.tape.QuantumTape() as tape:
                comm = qp.commutator(qp.X(0), qp.Y(0))

        In this example, both :class:`~PauliX` and :class:`PauliY` get recorded because they were created inside the
        recording context. To avoid this, create the input to ``qp.commutator`` outside the recording context / qnode
        or insert an extra ``stop_recording()`` context (see :class:`~QueuingManager`).

    """

    both_have_pauli_rep = op1.pauli_rep is not None and op2.pauli_rep is not None

    if pauli or both_have_pauli_rep:
        if not isinstance(op1, PauliSentence):
            op1 = qp.pauli.pauli_sentence(op1)
        if not isinstance(op2, PauliSentence):
            op2 = qp.pauli.pauli_sentence(op2)
        res = op1.commutator(op2)
        return res if pauli else res.operation(wire_order=res.wires)

    # If no pauli processing is possible, use operator arithmetic
    with qp.QueuingManager.stop_recording():
        if isinstance(op1, (PauliWord, PauliSentence)):
            op1 = op1.operation()
        if isinstance(op2, (PauliWord, PauliSentence)):
            op2 = op2.operation()

        res = qp.sum(qp.prod(op1, op2), qp.s_prod(-1.0, qp.prod(op2, op1)))
        res = res.simplify()
    return res


comm = commutator  # Slightly shorter alias

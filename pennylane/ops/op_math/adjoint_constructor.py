# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule applies the symbolic operation that indicates the adjoint of an operator through the `adjoint` transform.
"""
from functools import wraps

from pennylane.operation import Operator, AdjointUndefinedError
from pennylane.tape import QuantumTape, stop_recording

from .adjoint_class import Adjoint


def _single_op_eager(op):
    try:
        return op.adjoint()
    except AdjointUndefinedError:
        return Adjoint(op)


# pylint: disable=no-member
def adjoint(fn, lazy=True):
    """Create a function that applies the adjoint of the provided operation or template.

    This transform can be used to apply the adjoint of an arbitrary sequence of operations.

    Args:
        fn (function): A single operator or a quantum function that
            applies quantum operations.

    Keyword Args:
        lazy=True (bool): If the transform is behaving lazily, all operations are wrapped in a `Adjoint` class
            and handled later. If ``lazy=False``, operation-specific adjoint decompositions are first attempted.

    Returns:
        function: A new function that will apply the same operations but adjointed and in reverse order.

    .. note::

        While the adjoint and inverse are identical for Unitary gates, not all possible operators are Unitary.
        This transform can also act on Channels and Hamiltonians, for which the inverse and adjoint are different.

    .. seealso:: :class:`~.ops.arithmetic.Adjoint` and :meth:`~.operation.Operator.adjoint`

    **Example**

    The adjoint transforms can be used within a QNode to apply the adjoint of
    any quantum function. Consider the following quantum function, that applies two
    operations:

    .. code-block:: python3

        def my_ops(a, wire):
            qml.RX(a, wires=wire)
            qml.SX(wire)

    We can create a QNode that applies this quantum function,
    followed by the adjoint of this function:

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(a):
            my_ops(a, wire=0)
            qml.adjoint(my_ops)(a, wire=0)
            return qml.expval(qml.PauliZ(0))

    Printing this out, we can see that the inverse quantum
    function has indeed been applied:

    >>> print(qml.draw(circuit)(0.2))
    0: ──RX(0.20)──SX──SX†──RX(0.20)†─┤  <Z>

    .. details::
        :title: Usage Details

        **Adjoint of a function**

        Here, we apply the ``subroutine`` function, and then apply its adjoint.
        Notice that in addition to adjointing all of the operations, they are also
        applied in reverse construction order. Some `Adjoint` gates like those wrapping ``SX``, ``S``, and
        ``T`` are natively supported by ``default.qubit``. Other gates will be expanded either using a custom
        adjoint decomposition defined in :meth:`~.operation.Operator.adjoint`.

        .. code-block:: python3

            def subroutine(wire):
                qml.RX(0.123, wires=wire)
                qml.RY(0.456, wires=wire)

            dev = qml.device('default.qubit', wires=1)
            @qml.qnode(dev)
            def circuit():
                subroutine(0)
                qml.adjoint(subroutine)(0)
                return qml.expval(qml.PauliZ(0))

        This creates the following circuit:

        >>> print(qml.draw(circuit)())
        0: ──RX(0.12)──S──S†──RX(0.12)†─┤  <Z>
        >>> print(qml.draw(circuit, expansion_strategy="device")())
        0: ──RX(0.12)──S──S†──RX(-0.12)─┤  <Z>

        **Single operation**

        You can also easily adjoint a single operation just by wrapping it with ``adjoint``:

        .. code-block:: python3

            dev = qml.device('default.qubit', wires=1)
            @qml.qnode(dev)
            def circuit():
                qml.RX(0.123, wires=0)
                qml.adjoint(qml.RX)(0.123, wires=0)
                return qml.expval(qml.PauliZ(0))

        This creates the following circuit:

        >>> print(qml.draw(circuit)())
        0: ──RX(0.12)──RX(0.12)†─┤  <Z>


        :title: Developer details

        **Lazy Evaluation**

        When ``lazy=False``, the function first attempts operation-specific decomposition of the
        adjoint via the :meth:`.operation.Operator.adjoint` method. Only if an Operator doesn't have
        an :meth:`.operation.Operator.adjoint` method is the object wrapped with the :class:`~.ops.arithmetic.Adjoint`
        wrapper class.

        >>> qml.adjoint(qml.PauliZ, lazy=False)(0)
        PauliZ(wires=[0])
        >>> qml.adjoint(qml.RX, lazy=False)(1.0, wires=0)
        RX(-1.0, wires=[0])
        >>> qml.adjoint(qml.S, lazy=False)(0)
        Adjoint(S)(wires=[0])

    """
    if isinstance(fn, Operator):
        return Adjoint(fn) if lazy else _single_op_eager(fn)
    if not callable(fn):
        raise ValueError(
            f"The object {fn} of type {type(fn)} is not callable. "
            "This error might occur if you apply adjoint to a list "
            "of operations instead of a function or template."
        )

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with stop_recording(), QuantumTape() as tape:
            fn(*args, **kwargs)

        if lazy:
            adjoint_ops = [Adjoint(op) for op in reversed(tape.operations)]
        else:
            adjoint_ops = [_single_op_eager(op) for op in reversed(tape.operations)]

        return adjoint_ops[0] if len(adjoint_ops) == 1 else adjoint_ops

    return wrapper

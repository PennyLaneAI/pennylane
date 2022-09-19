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
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumTape

from .adjoint_class import Adjoint


def _single_op_eager(op, update_queue=False):
    try:
        adj = op.adjoint()
        if update_queue:
            QueuingManager.safe_update_info(op, owner=adj)
            QueuingManager.append(adj, owns=op)
        return adj
    except AdjointUndefinedError:
        return Adjoint(op)


# pylint: disable=no-member
def adjoint(fn, lazy=True):
    """Create the adjoint of an Operator or a function that applies the adjoint of the provided function.

    Args:
        fn (function or :class:`~.operation.Operator`): A single operator or a quantum function that
            applies quantum operations.

    Keyword Args:
        lazy=True (bool): If the transform is behaving lazily, all operations are wrapped in a ``Adjoint`` class
            and handled later. If ``lazy=False``, operation-specific adjoint decompositions are first attempted.

    Returns:
        (function or :class:`~.operation.Operator`): If an Operator is provided, returns an Operator that is the adjoint.
        If a function is provided, returns a function with the same call signature that returns the Adjoint of the
        provided function.

    .. note::

        The adjoint and inverse are identical for unitary gates, but not in general. For example, quantum channels and observables may have different adjoint and inverse operators.

    .. seealso:: :class:`~.ops.op_math.Adjoint` and :meth:`.Operator.adjoint`

    **Example**

    The adjoint transform can accept a single operator.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit2(y):
    ...     qml.adjoint(qml.RY(y, wires=0))
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(qml.draw(circuit2)("y"))
    0: ──RY(y)†─┤  <Z>
    >>> print(qml.draw(circuit2, expansion_strategy="device")(0.1))
    0: ──RY(-0.10)─┤  <Z>

    The adjoint transforms can also be used to apply the adjoint of
    any quantum function.  In this case, ``adjoint`` accepts a single function and returns
    a function with the same call signature.

    We can create a QNode that applies the ``my_ops`` function followed by its adjoint:

    .. code-block:: python3

        def my_ops(a, wire):
            qml.RX(a, wires=wire)
            qml.SX(wire)

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
        :title: Lazy Evaluation

        When ``lazy=False``, the function first attempts operation-specific decomposition of the
        adjoint via the :meth:`.Operator.adjoint` method. Only if an Operator doesn't have
        an :meth:`.Operator.adjoint` method is the object wrapped with the :class:`~.ops.op_math.Adjoint`
        wrapper class.

        >>> qml.adjoint(qml.PauliZ(0), lazy=False)
        PauliZ(wires=[0])
        >>> qml.adjoint(qml.RX, lazy=False)(1.0, wires=0)
        RX(-1.0, wires=[0])
        >>> qml.adjoint(qml.S, lazy=False)(0)
        Adjoint(S)(wires=[0])

    """
    if isinstance(fn, Operator):
        return Adjoint(fn) if lazy else _single_op_eager(fn, update_queue=True)
    if not callable(fn):
        raise ValueError(
            f"The object {fn} of type {type(fn)} is not callable. "
            "This error might occur if you apply adjoint to a list "
            "of operations instead of a function or template."
        )

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with QueuingManager.stop_recording(), QuantumTape() as tape:
            fn(*args, **kwargs)

        if lazy:
            adjoint_ops = [Adjoint(op) for op in reversed(tape.operations)]
        else:
            adjoint_ops = [_single_op_eager(op) for op in reversed(tape.operations)]

        return adjoint_ops[0] if len(adjoint_ops) == 1 else adjoint_ops

    return wrapper

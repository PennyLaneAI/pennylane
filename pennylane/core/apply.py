# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This module contains the :func:`apply` utility.

The ``apply`` utility records an instantiated operator or measurement in the active program,
whether PennyLane is using a queuing context or program capture.
"""

import copy

from pennylane import capture, pytrees  # tach-ignore
from pennylane.core.queuing import AnnotatedQueue, QueuingManager


def apply(op, context: type[QueuingManager] | AnnotatedQueue = QueuingManager):
    """Apply an instantiated operator or measurement to the active program.

    Args:
        op (.Operator or .MeasurementProcess): the operator or measurement to apply
        context (type[.QueuingManager] | AnnotatedQueue): The queuing context to queue the operator
            to when queuing is active. Note that if no context is specified, the operator is
            applied to the currently active queuing context.
    Returns:
        .Operator or .MeasurementProcess: the input operator is returned for convenience
    Raises:
        RuntimeError: if we try to use apply in a non-queuing/non-tracing context.

    **Example**

    In PennyLane, operations and measurements are 'queued' or added to a circuit
    when they are instantiated.

    The ``apply`` function can be used to add operations that might have
    already been instantiated elsewhere to the QNode:

    .. code-block:: python

        op = qp.RX(0.4, wires=0)
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(x):
            qp.RY(x, wires=0)  # applied during instantiation
            qp.apply(op)  # manually applied
            return qp.expval(qp.Z(0))

    >>> print(qp.draw(circuit)(0.6))
    0: ──RY(0.60)──RX(0.40)─┤  <Z>

    It can also be used to apply functions repeatedly:

    .. code-block:: python

        @qp.qnode(dev)
        def circuit(x):
            qp.apply(op)
            qp.RY(x, wires=0)
            qp.apply(op)
            return qp.expval(qp.Z(0))

    >>> print(qp.draw(circuit)(0.6))
    0: ──RX(0.40)──RY(0.60)──RX(0.40)─┤  <Z>

    .. warning::

        If you use ``apply`` on an operator that has already been queued, it will
        be queued for a second time. For example:

        .. code-block:: python

            @qp.qnode(dev)
            def circuit():
                op = qp.Hadamard(0)
                qp.apply(op)
                return qp.expval(qp.Z(0))

        >>> print(qp.draw(circuit)())
        0: ──H──H─┤  <Z>

    .. details::
        :title: Usage Details

        Instantiated measurements can also be applied to queuing contexts
        using ``apply``:

        .. code-block:: python

            meas = qp.expval(qp.Z(0) @ qp.Y(1))
            dev = qp.device("default.qubit", wires=2)

            @qp.qnode(dev)
            def circuit(x):
                qp.RY(x, wires=0)
                qp.CNOT(wires=[0, 1])
                return qp.apply(meas)

        >>> print(qp.draw(circuit)(0.6))
        0: ──RY(0.60)─╭●─┤ ╭<Z@Y>
        1: ───────────╰X─┤ ╰<Z@Y>

        By default, ``apply`` will queue operators to the currently
        active queuing context.

    """

    if capture.enabled():
        return _capture_apply(op)

    if not QueuingManager.recording():
        raise RuntimeError("No queuing context available to append operation to.")

    # Always make a copy since we don't want the provided op to be dequeued by a subsequent
    # PennyLane Operator/Function.
    # Note that queuing contexts can only contain unique objects.
    with QueuingManager.stop_recording():
        op = copy.copy(op)

    if hasattr(op, "queue"):
        # operator provides its own logic for queuing
        op.queue(context=context)
    else:
        # append the operator directly to the relevant queuing context
        context.append(op)

    return op


def _capture_apply(op):
    """Applies an op in a capture context."""

    if hasattr(op, "_bind_primitive"):
        # NOTE: Shallow-copy to avoid mutating the input operator
        op = copy.copy(op)
        # NOTE: Reset tracer attribute to prevent tracer leaks
        op.tracer = None
        op._bind_primitive()  # pylint: disable=protected-access
        if op.tracer is None:
            raise RuntimeError("Trying to use apply in a non-tracing context.")
        return op

    # Capture is active but the op has no _bind_primitive (e.g. minimal
    # legacy Operator subclass). Reconstruct via the constructor so the
    # new instance auto-binds its primitive.
    return pytrees.unflatten(*pytrees.flatten(op))


__all__ = ["apply"]

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

from collections import deque
from copy import copy


class QueuingError(Exception):
    """Exception that is raised when there is a queuing error"""


class TapeManager:

    _instance = None

    active_contexts = deque()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TapeManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def recording(cls):
        return bool(cls.active_contexts)

    @classmethod
    def active_context(cls):
        return cls.active_contexts[-1] if cls.recording() else None

    @classmethod
    def append(cls, obj, **kwargs):
        if cls.recording():
            cls.active_context().append(obj, **kwargs)

    @classmethod
    def remove(cls, obj):
        if cls.recording():
            cls.active_context().remove(obj)

    @classmethod
    def update_info(cls, obj, **kwargs):
        if cls.recording():
            cls.active_context().update_info(obj, **kwargs)

    @classmethod
    def get_info(cls, obj):
        if cls.recording():
            return cls.active_context().get_info(obj)
        return None


def apply(op, context=TapeManager):
    """Apply an instantiated operator or measurement to a queuing context.

    Args:
        op (.Operator or .MeasurementProcess): the operator or measurement to apply/queue
        context (.QueuingContext): The queuing context to queue the operator to.
            Note that if no context is specified, the operator is
            applied to the currently active queuing context.
    Returns:
        .Operator or .MeasurementProcess: the input operator is returned for convenience

    **Example**

    In PennyLane, **operations and measurements are 'queued' or added to a circuit
    when they are instantiated**.

    The ``apply`` function can be used to add operations that might have
    already been instantiated elsewhere to the QNode:

    .. code-block:: python

        op = qml.RX(0.4, wires=0)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)  # applied during instantiation
            qml.apply(op)  # manually applied
            return qml.expval(qml.PauliZ(0))

    >>> print(qml.draw(circuit)(0.6))
    0: ──RY(0.6)──RX(0.4)──┤ ⟨Z⟩

    It can also be used to apply functions repeatedly:

    .. code-block:: python

        @qml.qnode(dev)
        def circuit(x):
            qml.apply(op)
            qml.RY(x, wires=0)
            qml.apply(op)
            return qml.expval(qml.PauliZ(0))

    >>> print(qml.draw(circuit)(0.6))
    0: ──RX(0.4)──RY(0.6)──RX(0.4)──┤ ⟨Z⟩

    .. UsageDetails::

        Instantiated measurements can also be applied to queuing contexts
        using ``apply``:

        .. code-block:: python

            meas = qml.expval(qml.PauliZ(0) @ qml.PauliY(1))
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def circuit(x):
                qml.RY(x, wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.apply(meas)

        >>> print(qml.draw(circuit)(0.6))
         0: ──RY(0.6)──╭C──╭┤ ⟨Z ⊗ Y⟩
         1: ───────────╰X──╰┤ ⟨Z ⊗ Y⟩

        By default, ``apply`` will queue operators to the currently
        active queuing context.

        When working with low-level queuing contexts such as quantum tapes,
        the desired context to queue the operation to can be explicitly
        passed:

        .. code-block:: python

            with qml.tape.QuantumTape() as tape1:
                qml.Hadamard(wires=1)

                with qml.tape.QuantumTape() as tape2:
                    # Due to the nesting behaviour of queuing contexts,
                    # tape2 will be queued to tape1.

                    # The following PauliX operation will be queued
                    # to the active queuing context, tape2, during instantiation.
                    op1 = qml.PauliX(wires=0)

                    # We can use qml.apply to apply the same operation to tape1
                    # without leaving the tape2 context.
                    qml.apply(op1, context=tape1)

                    qml.RZ(0.2, wires=0)

                qml.CNOT(wires=[0, 1])

        >>> tape1.operations
        [Hadamard(wires=[1]), <QuantumTape: wires=[0], params=1>, PauliX(wires=[0]), CNOT(wires=[0, 1])]
        >>> tape2.operations
        [PauliX(wires=[0]), RZ(0.2, wires=[0])]
    """
    if not TapeManager.recording():
        raise RuntimeError("No queuing context available to append operation to.")

    if op in getattr(context, "queue", TapeManager.active_context().queue):
        # Queuing contexts can only contain unique objects.
        # If the object to be queued already exists, copy it.
        op = copy(op)

    if hasattr(op, "queue"):
        # operator provides its own logic for queuing
        op.queue(context=context)
    else:
        # append the operator directly to the relevant queuing context
        context.append(op)

    return op

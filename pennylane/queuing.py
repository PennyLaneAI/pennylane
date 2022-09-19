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
"""
This module contains the :class:`QueuingManager`.
"""

import copy
from collections import OrderedDict
from contextlib import contextmanager
from warnings import warn


def __getattr__(name):
    # for more information on overwriting `__getattr__`, see https://peps.python.org/pep-0562/
    if name == "QueuingContext":
        warn("QueuingContext has been renamed qml.queuing.QueuingManager.", UserWarning)
        return QueuingManager
    try:
        return globals()[name]
    except KeyError as e:
        raise AttributeError from e


class QueuingError(Exception):
    """Exception that is raised when there is a queuing error"""


class QueuingManager:
    """Singleton global entry point for managing active recording contexts.

    This class consists purely of class methods. It both maintains a list of
    recording queues and allows communication with the currently active object.

    Queueable objects, like :class:`~.operation.Operator` and :class:`~.measurements.MeasurementProcess`, should
    use ``QueuingManager`` as an entry point for accessing the active queue.

    See also: :class:`~.AnnotatedQueue`, :class:`~.tape.QuantumTape`, :meth:`~.operation.Operator.queue`.

    Recording queues, such as :class:`~.AnnotatedQueue`, must define the following methods:

    * ``append``: define an action to perform when an object append
      request is made.

    * ``remove``: define an action to perform when an object removal request is made.

    * ``get_info``: retrieve the object's metadata

    * ``update_info``: Update an object's metadata if it is already queued. Else, raise a ``QueuingError``.

    * ``safe_update_info``: Update an object's metadata without raising errors

    To start and end recording, the recording queue can use the :meth:`add_active_queue` and
    :meth:`remove_active_queue` methods.

    """

    _active_contexts = []
    """The stack of contexts that are currently active."""

    @classmethod
    def add_active_queue(cls, queue):
        """Makes a queue the currently active recording context."""
        cls._active_contexts.append(queue)

    @classmethod
    def remove_active_queue(cls):
        """Ends recording on the currently active recording queue."""
        return cls._active_contexts.pop()

    @classmethod
    def recording(cls):
        """Whether a queuing context is active and recording operations"""
        return bool(cls._active_contexts)

    @classmethod
    def active_context(cls):
        """Returns the currently active queuing context."""
        return cls._active_contexts[-1] if cls.recording() else None

    @classmethod
    @contextmanager
    def stop_recording(cls):
        """A context manager and decorator to ensure that contained logic is non-recordable
        or non-queueable within a QNode or quantum tape context.

        **Example:**

        Consider the function:

        >>> def qfunc(x):
        ...     qml.RX(x, 0)

        If executed in a recording context, the operations constructed in the function will be queued:

        >>> @qml.qnode(qml.device('default.qubit', wires=1))
        ... def circuit(x):
        ...     qfunc(x)
        ...     return qml.expval(qml.PauliZ(0))
        >>> print(qml.draw(circuit)("x"))
        ... 0: ──RX(x)─┤  <Z>

        Using the ``stop_recording`` context manager, all logic contained inside is not queued or recorded.

        >>> @qml.qnode(qml.device('default.qubit', wires=1))
        ... def circuit(x):
        ...     with qml.QueuingManager.stop_recording():
        ...         qfunc(x)
        ...     return qml.expval(qml.PauliZ(0))
        >>> print(qml.draw(circuit)("x"))
        0: ───┤  <Z>

        The context manager can also be used as a decorator on a function:

        >>> @qml.QueuingManager.stop_recording()
        ... def qfunc_stopped(y):
        ...     qml.RY(y, 0)
        >>> @qml.qnode(qml.device('default.qubit', wires=1))
        ... def circuit(x):
        ...     qfunc_stopped(x)
        ...     return qml.expval(qml.PauliZ(0))
        >>> print(qml.draw(circuit)("y"))
        0: ───┤  <Z>

        """
        previously_active_contexts = cls._active_contexts
        cls._active_contexts = []
        yield
        cls._active_contexts = previously_active_contexts

    @classmethod
    def append(cls, obj, **kwargs):
        """Append an object to the queue(s).

        Args:
            obj: the object to be appended
        """
        if cls.recording():
            cls.active_context().append(obj, **kwargs)

    @classmethod
    def remove(cls, obj):
        """Remove an object from the queue(s) if it is in the queue(s).

        Args:
            obj: the object to be removed
        """
        if cls.recording():
            cls.active_context().remove(obj)

    @classmethod
    def update_info(cls, obj, **kwargs):
        """Updates information of an object in the active queue.

        Args:
            obj: the object with metadata to be updated
        """
        if cls.recording():
            cls.active_context().update_info(obj, **kwargs)

    # pylint: disable=protected-access
    @classmethod
    def safe_update_info(cls, obj, **kwargs):
        """Updates information of an object in the active queue if it is already in the queue.

        Args:
            obj: the object with metadata to be updated
        """
        if cls.recording():
            cls.active_context().safe_update_info(obj, **kwargs)

    @classmethod
    def get_info(cls, obj):
        """Retrieves information of an object in the active queue.

        Args:
            obj: the object with metadata to be retrieved

        Returns:
            object metadata
        """
        return cls.active_context().get_info(obj) if cls.recording() else None


class AnnotatedQueue:
    """Lightweight class that maintains a basic queue of operations, in addition
    to metadata annotations."""

    def __init__(self):
        self._queue = OrderedDict()

    def __enter__(self):
        """Adds this instance to the global list of active contexts.

        Returns:
            AnnotatedQueue: this instance
        """
        QueuingManager.add_active_queue(self)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Remove this instance from the global list of active contexts."""
        QueuingManager.remove_active_queue()

    def append(self, obj, **kwargs):
        """Append ``obj`` into the queue with ``kwargs`` metadata."""
        self._queue[obj] = kwargs

    def remove(self, obj):
        """Remove ``obj`` from the queue.  Raises ``KeyError`` if ``obj`` is not already in the queue."""
        del self._queue[obj]

    def safe_update_info(self, obj, **kwargs):
        """Update ``obj``'s metadata with ``kwargs`` if it exists in the queue."""
        if obj in self._queue:
            self._queue[obj].update(kwargs)

    def update_info(self, obj, **kwargs):
        """Update ``obj``'s metadata with ``kwargs``.
        Raises a ``QueuingError`` if it doesn't exist in the queue."""
        if obj not in self._queue:
            raise QueuingError(f"Object {obj} not in the queue.")

        self._queue[obj].update(kwargs)

    def get_info(self, obj):
        """Retrieve the metadata for ``obj``.  Raises a ``QueuingError`` if obj is not in the queue."""
        if obj not in self._queue:
            raise QueuingError(f"Object {obj} not in the queue.")

        return self._queue[obj]

    @property
    def queue(self):
        """Returns a list of objects in the annotated queue"""
        return list(self._queue.keys())


def apply(op, context=QueuingManager):
    """Apply an instantiated operator or measurement to a queuing context.

    Args:
        op (.Operator or .MeasurementProcess): the operator or measurement to apply/queue
        context (.QueuingManager): The queuing context to queue the operator to.
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

    .. details::
        :title: Usage Details

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
         0: ──RY(0.6)──╭●──╭┤ ⟨Z ⊗ Y⟩
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
    if not QueuingManager.recording():
        raise RuntimeError("No queuing context available to append operation to.")

    if op in getattr(context, "queue", QueuingManager.active_context().queue):
        # Queuing contexts can only contain unique objects.
        # If the object to be queued already exists, copy it.
        op = copy.copy(op)

    if hasattr(op, "queue"):
        # operator provides its own logic for queuing
        op.queue(context=context)
    else:
        # append the operator directly to the relevant queuing context
        context.append(op)

    return op

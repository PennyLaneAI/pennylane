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
r"""
This module contains the classes for placing objects into queues.

Description
-----------

Users provide *quantum functions* which PennyLane needs to convert into a circuit representation capable
of being executed by a device. A quantum function is any callable that:

* accepts classical inputs
* constructs any number of quantum :class:`~.Operator` objects
* returns one or more :class:`~.MeasurementProcess` objects.

For example:

.. code-block:: python

    def qfunc(x, scale_value=1):
        qml.RX(x * scale_value, wires=0)
        if (1 != 2):
            qml.S(0)
        return qml.expval(qml.Z(0)), qml.expval(qml.X(1))

To convert from a quantum function to a representation of a circuit, we use queuing.

A *queuable object* is anything that can be placed into a queue. These will be :class:`~.Operator`,
:class:`~.MeasurementProcess`, and :class:`~.QuantumTape` objects. :class:`~.Operator` and
:class:`~.MeasurementProcess` objects achieve queuing via a :meth:`~.Operator.queue` method called upon construction.
Note that even though :class:`~.QuantumTape` is a queuable object, it does not have a ``queue`` method.

When an object is queued, it sends itself to the :class:`~.QueuingManager`. The :class:`~.QueuingManager`
is a global singleton class that facilitates placing objects in the queue. All of :class:`~.QueuingManager`'s methods
and properties are class methods and properties, so all instances will access the same information.

The :meth:`~.QueuingManager.active_context` is the queue where any new objects are placed.
The :class:`~.QueuingManager` is said to be *recording* if an active context exists.

Active contexts are :class:`~.AnnotatedQueue` instances. They are *context managers* where recording occurs
within a ``with`` block.

Let's take a look at an example. If we query the :class:`~.QueuingManager` outside of an
:class:`~.AnnotatedQueue`'s context, we can see that nothing is recording and no active context exists.

>>> print("Are we recording? ", qml.QueuingManager.recording())
Are we recording?  False
>>> print("What's the active context? ", qml.QueuingManager.active_context())
What's the active context?  None

Inside of a context, we can see the active recording context:

>>> with qml.queuing.AnnotatedQueue() as q:
...     print("Are we recording? ", qml.QueuingManager.recording())
...     print("Is q the active queue? ", q is qml.QueuingManager.active_context())
Are we recording?  True
Is q the active queue?  True

If we have nested :class:`~.AnnotatedQueue` contexts, only the innermost one will be recording.
Once the currently active queue exits, any outer queue will resume recording.

>>> with qml.queuing.AnnotatedQueue() as q1:
...     print("Is q1 recording? ", q1 is qml.QueuingManager.active_context())
...     with qml.queuing.AnnotatedQueue() as q2:
...         print("Is q1 recording? ", q1 is qml.QueuingManager.active_context())
...     print("Is q1 recording? ", q1 is qml.QueuingManager.active_context())
Is q1 recording?  True
Is q1 recording?  False
Is q1 recording?  True

If we construct an operator inside the recording context, we can see it is added to the queue:

>>> with qml.queuing.AnnotatedQueue() as q:
...     op = qml.X(0)
>>> q.queue
[X(0)]

If an operator is constructed outside of the context, we can manually add it to the queue by
calling the :meth:`~.Operator.queue` method. The :meth:`~.Operator.queue` method is automatically
called upon initialization, but it can also be manually called at a later time.

>>> op = qml.X(0)
>>> with qml.queuing.AnnotatedQueue() as q:
...     op.queue()
>>> q.queue
[X(0)]

An object can only exist up to *once* in the queue, so calling queue multiple times will
not do anything.

>>> op = qml.X(0)
>>> with qml.queuing.AnnotatedQueue() as q:
...     op.queue()
...     op.queue()
>>> q.queue
[X(0)]

The :func:`~.apply` method allows a single object to be queued multiple times in a circuit.
The function queues a copy of the original object if it already in the queue.

>>> op = qml.X(0)
>>> with qml.queuing.AnnotatedQueue() as q:
...     qml.apply(op)
...     qml.apply(op)
>>> q.queue
[X(0), X(0)]
>>> q.queue[0] is q.queue[1]
False

In the case of operators composed of other operators, like with :class:`~.SymbolicOp` and
:class:`~.CompositeOp`, the new nested operation removes its constituents from the queue.
Only the operators that will end up in the circuit will remain.

>>> with qml.queuing.AnnotatedQueue() as q:
...     base = qml.X(0)
...     print(q.queue)
...     pow_op = base ** 1.5
...     print(q.queue)
[X(0)]
[X(0)**1.5]

Once the queue is constructed, the :func:`~.process_queue` function converts it into the operations
and measurements in the final circuit. This step eliminates any object that has an owner.

>>> with qml.queuing.AnnotatedQueue() as q:
...     qml.StatePrep(np.array([1.0, 0]), wires=0)
...     base = qml.X(0)
...     pow_op = base ** 1.5
...     qml.expval(qml.Z(0) @ qml.X(1))
>>> ops, measurements = qml.queuing.process_queue(q)
>>> ops
[StatePrep(tensor([1., 0.], requires_grad=True), wires=[0]), X(0)**1.5]
>>> measurements
[expval(Z(0) @ X(1))]

These lists can be used to construct a :class:`~.QuantumScript`:

>>> qml.tape.QuantumScript(ops, measurements)
<QuantumScript: wires=[0, 1], params=1>

In order to construct new operators within a recording, but without queuing them
use the :meth:`~.queuing.QueuingManager.stop_recording` context upon construction:

>>> with qml.queuing.AnnotatedQueue() as q:
...     with qml.QueuingManager.stop_recording():
...         qml.Y(1)
>>> q.queue
[]

"""

import copy
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Optional


class QueuingError(Exception):
    """Exception that is raised when there is a queuing error"""


class WrappedObj:
    """Wraps an object to make its hash dependent on its identity"""

    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        if not isinstance(other, WrappedObj):
            return False
        return id(self.obj) == id(other.obj)

    def __repr__(self):
        return f"Wrapped({self.obj.__repr__()})"


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

    * ``update_info``: Update an object's metadata if it is already queued.

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
    def active_context(cls) -> Optional["AnnotatedQueue"]:
        """Returns the currently active queuing context."""
        return cls._active_contexts[-1] if cls.recording() else None

    @classmethod
    @contextmanager
    def stop_recording(cls):
        """A context manager and decorator to ensure that contained logic is non-recordable
        or non-queueable within a QNode or quantum tape context.

        **Example:**

        Consider the function:

        >>> def list_of_ops(params, wires):
        ...     return [
        ...         qml.RX(params[0], wires=wires),
        ...         qml.RY(params[1], wires=wires),
        ...         qml.RZ(params[2], wires=wires)
        ...     ]

        If executed in a recording context, the operations constructed in the function will be queued:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     ops = list_of_ops(params, wires=0)
        ...     qml.apply(ops[-1])  # apply the last operation from the list again
        ...     return qml.expval(qml.Z(0))
        >>> print(qml.draw(circuit)([1, 2, 3]))
        0: ──RX(1.00)──RY(2.00)──RZ(3.00)──RZ(3.00)─┤  <Z>

        Using the ``stop_recording`` context manager, all logic contained inside is not queued or recorded.

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     with qml.QueuingManager.stop_recording():
        ...         ops = list_of_ops(params, wires=0)
        ...     qml.apply(ops[-1])
        ...     return qml.expval(qml.Z(0))
        >>> print(qml.draw(circuit)([1, 2, 3]))
        0: ──RZ(3.00)─┤  <Z>

        The context manager can also be used as a decorator on a function:

        >>> @qml.QueuingManager.stop_recording()
        ... def list_of_ops(params, wires):
        ...     return [
        ...         qml.RX(params[0], wires=wires),
        ...         qml.RY(params[1], wires=wires),
        ...         qml.RZ(params[2], wires=wires)
        ...     ]
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     ops = list_of_ops(params, wires=0)
        ...     qml.apply(ops[-1])
        ...     return qml.expval(qml.Z(0))
        >>> print(qml.draw(circuit)([1, 2, 3]))
        0: ──RZ(3.00)─┤  <Z>

        """
        previously_active_contexts = cls._active_contexts
        cls._active_contexts = []
        try:
            yield
        finally:
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
        """Updates information of an object in the active queue if it is already in the queue.

        Args:
            obj: the object with metadata to be updated
        """
        if cls.recording():
            cls.active_context().update_info(obj, **kwargs)

    @classmethod
    def get_info(cls, obj):
        """Retrieves information of an object in the active queue.

        Args:
            obj: the object with metadata to be retrieved

        Returns:
            object metadata
        """
        return cls.active_context().get_info(obj) if cls.recording() else None


class AnnotatedQueue(OrderedDict):
    """Lightweight class that maintains a basic queue of operations, in addition
    to metadata annotations."""

    _lock = RLock()
    """threading.RLock: Used to synchronize appending to/popping from global QueueingContext."""

    def __enter__(self):
        """Adds this instance to the global list of active contexts.

        Returns:
            AnnotatedQueue: this instance
        """
        AnnotatedQueue._lock.acquire()
        QueuingManager.add_active_queue(self)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Remove this instance from the global list of active contexts."""
        QueuingManager.remove_active_queue()
        AnnotatedQueue._lock.release()

    def append(self, obj, **kwargs):
        """Append ``obj`` into the queue with ``kwargs`` metadata."""
        obj = obj if isinstance(obj, WrappedObj) else WrappedObj(obj)
        self[obj] = kwargs

    def remove(self, obj):
        """Remove ``obj`` from the queue. Passes silently if the object is not in the queue."""
        obj = obj if isinstance(obj, WrappedObj) else WrappedObj(obj)
        if obj in self:
            del self[obj]

    def update_info(self, obj, **kwargs):
        """Update ``obj``'s metadata with ``kwargs`` if it exists in the queue."""
        obj = obj if isinstance(obj, WrappedObj) else WrappedObj(obj)
        if obj in self:
            self[obj].update(kwargs)

    def get_info(self, obj):
        """Retrieve the metadata for ``obj``.  Raises a ``QueuingError`` if obj is not in the queue."""
        obj = obj if isinstance(obj, WrappedObj) else WrappedObj(obj)
        if obj not in self:
            raise QueuingError(f"Object {obj.obj} not in the queue.")

        return self[obj]

    def items(self):
        return tuple((key.obj, value) for key, value in super().items())

    @property
    def queue(self):
        """Returns a list of objects in the annotated queue"""
        return list(key.obj for key in self.keys())

    def __setitem__(self, key, value):
        key = key if isinstance(key, WrappedObj) else WrappedObj(key)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        key = key if isinstance(key, WrappedObj) else WrappedObj(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        key = key if isinstance(key, WrappedObj) else WrappedObj(key)
        return super().__contains__(key)


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
            return qml.expval(qml.Z(0))

    >>> print(qml.draw(circuit)(0.6))
    0: ──RY(0.6)──RX(0.4)──┤ ⟨Z⟩

    It can also be used to apply functions repeatedly:

    .. code-block:: python

        @qml.qnode(dev)
        def circuit(x):
            qml.apply(op)
            qml.RY(x, wires=0)
            qml.apply(op)
            return qml.expval(qml.Z(0))

    >>> print(qml.draw(circuit)(0.6))
    0: ──RX(0.4)──RY(0.6)──RX(0.4)──┤ ⟨Z⟩

    .. warning::

        If you use ``apply`` on an operator that has already been queued, it will
        be queued for a second time. For example:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit():
                op = qml.Hadamard(0)
                qml.apply(op)
                return qml.expval(qml.Z(0))

        >>> print(qml.draw(circuit)())
        0: ──H──H─┤  <Z>

    .. details::
        :title: Usage Details

        Instantiated measurements can also be applied to queuing contexts
        using ``apply``:

        .. code-block:: python

            meas = qml.expval(qml.Z(0) @ qml.Y(1))
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
                    op1 = qml.X(0)

                    # We can use qml.apply to apply the same operation to tape1
                    # without leaving the tape2 context.
                    qml.apply(op1, context=tape1)

                    qml.RZ(0.2, wires=0)

                qml.CNOT(wires=[0, 1])

        >>> tape1.operations
        [Hadamard(wires=[1]), <QuantumTape: wires=[0], params=1>, X(0), CNOT(wires=[0, 1])]
        >>> tape2.operations
        [X(0), RZ(0.2, wires=[0])]
    """
    if not QueuingManager.recording():
        raise RuntimeError("No queuing context available to append operation to.")

    # pylint: disable=unsupported-membership-test
    if op in getattr(context, "queue", QueuingManager.active_context()):
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


# pylint: disable=protected-access
def process_queue(queue: AnnotatedQueue):
    """Process the annotated queue, creating a list of quantum
    operations and measurement processes.

    Args:
        queue (.AnnotatedQueue): The queue to be processed into individual lists

    Returns:
        tuple[list(.Operation), list(.MeasurementProcess)]:
        The list of tape operations, the list of tape measurements
    """
    lists = {"_ops": [], "_measurements": []}
    list_order = {"_ops": 1, "_measurements": 2}
    current_list = "_ops"

    for obj, info in queue.items():
        if "owner" not in info and getattr(obj, "_queue_category", None) is not None:
            if list_order[obj._queue_category] > list_order[current_list]:
                current_list = obj._queue_category
            elif list_order[obj._queue_category] < list_order[current_list]:
                raise ValueError(
                    f"{obj._queue_category[1:]} operation {obj} must occur prior "
                    f"to {current_list[1:]}. Please place earlier in the queue."
                )
            lists[obj._queue_category].append(obj)

    return lists["_ops"], lists["_measurements"]

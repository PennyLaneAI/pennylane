# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`QueuingContext` abstract base class.
"""
import abc
from collections import OrderedDict, deque

import pennylane as qml


class QueuingContext(abc.ABC):
    """Abstract base class for classes that exposes a queue for objects.

    In PennyLane, the construction of quantum gates is separated from the specific
    quantum node (:class:`~.BaseQNode`) that they belong to. However, including
    logic for this when creating an instance of :class:`~.Operator` does not align
    with the current architecture. Therefore, there is a need to use a high level
    object that holds information about the relationship between quantum gates and
    a quantum node.

    The :class:`~.QueuingContext` class realizes this by providing access to
    the current QNode.  Furthermore, it provides the flexibility to have
    multiple objects record the creation of quantum gates.

    The ``QueuingContext`` class both acts as the abstract base class for all
    classes that expose a queue for Operations (so-called contexts), as well as the
    interface to said queues. The active contexts contain maximally one QNode and
    an arbitrary number of other contexts like the :class:`~.OperationRecorder`.
    """

    # TODO: update docstring

    _active_contexts = deque()
    """The stack of contexts that are currently active."""

    def __enter__(self):
        """Adds this instance to the global list of active contexts.

        Returns:
            QueuingContext: This instance
        """
        QueuingContext._active_contexts.append(self)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Remove this instance from the global list of active contexts."""
        QueuingContext._active_contexts.remove(self)

    @abc.abstractmethod
    def _append(self, obj, **kwargs):
        """Append an object to this QueuingContext instance.

        Args:
            obj: The object to be appended
        """

    @classmethod
    def active_context(cls):
        """Returns the currently active queuing context."""
        if cls._active_contexts:
            return cls._active_contexts[-1]

        return None

    @classmethod
    def append(cls, obj, **kwargs):
        """Append an object to the queue(s).

        Args:
            obj: the object to be appended
        """
        # TODO: this method should append only to `cls.active_context`, *not*
        # all active contexts. However this will require a refactor in
        # the template decorator and the operation recorder.
        for context in cls._active_contexts:
            context._append(obj, **kwargs)  # pylint: disable=protected-access

    @abc.abstractmethod
    def _remove(self, obj):
        """Remove an object from this QueuingContext instance.

        Args:
            obj: the object to be removed
        """

    @classmethod
    def remove(cls, obj):
        """Remove an object from the queue(s) if it is in the queue(s).

        Args:
            obj: the object to be removed
        """
        # TODO: this method should remove only from `cls.active_context`, *not*
        # all active contexts. However this will require a refactor in
        # the template decorator and the operation recorder.
        for context in cls._active_contexts:
            # We use the duck-typing approach to assume that the underlying remove
            # behaves like `list.remove(obj)` or `del dict[key]` and throws a
            # ValueError or KeyError if the operator is not present
            try:
                context._remove(obj)  # pylint: disable=protected-access
            except (ValueError, KeyError):
                pass

    @classmethod
    def update_info(cls, obj, **kwargs):
        """Updates information of an object in the active queue."""
        cls.active_context()._update_info(obj, **kwargs)  # pylint: disable=protected-access

    def _update_info(self, obj, **kwargs):
        """Updates information of an object in the queue instance."""
        raise NotImplementedError

    @classmethod
    def get_info(cls, obj):
        """Returns information of an object in the queue."""
        return cls.active_context().get_info(obj)


class Queue(QueuingContext):
    """Lightweight class that maintains a basic queue of operations and pre/post-processing steps
    for representing quantum circuits."""

    def __init__(self):
        self.queue = []

    def _append(self, obj, **kwargs):
        self.queue.append(obj)

    def _remove(self, obj):
        self.queue.remove(obj)


class AnnotatedQueue(QueuingContext):
    """Lightweight class that maintains a basic queue of operations, in addition
    to annotations."""

    def __init__(self):
        self._queue = OrderedDict()

    def _append(self, obj, **kwargs):
        self._queue[obj] = kwargs

    def _remove(self, obj):
        del self._queue[obj]

    def _update_info(self, obj, **kwargs):
        """Updates the annotated information of an object in the queue.

        Args:
            obj: the object to update
            kwargs: Keyword arguments and values to add to the annotation.
                If a particular keyword already exists in the annotation,
                the value is updated.
        """
        if obj not in self._queue:
            raise ValueError(f"Object {obj} not in the queue.")

        self._queue[obj].update(kwargs)

    def get_info(self, obj):
        """Returns the annotated information of an object in the queue.

        Args:
            obj: the object to query

        Returns:
            dict: the annotated information
        """
        if obj not in self._queue:
            raise ValueError(f"Object {obj} not in the queue.")

        return self._queue[obj]

    @property
    def queue(self):
        """Returns a list of objects in the annotated queue"""
        return list(self._queue.keys())


class OperationRecorder(Queue):
    """A template and quantum function inspector,
    allowing easy introspection of operators that have been
    applied without requiring a QNode.

    **Example**:

    The OperationRecorder is a context manager. Executing templates
    or quantum functions stores resulting applied operators in the
    recorder, which can then be printed.

    >>> weights = qml.init.strong_ent_layers_normal(n_layers=1, n_wires=2)
    >>>
    >>> with qml.utils.OperationRecorder() as rec:
    >>>    qml.templates.layers.StronglyEntanglingLayers(*weights, wires=[0, 1])
    >>>
    >>> print(rec)
    Operations
    ==========
    Rot(-0.10832656163640327, 0.14429091013664083, -0.010835826725765343, wires=[0])
    Rot(-0.11254523669444501, 0.0947222564914006, -0.09139600968423377, wires=[1])
    CNOT(wires=[0, 1])
    CNOT(wires=[1, 0])

    Alternatively, the :attr:`~.OperationRecorder.queue` attribute can be used
    to directly access the applied :class:`~.Operation` and :class:`~.Observable`
    objects.

    Attributes:
        queue (List[Operator]): list of operators applied within
            the OperatorRecorder context, includes operations and observables
        operations (List[Operation]): list of operations applied within
            the OperatorRecorder context
        observables (List[Observable]): list of observables applied within
            the OperatorRecorder context
    """

    def __init__(self):
        super().__init__()
        self.operations = None
        self.observables = None

    def __exit__(self, exception_type, exception_value, traceback):
        super().__exit__(exception_type, exception_value, traceback)

        # Remove duplicates that might have arisen from measurements
        self.queue = list(OrderedDict.fromkeys(self.queue))
        self.operations = list(
            filter(
                lambda op: not (
                    isinstance(op, qml.operation.Observable) and not op.return_type is None
                ),
                self.queue,
            )
        )
        self.observables = list(
            filter(
                lambda op: isinstance(op, qml.operation.Observable) and not op.return_type is None,
                self.queue,
            )
        )

    def __str__(self):
        output = ""
        output += "Operations\n"
        output += "==========\n"
        for op in self.operations:
            output += repr(op) + "\n"

        output += "\n"
        output += "Observables\n"
        output += "==========\n"
        for op in self.observables:
            output += repr(op) + "\n"

        return output

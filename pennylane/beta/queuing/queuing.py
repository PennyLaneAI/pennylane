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


class QueuingContext(abc.ABC):
    """Abstract base class for classes that exposes a queue for objects.

    This class provides a context manager that tracks queuable objects and queuing functions.
    Queuable objects are objects that queue themselves via ``QueuingContext.append(self)``, while
    queuing functions queue external objects.

    Queuable objects make use of the following ``QueuingContext`` methods and properties:

    * :attr:`~.recording`: determine whether a queuing context is actively recording operations
    * :meth:`~.append`: append an object to the active queuing context
    * :meth:`~.remove`: remove an object from the queuing context. If the object is queued multiple
      times, only the first occurrence is removed.
    * :meth:`~.update_info`: updates metadata attached to an object in the queue (not supported by
      all queuing contexts).
    * :meth:`~.get_info`: retrieves metadata attached to an object in the queue (not supported by
      all queuing contexts).

    Queuing context subclasses must define the following abstract methods:

    * :meth:`~._append`: define an action to perform when an object append
      request is made.

    * :meth:`~._remove`: define an action to perform when an object removal request is made.

    In addition, the optional methods :meth:`~._update_info` and ``get_info`` may also be
    defined as required.

    **Example**

    To create a queuing context, simply subclass from ``QueuingContext`` and define
    the required methods:

    >>> class MyQueue(QueuingContext):
    ...     def __init__(self):
    ...         self.queue = []
    ...     def _append(self, obj):
    ...         self.queue.append(obj)
    ...     def _remove(self, obj):
    ...         self.queue.remove(obj)

    Once defined, it can be used as a queuing context to track queuable objects
    and queuing functions:

    >>> with MyQueue() as q:
    ...     QueuingContext.append("object")
    >>> print(q.queue)
    ['object']

    Note that ``QueuingContext`` subclasses support nesting; objects are only queued to the
    first surrounding queuing context:

    >>> with MyQueue() as q1:
    ...     with MyQueue() as q2:
    ...         QueuingContext.append("first object")
    ...     QueuingContext.append("second object")
    ...
    >>> print(q1.queue)
    ['second object']
    >>> print(q2.queue)
    ['first object']

    Finally, queuing contexts *themselves* can be queuable objects:

    >>> class QueuableQueue(QueuingContext):
    ...     def __init__(self):
    ...         self.queue = []
    ...         QueuingContext.append(self)
    ...     def _append(self, obj):
    ...         self.queue.append(obj)
    ...     def _remove(self, obj):
    ...         self.queue.remove(obj)

    We can see that nested ``QueuableQueue`` objects are queued to their surrounding queuing
    context:

    >>> with QueuableQueue() as q1:
    ...     with QueuableQueue() as q2:
    ...         QueuingContext.append("first object")
    ...     QueuingContext.append("second object")
    >>> print(q1.queue)
    [<__main__.QueuableQueue object at 0x7f94c432b6d0>, 'second object']
    >>> print(q1.queue[0].queue)
    ['first object']
    """

    _active_contexts = deque()
    """The stack of contexts that are currently active."""

    def __enter__(self):
        """Adds this instance to the global list of active contexts.

        Returns:
            QueuingContext: this instance
        """
        QueuingContext._active_contexts.append(self)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Remove this instance from the global list of active contexts."""
        QueuingContext._active_contexts.pop()

    @abc.abstractmethod
    def _append(self, obj, **kwargs):
        """Append an object to this QueuingContext instance.

        Args:
            obj: The object to be appended
        """

    @classmethod
    def recording(cls):
        """Whether a queuing context is active and recording operations"""
        return bool(cls._active_contexts)

    @classmethod
    def active_context(cls):
        """Returns the currently active queuing context."""
        if cls.recording():
            return cls._active_contexts[-1]

        return None

    @classmethod
    def append(cls, obj, **kwargs):
        """Append an object to the queue(s).

        Args:
            obj: the object to be appended
        """
        if cls.recording():
            cls.active_context()._append(obj, **kwargs)  # pylint: disable=protected-access

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
        if cls.recording():
            cls.active_context()._remove(obj)  # pylint: disable=protected-access

    @classmethod
    def update_info(cls, obj, **kwargs):
        """Updates information of an object in the active queue.

        Args:
            obj: the object with metadata to be updated
        """
        if cls.recording():
            cls.active_context()._update_info(obj, **kwargs)  # pylint: disable=protected-access

    def _update_info(self, obj, **kwargs):
        """Updates information of an object in the queue instance."""
        raise NotImplementedError

    @classmethod
    def get_info(cls, obj):
        """Retrieves information of an object in the active queue.

        Args:
            obj: the object with metadata to be retrieved

        Returns:
            object metadata
        """
        if cls.recording():
            return cls.active_context()._get_info(obj)  # pylint: disable=protected-access

        return None

    def _get_info(self, obj):
        """Retrieves information of an object in the queue instance."""
        raise NotImplementedError


class Queue(QueuingContext):
    """Lightweight class that maintains a basic queue of objects."""

    def __init__(self):
        self.queue = []

    def _append(self, obj, **kwargs):
        self.queue.append(obj)

    def _remove(self, obj):
        self.queue.remove(obj)


class AnnotatedQueue(QueuingContext):
    """Lightweight class that maintains a basic queue of operations, in addition
    to metadata annotations."""

    def __init__(self):
        self._queue = OrderedDict()

    def _append(self, obj, **kwargs):
        self._queue[obj] = kwargs

    def _remove(self, obj):
        del self._queue[obj]

    def _update_info(self, obj, **kwargs):
        if obj not in self._queue:
            raise ValueError(f"Object {obj} not in the queue.")

        self._queue[obj].update(kwargs)

    def _get_info(self, obj):
        if obj not in self._queue:
            raise ValueError(f"Object {obj} not in the queue.")

        return self._queue[obj]

    @property
    def queue(self):
        """Returns a list of objects in the annotated queue"""
        return list(self._queue.keys())

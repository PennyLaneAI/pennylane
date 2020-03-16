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


class QueuingContext(abc.ABC):
    """Abstract base class for classes that exposes a queue for Operations.

    In PennyLane, the construction of quantum gates is separated from the
    specific quantum node (:class:`BaseQNode`) that they belong to. However,
    including logic for this when creating an instance of :class:`Operator`
    does not align with the current architecture. Therefore, there is a need to
    use a high level object that holds information about the relationship
    between quantum gates and a quantum node.

    The ``QueuingContext`` class realizes this by providing access to the
    current QNode.  Furthermore, it provides the flexibility to have multiple
    objects record the creation of quantum gates.

    The QueuingContext class both acts as the abstract base class for all
    classes that expose a queue for Operations (so-called contexts), as well
    as the interface to said queues. The active contexts contain maximally one QNode
    and an arbitrary number of other contexts like the OperationRecorder.
    """

    _active_contexts = []
    """The list of contexts that are currently active."""

    def __enter__(self):
        """Adds this instance to the global list of active contexts.

        Returns:
            QueuingContext: This instance
        """
        QueuingContext._active_contexts.append(self)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Remove this instance from the global list of active contexts.
        """
        QueuingContext._active_contexts.remove(self)

    @abc.abstractmethod
    def _append_operator(self, operator):
        """Append an operator to this QueuingContext instance.

        Args:
            operator (Operator): The Operator instance to be appended
        """

    @classmethod
    def append_operator(cls, operator):
        """Append an operator to the global queue(s).

        Args:
            operator (Operator): The Operator instance to be appended
        """
        for context in cls._active_contexts:
            context._append_operator(operator)  # pylint: disable=protected-access

    @abc.abstractmethod
    def _remove_operator(self, operator):
        """Remove an operator from this QueuingContext instance.

        Args:
            operator (Operator): The Operator instance to be removed
        """

    @classmethod
    def remove_operator(cls, operator):
        """Remove an operator from the global queue(s) if it is in the queue(s).

        Args:
            operator (Operator): The Operator instance to be removed
        """
        for context in cls._active_contexts:
            # We use the duck-typing approach to assume that the underlying remove
            # behaves like list.remove and throws a ValueError if the operator
            # is not in the list
            try:
                context._remove_operator(operator)  # pylint: disable=protected-access
            except ValueError:
                pass

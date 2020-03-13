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
This module contains the :class:`Device` abstract base class.
"""
import abc


class QueuingContext(abc.ABC):
    """Abstract base class for classes that exposes a queue for Operations.
    """

    _active_contexts = []
    """The list of contexts that are currently active."""

    def __enter__(self):
        QueuingContext._active_contexts.append(self)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
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
            # behaves like list.remove and throws an ValueError if the operator
            # is not in the list
            try:
                context._remove_operator(operator)  # pylint: disable=protected-access
            except ValueError:
                pass

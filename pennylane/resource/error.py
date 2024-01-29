# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Stores classes and logic to define and track algorithmic error in a quantum workflow.
"""
from abc import abstractmethod

from pennylane.operation import Operation


class AlgorithmicError:
    """Abstract base class representing different types of mathematical errors."""

    def __init__(self, error: float):
        self.error = error

    @abstractmethod
    def combine(self, other):
        """A method to combine two errors of the same type.
        (eg. additive, square additive, multiplicative, etc.)

        Args:
            other (AlgorithmicError): The other instance of error being combined.

        Returns:
            AlgorithmicError: The total error after combination.
        """
        raise NotImplementedError

    @staticmethod
    def get_error(approximate_op, exact_op, **kwargs):
        """A method to allow users to compute this type of error
        between two operators.

        Args:
            approximate_op (.Operator): The approximate operator.
            exact_op (.Operator): The exact operator.

        Returns:
            float: The error between the exact operator and its
            approximation.
        """
        raise NotImplementedError


class ErrorOperation(Operation):
    r"""Base class that represents quantum gates or channels applied to quantum
    states and stores the error of the quantum gate.

    .. note::
        Child classes must implement the :func:`~.ErrorOperation.error` method which computes
        the error of the operation.
    """

    @abstractmethod
    def error(self) -> AlgorithmicError:
        """Computes the error of the operation.

        Returns:
            AlgorithmicError: The error.
        """
        raise NotImplementedError

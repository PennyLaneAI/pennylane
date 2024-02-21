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
from abc import ABC, abstractmethod

from pennylane.operation import Operation


class AlgorithmicError(ABC):
    """Abstract base class representing an abstract type of error.
    This class can be used to create objects that track and propagate errors introduced by approximations and other algorithmic inaccuracies.

    Args:
        error (float): The numerical value of the error

    .. note::
        Child classes must implement the :func:`~.AlgorithmicError.combine` method which combines two
        instances of this error type (as if the associated gates were applied in series).
    """

    def __init__(self, error: float):
        self.error = error

    @abstractmethod
    def combine(self, other):
        """A method to combine two errors of the same type.
        (e.g., additive, square additive, multiplicative, etc.)

        Args:
            other (AlgorithmicError): The other instance of error being combined.

        Returns:
            AlgorithmicError: The total error after combination.
        """

    @staticmethod
    def get_error(approximate_op, exact_op, **kwargs):
        """A method to allow users to compute this type of error between two operators.

        Args:
            approximate_op (.Operator): The approximate operator.
            exact_op (.Operator): The exact operator.

        Returns:
            float: The error between the exact operator and its
            approximation.
        """
        raise NotImplementedError


class ErrorOperation(Operation):
    r"""Base class that represents quantum operations which carry some form of algorithmic error.

    .. note::
        Child classes must implement the :func:`~.ErrorOperation.error` method which computes
        the error of the operation.
    """

    @property
    @abstractmethod
    def error(self) -> AlgorithmicError:
        """Computes the error of the operation.

        Returns:
            AlgorithmicError: The error.
        """

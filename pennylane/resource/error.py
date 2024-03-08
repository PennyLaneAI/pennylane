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

import pennylane as qml
from pennylane.operation import Operation, Operator


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
    def get_error(approximate_op, exact_op):
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


class SpectralNormError(AlgorithmicError):
    """Class representing the spectral norm error.

    The spectral norm error is defined as the distance, in spectral norm, between the true unitary
    we intend to apply and the approximate unitary that is actually applied.

    Args:
        error (float): The numerical value of the error

    **Example**

    >>> s1 = SpectralNormError(0.01)
    >>> s2 = SpectralNormError(0.02)
    >>> s1.combine(s2)
    <SpectralNormError(0.03)>
    """

    def __repr__(self):
        """Return formal string representation."""

        return f"SpectralNormError({self.error})"

    def combine(self, other: "SpectralNormError"):
        """Combine two spectral norm errors.

        Args:
            other (SpectralNormError): The other instance of error being combined.

        Returns:
            SpectralNormError: The total error after combination.

        **Example**

        >>> s1 = SpectralNormError(0.01)
        >>> s2 = SpectralNormError(0.02)
        >>> s1.combine(s2)
        SpectralNormError(0.03)
        """
        return self.__class__(self.error + other.error)

    @staticmethod
    def get_error(approximate_op: Operator, exact_op: Operator):
        """Compute spectral norm error between two operators.

        Args:
            approximate_op (Operator): The approximate operator.
            exact_op (Operator): The exact operator.

        Returns:
            float: The error between the exact operator and its
            approximation.

        **Example**

        >>> Op1 = qml.RY(0.40, 0)
        >>> Op2 = qml.RY(0.41, 0)
        >>> SpectralNormError.get_error(Op1, Op2)
        0.004999994791668309
        """
        wire_order = exact_op.wires
        m1 = qml.matrix(exact_op, wire_order=wire_order)
        m2 = qml.matrix(approximate_op, wire_order=wire_order)
        return qml.math.max(qml.math.svd(m1 - m2, compute_uv=False))

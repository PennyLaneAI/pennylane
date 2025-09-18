# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This module contains all the custom exceptions and warnings used in PennyLane.

.. warning::

    Unless you are a PennyLane or plugin developer, you will likely not need
    to use these classes directly. They are raised by PennyLane functions
    when errors are encountered.

Contents
--------

The exceptions and warnings are organized by their category of use.

.. currentmodule:: pennylane.exceptions

General Execution Errors
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~AllocationError
    ~CaptureError
    ~DeviceError
    ~QuantumFunctionError
    ~TransformError
    ~ConditionalTransformError
    ~QueuingError
    ~WireError
    ~MeasurementShapeError
    ~AutoGraphError
    ~CompileError
    ~DecompositionError
    ~InvalidCapabilitiesError
    ~NonDifferentiableError

Operator Property Errors
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~OperatorPropertyUndefined
    ~DecompositionUndefinedError
    ~TermsUndefinedError
    ~MatrixUndefinedError
    ~SparseMatrixUndefinedError
    ~EigvalsUndefinedError
    ~DiagGatesUndefinedError
    ~AdjointUndefinedError
    ~PowUndefinedError
    ~GeneratorUndefinedError
    ~ParameterFrequenciesUndefinedError

User Warnings
~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~PennyLaneDeprecationWarning
    ~ExperimentalWarning
    ~AutoGraphWarning

"""  # pragma: no cover

# =============================================================================
# General Execution and Quantum Function Errors
# =============================================================================


class AllocationError(RuntimeError):
    """An error arising from trying handling a dynamically allocated wire."""


class CaptureError(Exception):
    """Errors related to PennyLane's Program Capture execution pipeline."""


class DeviceError(Exception):
    """Exception raised when it encounters an illegal operation in the quantum circuit."""


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""


class TransformError(Exception):
    """Raised when there is an error with the transform logic."""


class ConditionalTransformError(ValueError):
    """Error for using qml.cond incorrectly"""


class QueuingError(Exception):
    """Exception that is raised when there is a queuing error"""


class WireError(Exception):
    """Exception raised by a :class:`~.pennylane.wires.Wire` object when it is unable to process wires."""


class MeasurementShapeError(ValueError):
    """An error raised when an unsupported operation is attempted with a
    quantum tape."""


# =============================================================================
# Operator Property Errors
# =============================================================================


class OperatorPropertyUndefined(Exception):
    """Generic exception to be used for undefined
    Operator properties or methods."""


class DecompositionUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's representation as a decomposition is undefined."""


class TermsUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's representation as a linear combination is undefined."""


class MatrixUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's matrix representation is undefined."""


class SparseMatrixUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's sparse matrix representation is undefined."""


class EigvalsUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's eigenvalues are undefined."""


class DiagGatesUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's diagonalizing gates are undefined."""


class AdjointUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's adjoint version is undefined."""


class PowUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's power is undefined."""


class GeneratorUndefinedError(OperatorPropertyUndefined):
    """Exception used to indicate that an operator
    does not have a generator"""


class ParameterFrequenciesUndefinedError(OperatorPropertyUndefined):
    """Exception used to indicate that an operator
    does not have parameter_frequencies"""


class ResourcesUndefinedError(Exception):
    r"""Exception to be raised when a ``ResourceOperator`` does not implement resource_decomp"""


# =============================================================================
# Warnings
# =============================================================================


class PennyLaneDeprecationWarning(UserWarning):
    """Warning raised when a PennyLane feature is being deprecated."""


class ExperimentalWarning(UserWarning):
    """Warning raised to indicate experimental/non-stable feature or support."""


class AutoGraphWarning(Warning):
    """Warnings related to PennyLane's AutoGraph submodule."""


# =============================================================================
# Autograph and Compilation Errors
# =============================================================================


class AutoGraphError(Exception):
    """Errors related to PennyLane's AutoGraph submodule."""


class CompileError(Exception):
    """Error encountered in the compilation phase."""


class DecompositionError(Exception):
    """Base class for decomposition errors."""


class InvalidCapabilitiesError(Exception):
    """Exception raised from invalid TOML files."""


class NonDifferentiableError(Exception):
    """Exception raised if attempting to differentiate non-trainable
    :class:`~.tensor` using Autograd."""


_TF_DEPRECATION_MSG = (
    "Support for the TensorFlow interface is deprecated and will be removed in v0.44. "
    "Future versions of PennyLane are not guaranteed to work with TensorFlow. "
    "Please migrate your workflows to JAX or Pytorch to benefit from enhanced support and features."
)

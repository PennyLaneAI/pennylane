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
The former location of pennylane2/core/operator.
"""

import pennylane2 as qp

# pylint: disable-next=wildcard-import,unused-wildcard-import
from pennylane2.core.operator import *  # tach-ignore
from pennylane2.core.operator.base import (  # pylint: disable=unused-import # tach-ignore
    _UNSET_BATCH_SIZE,
)
from pennylane2.exceptions import (  # pylint: disable=unused-import
    AdjointUndefinedError,
    DecompositionUndefinedError,
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    GeneratorUndefinedError,
    MatrixUndefinedError,
    OperatorPropertyUndefined,
    ParameterFrequenciesUndefinedError,
    PowUndefinedError,
    SparseMatrixUndefinedError,
    TermsUndefinedError,
)
from pennylane2.typing import TensorLike
from pennylane2.wires import Wires  # pylint: disable=unused-import


def operation_derivative(operation: Operation) -> TensorLike:
    r"""Calculate the derivative of an operation.

    For an operation :math:`e^{i \hat{H} \phi t}`, this function returns the matrix representation
    in the standard basis of its derivative with respect to :math:`t`, i.e.,

    .. math:: \frac{d \, e^{i \hat{H} \phi t}}{dt} = i \phi \hat{H} e^{i \hat{H} \phi t},

    where :math:`\phi` is a real constant.

    Args:
        operation (.Operation): The operation to be differentiated.

    Returns:
        array: the derivative of the operation as a matrix in the standard basis

    Raises:
        ValueError: if the operation does not have a generator or is not composed of a single
            trainable parameter
    """
    generator = qp.matrix(qp.generator(operation, format="observable"), wire_order=operation.wires)
    return 1j * generator @ operation.matrix()


def __getattr__(name):
    """To facilitate StatePrep rename"""
    if name == "StatePrep":
        return StatePrepBase
    raise AttributeError(f"module 'pennylane2.operation' has no attribute '{name}'")

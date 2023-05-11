# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This module contains the qml.bind_parameters function.
"""

from typing import Optional, Sequence
import copy
from functools import singledispatch

import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.operation import Operator

from ..op_math import CompositeOp, SymbolicOp, ScalarSymbolicOp


def _validate_params(op: Operator, params: Optional[Sequence[TensorLike]]) -> Sequence[TensorLike]:
    """Validate that the list of new parameters has the correct length and shape.

    Args:
        op (.Operator): Operator to update
        params (Optional[Sequence[TensorLike]]): New parameters to create operator with

    Returns:
        Sequence[TensorLike]: Validated parameters

    Raises:
        ValueError: If the new parameters don't have the expected shape.
    """
    if params is None:
        params = []

    if isinstance(op, qml.Hamiltonian):
        invalid_params = len(params) != len(op.coeffs)
    else:
        invalid_params = len(params) != op.num_params

    if invalid_params:
        raise ValueError(
            "The length of the new parameters does not match the expected shape; "
            f"got {len(params)}, expected {op.num_params}."
        )

    return params


@singledispatch
def bind_parameters(op: Operator, params: Optional[Sequence[TensorLike]] = None) -> Operator:
    """Create a new operator with updated parameters

    This function takes an :class:`~.Operator` and new parameters as input and
    returns a new operator of the same type with the new parameters. This function
    does not mutate the original operator.

    Args:
        op (.Operator): Operator to update
        params (Sequence[TensorLike]): New parameters to create operator with

    Returns:
        .Operator: New operator with updated parameters

    Raises:
        ValueError: If the shape of the old and new operator parameters don't match
    """
    params = _validate_params(op, params)

    return op.__class__(*params, wires=op.wires, **copy.deepcopy(op.hyperparameters))


# pylint: disable=missing-docstring
@bind_parameters.register
def bind_parameters_composite_op(op: CompositeOp, params: Optional[Sequence[TensorLike]] = None):
    params = _validate_params(op, params)
    new_operands = []

    for operand in op.operands:
        sub_params = params[: operand.num_params]
        params = params[operand.num_params :]
        new_operands.append(bind_parameters(operand, sub_params))

    return op.__class__(*new_operands)


# pylint: disable=missing-docstring
@bind_parameters.register
def bind_parameters_symbolic_op(op: SymbolicOp, params: Optional[Sequence[TensorLike]] = None):
    params = _validate_params(op, params)
    new_scalar = []

    if isinstance(op, ScalarSymbolicOp):
        new_scalar.append(params[0])
        params = params[1:]

    new_base = bind_parameters(op.base, params)
    return op.__class__(new_base, *new_scalar)


# pylint: disable=missing-docstring
@bind_parameters.register
def bind_parameters_hamiltonian(op: qml.Hamiltonian, params: Optional[Sequence[TensorLike]] = None):
    params = _validate_params(op, params)
    new_observables = copy.deepcopy(op.coeffs)

    return qml.Hamiltonian(params, new_observables)

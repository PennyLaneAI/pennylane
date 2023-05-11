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
This module contains the qml.new_parameters function.
"""

from typing import Union, Sequence
import copy

import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.operation import Operator

from ..op_math import CompositeOp, SymbolicOp, ScalarSymbolicOp


def new_parameters(op: Operator, params: Sequence[Union[float, TensorLike]] = None) -> Operator:
    """Create a new operator with updated parameters

    This function takes an :class:`~.Operator` and new parameters as input and
    returns a new :class:`~.Operator` of the same type with the new parameters.

    Args:
        op (.Operator): Operator to update
        params (Sequence[Union[float, Sequence]]): New parameters to create operator with

    Returns:
        .Operator: New operator with updated parameters

    Raises:
        ValueError: If the shape of the old and new operator parameters don't match
    """
    if params is None:
        params = []

    if len(params) != op.num_params:
        raise ValueError(
            "The length of the new parameters does not match the expected shape; "
            f"got {len(params)}, expected {op.num_params}."
        )

    if isinstance(op, CompositeOp):
        new_operands = []

        for operand in op.operands:
            sub_params = params[:operand.num_params]
            params = params[operand.num_params:]
            new_operands.append(new_parameters(operand, sub_params))

        return op.__class__(*new_operands)

    if isinstance(op, ScalarSymbolicOp):
        new_scalar = params[0]
        params = params[1:]
        new_base = new_parameters(op.base, params)

        return op.__class__(new_base, new_scalar)

    if isinstance(op, SymbolicOp):
        new_base = new_parameters(op.base, params)
        return op.__class__(new_base)

    return op.__class__(*params, wires=op.wires, **copy.deepcopy(op.hyperparameters))

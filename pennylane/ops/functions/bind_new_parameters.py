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
This module contains the qml.bind_new_parameters function.
"""
# pylint: disable=missing-docstring

from typing import Sequence
import copy
from functools import singledispatch

import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.operation import Operator, Tensor

from ..identity import Identity
from ..op_math import CompositeOp, SymbolicOp, ScalarSymbolicOp, Adjoint, Pow, SProd


@singledispatch
def bind_new_parameters(op: Operator, params: Sequence[TensorLike]) -> Operator:
    """Create a new operator with updated parameters

    This function takes an :class:`~.Operator` and new parameters as input and
    returns a new operator of the same type with the new parameters. This function
    does not mutate the original operator.

    Args:
        op (.Operator): Operator to update
        params (Sequence[TensorLike]): New parameters to create operator with

    Returns:
        .Operator: New operator with updated parameters
    """

    return op.__class__(*params, wires=op.wires, **copy.deepcopy(op.hyperparameters))


@bind_new_parameters.register
def bind_new_parameters_identity(op: Identity, params: Sequence[TensorLike]):
    return qml.Identity(*params, wires=op.wires)


@bind_new_parameters.register
def bind_new_parameters_composite_op(op: CompositeOp, params: Sequence[TensorLike]):
    new_operands = []

    for operand in op.operands:
        op_num_params = operand.num_params
        sub_params = params[:op_num_params]
        params = params[op_num_params:]
        new_operands.append(bind_new_parameters(operand, sub_params))

    return op.__class__(*new_operands)


@bind_new_parameters.register
def bind_new_parameters_symbolic_op(op: SymbolicOp, params: Sequence[TensorLike]):
    new_base = bind_new_parameters(op.base, params)
    new_hyperparameters = copy.deepcopy(op.hyperparameters)
    _ = new_hyperparameters.pop("base")

    return op.__class__(new_base, **new_hyperparameters)


@bind_new_parameters.register
def bind_new_parameters_adjoint(op: Adjoint, params: Sequence[TensorLike]):
    # Need a separate dispatch for `Adjoint` because using a more general class
    # signature results in a call to `Adjoint.__new__` which doesn't raise an
    # error but does return an unusable object.
    return Adjoint(bind_new_parameters(op.base, params))


@bind_new_parameters.register
def bind_new_parameters_scalar_symbolic_op(op: ScalarSymbolicOp, params: Sequence[TensorLike]):
    new_scalar = params[0]
    params = params[1:]

    new_base = bind_new_parameters(op.base, params)
    new_hyperparameters = copy.deepcopy(op.hyperparameters)
    _ = new_hyperparameters.pop("base")

    return op.__class__(new_base, new_scalar, **new_hyperparameters)


@bind_new_parameters.register
def bind_new_parameters_sprod(op: SProd, params: Sequence[TensorLike]):
    # Separate dispatch for `SProd` since its constructor has a different interface
    new_scalar = params[0]
    params = params[1:]
    new_base = bind_new_parameters(op.base, params)

    return SProd(new_scalar, new_base)


@bind_new_parameters.register
def bind_new_parameters_pow(op: Pow, params: Sequence[TensorLike]):
    # Need a separate dispatch for `Pow` because using a more general class
    # signature results in a call to `Pow.__new__` which doesn't raise an
    # error but does return an unusable object.
    return Pow(bind_new_parameters(op.base, params), op.scalar)


@bind_new_parameters.register
def bind_new_parameters_hamiltonian(op: qml.Hamiltonian, params: Sequence[TensorLike]):
    return qml.Hamiltonian(params, op.ops)


@bind_new_parameters.register
def bind_new_parameters_tensor(op: Tensor, params: Sequence[TensorLike]):
    new_obs = []

    for obs in op.obs:
        sub_params = params[: obs.num_params]
        params = params[obs.num_params :]
        new_obs.append(bind_new_parameters(obs, sub_params))

    return Tensor(*new_obs)

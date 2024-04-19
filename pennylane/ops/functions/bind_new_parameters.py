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

from typing import Sequence, Union
import copy
from functools import singledispatch

import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.operation import Operator, Tensor

from ..identity import Identity
from ..qubit import Projector
from ..op_math import CompositeOp, SymbolicOp, ScalarSymbolicOp, Adjoint, Pow, SProd


@singledispatch
def bind_new_parameters(op: Operator, params: Sequence[TensorLike]) -> Operator:
    """Create a new operator with updated parameters

    This function takes an :class:`~.Operator` and new parameters as input and
    returns a new operator of the same type with the new parameters. This function
    does not mutate the original operator.

    Args:
        op (.Operator): Operator to update
        params (Sequence[TensorLike]): New parameters to create operator with. This
            must have the same shape as `op.data`.

    Returns:
        .Operator: New operator with updated parameters
    """
    try:
        return op.__class__(*params, wires=op.wires, **copy.deepcopy(op.hyperparameters))
    except (TypeError, ValueError):
        # operation is doing something different with its call signature.
        new_op = copy.deepcopy(op)
        new_op.data = tuple(params)
        return new_op


@bind_new_parameters.register
def bind_new_parameters_approx_time_evolution(
    op: qml.ApproxTimeEvolution, params: Sequence[TensorLike]
):
    new_hamiltonian = bind_new_parameters(op.hyperparameters["hamiltonian"], params[:-1])
    time = params[-1]
    n = op.hyperparameters["n"]

    return qml.ApproxTimeEvolution(new_hamiltonian, time, n)


@bind_new_parameters.register
def bind_new_parameters_commuting_evolution(
    op: qml.CommutingEvolution, params: Sequence[TensorLike]
):
    new_hamiltonian = bind_new_parameters(op.hyperparameters["hamiltonian"], params[1:])
    freq = op.hyperparameters["frequencies"]
    shifts = op.hyperparameters["shifts"]
    time = params[0]

    return qml.CommutingEvolution(new_hamiltonian, time, frequencies=freq, shifts=shifts)


@bind_new_parameters.register
def bind_new_parameters_fermionic_double_excitation(
    op: qml.FermionicDoubleExcitation, params: Sequence[TensorLike]
):
    wires1 = op.hyperparameters["wires1"]
    wires2 = op.hyperparameters["wires2"]

    return qml.FermionicDoubleExcitation(params[0], wires1=wires1, wires2=wires2)


@bind_new_parameters.register
def bind_new_parameters_angle_embedding(op: qml.AngleEmbedding, params: Sequence[TensorLike]):
    rotation = op.hyperparameters["rotation"].basis
    return qml.AngleEmbedding(params[0], wires=op.wires, rotation=rotation)


@bind_new_parameters.register
def bind_new_parameters_identity(op: Identity, params: Sequence[TensorLike]):
    return qml.Identity(*params, wires=op.wires)


@bind_new_parameters.register
def bind_new_parameters_linear_combination(
    op: qml.ops.LinearCombination, params: Sequence[TensorLike]
):
    new_coeffs, new_ops = [], []
    i = 0
    for o in op.ops:
        new_coeffs.append(params[i])
        i += 1
        if o.data:
            sub_data = params[i : i + len(o.data)]
            new_ops.append(bind_new_parameters(o, sub_data))
            i += len(sub_data)
        else:
            new_ops.append(o)

    new_H = qml.ops.LinearCombination(new_coeffs, new_ops)

    if op.grouping_indices is not None:
        new_H.grouping_indices = op.grouping_indices

    return new_H


@bind_new_parameters.register
def bind_new_parameters_composite_op(op: CompositeOp, params: Sequence[TensorLike]):
    new_operands = []

    for operand in op.operands:
        op_num_params = operand.num_params
        sub_params = params[:op_num_params]
        params = params[op_num_params:]
        new_operands.append(bind_new_parameters(operand, sub_params))

    return op.__class__(*new_operands)


@bind_new_parameters.register(qml.CY)
@bind_new_parameters.register(qml.CZ)
@bind_new_parameters.register(qml.CH)
@bind_new_parameters.register(qml.CCZ)
@bind_new_parameters.register(qml.CSWAP)
@bind_new_parameters.register(qml.CNOT)
@bind_new_parameters.register(qml.Toffoli)
@bind_new_parameters.register(qml.MultiControlledX)
def bind_new_parameters_copy(op, params: Sequence[TensorLike]):  # pylint:disable=unused-argument
    return copy.copy(op)


@bind_new_parameters.register(qml.CRX)
@bind_new_parameters.register(qml.CRY)
@bind_new_parameters.register(qml.CRZ)
@bind_new_parameters.register(qml.CRot)
@bind_new_parameters.register(qml.ControlledPhaseShift)
def bind_new_parameters_parametric_controlled_ops(
    op: Union[qml.CRX, qml.CRY, qml.CRZ, qml.CRot, qml.ControlledPhaseShift],
    params: Sequence[TensorLike],
):
    return op.__class__(*params, wires=op.wires)


@bind_new_parameters.register
def bind_new_parameters_symbolic_op(op: SymbolicOp, params: Sequence[TensorLike]):
    new_base = bind_new_parameters(op.base, params)
    new_hyperparameters = copy.deepcopy(op.hyperparameters)
    _ = new_hyperparameters.pop("base")

    return op.__class__(new_base, **new_hyperparameters)


@bind_new_parameters.register
def bind_new_parameters_controlled_sequence(
    op: qml.ControlledSequence, params: Sequence[TensorLike]
):
    new_base = bind_new_parameters(op.base, params)
    return op.__class__(new_base, control=op.control)


@bind_new_parameters.register
def bind_new_parameters_adjoint(op: Adjoint, params: Sequence[TensorLike]):
    # Need a separate dispatch for `Adjoint` because using a more general class
    # signature results in a call to `Adjoint.__new__` which doesn't raise an
    # error but does return an unusable object.
    return Adjoint(bind_new_parameters(op.base, params))


@bind_new_parameters.register
def bind_new_parameters_projector(op: Projector, params: Sequence[TensorLike]):
    # Need a separate dispatch for `Projector` because using a more general class
    # signature results in a call to `Projector.__new__` which doesn't raise an
    # error but does return an unusable object.
    return Projector(*params, wires=op.wires)


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
def bind_new_parameters_hamiltonian(op: qml.ops.Hamiltonian, params: Sequence[TensorLike]):
    new_H = qml.ops.Hamiltonian(params, op.ops)
    if op.grouping_indices is not None:
        new_H.grouping_indices = op.grouping_indices
    return new_H


@bind_new_parameters.register
def bind_new_parameters_tensor(op: Tensor, params: Sequence[TensorLike]):
    new_obs = []

    for obs in op.obs:
        sub_params = params[: obs.num_params]
        params = params[obs.num_params :]
        new_obs.append(bind_new_parameters(obs, sub_params))

    return Tensor(*new_obs)


@bind_new_parameters.register
def bind_new_parameters_conditional(op: qml.ops.Conditional, params: Sequence[TensorLike]):
    then_op = bind_new_parameters(op.then_op, params)
    mv = copy.deepcopy(op.meas_val)

    return qml.ops.Conditional(mv, then_op)

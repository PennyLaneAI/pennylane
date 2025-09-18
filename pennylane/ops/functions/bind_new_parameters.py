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
# pylint:disable=missing-function-docstring

import copy
from collections.abc import Sequence
from functools import singledispatch

from pennylane import ops
from pennylane.operation import Operator
from pennylane.ops import (
    Adjoint,
    CompositeOp,
    Identity,
    Pow,
    Projector,
    ScalarSymbolicOp,
    SProd,
    SymbolicOp,
)
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.subroutines import (
    ApproxTimeEvolution,
    CommutingEvolution,
    ControlledSequence,
    FermionicDoubleExcitation,
    QDrift,
    TrotterProduct,
)
from pennylane.typing import TensorLike


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
    op: ApproxTimeEvolution, params: Sequence[TensorLike]
):
    new_hamiltonian = bind_new_parameters(op.hyperparameters["hamiltonian"], params[:-1])
    time = params[-1]
    n = op.hyperparameters["n"]

    return ApproxTimeEvolution(new_hamiltonian, time, n)


@bind_new_parameters.register
def _(op: TrotterProduct, params: Sequence[TensorLike]):
    new_hamiltonian = bind_new_parameters(op.hyperparameters["base"], params[:-1])
    time = params[-1]

    hp = op.hyperparameters
    return TrotterProduct(new_hamiltonian, time, n=hp["n"], order=hp["order"])


@bind_new_parameters.register
def bind_new_parameters_commuting_evolution(op: CommutingEvolution, params: Sequence[TensorLike]):
    new_hamiltonian = bind_new_parameters(op.hyperparameters["hamiltonian"], params[1:])
    freq = op.hyperparameters["frequencies"]
    shifts = op.hyperparameters["shifts"]
    time = params[0]

    return CommutingEvolution(new_hamiltonian, time, frequencies=freq, shifts=shifts)


@bind_new_parameters.register
def bind_new_parameters_qdrift(op: QDrift, params: Sequence[TensorLike]):
    new_hamiltonian = bind_new_parameters(op.hyperparameters["base"], params[:-1])
    time = params[-1]
    n = op.hyperparameters["n"]
    seed = op.hyperparameters["seed"]

    return QDrift(new_hamiltonian, time, n=n, seed=seed)


@bind_new_parameters.register
def bind_new_parameters_fermionic_double_excitation(
    op: FermionicDoubleExcitation, params: Sequence[TensorLike]
):
    wires1 = op.hyperparameters["wires1"]
    wires2 = op.hyperparameters["wires2"]

    return FermionicDoubleExcitation(params[0], wires1=wires1, wires2=wires2)


@bind_new_parameters.register
def bind_new_parameters_angle_embedding(op: AngleEmbedding, params: Sequence[TensorLike]):
    rotation = op.hyperparameters["rotation"].basis
    return AngleEmbedding(params[0], wires=op.wires, rotation=rotation)


@bind_new_parameters.register
def bind_new_parameters_identity(op: Identity, params: Sequence[TensorLike]):
    return Identity(*params, wires=op.wires)


@bind_new_parameters.register
def bind_new_parameters_linear_combination(op: ops.LinearCombination, params: Sequence[TensorLike]):
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

    new_H = ops.LinearCombination(new_coeffs, new_ops)

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


@bind_new_parameters.register(ops.CY)
@bind_new_parameters.register(ops.CZ)
@bind_new_parameters.register(ops.CH)
@bind_new_parameters.register(ops.CCZ)
@bind_new_parameters.register(ops.CSWAP)
@bind_new_parameters.register(ops.CNOT)
@bind_new_parameters.register(ops.Toffoli)
@bind_new_parameters.register(ops.MultiControlledX)
def bind_new_parameters_copy(op, params: Sequence[TensorLike]):
    return copy.copy(op)


@bind_new_parameters.register(ops.CRX)
@bind_new_parameters.register(ops.CRY)
@bind_new_parameters.register(ops.CRZ)
@bind_new_parameters.register(ops.CRot)
@bind_new_parameters.register(ops.ControlledPhaseShift)
@bind_new_parameters.register(ops.ControlledQubitUnitary)
def bind_new_parameters_parametric_controlled_ops(
    op: ops.CRX | ops.CRY | ops.CRZ | ops.CRot | ops.ControlledPhaseShift,
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
def bind_new_parameters_controlled_sequence(op: ControlledSequence, params: Sequence[TensorLike]):
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
def bind_new_parameters_conditional(op: ops.Conditional, params: Sequence[TensorLike]):
    then_op = bind_new_parameters(op.base, params)
    mv = copy.deepcopy(op.meas_val)

    return ops.Conditional(mv, then_op)

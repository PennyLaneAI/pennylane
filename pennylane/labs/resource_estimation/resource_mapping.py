# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Mapping PL operations to their ResourceOperator."""
from __future__ import annotations

import copy
from collections import defaultdict
from functools import singledispatch
from typing import Hashable, Optional, Type

from pennylane.labs.resource_estimation.ops.identity import ResourceGlobalPhase, ResourceIdentity
from pennylane.labs.resource_estimation.ops.op_math import (
    ResourceCCZ,
    ResourceCH,
    ResourceCNOT,
    ResourceControlledPhaseShift,
    ResourceCRot,
    ResourceCRX,
    ResourceCRY,
    ResourceCRZ,
    ResourceCSWAP,
    ResourceCY,
    ResourceCZ,
    ResourceMultiControlledX,
    ResourceProd,
    ResourceToffoli,
)
from pennylane.labs.resource_estimation.ops.qubit import (
    ResourceDoubleExcitation,
    ResourceDoubleExcitationMinus,
    ResourceDoubleExcitationPlus,
    ResourceFermionicSWAP,
    ResourceHadamard,
    ResourceIsingXX,
    ResourceIsingXY,
    ResourceIsingYY,
    ResourceIsingZZ,
    ResourceMultiRZ,
    ResourceOrbitalRotation,
    ResourcePauliRot,
    ResourcePhaseShift,
    ResourcePSWAP,
    ResourceRot,
    ResourceRX,
    ResourceRY,
    ResourceRZ,
    ResourceS,
    ResourceSingleExcitation,
    ResourceSingleExcitationMinus,
    ResourceSingleExcitationPlus,
    ResourceSWAP,
    ResourceT,
    ResourceX,
    ResourceY,
    ResourceZ,
)
from pennylane.operation import DecompositionUndefinedError, Operation
from pennylane.ops.identity import GlobalPhase, Identity
from pennylane.ops.op_math import (
    CCZ,
    CH,
    CNOT,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CY,
    CZ,
    ControlledPhaseShift,
    CRot,
    MultiControlledX,
    Toffoli,
)
from pennylane.ops.qubit import (
    PSWAP,
    RX,
    RY,
    RZ,
    SWAP,
    DoubleExcitation,
    DoubleExcitationMinus,
    DoubleExcitationPlus,
    FermionicSWAP,
    Hadamard,
    IsingXX,
    IsingXY,
    IsingYY,
    IsingZZ,
    MultiRZ,
    OrbitalRotation,
    PauliRot,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    Rot,
    S,
    SingleExcitation,
    SingleExcitationMinus,
    SingleExcitationPlus,
    T,
)


@singledispatch
def map_to_resource_op(op):
    r"""A function which maps an instance of :class:`~.Operation` to
    its associated :class:`~.ResourceOperator`.

    Args:
        op (~.Operation): base operation to be mapped

    Raise:
        TypeError: The op is not a valid operation
        NotImplementedError: Operation doesn't have a resource equivalent and doesn't define
            a decomposition.

    Return:
        (~.ResourceOperator): the resource operator equivalent of the base operator
    """
    if not isinstance(op, Operation):
        raise TypeError(f"The op {op} is not a valid operation.")

    try:
        mapped_ops = tuple(map_to_resource_op(sub_op) for sub_op in op.decomposition())
        return ResourceProd.resource_rep(mapped_ops)

    except DecompositionUndefinedError as e:
        raise NotImplementedError(
            "Operation doesn't have a resource equivalent and doesn't define a decomposition."
        ) from e


# Parametric Single Qubit:
@map_to_resource_op.register
def _(op: PhaseShift):
    return ResourcePhaseShift.resource_rep()


@map_to_resource_op.register
def _(op: RX):
    return ResourceRX.resource_rep()


@map_to_resource_op.register
def _(op: RY):
    return ResourceRY.resource_rep()


@map_to_resource_op.register
def _(op: RZ):
    return ResourceRZ.resource_rep()


@map_to_resource_op.register
def _(op: Rot):
    return ResourceRot.resource_rep()


# Controlled Ops:
@map_to_resource_op.register
def _(op: CH):
    return ResourceCH.resource_rep()


@map_to_resource_op.register
def _(op: CY):
    return ResourceCY.resource_rep()


@map_to_resource_op.register
def _(op: CZ):
    return ResourceCZ.resource_rep()


@map_to_resource_op.register
def _(op: CSWAP):
    return ResourceCSWAP.resource_rep()


@map_to_resource_op.register
def _(op: CCZ):
    return ResourceCCZ.resource_rep()


@map_to_resource_op.register
def _(op: CRX):
    return ResourceCRX.resource_rep()


@map_to_resource_op.register
def _(op: CRY):
    return ResourceCRY.resource_rep()


@map_to_resource_op.register
def _(op: CRZ):
    return ResourceCRZ.resource_rep()


@map_to_resource_op.register
def _(op: CRot):
    return ResourceCRot.resource_rep()


@map_to_resource_op.register
def _(op: ControlledPhaseShift):
    return ResourceControlledPhaseShift.resource_rep()


@map_to_resource_op.register
def _(op: MultiControlledX):
    num_control = len(op.hyperparameters["control_wires"])
    num_work_wires = len(op.hyperparameters["work_wires"])
    num_control_values = len([val for val in op.hyperparameters["control_values"] if not val])

    params = {
            "num_ctrl_wires": num_control,
            "num_ctrl_values": num_control_values,
            "num_work_wires": num_work_wires,
        }
    return ResourceMultiControlledX.resource_rep(**params)


@map_to_resource_op.register
def _(op: Toffoli):
    return ResourceToffoli.resource_rep()


@map_to_resource_op.register
def _(op: CNOT):
    return ResourceCNOT.resource_rep()


# Non-parametric Ops:
@map_to_resource_op.register
def _(op: PauliX):
    return ResourceX.resource_rep()


@map_to_resource_op.register
def _(op: PauliY):
    return ResourceY.resource_rep()


@map_to_resource_op.register
def _(op: PauliZ):
    return ResourceZ.resource_rep()


@map_to_resource_op.register
def _(op: T):
    return ResourceT.resource_rep()


@map_to_resource_op.register
def _(op: S):
    return ResourceS.resource_rep()


@map_to_resource_op.register
def _(op: Hadamard):
    return ResourceHadamard.resource_rep()


@map_to_resource_op.register
def _(op: SWAP):
    return ResourceSWAP.resource_rep()


# Identity Ops: 
@map_to_resource_op.register
def _(op: Identity):
    return ResourceIdentity.resource_rep()


@map_to_resource_op.register
def _(op: GlobalPhase):
    return ResourceGlobalPhase.resource_rep()


# Qchem Ops:
@map_to_resource_op.register
def _(op: SingleExcitation):
    return ResourceSingleExcitation.resource_rep()


@map_to_resource_op.register
def _(op: SingleExcitationPlus):
    return ResourceSingleExcitationPlus.resource_rep()


@map_to_resource_op.register
def _(op: SingleExcitationMinus):
    return ResourceSingleExcitationMinus.resource_rep()


@map_to_resource_op.register
def _(op: DoubleExcitation):
    return ResourceDoubleExcitation.resource_rep()


@map_to_resource_op.register
def _(op: DoubleExcitationPlus):
    return ResourceDoubleExcitationPlus.resource_rep()


@map_to_resource_op.register
def _(op: DoubleExcitationMinus):
    return ResourceDoubleExcitationMinus.resource_rep()


@map_to_resource_op.register
def _(op: OrbitalRotation):
    return ResourceOrbitalRotation.resource_rep()


@map_to_resource_op.register
def _(op: FermionicSWAP):
    return ResourceFermionicSWAP.resource_rep()


# Parametric Multi-qubit Ops:
@map_to_resource_op.register
def _(op: IsingXX):
    return ResourceIsingXX.resource_rep()


@map_to_resource_op.register
def _(op: IsingXY):
    return ResourceIsingXY.resource_rep()


@map_to_resource_op.register
def _(op: IsingYY):
    return ResourceIsingYY.resource_rep()


@map_to_resource_op.register
def _(op: IsingZZ):
    return ResourceIsingZZ.resource_rep()


@map_to_resource_op.register
def _(op: PSWAP):
    return ResourcePSWAP.resource_rep()


@map_to_resource_op.register
def _(op: PauliRot):
    pauli_string = op.hyperparameters["pauli_word"]
    return ResourcePauliRot.resource_rep(pauli_string=pauli_string)


@map_to_resource_op.register
def _(op: MultiRZ):
    num_wires = len(op.wires)
    return ResourceMultiRZ.resource_rep(num_wires=num_wires)



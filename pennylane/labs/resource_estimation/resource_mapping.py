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
from typing import Optional, Type, Hashable

from pennylane.operation import Operation, DecompositionUndefinedError
from pennylane.ops.identity import Identity, GlobalPhase
from pennylane.ops.qubit import (
    S,
    T,
    PauliX,
    PauliY,
    PauliZ,
    SWAP,
    Hadamard,
    RX,
    RY,
    RZ,
    Rot,
    PhaseShift,
    MultiRZ,
    PauliRot,
    IsingXX,
    IsingXY,
    IsingYY,
    IsingZZ,
    PSWAP,
    SingleExcitation,
    SingleExcitationMinus,
    SingleExcitationPlus,
    DoubleExcitation,
    DoubleExcitationMinus,
    DoubleExcitationPlus,
    FermionicSWAP, 
    OrbitalRotation,
)
from pennylane.ops.op_math import (
    CH,
    CY,
    CZ,
    CCZ,
    CNOT,
    Toffoli,
    CSWAP,
    MultiControlledX,
    CRX,
    CRY,
    CRZ,
    CRot,
    ControlledPhaseShift,
)


from pennylane.labs.resource_estimation.ops.identity import ResourceIdentity, ResourceGlobalPhase
from pennylane.labs.resource_estimation.ops.qubit import (
    ResourceS,
    ResourceT,
    ResourceX,
    ResourceY,
    ResourceZ,
    ResourceSWAP,
    ResourceHadamard,
    ResourceRX,
    ResourceRY,
    ResourceRZ,
    ResourceRot,
    ResourcePhaseShift,
    ResourceMultiRZ,
    ResourcePauliRot,
    ResourceIsingXX,
    ResourceIsingXY,
    ResourceIsingYY,
    ResourceIsingZZ,
    ResourcePSWAP,
    ResourceSingleExcitation,
    ResourceSingleExcitationMinus,
    ResourceSingleExcitationPlus,
    ResourceDoubleExcitation,
    ResourceDoubleExcitationMinus,
    ResourceDoubleExcitationPlus,
    ResourceFermionicSWAP, 
    ResourceOrbitalRotation,
)
from pennylane.labs.resource_estimation.ops.op_math import (
    ResourceProd,
    ResourceCH,
    ResourceCY,
    ResourceCZ,
    ResourceCCZ,
    ResourceCNOT,
    ResourceToffoli,
    ResourceCSWAP,
    ResourceMultiControlledX,
    ResourceCRX,
    ResourceCRY,
    ResourceCRZ,
    ResourceCRot,
    ResourceControlledPhaseShift,
)

@singledispatch
def map_to_resource_op(op):
    r"""A function which maps an instance of :class:`~.Operation` to
    its associated :class:`~.ResourceOperator`.

    Args:
        op (~.Operation): base operation to be mapped

    Raise:
        TypeError: The op is not a valid operation 
        ValueError: Operation doesn't have a resource equivalent and doesn't define
            a decomposition.
        
    Return:
        (~.ResourceOperator): the resource operator equivalent of the base operator
    """
    if not isinstance(op, Operation):
        raise TypeError(f"The op {op} is not a valid operation.")
    
    try:
        mapped_ops = tuple(map_to_resource_op(sub_op) for sub_op in op.decomposition())
        return ResourceProd(*mapped_ops)

    except DecompositionUndefinedError as e: 
        raise ValueError(
            "Operation doesn't have a resource equivalent and doesn't define a decomposition."
        ) from e 

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

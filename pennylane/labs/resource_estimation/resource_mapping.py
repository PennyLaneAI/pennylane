# Copyright 2025 Xanadu Quantum Technologies Inc.

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

from functools import singledispatch

import pennylane.labs.resource_estimation.ops as re_ops
import pennylane.labs.resource_estimation.templates as re_temps
import pennylane.ops as qops
import pennylane.templates as qtemps
from pennylane.labs.resource_estimation import ResourceOperator
from pennylane.operation import Operation


@singledispatch
def map_to_resource_op(op: Operation) -> ResourceOperator:
    r"""Maps an instance of :class:`~.Operation` to its associated :class:`~.pennylane.labs.resource_estimation.ResourceOperator`.

    Args:
        op (~.Operation): base operation to be mapped

    Return:
        (~.pennylane.labs.resource_estimation.ResourceOperator): the resource operator equal of the base operator

    Raises:
        TypeError: The op is not a valid operation
        NotImplementedError: Operation doesn't have a resource equivalent and doesn't define
            a decomposition.
    """

    if not isinstance(op, Operation):
        raise TypeError(f"The op {op} is not a valid operation.")

    raise NotImplementedError(
        "Operation doesn't have a resource equivalent and doesn't define a decomposition."
    )


@map_to_resource_op.register
def _(op: qops.Identity):
    return re_ops.ResourceIdentity()


@map_to_resource_op.register
def _(op: qops.Hadamard):
    return re_ops.ResourceHadamard()


@map_to_resource_op.register
def _(op: qops.S):
    return re_ops.ResourceS()


@map_to_resource_op.register
def _(op: qops.T):
    return re_ops.ResourceT()


@map_to_resource_op.register
def _(op: qops.X):
    return re_ops.ResourceX()


@map_to_resource_op.register
def _(op: qops.Y):
    return re_ops.ResourceY()


@map_to_resource_op.register
def _(op: qops.Z):
    return re_ops.ResourceZ()


@map_to_resource_op.register
def _(op: qops.SWAP):
    return re_ops.ResourceSWAP()


@map_to_resource_op.register
def _(op: qops.PhaseShift):
    return re_ops.ResourcePhaseShift()


@map_to_resource_op.register
def _(op: qops.Rot):
    return re_ops.ResourceRot()


@map_to_resource_op.register
def _(op: qops.RX):
    return re_ops.ResourceRX()


@map_to_resource_op.register
def _(op: qops.RY):
    return re_ops.ResourceRY()


@map_to_resource_op.register
def _(op: qops.RZ):
    return re_ops.ResourceRZ()


@map_to_resource_op.register
def _(op: qops.MultiRZ):
    return re_ops.ResourceMultiRZ(num_wires=len(op.wires))


@map_to_resource_op.register
def _(op: qops.PauliRot):
    return re_ops.ResourcePauliRot(pauli_string=op.hyperparameters["pauli_word"])


@map_to_resource_op.register
def _(op: qops.IsingXX):
    return re_ops.ResourceIsingXX()


@map_to_resource_op.register
def _(op: qops.IsingYY):
    return re_ops.ResourceIsingYY()


@map_to_resource_op.register
def _(op: qops.IsingXY):
    return re_ops.ResourceIsingXY()


@map_to_resource_op.register
def _(op: qops.IsingZZ):
    return re_ops.ResourceIsingZZ()


@map_to_resource_op.register
def _(op: qops.PSWAP):
    return re_ops.ResourcePSWAP()


@map_to_resource_op.register
def _(op: qops.SingleExcitation):
    return re_ops.ResourceSingleExcitation()


@map_to_resource_op.register
def _(op: qops.CCZ):
    return re_ops.ResourceCCZ()


@map_to_resource_op.register
def _(op: qops.CH):
    return re_ops.ResourceCH()


@map_to_resource_op.register
def _(op: qops.CNOT):
    return re_ops.ResourceCNOT()


@map_to_resource_op.register
def _(op: qops.ControlledPhaseShift):
    return re_ops.ResourceControlledPhaseShift()


@map_to_resource_op.register
def _(op: qops.CRot):
    return re_ops.ResourceCRot()


@map_to_resource_op.register
def _(op: qops.CRX):
    return re_ops.ResourceCRX()


@map_to_resource_op.register
def _(op: qops.CRY):
    return re_ops.ResourceCRY()


@map_to_resource_op.register
def _(op: qops.CRZ):
    return re_ops.ResourceCRZ()


@map_to_resource_op.register
def _(op: qops.CSWAP):
    return re_ops.ResourceCSWAP()


@map_to_resource_op.register
def _(op: qops.CY):
    return re_ops.ResourceCY()


@map_to_resource_op.register
def _(op: qops.CZ):
    return re_ops.ResourceCZ()


@map_to_resource_op.register
def _(op: qops.MultiControlledX):
    return re_ops.ResourceMultiControlledX(
        num_ctrl_wires=len(op.wires) - 1, num_ctrl_values=sum(op.control_values)
    )


@map_to_resource_op.register
def _(op: qtemps.TemporaryAND):
    return re_ops.ResourceTempAND()


@map_to_resource_op.register
def _(op: qops.Toffoli):
    return re_ops.ResourceToffoli()

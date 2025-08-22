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

from pennylane.labs.resource_estimation import ResourceOperator
from pennylane.operation import Operation
import pennylane.ops as qops
import pennylane.labs.resource_estimation.ops as re_ops


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
def _(op: qops.GlobalPhase):
    return re_ops.ResourceGlobalPhase()


@map_to_resource_op.register
def _(op: qops.Hadamard):
    return re_ops.ResourceHadamard()


@map_to_resource_op.register
def _(op: qops.Identity):
    return re_ops.ResourceIdentity()


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
def _(op: qops.S):
    return re_ops.ResourceS()

@map_to_resource_op.register
def _(op: qops.SWAP):
    return re_ops.ResourceSWAP()


@map_to_resource_op.register
def _(op: qops.CSWAP):
    return re_ops.ResourceCSWAP()


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
def _(op: qops.CY):
    return re_ops.ResourceCY()

@map_to_resource_op.register
def _(op: qops.Z):
    return re_ops.ResourceZ()

@map_to_resource_op.register
def _(op: qops.CZ):
    return re_ops.ResourceCZ()

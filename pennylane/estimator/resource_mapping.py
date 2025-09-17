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
r"""Mapping PL operations to their associated ResourceOperator."""
from __future__ import annotations

from functools import singledispatch

import pennylane.estimator.ops as re_ops
import pennylane.ops as qops
from pennylane.estimator import ResourceOperator
from pennylane.operation import Operation


@singledispatch
def map_to_resource_op(op: Operation) -> ResourceOperator:
    r"""Maps an instance of :class:`~.Operation` to its associated :class:`~.estimator.ResourceOperator`.

    Args:
        op (~.Operation): base operation to be mapped

    Return:
        (~.estimator.ResourceOperator): the resource operator equivalent of the base operator

    Raises:
        TypeError: The op is not a valid operation
        NotImplementedError: Operation doesn't have a resource equivalent and doesn't define
            a decomposition.
    """

    if not isinstance(op, Operation):
        raise TypeError(f"Operator of type {type(op)} is not a valid operation.")

    raise NotImplementedError(
        "Operation doesn't have a resource equivalent and doesn't define a decomposition."
    )


@map_to_resource_op.register
def _(op: qops.Identity):
    return re_ops.Identity()


@map_to_resource_op.register
def _(op: qops.GlobalPhase):
    return re_ops.GlobalPhase()


@map_to_resource_op.register
def _(op: qops.Hadamard):
    return re_ops.Hadamard()


@map_to_resource_op.register
def _(op: qops.S):
    return re_ops.S()


@map_to_resource_op.register
def _(op: qops.T):
    return re_ops.T()


@map_to_resource_op.register
def _(op: qops.X):
    return re_ops.X()


@map_to_resource_op.register
def _(op: qops.Y):
    return re_ops.Y()


@map_to_resource_op.register
def _(op: qops.Z):
    return re_ops.Z()


@map_to_resource_op.register
def _(op: qops.SWAP):
    return re_ops.SWAP()

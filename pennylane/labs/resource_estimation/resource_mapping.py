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

from functools import singledispatch

from pennylane.operation import Operation


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

    # try:
    #     mapped_ops = tuple(map_to_resource_op(sub_op) for sub_op in op.decomposition())
    #     return ResourceProd.resource_rep(mapped_ops)
    #
    # except DecompositionUndefinedError as e:
    #     raise NotImplementedError(
    #         "Operation doesn't have a resource equivalent and doesn't define a decomposition."
    #     ) from e

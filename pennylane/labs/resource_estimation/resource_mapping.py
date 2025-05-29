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

from pennylane.operation import Operation


@singledispatch
def map_to_resource_op(op: Operation):
    r"""A function which maps an instance of :class:`~.Operation` to
    its associated :class:`~.ResourceOperator`.

    Args:
        op (~.Operation): base operation to be mapped

    Return:
        (~.ResourceOperator): the resource operator equal of the base operator

    Raises:
        TypeError: The op is not a valid operation
        NotImplementedError: Operation doesn't have a resource equivalent and doesn't define
            a decomposition.
    """

    if not isinstance(op, Operation):
        raise TypeError(f"The op {op} is not a valid operation.")

    raise NotImplementedError(
        "Operation doesn't have a resource equivalent and doesn't define" + " a decomposition."
    )

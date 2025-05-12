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
r"""Base classes for resource estimation."""
from __future__ import annotations

from typing import Hashable, Optional, Type
from pennylane.labs.resource_estimation.resource_operator import ResourceOperator

class CompressedResourceOp:  # pylint: disable=too-few-public-methods
    r"""Instantiate the light weight class corresponding to the operator type and parameters.

    Args:
        op_type (Type): the class object of an operation which inherits from '~.ResourceOperator'
        params (dict): a dictionary containing the minimal pairs of parameter names and values
                    required to compute the resources for the given operator

    .. details::

        This representation is the minimal amount of information required to estimate resources for the operator.

        **Example**

        >>> op_tp = CompressedResourceOp(ResourceHadamard, {"num_wires":1})
        >>> print(op_tp)
        Hadamard(num_wires=1)
    """

    def __init__(
        self, op_type: Type[ResourceOperator], params: Optional[dict] = None, name: str = None
    ):

        if not issubclass(op_type, ResourceOperator):
            raise TypeError(f"op_type must be a subclass of ResourceOperator. Got {op_type}.")
        self.op_type = op_type
        self.params = params or {}
        self._hashable_params = _make_hashable(params) if params else ()
        self._name = name or op_type.tracking_name(**self.params)

    def __hash__(self) -> int:
        return hash((self.op_type, self._hashable_params))

    def __eq__(self, other: CompressedResourceOp) -> bool:
        return (
            isinstance(other, CompressedResourceOp)
            and self.op_type == other.op_type
            and self.params == other.params
        )

    def __repr__(self) -> str:
        return self._name


def _make_hashable(d) -> tuple:
    if isinstance(d, Hashable):
        return d
    sorted_keys = sorted(d)
    return tuple((k, _make_hashable(d[k])) for k in sorted_keys)

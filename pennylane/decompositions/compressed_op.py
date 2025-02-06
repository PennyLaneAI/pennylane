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

"""Defines the compressed op class in a decomposition graph."""

from __future__ import annotations


class CompressedResourceOp:
    """A lightweight class representing an operator to be decomposed.

    An object of this class represents an operator in the decomposition graph. If the decomposition
    of this operator is independent of its parameters, e.g., ``Rot`` can be decomposed into two
    ``RZ`` and an ``RY`` regardless of the angles, then every occurrence of this operator in the
    circuit is represented by the same ``CompressedOp``.

    On the other hand, for more complex ops such as ``QFT``, for which the number of each type of
    gate depend on parameters such as the number of wires, each occurrence of this operator with
    different values for this parameters will be represented by different ``CompressedOp`` objects
    and thus different nodes in the decomposition graph.

    Args:
        op_type: the operator type
        params (dict): the parameters of the operator relevant to the resource estimation of
            its decompositions. This should only include parameters that affect the gate counts.

    """

    def __init__(self, op_type, params: dict = None):
        self.op_type = op_type
        self.params = params or {}
        self._hashable_params = _make_hashable(params)

    def __hash__(self) -> int:
        return hash((self.op_type, self._hashable_params))

    def __eq__(self, other: CompressedResourceOp) -> bool:
        return (
            isinstance(other, CompressedResourceOp)
            and (self.op_type == other.op_type)
            and (self.params == other.params)
        )

    def __repr__(self):
        return self.op_type.__name__


def _make_hashable(d) -> tuple:
    return tuple((k, _make_hashable(v)) for k, v in d.items()) if isinstance(d, dict) else d

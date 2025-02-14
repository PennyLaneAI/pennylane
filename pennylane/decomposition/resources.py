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

"""Defines the data structure that stores resource estimates for each decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Resources:
    r"""Stores resource estimates.

    Args:
        num_gates (int): the total number of gates.
        gate_counts (dict): dictionary mapping compressed ops to number of occurrences.

    """

    num_gates: int = 0
    gate_counts: dict[CompressedResourceOp, int] = field(default_factory=dict)

    def __add__(self, other: Resources):
        return Resources(
            self.num_gates + other.num_gates,
            _combine_dict(self.gate_counts, other.gate_counts),
        )

    def __mul__(self, scalar: int):
        return Resources(self.num_gates * scalar, _scale_dict(self.gate_counts, scalar))

    __rmul__ = __mul__


def _combine_dict(dict1: dict, dict2: dict):
    r"""Combines two dictionaries and adds values of common keys."""

    combined_dict = dict1.copy()

    for k, v in dict2.items():
        combined_dict[k] = combined_dict.get(k, 0) + v

    return combined_dict


def _scale_dict(dict1: dict, scalar: int):
    r"""Scales the values in a dictionary with a scalar."""

    scaled_dict = dict1.copy()

    for k in scaled_dict:
        scaled_dict[k] *= scalar

    return scaled_dict


class CompressedResourceOp:
    """A lightweight class representing an operator to be decomposed.

    An object of this class represents an operator in the decomposition graph. If the decomposition
    of this operator is independent of its parameters, e.g., ``Rot`` can be decomposed into two
    ``RZ`` and an ``RY`` regardless of the angles, then every occurrence of this operator in the
    circuit is represented by the same ``CompressedOp``.

    On the other hand, for more complex ops such as ``PauliRot``, for which the numbers of each
    gate depend on its pauli word, each occurrence of this operator with a different pauli word
    will be a different ``CompressedOp`` object, thus a new node in the decomposition graph.

    Args:
        op_type: the operator type
        params (dict): the parameters of the operator relevant to the resource estimation of
            its decompositions. This should only include parameters that affect the gate counts.

    """

    def __init__(self, op_type, params: dict = None):
        self.op_type = op_type
        self.params = params or {}
        self._hashable_params = _make_hashable(params) if params else ()

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

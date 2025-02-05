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

"""Module defining the class to store resource estimates for each decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field

from .compressed_op import CompressedResourceOp


@dataclass(frozen=True)
class Resources:
    r"""Stores resource estimates.

    Args:
        num_gates (int): number of gates.
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

    @property
    def num_gate_types(self):
        r"""Returns the number of gate types."""
        return len(self.gate_counts)


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

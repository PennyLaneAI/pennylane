# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains the 'label' function for customizing operator labels.
"""

from pennylane.operation import Operator
from pennylane.ops.op_math import SymbolicOp


class LabelledOp(SymbolicOp):
    """Creates a labelled operator."""

    def __init__(self, base: Operator, custom_label: str):
        super().__init__(base)
        self.hyperparameters["custom_label"] = custom_label

    def label(self, decimals=None, base_label=None, cache=None):
        return f"{self.base.label(decimals, base_label, cache)}[{self.hyperparameters['custom_label']}]"


def label(op: Operator, new_label: str) -> LabelledOp:
    """Labels an operator with a custom label."""
    return LabelledOp(op, new_label)

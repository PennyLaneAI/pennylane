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


class MarkedOp(SymbolicOp):
    """Creates a marked operator."""

    resource_keys = {"base_class", "base_params"}

    def __init__(self, base: Operator, tag: str):
        super().__init__(base)
        self.hyperparameters["tag"] = tag

    @property
    def resource_params(self) -> dict:
        return {"base_class": type(self.base), "base_params": self.base.resource_params}

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_generator(self) -> bool:
        return self.base.has_generator

    def generator(self):
        return self.base.generator()

    def label(self, decimals=None, base_label=None, cache=None):
        base_label = self.base.label(decimals, base_label, cache)
        tag = self.hyperparameters["tag"]

        # If base label already has parameters, e.g., "RX(0.5)"
        if base_label.endswith(")"):
            return f'{base_label[:-1]}, "{tag}")'

        # If base label is a simple label, e.g., "X"
        return f'{base_label}("{tag}")'


def mark(op: Operator, tag: str) -> MarkedOp:
    """Marks an operator with a custom tag."""
    return MarkedOp(op, tag)

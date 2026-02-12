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
from pennylane.ops.functions.equal import (
    BASE_OPERATION_MISMATCH_ERROR_MESSAGE,
    _equal,
    _equal_dispatch,
)
from pennylane.ops.op_math import SymbolicOp


class MarkedOp(SymbolicOp):
    """Creates a marked operator.

    Args:
        base (Operator): The operator you wish to mark.
        marker (str): The custom marker to give to your operator.

    **Example:**

    >>> op = qml.RX(1.23456, wires=0)
    >>> marked_op = MarkedOp(op, "my-rx")
    >>> print(marked_op.marker)
    my-rx

    """

    def _flatten(self):
        hyperparameters = (("marker", self.hyperparameters["marker"]),)
        return (self.base,), hyperparameters

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(data[0], **hyperparams_dict)

    resource_keys = {"base_class", "base_params"}

    def __init__(self, base: Operator, marker: str):
        super().__init__(base)
        self.hyperparameters["marker"] = marker

    def __repr__(self):
        return f'MarkedOp({self.base}, marker="{self.marker}")'

    @property
    def resource_params(self) -> dict:
        return {"base_class": type(self.base), "base_params": self.base.resource_params}

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_generator(self) -> bool:
        return self.base.has_generator

    def generator(self):
        return self.base.generator()

    @property
    def marker(self) -> str:
        """Retrieve the marker set on this operator."""
        return self.hyperparameters["marker"]

    def label(self, decimals=None, base_label=None, cache=None):
        base_label = self.base.label(decimals, base_label, cache)
        marker = self.hyperparameters["marker"]

        # If base label already has parameters, e.g., "RX(0.5)"
        if base_label.endswith(")"):
            return f'{base_label[:-1]}, "{marker}")'

        # If base label is a simple label, e.g., "X"
        return f'{base_label}("{marker}")'

    def matrix(self, wire_order=None):
        return self.base.matrix(wire_order=wire_order)


@_equal_dispatch.register
def _equal_marked_op(op1: MarkedOp, op2: MarkedOp, **kwargs):
    if op1.marker != op2.marker:
        return (
            f"op1 and op2 have different markers. Got {op1.marker} and {op2.marker} respectively."
        )

    base_equal_check = _equal(op1.base, op2.base, **kwargs)
    if isinstance(base_equal_check, str):
        return BASE_OPERATION_MISMATCH_ERROR_MESSAGE + base_equal_check

    return True


def mark(op: Operator, marker: str) -> MarkedOp:
    """Marks an operator with a custom tag.

    Args:
        op (Operator): The operator you wish to mark.
        marker (str): The marker to give to the operator.

    **Example:**

    >>> op = qml.X(0)
    >>> marked_op = mark(op, "my-x")
    >>> print(marked_op.marker)
    my-x

    """
    return MarkedOp(op, marker)

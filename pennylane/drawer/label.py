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


class LabelledOp(SymbolicOp):
    """Creates a labelled operator.

    Args:
        base (Operator): The operator you wish to label.
        custom_label (str): The custom label to label your operator with.

    **Example:**

    >>> op = qml.RX(1.23456, wires=0)
    >>> labelled_op = LabelledOp(op, "my-rx")
    >>> print(labelled_op.hyperparameters["custom_label"])
    my-rx
    >>> labelled_op.label()
    'RX("my-rx")'
    >>> labelled_op.label(decimals=2)
    'RX\\n(1.23, "my-rx")'

    """

    def _flatten(self):
        hyperparameters = (("custom_label", self.hyperparameters["custom_label"]),)
        return (self.base,), hyperparameters

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(data[0], **hyperparams_dict)

    def __init__(self, base: Operator, custom_label: str):
        super().__init__(base)
        self.hyperparameters["custom_label"] = custom_label

    def __repr__(self):
        return f'LabelledOp({self.base}, custom_label="{self.custom_label}")'

    @property
    def custom_label(self) -> str:
        """Retrieve the custom label set on this operator."""
        return self.hyperparameters["custom_label"]

    def label(self, decimals=None, base_label=None, cache=None) -> str:
        """Retrieve the label for this operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings
        """
        base_label = self.base.label(decimals, base_label, cache)
        custom_label = self.hyperparameters["custom_label"]

        # If base label already has parameters, e.g., "RX(0.5)"
        if base_label.endswith(")"):
            return f'{base_label[:-1]}, "{custom_label}")'

        # If base label is a simple label, e.g., "X"
        return f'{base_label}("{custom_label}")'

    def matrix(self, wire_order=None):
        return self.base.matrix(wire_order=wire_order)


@_equal_dispatch.register
def _equal_labelled_op(op1: LabelledOp, op2: LabelledOp, **kwargs):
    if op1.custom_label != op2.custom_label:
        return f"op1 and op2 have different custom labels. Got {op1.custom_label} and {op2.custom_label} respectively."

    base_equal_check = _equal(op1.base, op2.base, **kwargs)
    if isinstance(base_equal_check, str):
        return BASE_OPERATION_MISMATCH_ERROR_MESSAGE + base_equal_check

    return True


def label(op: Operator, new_label: str) -> LabelledOp:
    """Labels an operator with a custom label.

    Args:
        op (Operator): The operator you wish to mark.
        new_label (str): The label you wish to give to the operator.

    **Example:**

    >>> op = qml.X(0)
    >>> labelled_op = label(op, "my-x")
    >>> print(labelled_op.custom_label)
    my-x

    The custom label will be displayed in the circuit diagram when using :func:`~.draw`

    .. code-block:: python

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            label(qml.H(0), "my-h")
            qml.CNOT([0,1])
            return qml.probs()

    >>> print(qml.draw(circuit)())
    0: ──H("my-h")─╭●─┤  Probs
    1: ────────────╰X─┤  Probs

    """
    return LabelledOp(op, new_label)

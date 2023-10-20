# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the CtrlSequence template.
"""
from copy import copy
import numpy as np
import pennylane as qml

from pennylane.operation import AnyWires
from pennylane.wires import Wires
from pennylane.ops.op_math.symbolicop import SymbolicOp


class CtrlSequence(SymbolicOp):
    """docstring"""

    num_wires = AnyWires
    grad_method = None

    def _flatten(self):
        return (self.base,), (self.control,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], control=metadata[0])

    def __init__(self, base, control, id=None):

        control_wires = Wires(control)

        if len(Wires.shared_wires([base.wires, control_wires])) != 0:
            raise ValueError("The control wires must be different from the base operation wires.")

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["base"] = base

        self._name = "CtrlSequence"

        super().__init__(base, id=id)

    @property
    def hash(self):
        return hash(
            (
                str(self.name),
                self.control,
                self.base.hash,
            )
        )

    @property
    def control(self):
        """The control wires for the sequence"""
        return self.hyperparameters["control_wires"]

    @property
    def wires(self):
        return self.control + self.base.wires

    @property
    def has_matrix(self):
        return False

    def __repr__(self):
        return f"CtrlSequence({self.base}, control={list(self.control)})"

    def map_wires(self, wire_map: dict):
        # pylint:disable=protected-access
        new_op = copy(self)
        new_op.hyperparameters["base"] = self.base.map_wires(wire_map=wire_map)
        new_op.hyperparameters["control_wires"] = Wires(
            [wire_map.get(wire, wire) for wire in self.control]
        )
        return new_op

    def decomposition(self):  # pylint:disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.CtrlSequence.decomposition`.

        Args:
            base (Operator): the operator that acts as the base for the sequence
            control_wires (Any or Iterable[Any]): the control wires for the sequence

        Returns:
            list[.Operator]: decomposition of the operator
        """
        # ToDo: docstring missing example
        return self.compute_decomposition(self.base, self.control)

    @staticmethod
    def compute_decomposition(base, control_wires):  # pylint:disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.CtrlSequence.decomposition`.

        Args:
            base (Operator): the operator that acts as the base for the sequence
            control_wires (Any or Iterable[Any]): the control wires for the sequence

        Returns:
            list[.Operator]: decomposition of the operator
        """
        # ToDo: docstring missing example
        ops = []
        powers_of_two = 1 << np.arange(len(control_wires))

        for z, ctrl_wire in zip(powers_of_two[::-1], control_wires):
            ops.append(qml.ctrl(qml.pow(base, z=z), control=ctrl_wire))

        return ops

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
"""
Contains the Select template.
"""
# pylint: disable=too-many-arguments

import copy
import itertools
import pennylane as qml
from pennylane.operation import Operation
from pennylane import math


class Select(Operation):
    r"""Applies specific input operations depending on the state of
    the designated control qubits.

    .. math:: Select|i\rangle \otimes |\psi\rangle = |i\rangle \otimes U_i |\psi\rangle

    .. figure:: ../../../doc/_static/templates/subroutines/select.png
                    :align: center
                    :width: 60%
                    :target: javascript:void(0);

    Args:
        ops (list[Operator]): operations to apply
        control (Sequence[int]): the wires controlling which operation is applied
        id (str or None): String representing the operation (optional)

    .. note::
        The position of the operation in the list determines which qubit state implements that operation.
        For example, when the qubit register is in the state :math:`|00\rangle`, we will apply ``ops[0]``.
        When the qubit register is in the state :math:`|10\rangle`, we will apply ``ops[2]``. To obtain the
        binary bitstring representing the state for list position ``index`` we can use the following relationship:
        ``index = int(state_string, 2)``. For example, ``2 = int('10', 2)``.

    **Example**

    >>> dev = qml.device('default.qubit', wires=4)
    >>> ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2,3])]
    >>> @qml.qnode(dev)
    >>> def circuit():
    >>>     qml.Select(ops, control=[0,1])
    >>>     return qml.state()
    ...
    >>> print(qml.draw(circuit, expansion_strategy='device')())
    0: ─╭○─╭○─╭●─╭●────┤  State
    1: ─├○─├●─├○─├●────┤  State
    2: ─╰X─│──╰Y─├SWAP─┤  State
    3: ────╰X────╰SWAP─┤  State

    """

    num_wires = qml.operation.AnyWires

    def _flatten(self):
        return (self.ops), (self.control)

    @classmethod
    def _unflatten(cls, data, metadata) -> "Select":
        return cls(data, metadata)

    def __repr__(self):
        return f"Select(ops={self.ops}, control={self.control})"

    def __init__(self, ops, control, id=None):
        control = qml.wires.Wires(control)
        self.hyperparameters["ops"] = tuple(ops)
        self.hyperparameters["control"] = control

        if 2 ** len(control) < len(ops):
            raise ValueError(
                f"Not enough control wires ({len(control)}) for the desired number of "
                + f"operations ({len(ops)}). At least {int(math.ceil(math.log2(len(ops))))} control "
                + "wires required."
            )

        if any(
            control_wire in qml.wires.Wires.all_wires([op.wires for op in ops])
            for control_wire in control
        ):
            raise ValueError("Control wires should be different from operation wires.")

        for op in ops:
            qml.QueuingManager.remove(op)

        target_wires = qml.wires.Wires.all_wires([op.wires for op in ops])
        self.hyperparameters["target_wires"] = target_wires

        all_wires = target_wires + control
        super().__init__(*self.data, wires=all_wires, id=id)

    def map_wires(self, wire_map: dict) -> "Select":
        new_ops = [o.map_wires(wire_map) for o in self.hyperparameters["ops"]]
        new_control = [wire_map.get(wire, wire) for wire in self.hyperparameters["control"]]
        return Select(new_ops, new_control)

    def __copy__(self):
        """Copy this op"""
        cls = self.__class__
        copied_op = cls.__new__(cls)

        new_data = copy.copy(self.data)

        for attr, value in vars(self).items():
            if attr != "data":
                setattr(copied_op, attr, value)

        copied_op.data = new_data

        return copied_op

    @property
    def data(self):
        """Create data property"""
        return tuple(d for op in self.ops for d in op.data)

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        for op in self.ops:
            op_num_params = op.num_params
            if op_num_params > 0:
                op.data = new_data[:op_num_params]
                new_data = new_data[op_num_params:]

    def decomposition(self):
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n

        A ``DecompositionUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_decomposition`.

        Returns:
            list[Operator]: decomposition of the operator

        **Example**

        >>> ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2,3])]
        >>> op = qml.Select(ops, control=[0,1])
        >>> op.decomposition()
        [MultiControlledX(wires=[0, 1, 2], control_values=[0, 0]),
         MultiControlledX(wires=[0, 1, 3], control_values=[0, 1]),
         Controlled(Y(2), control_wires=[0, 1], control_values=[True, False]),
         Controlled(SWAP(wires=[2, 3]), control_wires=[0, 1])]
        """
        return self.compute_decomposition(self.ops, control=self.control)

    @staticmethod
    def compute_decomposition(
        ops,
        control,
    ):  # pylint: disable=arguments-differ, unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            ops (list[Operator]): operations to apply
            control (Sequence[int]): the wires controlling which operation is applied

        Returns:
            list[Operator]: decomposition of the operator

        **Example**

        >>> ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2,3])]
        >>> qml.Select.compute_decomposition(ops, control=[0,1])
        [MultiControlledX(wires=[0, 1, 2], control_values=[0, 0]),
         MultiControlledX(wires=[0, 1, 3], control_values=[0, 1),
         Controlled(Y(2), control_wires=[0, 1], control_values=[True, False]),
         Controlled(SWAP(wires=[2, 3]), control_wires=[0, 1])]
        """
        states = list(itertools.product([0, 1], repeat=len(control)))
        decomp_ops = [
            qml.ctrl(op, control, control_values=states[index]) for index, op in enumerate(ops)
        ]
        return decomp_ops

    @property
    def ops(self):
        """Operations to be applied."""
        return self.hyperparameters["ops"]

    @property
    def control(self):
        """The control wires."""
        return self.hyperparameters["control"]

    @property
    def target_wires(self):
        """The wires of the input operators."""
        return self.hyperparameters["target_wires"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.hyperparameters["control"] + self.hyperparameters["target_wires"]

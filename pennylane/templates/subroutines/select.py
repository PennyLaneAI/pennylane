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

import pennylane as qml
from pennylane.operation import Operation
from pennylane import math


class Select(Operation):
    r"""Applies specific input operations depending on the state of
    the designated control qubits.

    .. math:: Select|i\rangle \otimes |\psi\rangle = |i\rangle \otimes U_i |\psi\rangle

    Args:
        ops (list[Operator]): operations to apply
        control_wires (Sequence[int]): the wires controlling which operation is applied
        id (str or None): String representing the operation (optional)

    .. note::
        The position of the operation in the list determines which qubit state implements that operation.
        For example, when the qubit register is in the state :math:`|00\rangle`, we will apply ``ops[0]``.
        When the qubit register is in the state :math:`|10\rangle`, we will apply ``ops[2]``. To obtain the
        binary bitstring representing the state for list position ``index`` we can use the following relationship:
        ``index = int(state_string, 2)``. For example, ``2 = int('10', 2)``.

    **Example**

    >>> dev = qml.device('default.qubit',wires=4)
    >>> ops = [qml.PauliX(wires=2),qml.PauliX(wires=3),qml.PauliY(wires=2),qml.SWAP([2,3])]
    >>> @qml.qnode(dev)
    >>> def circuit():
    >>>     qml.Select(ops,control_wires=[0,1])
    >>>     return qml.state()
    ...
    >>> print(qml.draw(circuit,expansion_strategy='device')())
    0: ─╭○─╭○─╭●─╭●────┤  State
    1: ─├○─├●─├○─├●────┤  State
    2: ─╰X─│──╰Y─├SWAP─┤  State
    3: ────╰X────╰SWAP─┤  State

    """

    num_wires = qml.operation.AnyWires

    def _flatten(self):
        return tuple(self.ops), tuple()

    @classmethod
    def _unflatten(cls, data, _) -> "Select":
        return cls(*data)

    def __init__(self, ops, control_wires, id=None):
        self.ops = ops
        control_wires = qml.wires.Wires(control_wires)
        if 2 ** len(control_wires) < len(ops):
            raise ValueError(
                f"Not enough control wires ({len(control_wires)}) for the desired number of "
                + f"operations ({len(ops)}). At least {int(math.ceil(math.log2(len(ops))))} control "
                + "wires required."
            )

        if any(
            control_wire in qml.wires.Wires.all_wires([op.wires for op in ops])
            for control_wire in control_wires
        ):
            raise ValueError("Control wires should be different from operation wires.")

        for op in ops:
            qml.QueuingManager.remove(op)

        all_wires = qml.wires.Wires.all_wires([op.wires for op in ops]) + control_wires
        super().__init__(ops, control_wires, wires=all_wires, id=id)

    @staticmethod
    def compute_decomposition(
        ops, control_wires, wires=None
    ):  # pylint: disable=arguments-differ, unused-argument
        states = [
            [int(i) for i in list(bitstring)]
            for bitstring in [
                format(i, f"0{len(control_wires)}b") for i in range(2 ** len(control_wires))
            ]
        ]
        decomp_ops = [
            qml.ctrl(op, control_wires, control_values=states[index])
            for index, op in enumerate(ops)
        ]
        return decomp_ops

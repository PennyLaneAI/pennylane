# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the FlipSign template.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class FlipSign(Operation):
    r"""FlipSign operator flips the sign for a given state.

    In a nutshell, this class perform the following operation:

    FlipSign(n):math:`|m\rangle = -|m\rangle` if len(m) = len(n)
    FlipSign(n):math:`|m\rangle = |m\rangle` if len(m) != len(n)

    Where m is the state to flip and n is the array of qubits (wires).
    It flips the sign of the state.

    Args:
        wires (array[int]): wires that the operator acts on
        bin_arr (array[int]): binary array vector representing the state to flip the sign

    Raises:
        ValueError: "expected at integer binary array "
        ValueError: "expected at integer binary array for wires "
        ValueError: "expected at integer binary array not empty "
        ValueError: "expected at least one wire representing the qubit "

    .. seealso:: :func:`~.relevant_func`, :class:`~.RelevantClass` (optional)

    .. details::

        :title: Usage Details

        The template is used inside a qnode.
        The number of shots has to be explicitly set on the device when using sample-based measurements:

        .. code-block:: python

            dev = qml.device("default.qubit", wires=5, shots = 1000)

            @qml.qnode(dev)
            def circuit():
               for wire in list(range(5)):
                    qml.Hadamard(wires = wire)
               qml.FlipSign([1,0,1,0,0], wires = list(range(5)))
               for wire in list(range(5)):
                    qml.Hadamard(wires = wire)
               return qml.sample()

            drawer = qml.draw(circuit, show_all_wires = True)

        The result for the above circuit is:

            >>> print(drawer())
            0: ──H─╭FlipSign──H─┤  Sample
            1: ──H─├FlipSign──H─┤  Sample
            2: ──H─├FlipSign──H─┤  Sample
            3: ──H─├FlipSign──H─┤  Sample
            4: ──H─╰FlipSign──H─┤  Sample

    """

    num_wires = AnyWires

    def __init__(self, bin_arr, wires, do_queue=True, id=None):

        if not isinstance(bin_arr, list):
            raise ValueError("expected at integer binary array ")

        if np.array(bin_arr).dtype != np.dtype("int"):
            raise ValueError("expected at integer binary array ")

        if not isinstance(wires, list):
            raise ValueError("expected at integer binary array for wires ")

        if np.array(wires).dtype != np.dtype("int"):
            raise ValueError("expected a integer binary array for wires ")

        if len(bin_arr) == 0:
            raise ValueError("expected at integer binary array not empty ")

        if len(wires) == 0:
            raise ValueError("expected at least one wire representing the qubit ")

        self._hyperparameters = {"bin_arr": bin_arr}
        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(wires, bin_arr):
        r"""Representation of the operator

        .. math::

        .. seealso:: :meth:`~.FlipSign.decomposition`.

        Args:
            wires (array[int]): wires that the operator acts on
            bin_arr (array[int]): binary array vector representing the state to flip the sign

        Returns:
            list[Operator]: decomposition of the operator
        """

        op_list = []

        if len(wires) == len(bin_arr):
            if bin_arr[-1] == 0:
                op_list.append(qml.PauliX(wires[-1]))

            op_list.append(
                qml.ctrl(qml.PauliZ, control=wires[:-1], control_values=bin_arr[:-1])(
                    wires=wires[-1]
                )
            )

            if bin_arr[-1] == 0:
                op_list.append(qml.PauliX(wires[-1]))
        else:
            for wire in list(range(len(wires))):
                op_list.append(qml.Identity(wire))

        return op_list

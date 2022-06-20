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
from pennylane.ops import BasisState

class FlipSign(Operation):
    r"""FlipSign operator flips the sign for a given state.

    In a nutshell, this class perform the following operation:

    FlipSign(n)|m> = -|m> if m = n
    FlipSign(n)|m> = |m> if m != n

    Where m is the status length to flip and n is the length of the number of qubits.
    It flips the sign of the state.

    Args:
        wires (int): wires that the operator acts on
        bin_arr (array[int]): binary array vector representing the state to flip the sign

    Raises:
        ValueError: "expected at integer binary array "
        ValueError: "expected at integer binary array for wires "
        ValueError: "expected at integer binary array not empty "
        ValueError: "expected at least one wire representing the qubit "

    .. seealso:: :func:`~.relevant_func`, :class:`~.RelevantClass` (optional)

    .. details::

        :title: Usage Details

        The template is used inside a qnode:

        .. code-block:: python

            dev = qml.device("default.qubit", wires=1)

            @qml.qnode(dev)
            def circuit():
               qml.Hadamard(wires = 0)
               qml.FlipSign([1,0,1,0,0], wires = list(range(5)))
               qml.Hadamard(wires = 0)
               return qml.sample()

            drawer = qml.draw(circuit, show_all_wires = True)
            >>> print(drawer())
            0: ──H─╭FlipSign──H─┤  Sample
            1: ────├FlipSign────┤  Sample
            2: ────├FlipSign────┤  Sample
            3: ────├FlipSign────┤  Sample
            4: ────╰FlipSign────┤  Sample

    """

    num_wires = AnyWires

    def __init__(self, bin_arr, wires, do_queue=True, id=None):

        if not isinstance(bin_arr, list):
            raise ValueError(
                f"expected at integer binary array "
            )

        if np.array(bin_arr).dtype != np.dtype("int"):
            raise ValueError(
                f"expected at integer binary array "
            )

        if not isinstance(wires, list):
            raise ValueError(
                f"expected at integer binary array for wires "
            )

        if np.array(wires).dtype != np.dtype("int"):
            raise ValueError(
                f"expected a integer binary array for wires "
            )

        if len(bin_arr) == 0:
            raise ValueError(
                f"expected at integer binary array not empty "
            )

        if len(wires) == 0:
            raise ValueError(
                f"expected at least one wire representing the qubit "
            )

        self._hyperparameters = {"bin_arr": bin_arr}
        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def find_occur(value, arr):
        r"""Find index of occurrences for value in binary array.

        Args:
            value (int): value to search in binary arrays
            arr (array[int]): binary array to search into

        Returns:
            list[int]: value indexes of search value present in array
        """

        narray = np.array(arr)
        res_occur = list(np.where(narray == value)[0])
        return res_occur

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(wires, bin_arr):
        r"""Representation of the operator

        .. math::

        .. seealso:: :meth:`~.FlipSign.decomposition`.

        Args:
            wires (int): wires that the operator acts on
            bin_arr (array[int]): binary array vector representing the state to flip the sign

        Returns:
            list[Operator]: decomposition of the operator
        """

        zeros_idx = find_occur(0, bin_arr)

        op_list = []

        if len(zeros_idx) > 0:
            for wire in zeros_idx[:-1]:
                op_list.append(PauliX(wire))
                op_list.append(ctrl(PauliZ, control = wires[:-1], control_values = bin_arr[:-1])(wires = wire))
                op_list.append(PauliX(wire))
        else:
            for wire in list(range(len(wires))):
                op_list.append(Identity(wire))

        raise op_list

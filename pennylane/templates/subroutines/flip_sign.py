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
    r"""FlipSign operator flips the sign for a given basic state.

    In a nutshell, this class perform the following operation:

    FlipSign(n):math:`|m\rangle = -|m\rangle` if m = n
    FlipSign(n):math:`|m\rangle = |m\rangle` if m != n

    Where n is the basic state to flip and m is the input.
    It flips the sign of the state.

    Args:
        n (array[int] or int): binary array or integer value representing the state to flip the sign
        wires (array[int]): number of wires that the operator acts on

    Raises:
        ValueError: "expected an integer array for wires "
        ValueError: "expected at least one wire representing the qubit "
        ValueError: "expected an integer binary array or integer number for basic flipping state "
        ValueError: "expected an integer equal or greater than zero for basic flipping state "

    .. seealso:: :func:`~.relevant_func`, :class:`~.RelevantClass` (optional)

    .. details::

        :title: Usage Details

        The template is used inside a qnode.
        The number of shots has to be explicitly set on the device when using sample-based measurements:

        .. code-block:: python

            dev = qml.device("default.qubit", wires=4, shots = 1)

            @qml.qnode(dev)
            def circuit():
               for wire in list(range(4)):
                    qml.Hadamard(wires = wire)
               qml.FlipSign([1,0,0,0], wires = list(range(4)))
               return qml.sample()

            circuit()

        The result for the above circuit is:

            >>> print(circuit.draw())
            0: ──H─╭FlipSign───┤  Sample
            1: ──H─├FlipSign───┤  Sample
            2: ──H─├FlipSign───┤  Sample
            3: ──H─╰FlipSign───┤  Sample

    """

    num_wires = AnyWires

    def __init__(self, n, wires, do_queue=True, id=None):

        if type(wires) is not int and len(wires) == 0:
            raise ValueError("expected at least one wire representing the qubit ")

        if type(n) is int:
            if n >= 0:
                n = self.to_list(n, len(wires))
            else:
                raise ValueError(
                    "expected an integer equal or greater than zero for basic flipping state"
                )
        else:

            if self.is_list_typeof(n, int):
                n = self.to_number(n)
            else:
                raise ValueError(
                    "expected an integer binary array or integer number for basic flipping state "
                )

        self._hyperparameters = {"arr_bin": n}
        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def is_list_typeof(arr, type_val):
        r"""Check if array is for a given type or not
        Args:
            type_val (type): Type to infer and check for array
            arr (array[object]): Integer binary array with regarding basis state

        Returns:
            (bool): boolean that checks whether array is binary or not
        """
        return np.array_equal(arr, np.array(arr).astype(type_val))

    @staticmethod
    def to_list(n, n_wires):
        r"""Convert an integer into a binary integer list
        Args:
            n (int): Basis state as integer
            n_wires (int): Numer of wires to transform the basis state

        Raises:
            ValueError: "cannot encode n with n wires "

        Returns:
            (array[int]): integer binary array
        """
        if n >= 2**n_wires:
            raise ValueError(f"cannot encode {n} with {n_wires} wires ")

        b_str = f"{n:b}".zfill(n_wires)
        bin_list = [int(i) for i in b_str]
        return bin_list

    @staticmethod
    def to_number(arr_bin):
        r"""Convert a binary array to integer number

        Args:
            arr_bin (array[int]): Integer binary array that represent the basis state
        Returns:
            (int): integer number
        """
        return sum([arr_bin[i] * 2 ** (len(arr_bin) - i - 1) for i in range(len(arr_bin))])

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(wires, arr_bin):  # pylint: disable=arguments-differ
        r"""Representation of the operator

        .. seealso:: :meth:`~.FlipSign.decomposition`.

        Args:
            wires (array[int]): wires that the operator acts on
            arr_bin (array[int]): binary array vector representing the state to flip the sign

        Raises:
            ValueError: "Wires length and flipping state length does not match, they must be equal length "

        Returns:
            list[Operator]: decomposition of the operator
        """

        op_list = []

        if len(wires) == len(arr_bin):
            if arr_bin[-1] == 0:
                op_list.append(qml.PauliX(wires[-1]))

            op_list.append(
                qml.ctrl(qml.PauliZ, control=wires[:-1], control_values=arr_bin[:-1])(
                    wires=wires[-1]
                )
            )

            if arr_bin[-1] == 0:
                op_list.append(qml.PauliX(wires[-1]))
        else:
            raise ValueError(
                "Wires length and flipping state length does not match, they must be equal length "
            )

        return op_list

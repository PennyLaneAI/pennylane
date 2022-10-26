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

import pennylane as qml
from pennylane.operation import Operation, AnyWires


class FlipSign(Operation):
    r"""FlipSign operator flips the sign for a given basic state.

    In a nutshell, this class perform the following operation:

    FlipSign(n) :math:`|m\rangle = -|m\rangle` if :math:`m = n`

    FlipSign(n) :math:`|m\rangle = |m\rangle` if :math:`m \not = n`,

    where n is the basic state to flip and m is the input.

    Args:
        n (array[int] or int): binary array or integer value representing the state to flip the sign
        wires (array[int]): number of wires that the operator acts on


    .. details::
        :title: Usage Details

        The template is used inside a qnode.
        The number of shots has to be explicitly set on the device when using sample-based measurements:

        .. code-block:: python

            basis_state = [1, 0]

            dev = qml.device("default.qubit", wires=2)


            @qml.qnode(dev)
            def circuit():
                for wire in list(range(2)):
                    qml.Hadamard(wires=wire)
                qml.FlipSign(basis_state, wires=list(range(2)))
                return qml.state()


            circuit()

        The result for the above circuit is:

        .. code-block:: python

            >>> print(circuit())
            [ 0.5+0.j  0.5+0.j -0.5+0.j  0.5+0.j]

    """

    num_wires = AnyWires

    def __init__(self, n, wires, do_queue=True, id=None):

        if not isinstance(wires, int) and len(wires) == 0:
            raise ValueError("expected at least one wire representing the qubit ")

        if isinstance(wires, int):
            wires = 1 if wires == 0 else wires
            wires = list(range(wires))

        if isinstance(n, int):
            if n >= 0:
                n = self.to_list(n, len(wires))
            else:
                raise ValueError(
                    "expected an integer equal or greater than zero for basic flipping state"
                )

        if len(wires) != len(n):
            raise ValueError(
                "Wires length and flipping state length does not match, they must be equal length "
            )

        self._hyperparameters = {"arr_bin": n}
        super().__init__(wires=wires, do_queue=do_queue, id=id)

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

        if arr_bin[-1] == 0:
            op_list.append(qml.PauliX(wires[-1]))

        op_list.append(
            qml.ctrl(qml.PauliZ, control=wires[:-1], control_values=arr_bin[:-1])(wires=wires[-1])
        )

        if arr_bin[-1] == 0:
            op_list.append(qml.PauliX(wires[-1]))

        return op_list

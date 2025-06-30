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
import warnings

import pennylane as qml
from pennylane.operation import Operation


class FlipSign(Operation):
    r"""Flips the sign of a given basis state.

    This template performs the following operation:

    FlipSign(n) :math:`|m\rangle = -|m\rangle` if :math:`m = n`

    FlipSign(n) :math:`|m\rangle = |m\rangle` if :math:`m \not = n`,

    where n is the basis state to flip and m is the input.

    Args:
        n (array[int] or int): binary array or integer value representing the state on which to flip the sign
        wires (array[int] or int): wires that the template acts on

    .. warning::

        Passing an integer `m` as the wires argument is now interpreted as a single wire (i.e. wires=[`m`]).
        This is different from the previous interpretation of wires=range(`m`). This may lead to unexpected errors.


    **Example**

    This template changes the sign of the basis state passed as an argument.
    In this example, when passing the element ``[1, 0]``, we will change the sign of the state :math:`|10\rangle`.
    We could alternatively pass the integer ``2`` and get the same result since its binary representation is ``[1, 0]``.

    .. code-block:: python

        num_wires = 2
        dev = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(dev)
        def circuit():
            for wire in range(num_wires):
                qml.Hadamard(wire)
            qml.FlipSign([1, 0], wires=range(num_wires))
            return qml.state()

    The result for the above circuit is:

    .. code-block:: python

        >>> circuit()
        array([ 0.5+0.j,  0.5+0.j, -0.5+0.j,  0.5+0.j])

    """

    def _flatten(self):
        hyperparameters = (("n", tuple(self.hyperparameters["arr_bin"])),)
        return tuple(), (self.wires, hyperparameters)

    def __repr__(self):
        return f"FlipSign({self.hyperparameters['arr_bin']}, wires={self.wires.tolist()})"

    def __init__(self, n, wires, id=None):
        if not isinstance(wires, int) and len(wires) == 0:
            raise ValueError("At least one valid wire is required.")

        if isinstance(wires, int):
            warnings.warn(
                "Passing an integer `m` as the wires argument is now interpreted as a single wire (i.e. wires=[`m`])."
                "This is different from the previous interpretation of wires=range(`m`). This may lead to unexpected errors."
            )
            wires = [wires]

        if isinstance(n, int):
            if n < 0:
                raise ValueError("The given basis state cannot be a negative integer number.")
            n = self.to_list(n, len(wires))

        n = tuple(n)

        if len(wires) != len(n):
            raise ValueError(f"The basis state {n} and wires {wires} must be of equal length.")

        self._hyperparameters = {"arr_bin": n}
        super().__init__(wires=wires, id=id)

    @staticmethod
    def to_list(n, n_wires):
        r"""Convert the given basis state from integer number into list of bits.

        Args:
            n (int): basis state as integer number
            n_wires (int): number of wires

        Raises:
            ValueError: "Cannot encode basis state ``n`` on ``n_wires`` wires."

        Returns:
            list[int]: basis state as list of bits
        """
        if n >= 2**n_wires:
            raise ValueError(f"Cannot encode basis state {n} on {n_wires} wires.")

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
            op_list.append(qml.X(wires[-1]))

        op_list.append(qml.ctrl(qml.Z(wires[-1]), control=wires[:-1], control_values=arr_bin[:-1]))

        if arr_bin[-1] == 0:
            op_list.append(qml.X(wires[-1]))

        return op_list

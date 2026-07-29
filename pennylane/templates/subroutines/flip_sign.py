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

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import X, Z, ctrl
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.typing import Wire
from pennylane.wires import Wires


class FlipSign(Operator2):
    r"""Flips the sign of a given basis state.

    This template performs the following operation:

    FlipSign(n) :math:`|m\rangle = -|m\rangle` if :math:`m = n`

    FlipSign(n) :math:`|m\rangle = |m\rangle` if :math:`m \not = n`,

    where n is the basis state to flip and m is the input.

    Args:
        n (array[int] or int): binary array or integer value representing the state on which to flip the sign
        wires (array[int] or int): wires that the template acts on

    **Example**

    This template changes the sign of the basis state passed as an argument.
    In this example, when passing the element ``[1, 0]``, we will change the sign of the state :math:`|10\rangle`.
    We could alternatively pass the integer ``2`` and get the same result since its binary representation is ``[1, 0]``.

    .. code-block:: python

        num_wires = 2
        dev = qp.device("default.qubit", wires=num_wires)

        @qp.qnode(dev)
        def circuit():
            for wire in range(num_wires):
                qp.Hadamard(wire)
            qp.FlipSign([1, 0], wires=range(num_wires))
            return qp.state()

    The result for the above circuit is:

    >>> circuit()
    array([ 0.5+0.j,  0.5+0.j, -0.5+0.j,  0.5+0.j])

    """

    wires_argnames = ("wires",)
    compilable_argnames = ("n",)
    arg_specs = {"wires": Wire[-1]}
    wire_sizes = (None,)

    def __init__(self, n: int, wires):  # TODO: DO we rename n->state?
        wires = Wires(wires)
        num_wires = len(wires)
        if num_wires == 0:
            raise ValueError("At least one wire is required.")

        if isinstance(n, int):
            if not 0 <= n < 2**num_wires:
                raise ValueError(
                    "The given basis state must be a non-negative integer smaller "
                    f"than {2**num_wires}, but got {n}."
                )
            n = tuple(map(int, math.int_to_binary(n, num_wires)))
        else:
            if num_wires != len(n):
                raise ValueError(f"The basis state {n} and wires {wires} must be of equal length.")
            n = tuple(n)

        super().__init__(n, wires)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(n, wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator

        .. seealso:: :meth:`~.FlipSign.decomposition`.

        Args:
            n (tuple[int]): binary array vector representing the state to flip the sign on
            wires (WiresLike): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator
        """

        op_list = []

        if n[-1] == 0:
            op_list.append(X(wires[-1]))

        op_list.append(ctrl(Z(wires[-1]), control=wires[:-1], control_values=n[:-1]))

        if n[-1] == 0:
            op_list.append(X(wires[-1]))

        return op_list


def _flip_sign_resources(n, wires):
    num_wires = len(wires)
    num_ctrl_wires = num_wires - 1
    num_zeros = num_ctrl_wires - sum(n[:-1])
    res = {_ctrl_abstract(Z, Wire[num_ctrl_wires], num_zero_control_values=num_zeros): 1}
    if n[-1] == 0:
        res[X] = 2
    return res


@register_resources(_flip_sign_resources)
def _flip_sign_decomposition(n, wires):
    if n[-1] == 0:
        X(wires[-1])

    ctrl(Z(wires[-1]), control=wires[:-1], control_values=n[:-1])

    if n[-1] == 0:
        X(wires[-1])


add_decomps(FlipSign, _flip_sign_decomposition)

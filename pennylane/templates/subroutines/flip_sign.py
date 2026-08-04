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
from pennylane.wires import Wires, WiresLike


class FlipSign(Operator2):
    r"""Flips the sign of a given basis state.

    This template performs the following operation:

    FlipSign(n) :math:`|m\rangle = -|m\rangle` if :math:`m = n`

    FlipSign(n) :math:`|m\rangle = |m\rangle` if :math:`m \not = n`,

    where :math:`n` is the basis state (argument ``state``) to flip and :math:`m` is the input.

    Args:
        state (tuple[int] or list[int] or int): integer or binary sequence representing
            the basis state whose sign is to be flipped
        wires (WiresLike): wires that the template acts on

    **Example**

    This template changes the sign of the basis state passed as an argument. In this example,
    when passing the element ``[1, 0]``, we will change the sign of the state :math:`|10\rangle`
    on two qubits. We could alternatively pass the integer ``2`` and get the same result since
    its two-bit binary representation is ``[1, 0]``.

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

    compilable_argnames = ("state",)
    arg_specs = {"wires": Wire[-1]}
    wire_sizes = (None,)

    @staticmethod
    def _canonicalize_state(state: int | list[int] | tuple[int], num_wires: int) -> tuple[int]:
        """Canonicalize the input state into a tuple of integers."""
        if isinstance(state, int):
            if not 0 <= state < 2**num_wires:
                raise ValueError(
                    "The given basis state must be a non-negative integer smaller "
                    f"than {2**num_wires}, but got {state}."
                )
            return tuple(map(int, math.int_to_binary(state, num_wires)))
        if num_wires != len(state):
            raise ValueError(
                "The basis state and wires must have equal length, "
                f"but got {len(state)} and {num_wires}."
            )
        return tuple(state)

    def __init__(self, state: int | list[int] | tuple[int], wires: WiresLike):
        wires = Wires(wires)
        num_wires = len(wires)
        if num_wires == 0:
            raise ValueError("At least one wire is required.")
        state = self._canonicalize_state(state, num_wires)
        super().__init__(state, wires)

    def __abstract_init__(self, state, wires):  # pylint: disable=arguments-differ
        state = self._canonicalize_state(state, len(wires))
        super().__abstract_init__(state, wires)


def _flip_sign_resources(state: tuple[int], wires: WiresLike):
    num_wires = len(wires)
    num_ctrl_wires = num_wires - 1
    num_zeros = num_ctrl_wires - sum(state[:-1])
    res = {_ctrl_abstract(Z, Wire[num_ctrl_wires], num_zero_control_values=num_zeros): 1}
    if state[-1] == 0:
        res[X] = 2
    return res


@register_resources(_flip_sign_resources)
def _flip_sign_decomposition(state: tuple[int], wires: WiresLike):
    if state[-1] == 0:
        X(wires[-1])

    if len(wires) == 1:
        Z(wires[-1])
    else:
        ctrl(Z(wires[-1]), control=wires[:-1], control_values=values_state[:-1])

    if state[-1] == 0:
        X(wires[-1])


add_decomps(FlipSign, _flip_sign_decomposition)

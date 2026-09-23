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

from collections.abc import Sequence
from typing import Literal

from pennylane import math
from pennylane.core.operator import Operator2, abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.resources import resolve_work_wire_type
from pennylane.ops import X, Z
from pennylane.ops.op_math.condition import cond
from pennylane.ops.op_math.controlled2 import ControlledOp2, _validate_work_wire_type
from pennylane.typing import AbstractArray, Int, Wire
from pennylane.wires import Wires, WiresLike, concatenate_wires


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
        work_wires (WiresLike): optional auxiliary wires that can be used in the decomposition
            of the multi-controlled :class:`~.Z` gate. They are restored to their original state.
        work_wire_type (str): whether the work wires are ``"zeroed"`` or ``"borrowed"``.
            ``"zeroed"`` indicates that the work wires are in the :math:`|0\rangle` state;
            ``"borrowed"`` work wires can be in any arbitrary state. Defaults to ``"zeroed"``.

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

    dynamic_argnames = ("state",)
    compilable_argnames = ("work_wire_type",)
    wire_argnames = ("wires", "work_wires")
    arg_specs = {"state": Int[-1], "wires": Wire[-1], "work_wires": Wire[-1]}

    @staticmethod
    def _canonicalize_state(
        state: int | Sequence[int] | AbstractArray, num_wires: int
    ) -> tuple[int] | AbstractArray:
        """Canonicalize the input state into a tuple of integers."""

        if isinstance(state, int):
            if not 0 <= state < 2**num_wires:
                raise ValueError(
                    "The given basis state must be a non-negative integer smaller "
                    f"than {2**num_wires}, but got {state}."
                )
            return math.asarray(list(map(int, math.int_to_binary(state, num_wires))), dtype=int)

        if num_wires != len(state):
            raise ValueError(
                "The basis state and wires must have equal length, "
                f"but got {len(state)} and {num_wires}."
            )

        if isinstance(state, AbstractArray) or math.is_abstract(state):
            return state
        if isinstance(state, (list, tuple)) and state and math.is_abstract(state[0]):
            return math.stack([math.asarray(v) for v in state])
        return math.asarray([int(v) for v in state], dtype=int)

    def __init__(
        self,
        state: int | list[int] | tuple[int],
        wires: WiresLike,
        work_wires: WiresLike = None,
        work_wire_type: Literal["zeroed", "borrowed"] = "zeroed",
    ):
        wires = Wires(wires)
        num_wires = len(wires)
        if num_wires == 0:
            raise ValueError("At least one wire is required.")
        state = self._canonicalize_state(state, num_wires)
        work_wires = () if work_wires is None else work_wires
        _validate_work_wire_type(work_wire_type)
        super().__init__(state, wires, work_wires, work_wire_type)


def _is_abstract_state(state) -> bool:
    if isinstance(state, AbstractArray) or math.is_abstract(state):
        return True
    # Capture may pass a tuple of tracers; the container itself is not abstract.
    if isinstance(state, (list, tuple)) and state:
        return math.is_abstract(state[0])
    return False


def _controlled_z_rep(num_ctrl_wires, num_work_wires, work_wire_type):
    """Resource key for a multi-controlled Z, matching ``ControlledOp2`` abstractify."""
    target = num_ctrl_wires
    controls = list(range(num_ctrl_wires))
    work = list(range(target + 1, target + 1 + num_work_wires))
    return abstractify(
        ControlledOp2(
            Z(target),
            control_wires=controls,
            control_values=[1] * num_ctrl_wires,
            work_wires=work,
            work_wire_type=work_wire_type,
        )
    )


def _flip_sign_resources(
    state: tuple[int],
    wires: WiresLike,
    work_wires: WiresLike,
    work_wire_type: Literal["zeroed", "borrowed"] = "zeroed",
):
    num_wires = len(wires)
    num_ctrl_wires = num_wires - 1

    if num_ctrl_wires == 0:
        res = {Z: 1}
    else:
        # Always the Controlled(PauliZ) form so keys match ControlledOp2 emission.
        res = {_controlled_z_rep(num_ctrl_wires, len(work_wires), work_wire_type): 1}

    if _is_abstract_state(state) or state[-1] == 0:
        res[X] = 2
    return res


@register_resources(_flip_sign_resources, exact=False)
def _flip_sign_decomposition(
    state: tuple[int],
    wires: WiresLike,
    work_wires: WiresLike,
    work_wire_type: Literal["zeroed", "borrowed"] = "zeroed",
):
    def _flip_target():
        X(wires[-1])

    cond(math.equal(state[-1], 0), _flip_target)()

    if len(wires) == 1:
        Z(wires)
    else:
        # ControlledOp2 accepts traced control_values under capture; qp.ctrl does not.
        ControlledOp2(
            Z(wires[-1]),
            control_wires=wires[:-1],
            control_values=state[:-1],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    cond(math.equal(state[-1], 0), _flip_target)()


def _merge_flip_sign_state(control_values, base_state, n_wires):
    """Concatenate control values with the base FlipSign state for resource keys."""
    if _is_abstract_state(control_values) or _is_abstract_state(base_state):
        return Int[n_wires]
    return tuple(int(v) for v in control_values) + tuple(base_state)


def _ctrl_flip_sign_resource(base, control_wires, control_values, work_wires, work_wire_type):
    work_wire_type = resolve_work_wire_type(
        base.work_wires,
        base.work_wire_type,
        work_wires,
        work_wire_type,
    )
    n_wires = len(control_wires) + len(base.wires)
    n_work = len(work_wires) + len(base.work_wires)
    state = _merge_flip_sign_state(control_values, base.state, n_wires)
    return {
        FlipSign(
            state,
            Wire[n_wires],
            work_wires=Wire[n_work],
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_flip_sign_resource)
def _ctrl_flip_sign_to_flip_sign(base, control_wires, control_values, work_wires, work_wire_type):
    work_wire_type = resolve_work_wire_type(
        base.work_wires,
        base.work_wire_type,
        work_wires,
        work_wire_type,
    )
    # Always concatenate via math so capture tracers are handled.
    state = math.concatenate(
        [math.astype(math.atleast_1d(control_values), int), math.atleast_1d(base.state)]
    )

    FlipSign(
        state,
        concatenate_wires(control_wires, base.wires),
        work_wires=concatenate_wires(work_wires, base.work_wires),
        work_wire_type=work_wire_type,
    )


add_decomps(FlipSign, _flip_sign_decomposition)
add_decomps("C(FlipSign)", _ctrl_flip_sign_to_flip_sign)

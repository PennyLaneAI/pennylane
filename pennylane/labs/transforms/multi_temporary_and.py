# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Multi target X operation
"""

from typing import Literal

import pennylane as qp
from pennylane.ops.op_math.controlled import ControlledOp
from pennylane.ops.op_math.controlled_ops import _check_and_convert_control_values
from pennylane.wires import Wires, WiresLike


class MultiTemporaryAND(ControlledOp):
    r"""
    Multi target X gate :math:`\bigotimes_i X_i`

    The main benefit of this gate is that it has eficient controlled operators.
    In particular, we can use it to create a fanout operation ``qp.ctrl(MultiX(range(4)), control=[f"c{i}" for i in range(4)])``:

    .. code-block::

        c0: ─╭●─┤   ─╭●─╭●─╭●─╭●─┤
        c1: ─├●─┤   ─├●─├●─├●─├●─┤
        c2: ─├●─┤   ─├●─├●─├●─├●─┤
        c3: ─├●─┤   ─├●─├●─├●─├●─┤
         0: ─│X─┤ = ─╰X─│──│──│──┤
         1: ─│X─┤   ────╰X─│──│──┤
         2: ─│X─┤   ───────╰X─│──┤
         3: ─╰X─┤   ──────────╰X─┤

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        wires (Union[Wires, Sequence[int], or int]): The wires that are targeted by ``X`` operators.

    **Example**

    On its own, the operator simply applies :class:`~.X` gates to the provided wires.

    >>> n = 4
    >>> wires = range(n)
    >>> @qp.transforms.decompose(gate_set="X")
    >>> @qp.qnode(qp.device("default.qubit"))
    >>> def qnode():
    ...     MultiX(wires)
    ...     return qp.state()
    >>> print(qp.draw(qnode)())
    0: ──X─┤  State
    1: ──X─┤  State
    2: ──X─┤  State
    3: ──X─┤  State

    It can be used to construct a fanout operator using :func:`~.ctrl` like so:

    .. code-block:: python

        control = [f"c{i}" for i in range(n)]

        @qp.transforms.decompose(
            gate_set={"MultiControlledX": 1000}
        )
        @qp.qnode(qp.device("default.qubit"))
        def circuit_no_work():
            qp.ctrl(MultiX(wires), control=control)
            return qp.state()

    This yields a decomposition in terms of :class:`MultiControlledX` gates.

    >>> print(qp.draw(circuit_no_work)())
    c0: ─╭●─╭●─╭●─╭●─┤  State
    c1: ─├●─├●─├●─├●─┤  State
    c2: ─├●─├●─├●─├●─┤  State
    c3: ─├●─├●─├●─├●─┤  State
     0: ─╰X─│──│──│──┤  State
     1: ────╰X─│──│──┤  State
     2: ───────╰X─│──┤  State
     3: ──────────╰X─┤  State

    Things get interesting when we allow for additional work wires.
    This turns the control structure into a :class:`TemporaryAND` ladder
    and uses :class:`CNOT` operators to distribute the targets clevery from the last work wire, so we do not have
    to re-use the control structure multiple times:

    .. code-block:: python

        @qp.transforms.decompose(
            gate_set={
                "TemporaryAND": 1,
                "Adjoint(TemporaryAND)": 1,
                "CNOT": 1,
                "Toffoli": 10000,
            },
            num_work_wires=4,
        )
        @qp.qnode(qp.device("default.qubit"))
        def circuit_with_work():
            qp.ctrl(MultiX(wires), control=control)
            return qp.state()


    >>> print(qp.draw(circuit_with_work)())
    <DynamicWire>: ─╭Allocate─╭⊕─╭●─────────────────────●╮──⊕╮─╭Deallocate─┤  State
    <DynamicWire>: ─├Allocate─│──├⊕─╭●──────────────●╮──⊕┤───│─├Deallocate─┤  State
    <DynamicWire>: ─╰Allocate─│──│──├⊕─╭●─╭●─╭●─╭●──⊕┤───│───│─╰Deallocate─┤  State
               c0: ───────────├●─│──│──│──│──│──│────│───│──●┤─────────────┤  State
               c1: ───────────╰●─│──│──│──│──│──│────│───│──●╯─────────────┤  State
               c2: ──────────────╰●─│──│──│──│──│────│──●╯─────────────────┤  State
               c3: ─────────────────╰●─│──│──│──│───●╯─────────────────────┤  State
                0: ────────────────────╰X─│──│──│──────────────────────────┤  State
                1: ───────────────────────╰X─│──│──────────────────────────┤  State
                2: ──────────────────────────╰X─│──────────────────────────┤  State
                3: ─────────────────────────────╰X─────────────────────────┤  State
    """

    num_params = 0
    resource_keys = {
        "num_control_wires",
        "num_zero_control_values",
        "num_work_wires",
        "work_wire_type",
    }

    def __init__(
        self,
        wires: WiresLike,
        control_values: None | bool | list[bool] | int | list[int] = None,
        work_wires: WiresLike = (),
        work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
    ):
        wires = Wires(() if wires is None else wires)
        work_wires = Wires(() if work_wires is None else work_wires)
        self._validate_control_values(control_values)

        if len(wires) == 0:
            raise ValueError("Must specify the wires where the operation acts on")

        if len(wires) < 2:
            raise ValueError(
                f"MultiControlledX: wrong number of wires. {len(wires)} wire(s) given. "
                f"Need at least 2."
            )
        control_wires = wires[:-1]
        wires = wires[-1:]

        control_values = _check_and_convert_control_values(control_values, control_wires)

        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qp.X, wires=wires)
        super().__init__(
            base,
            control_wires=control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    @property
    def resource_params(self) -> dict:
        return {
            "num_control_wires": len(self.control_wires),
            "num_zero_control_values": len([val for val in self.control_values if not val]),
            "num_work_wires": len(self.work_wires),
            "work_wire_type": self.work_wire_type,
        }

    @staticmethod
    def _validate_control_values(control_values):
        if control_values is not None:
            if not (
                isinstance(control_values, (bool, int))
                or (
                    isinstance(control_values, (list, tuple))
                    and all(isinstance(val, (bool, int)) for val in control_values)
                )
            ):
                raise ValueError(f"control_values must be boolean or int. Got: {control_values}")


# Add controlled decomposition
def _multi_temporary_and_resources(
    *_,
    num_control_wires,
    **__,
):
    return {qp.resource_rep(qp.TemporaryAND): num_control_wires - 1}


@qp.register_condition(
    lambda *_, num_control_wires, num_work_wires, **__: num_work_wires >= num_control_wires - 1
)  # and work_wire_type == "zeroed")
@qp.register_resources(_multi_temporary_and_resources)
def _multi_temporary_and_decomp_with_work_wires(
    wires,
    control_values,
    work_wires,
    **__,
):
    """Controlled decomposition using TemporaryAND ladder (needs work wires)."""
    control_wires = wires[:-1]
    c = len(control_wires)
    if c == 1:
        # essentially, a CNOT, but with potentially different control value
        qp.ctrl(
            qp.X(wires[-1]),
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )
        return

    # build AND ladder
    num_needed = c - 1
    qp.TemporaryAND(
        wires=[control_wires[0], control_wires[1], work_wires[0]],
        control_values=(control_values[0], control_values[1]),
    )
    ii_ = 0

    for i in range(1, num_needed):
        _wires = (
            [work_wires[i - 1], control_wires[i + 1], work_wires[i]]
            if i < num_needed - 1
            else [work_wires[i - 1], control_wires[i + 1], wires[-1]]
        )
        qp.TemporaryAND(
            wires=_wires,
            control_values=(True, control_values[i + 1]),
        )
        ii_ += 1


qp.add_decomps(
    MultiTemporaryAND,
    _multi_temporary_and_decomp_with_work_wires,
)

# def _ctrl_no_work_resources(
#     *_,
#     base_params,
#     num_control_wires,
#     num_zero_control_values,
#     work_wire_type,
#     **__,
# ):
#     num_wires = base_params["num_wires"]
#     return {
#         qp.decomposition.controlled_resource_rep(
#             qp.X,
#             {},
#             num_control_wires=num_control_wires,
#             num_zero_control_values=num_zero_control_values,
#             num_work_wires=0,
#             work_wire_type=work_wire_type,
#         ): num_wires,
#     }


# @qp.register_condition(lambda num_control_wires, **_: num_control_wires > 1)
# @qp.register_resources(_ctrl_no_work_resources)
# def ctrl_decomp_no_work_wires(
#     *_,
#     control_wires,
#     control_values,
#     base,
#     **__,
# ):
#     """Controlled decomposition without work wires — emits multicontrolled X gates."""
#     base_wires = base.wires
#     for w in base_wires:
#         qp.ctrl(qp.X(w), control=control_wires, control_values=control_values)

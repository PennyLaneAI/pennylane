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

import pennylane as qp


class MultiX(qp.operation.Operator):
    r"""
    Multi target X gate :math:`\bigotimes_i X_i`

    The main benefit of this gate is that it has eficient controlled operators.
    In particular, we can use it to create a fanout operation ``qp.ctrl(MultiX(range(4)), control=[f"c{i}" for i in range(4)])``:

    .. code-block:: python

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
    resource_keys = {"num_wires"}

    def __init__(self, wires):
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}


def _resources(num_wires):
    return {qp.resource_rep(qp.X): num_wires}


@qp.register_resources(_resources)
def decomp(wires, **_):
    """Decomposition of uncontrolled MultiX operation"""
    for wire in wires:
        qp.X(wire)


qp.add_decomps(MultiX, decomp)


def _ctrl_and_ladder_resources(
    *_,
    base_params,
    num_control_wires,
    **__,
):
    num_wires = base_params["num_wires"]
    return {
        qp.resource_rep(qp.TemporaryAND): num_control_wires - 1,
        qp.decomposition.adjoint_resource_rep(qp.TemporaryAND): num_control_wires - 1,
        qp.resource_rep(qp.CNOT): num_wires,
    }


def _ctrl_work_wires(num_control_wires, **_):
    """Declare work wire requirements: (num_control_wires - 1) zeroed wires."""
    return {"zeroed": max(0, num_control_wires - 1)}


@qp.register_condition(lambda num_control_wires, **_: num_control_wires > 1)
@qp.register_resources(_ctrl_and_ladder_resources, work_wires=_ctrl_work_wires)
def ctrl_decomp_with_work_wires(
    *_,
    control_wires,
    control_values,
    work_wires,
    base,
    **__,
):
    """Controlled decomposition using TemporaryAND ladder (needs work wires)."""
    c = len(control_wires)
    if c == 1:
        base_op = qp.pytrees.unflatten(*qp.pytrees.flatten(base))
        qp.ctrl(
            base_op,
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )
        return

    num_needed = c - 1
    with qp.allocation.allocate(num_needed, state="zero", restored=True) as aw:
        # Forward pass: build AND ladder
        qp.TemporaryAND(
            wires=[control_wires[0], control_wires[1], aw[0]],
            control_values=(control_values[0], control_values[1]),
        )
        for i in range(1, num_needed):
            qp.TemporaryAND(
                wires=[aw[i - 1], control_wires[i + 1], aw[i]],
                control_values=(True, control_values[i + 1]),
            )

        # Apply base operation (CNOTs) controlled on last ancilla
        base_wires = base.wires
        for w in base_wires:
            qp.CNOT(wires=[aw[-1], w])

        # Reverse pass: uncompute AND ladder
        for i in range(num_needed - 1, 0, -1):
            qp.adjoint(qp.TemporaryAND)(
                wires=[aw[i - 1], control_wires[i + 1], aw[i]],
                control_values=(True, control_values[i + 1]),
            )
        qp.adjoint(qp.TemporaryAND)(
            wires=[control_wires[0], control_wires[1], aw[0]],
            control_values=(control_values[0], control_values[1]),
        )


def _ctrl_no_work_resources(
    *_,
    base_params,
    num_control_wires,
    num_zero_control_values,
    work_wire_type,
    **__,
):
    num_wires = base_params["num_wires"]
    return {
        qp.decomposition.controlled_resource_rep(
            qp.X,
            {},
            num_control_wires=num_control_wires,
            num_zero_control_values=num_zero_control_values,
            num_work_wires=0,
            work_wire_type=work_wire_type,
        ): num_wires,
    }


@qp.register_condition(lambda num_control_wires, **_: num_control_wires > 1)
@qp.register_resources(_ctrl_no_work_resources)
def ctrl_decomp_no_work_wires(
    *_,
    control_wires,
    control_values,
    base,
    **__,
):
    """Controlled decomposition without work wires — emits multicontrolled X gates."""
    base_wires = base.wires
    for w in base_wires:
        qp.ctrl(qp.X(w), control=control_wires, control_values=control_values)


qp.add_decomps("C(MultiX)", ctrl_decomp_with_work_wires, ctrl_decomp_no_work_wires)

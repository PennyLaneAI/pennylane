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
    Multi target X gate :math:`\bigotimes_i X_i`, which can be used to create a fanout operation via :func:`~.ctrl`.

    The main benefit of this gate is that it has efficient controlled operators.
    In particular, we can use it to create a fanout operation

    .. code-block:: python

        qp.ctrl(MultiX(range(4)),
                control=["c0, c1, c2, c3"]
                work_wires=["w0, w1, w2"]
                )

    with the following decompositions:

    .. code-block::

        c0: ─╭●─┤   ─╭●─╭●─╭●─╭●─┤   ─╭●────────────────────────────●╮─┤
        c1: ─├●─┤   ─├●─├●─├●─├●─┤   ─├●────────────────────────────●┤─┤
        c2: ─├●─┤   ─├●─├●─├●─├●─┤   ─│──╭●─────────────────────●╮───│─┤
        c3: ─├●─┤   ─├●─├●─├●─├●─┤   ─│──│──╭●──────────────●╮───│───│─┤
         0: ─│X─┤ = ─╰X─│──│──│──┤ = ─│──│──│──╭X────────────│───│───│─┤
         1: ─│X─┤   ────╰X─│──│──┤   ─│──│──│──│──╭X─────────│───│───│─┤
         2: ─│X─┤   ───────╰X─│──┤   ─│──│──│──│──│──╭X──────│───│───│─┤
         3: ─╰X─┤   ──────────╰X─┤   ─│──│──│──│──│──│──╭X───│───│───│─┤
        w0:                          ─╰⊕─├●─│──│──│──│──│────│──●┤──⊕╯─┤
        w1:                          ────╰⊕─├●─│──│──│──│───●┤──⊕╯─────┤
        w2:                          ───────╰⊕─╰●─╰●─╰●─╰●──⊕╯─────────┤

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        wires (Union[Wires, Sequence[int], or int]): The wires that are targeted by ``X`` operators.

    **Example**

    On its own, the operator simply applies :class:`~.X` gates to the provided wires.

    >>> import pennylane as qp
    >>> qp.decomposition.enable_graph()
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

        n = 4
        control = list(range(n))
        wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n, 3 * n -1))

        all_wires = control + wires + work_wires

        @qp.qjit(capture=False)
        @qp.transforms.decompose(
            gate_set={
                "StatePrep",
                "TemporaryAND",
                "Adjoint(TemporaryAND)",
                "CNOT",
            }
        )
        @qp.qnode(qp.device("null.qubit", wires=all_wires))
        def qnode():
            qp.ctrl(MultiX(wires), control=control, work_wires=work_wires)
            return qp.state()


    >>> print(qp.draw(qnode)())
     0: ─╭●────────────────────────────●╮─┤  State
     1: ─├●────────────────────────────●┤─┤  State
     2: ─│──╭●─────────────────────●╮───│─┤  State
     3: ─│──│──╭●──────────────●╮───│───│─┤  State
     4: ─│──│──│──╭X────────────│───│───│─┤  State
     5: ─│──│──│──│──╭X─────────│───│───│─┤  State
     6: ─│──│──│──│──│──╭X──────│───│───│─┤  State
     7: ─│──│──│──│──│──│──╭X───│───│───│─┤  State
     8: ─╰⊕─├●─│──│──│──│──│────│──●┤──⊕╯─┤  State
     9: ────╰⊕─├●─│──│──│──│───●┤──⊕╯─────┤  State
    10: ───────╰⊕─╰●─╰●─╰●─╰●──⊕╯─────────┤  State

    In the previous example, we explicitly provided the required ``work_wires``, but we can also let
    the decompose transform automatically assign the necessary amount of additional work wires.

    .. code-block:: python

        @qp.transforms.decompose(
            gate_set={
                "TemporaryAND",
                "Adjoint(TemporaryAND)",
                "CNOT",
            },
            num_work_wires=n-1,
        )
        @qp.qnode(qp.device("default.qubit", wires=all_wires))
        def circuit_with_work():
            qp.ctrl(MultiX(wires), control=control)
            return qp.state()


    >>> print(qp.draw(circuit_with_work)())
                0: ───────────╭●────────────────────────────●╮─────────────┤  State
                1: ───────────├●────────────────────────────●┤─────────────┤  State
                2: ───────────│──╭●─────────────────────●╮───│─────────────┤  State
                3: ───────────│──│──╭●──────────────●╮───│───│─────────────┤  State
                4: ───────────│──│──│──╭X────────────│───│───│─────────────┤  State
                5: ───────────│──│──│──│──╭X─────────│───│───│─────────────┤  State
                6: ───────────│──│──│──│──│──╭X──────│───│───│─────────────┤  State
                7: ───────────│──│──│──│──│──│──╭X───│───│───│─────────────┤  State
    <DynamicWire>: ─╭Allocate─╰⊕─├●─│──│──│──│──│────│──●┤──⊕╯─╭Deallocate─┤  State
    <DynamicWire>: ─├Allocate────╰⊕─├●─│──│──│──│───●┤──⊕╯─────├Deallocate─┤  State
    <DynamicWire>: ─╰Allocate───────╰⊕─╰●─╰●─╰●─╰●──⊕╯─────────╰Deallocate─┤  State
    """

    num_params = 0
    resource_keys = {"num_wires"}

    def __init__(self, wires):
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}


# Add decomposition
def _resources(num_wires):
    return {qp.resource_rep(qp.X): num_wires}


@qp.register_resources(_resources)
def decomp(wires, **_):
    """Decomposition of uncontrolled MultiX operation"""
    for wire in wires:
        qp.X(wire)


qp.add_decomps(MultiX, decomp)


# Add controlled decomposition
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
    return {"zeroed": num_control_wires - 1}


@qp.register_condition(lambda num_control_wires, **_: num_control_wires > 1)
@qp.register_resources(_ctrl_and_ladder_resources, work_wires=_ctrl_work_wires)
def ctrl_decomp_with_allocate(
    *_,
    control_wires,
    control_values,
    base,
    **__,
):
    """Controlled decomposition using TemporaryAND ladder (needs work wires)."""
    c = len(control_wires)

    base_wires = base.wires
    num_needed = c - 1
    with qp.allocation.allocate(num_needed, state="zero", restored=True) as aw:
        _fanout(num_needed, base_wires, control_wires, control_values, work_wires=aw)


def _enough_work_wires(*_, num_work_wires, num_control_wires, **__):
    """Declare work wire requirements: (num_control_wires - 1) zeroed wires."""
    return num_work_wires >= num_control_wires - 1


@qp.register_condition(lambda num_control_wires, **_: num_control_wires > 1)
@qp.register_condition(_enough_work_wires)
@qp.register_resources(_ctrl_and_ladder_resources)
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

    base_wires = base.wires
    num_needed = c - 1

    _fanout(num_needed, base_wires, control_wires, control_values, work_wires)


def _fanout(num_needed, base_wires, control_wires, control_values, work_wires):
    qp.TemporaryAND(
        wires=[control_wires[0], control_wires[1], work_wires[0]],
        control_values=(control_values[0], control_values[1]),
    )
    for i in range(1, num_needed):
        qp.TemporaryAND(
            wires=[work_wires[i - 1], control_wires[i + 1], work_wires[i]],
            control_values=(True, control_values[i + 1]),
        )

    # Apply base operation (CNOTs) controlled on last ancilla
    for w in base_wires:
        qp.CNOT(wires=[work_wires[-1], w])

    # Reverse pass: uncompute AND ladder
    for i in range(num_needed - 1, 0, -1):
        qp.adjoint(qp.TemporaryAND)(
            wires=[work_wires[i - 1], control_wires[i + 1], work_wires[i]],
            control_values=(True, control_values[i + 1]),
        )
    qp.adjoint(qp.TemporaryAND)(
        wires=[control_wires[0], control_wires[1], work_wires[0]],
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


qp.add_decomps(
    "C(MultiX)", ctrl_decomp_with_work_wires, ctrl_decomp_with_allocate, ctrl_decomp_no_work_wires
)

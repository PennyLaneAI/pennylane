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
Factory that produces a decomposition rule for CRZ in terms of
`phase gradient states <https://pennylane.ai/compilation/phase-gradient/c-control-rotations>`__
"""

import numpy as np

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.decomposition import change_op_basis_resource_rep
from pennylane.ops import Prod
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.typing import Bool, Wire

from .rz_phase_gradient import validate_phase_gradient_wires


def make_crz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires):
    r"""
    Create a custom decomposition rule for :class:`~.CRZ` gates.

    This is a temporary workaround before moving to `capture` as default frontend, which unlocks dynamic wire allocation.
    Here, we explicitly provide the necessary wires for the
    `phase gradient decomposition of SelectPauliRot <https://pennylane.ai/compilation/phase-gradient/c-control-rotations>`__.
    This way, this function can be used in a workflow context that explicitly uses those wires to
    generate the decomposition rule, which can then be used
    as ``alt_decomps`` or ``fixed_decomp`` within :func:`~.pennylane.decompose` (with the
    graph-based decomposition system).

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state. Should have the same
            length as ``angle_wires``.
        work_wires (Wires): additional work wires for :class:`~.SemiAdder` decomposition.
            At least ``len(angle_wires)-1`` work wires are required.

    Returns:
        func: decomposition rule to be used within :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp`, :func:`~.make_selectpaulirot_to_phase_gradient_decomp`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~.CRZ`
    gate using the custom decomposition rule that we generate from within the context of the
    example, where all auxiliary wires exist.

    .. code-block:: python

        import pennylane as qp
        from pennylane.transforms.decompositions import make_crz_to_phase_gradient_decomp
        import numpy as np

        qp.decomposition.enable_graph()

        prec = 3
        phi = (1/2 + 1/4 + 1/8) * 4 * np.pi # binary rep is (111)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_crz_to_phase_gradient_decomp(
            angle_wires, phase_grad_wires, work_wires
        )

        @qp.transforms.decompose(
            gate_set={"CNOT", "SemiAdder", "PauliX"},
            fixed_decomps={qp.CRZ: custom_decomp}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.CRZ(phi, [0, 1])
            return qp.state()

        specs = qp.specs(circuit)()["resources"].quantum_operations

    The resulting circuit corresponds to the phase gradient decomposition of ``CRZ``. The compute
    and uncompute fanouts load the binary representation of the angle (111 in this case) onto the
    ``angle_wires`` with a :class:`~.MultiX` controlled by the control wire (rendering as ``CNOT``,
    since the control is always on |1>), and flip the ``phase_grad_wires`` with controlled-``X``
    gates controlled by the target wire on |0> (rendering as ``CNOT`` plus a ``PauliX`` flip of the
    target wire), and enclose the :class:`~.SemiAdder`.

    >>> specs
    {'CNOT': 12, 'PauliX': 6, 'SemiAdder': 1}
    >>> wire_order = [0, 1] + angle_wires + phase_grad_wires + work_wires
    >>> print(qp.draw(circuit, wire_order=wire_order)())
         0: ─╭●─╭●─╭●──────────────────────────────╭●─╭●─╭●───────────────────┤ ╭State
         1: ─│──│──│──╭●────╭●────╭●───────────────│──│──│──╭●────╭●────╭●────┤ ├State
     aux_0: ─╰X─│──│──│─────│─────│─────╭SemiAdder─╰X─│──│──│─────│─────│─────┤ ├State
     aux_1: ────╰X─│──│─────│─────│─────├SemiAdder────╰X─│──│─────│─────│─────┤ ├State
     aux_2: ───────╰X─│─────│─────│─────├SemiAdder───────╰X─│─────│─────│─────┤ ├State
     qft_0: ──────────╰X──X─│─────│─────├SemiAdder──────────╰X──X─│─────│─────┤ ├State
     qft_1: ────────────────╰X──X─│─────├SemiAdder────────────────╰X──X─│─────┤ ├State
     qft_2: ──────────────────────╰X──X─├SemiAdder──────────────────────╰X──X─┤ ├State
    work_0: ────────────────────────────├SemiAdder────────────────────────────┤ ├State
    work_1: ────────────────────────────╰SemiAdder────────────────────────────┤ ╰State

    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )

    def _resource_fn(phi, wires):  # pylint: disable=unused-argument
        # Full-precision cost from the wires in the outer scope. The angle-load fanout emits one
        # gate per *set* bit of the (concrete) angle, so this is an upper bound (exact=False below).
        precision = len(angle_wires)
        target_op = qp.SemiAdder(Wire[precision], Wire[precision], Wire[len(work_wires)])
        # compute/uncompute fanout: a controlled MultiX loads the angle bits onto the angle wires
        # (controlled by the CRZ control wire) and controlled-X gates flip the phase-gradient wires
        # (controlled by the target wire on |0>). Both lower under program capture, unlike a
        # controlled BasisState.
        angle_fanout = abstractify(
            qp.ctrl(qp.MultiX(Bool[precision], Wire[precision]), control=Wire[1])
        )
        ctrl_x_rep = _ctrl_abstract(qp.X, Wire[1], num_zero_control_values=1)
        fanout = qp.resource_rep(Prod, resources={angle_fanout: 1, ctrl_x_rep: precision})
        change_basis_rep = change_op_basis_resource_rep(fanout, target_op, fanout)
        return {change_basis_rep: 1}

    @qp.register_resources(_resource_fn, exact=False)
    def _decomp_fn(phi, wires):
        precision = len(angle_wires)
        binary_int = qp.math.binary_decimals(phi, precision, unit=4 * np.pi)
        control_wire, target_wire = wires[0], wires[1]

        def _compute_fn():
            # Load the angle bits onto the angle wires controlled by the CRZ control wire: a MultiX
            # (one X per set bit) controlled by the control wire is a per-set-bit fanout.
            qp.ctrl(qp.MultiX(binary_int, angle_wires), control=control_wire)
            # Flip the phase-gradient wires when the target wire is |0> (double-phase trick).
            for w in phase_grad_wires:
                qp.ctrl(qp.X(w), control=target_wire, control_values=[0])

        target_op = qp.SemiAdder(angle_wires, phase_grad_wires, work_wires=work_wires)
        qp.change_op_basis(_compute_fn, target_op, _compute_fn)

    return _decomp_fn

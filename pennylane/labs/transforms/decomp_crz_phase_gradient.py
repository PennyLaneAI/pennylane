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
from pennylane.decomposition import change_op_basis_resource_rep, controlled_resource_rep
from pennylane.ops import Prod

from .decomp_rz_phase_gradient import validate_phase_gradient_wires


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
        from pennylane.labs.transforms import make_selectpaulirot_to_phase_gradient_decomp
        import numpy as np

        qp.decomposition.enable_graph()

        prec = 3
        np.random.seed(35)
        angles = np.random.rand(2**3)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
            angle_wires, phase_grad_wires, work_wires
        )

        @qp.decompose(
            gate_set={"QROM", "Adjoint(QROM)", "SemiAdder", "MultiControlledX", "GlobalPhase"},
            fixed_decomps={qp.SelectPauliRot: custom_decomp}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit(angles):
            qp.SelectPauliRot(angles, control_wires=range(3), target_wire=3)
            return qp.state()

        specs = qp.specs(circuit)(angles)["resources"].gate_types

    The resulting circuit corresponds to the phase gradient decomposition
    of ``CRZ``, containing four fanouts corresponding to the binary representation
    of the angle (111 in this case), the :class:`~.SemiAdder`, and a :class:`~.GlobalPhase`.

    >>> specs
    {'QROM': 2, 'MultiControlledX': 6, 'SemiAdder': 1}
    >>> wire_order = [0, 1, 2, 3] + angle_wires + phase_grad_wires + work_wires
    >>> print(qp.draw(circuit, wire_order=wire_order, show_matrices=False)(angles))
         0: ─╭◑────────────────────────────╭◑─────────────────┤ ╭State
         1: ─├◑────────────────────────────├◑─────────────────┤ ├State
         2: ─├◑────────────────────────────├◑─────────────────┤ ├State
         3: ─│─────────╭○─╭○─╭○────────────│─────────╭○─╭○─╭○─┤ ├State
     aux_0: ─├QROM(M0)─│──│──│──╭SemiAdder─├QROM(M0)─│──│──│──┤ ├State
     aux_1: ─├QROM(M0)─│──│──│──├SemiAdder─├QROM(M0)─│──│──│──┤ ├State
     aux_2: ─├QROM(M0)─│──│──│──├SemiAdder─├QROM(M0)─│──│──│──┤ ├State
     qft_0: ─│─────────╰X─│──│──├SemiAdder─│─────────╰X─│──│──┤ ├State
     qft_1: ─│────────────╰X─│──├SemiAdder─│────────────╰X─│──┤ ├State
     qft_2: ─│───────────────╰X─├SemiAdder─│───────────────╰X─┤ ├State
    work_0: ─├work──────────────├SemiAdder─├work──────────────┤ ├State
    work_1: ─╰work──────────────╰SemiAdder─╰work──────────────┤ ╰State

    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )

    def _resource_fn():
        precision = len(angle_wires)
        # decomposition costs, using information about angle_wires etc from the outer scope
        target_op = qp.resource_rep(
            qp.SemiAdder,
            num_x_wires=precision,
            num_y_wires=precision,
            num_work_wires=len(work_wires),
        )
        fanout_angle = controlled_resource_rep(
            qp.BasisState, {"num_wires": precision}, num_control_wires=1, num_zero_control_values=0
        )
        fanout_addsub = controlled_resource_rep(
            qp.BasisState, {"num_wires": precision}, num_control_wires=1, num_zero_control_values=1
        )
        compute_op = uncompute_op = qp.resource_rep(
            Prod, resources={fanout_angle: 1, fanout_addsub: 1}
        )
        change_basis_rep = change_op_basis_resource_rep(compute_op, target_op, uncompute_op)
        return {change_basis_rep: 1}

    @qp.register_resources(_resource_fn)
    def _decomp_fn(angle, wires, **_):
        precision = len(angle_wires)
        binary_int = qp.math.binary_decimals(angle, precision, unit=4 * np.pi)

        def compute_fn():
            qp.ctrl(qp.BasisState(binary_int, angle_wires), control=wires[0])
            qp.ctrl(
                qp.BasisState([1] * precision, phase_grad_wires),
                control=wires[1],
                control_values=[0],
            )

        def target_fn():
            qp.SemiAdder(angle_wires, phase_grad_wires, work_wires=work_wires)

        qp.change_op_basis(compute_fn, target_fn, compute_fn)

    return _decomp_fn

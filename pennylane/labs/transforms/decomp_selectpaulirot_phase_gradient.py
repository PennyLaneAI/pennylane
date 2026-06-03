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
Decomposition rule for SelectPauliRot in terms of `phase gradient states <https://pennylane.ai/compilation/phase-gradient/d-multiplex-rotations>`__
"""

# pylint: disable=too-many-branches
import numpy as np

import pennylane as qp
from pennylane.decomposition import (
    adjoint_resource_rep,
    change_op_basis_resource_rep,
    controlled_resource_rep,
    resource_rep,
)
from pennylane.operation import Operator
from pennylane.ops import Prod
from pennylane.ops.op_math import change_op_basis
from pennylane.wires import WireError, Wires


# pylint: disable=too-many-arguments
def _select_pauli_rot_phase_gradient(
    phis: np.ndarray,
    rot_axis: str,
    control_wires: Wires,
    target_wire: Wires,
    angle_wires: Wires,
    phase_grad_wires: Wires,
    work_wires: Wires,
) -> Operator:
    """Function that transforms the SelectPauliRot gate to the phase gradient circuit
    The precision is implicitly defined by the length of ``angle_wires``
    """

    precision = len(angle_wires)
    binary_int = qp.math.binary_decimals(phis, precision, unit=4 * np.pi)

    def compute_fn():
        # we can set clean=False because we are doing QROM - something - QROM†
        qp.QROM(
            binary_int,
            control_wires,
            angle_wires,
            work_wires=work_wires[len(angle_wires) - 1 :],
            clean=False,
        )
        for wire in phase_grad_wires:
            qp.ctrl(qp.X(wire), control=target_wire, control_values=[0])

    def target_fn():
        qp.SemiAdder(angle_wires, phase_grad_wires, work_wires=work_wires[: len(angle_wires) - 1])

    def inner_cob():
        return change_op_basis(compute_fn, target_fn)

    match rot_axis:
        case "X":

            def x_basis_comp():
                qp.Hadamard(target_wire)

            return qp.change_op_basis(x_basis_comp, inner_cob)
        case "Y":

            def y_basis_comp():
                qp.adjoint(qp.S(target_wire))
                qp.Hadamard(target_wire)

            return qp.change_op_basis(y_basis_comp, inner_cob)

    return inner_cob()


def make_selectpaulirot_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires):
    r"""
    Custom decomposition rule for :class:`~.SelectPauliRot` gates

    This is a temporary workaround before moving to `capture` as default frontend, which unlocks dynamic wire allocation.
    Here, we explicitly provide the necessary wires for the `phase gradient decomposition of SelectPauliRot <https://pennylane.ai/compilation/phase-gradient/d-multiplex-rotations>`__.
    This way, this function can be used in a workflow context that explicitly uses those wires to generate this decomposition rule, which can then be used
    as ``alt_decomps`` or ``fixed_decomp`` within :func:`~.pennylane.decompose`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state
        work_wires (Wires): additional work wires for :class:`~SemiAdder` decomposition

    Returns:
        func: decomposition rule to be used within :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~.SelectPauliRot` gate using the custom decomposition rule
    that we generate from within the context of the example, where all auxiliary wires exist.

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

    The resulting circuit corresponds to the `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/d-multiplex-rotations>`__ of ``SelectPauliRot``,
    containing two CNOT fanouts corresponding to the binary representation of the angle (111 in this case), the :class:`~SemiAdder`, and a :class:`~GlobalPhase`.

    >>> specs
    {'QROM': 1, 'MultiControlledX': 6, 'SemiAdder': 1, 'Adjoint(QROM)': 1}
    >>> print(qp.draw(circuit, wire_order=[0, 1, 2, 3] + angle_wires + phase_grad_wires + work_wires, show_matrices=False)(angles))
         0: ─╭QROM(M0)──────────────────────────────╭QROM(M0)†─┤ ╭State
         1: ─├QROM(M0)──────────────────────────────├QROM(M0)†─┤ ├State
         2: ─├QROM(M0)──────────────────────────────├QROM(M0)†─┤ ├State
         3: ─│─────────╭○─╭○─╭○────────────╭○─╭○─╭○─│──────────┤ ├State
     aux_0: ─├QROM(M0)─│──│──│──╭SemiAdder─│──│──│──├QROM(M0)†─┤ ├State
     aux_1: ─├QROM(M0)─│──│──│──├SemiAdder─│──│──│──├QROM(M0)†─┤ ├State
     aux_2: ─╰QROM(M0)─│──│──│──├SemiAdder─│──│──│──╰QROM(M0)†─┤ ├State
     qft_0: ───────────╰X─│──│──├SemiAdder─│──│──╰X────────────┤ ├State
     qft_1: ──────────────╰X─│──├SemiAdder─│──╰X───────────────┤ ├State
     qft_2: ─────────────────╰X─├SemiAdder─╰X──────────────────┤ ├State
    work_0: ────────────────────├SemiAdder─────────────────────┤ ├State
    work_1: ────────────────────╰SemiAdder─────────────────────┤ ╰State

    """
    # Sanitize wires
    angle_wires = Wires(angle_wires)
    phase_grad_wires = Wires(phase_grad_wires)
    work_wires = Wires(work_wires)

    if len(angle_wires) != len(phase_grad_wires):
        raise WireError(
            f"angle_wires and phase_grad wires must be of same size, received {len(angle_wires)} and {len(phase_grad_wires-1)}"
        )
    if len(phase_grad_wires) - 1 > len(work_wires):
        raise WireError(
            f"work_wires need to be at least of size phase_grad_wires - 1, received {len(work_wires)} but require {len(phase_grad_wires-1)}"
        )

    def _resource_fn(num_wires, rot_axis):
        # decomposition costs, using information about angle_wires etc from the outer scope

        num_control_wires = num_wires - 1
        if num_control_wires == 0:
            match rot_axis:
                case "X":
                    return {resource_rep(qp.RX): 1}
                case "Y":
                    return {resource_rep(qp.RY): 1}
                case "Z":
                    return {resource_rep(qp.RZ): 1}

        # 1. QROM compressed rep
        qrom_rep = resource_rep(
            qp.QROM,
            clean=False,
            num_bitstrings=2**num_control_wires,
            num_control_wires=num_control_wires,
            num_target_wires=len(angle_wires),
            num_work_wires=len(work_wires) - len(angle_wires) + 1,
        )

        # 2. ctrl(X, control=target_wire, control_values=[0])
        #    -> Controlled X with 1 control, 1 zero-ctrl
        ctrl_x_rep = controlled_resource_rep(
            qp.X, base_params={}, num_control_wires=1, num_zero_control_values=1
        )

        # 3. Prod: MUST be a dict {CompressedResourceOp: count}
        prod_res = {
            qrom_rep: 1,
            ctrl_x_rep: len(phase_grad_wires),
        }
        prod_rep = resource_rep(Prod, resources=prod_res)

        # 4. SemiAdder as the target_op
        semi_adder_rep = resource_rep(
            qp.SemiAdder,
            num_x_wires=len(angle_wires),
            num_y_wires=len(phase_grad_wires),
            num_work_wires=len(angle_wires) - 1,
        )

        # 5. change_op_basis(compute_op, target_op)
        #    compute_op = prod (the QROM + ctrl-X product)
        #    target_op  = SemiAdder
        change_basis_rep = change_op_basis_resource_rep(
            compute_op=prod_rep,
            target_op=semi_adder_rep,
        )

        # 6. Basis adaptation depending on rot_axis
        match rot_axis:
            case "X":
                change_basis_rep_basis_adapted = change_op_basis_resource_rep(
                    resource_rep(qp.Hadamard),
                    change_basis_rep,
                    resource_rep(qp.Hadamard),
                )
            case "Y":
                comp_rep = resource_rep(
                    Prod,
                    resources={
                        resource_rep(qp.Hadamard): 1,
                        adjoint_resource_rep(qp.S): 1,
                    },
                )
                uncomp_rep = resource_rep(
                    Prod,
                    resources={
                        resource_rep(qp.S): 1,
                        resource_rep(qp.Hadamard): 1,
                    },
                )
                change_basis_rep_basis_adapted = change_op_basis_resource_rep(
                    comp_rep, change_basis_rep, uncomp_rep
                )
            case "Z":
                change_basis_rep_basis_adapted = change_basis_rep

        return {change_basis_rep_basis_adapted: 1}

    @qp.register_resources(_resource_fn)
    def _decomp_fn(angles, control_wires, target_wire, rot_axis, **_):

        if len(control_wires) == 0:

            match rot_axis:
                case "X":
                    qp.RX(angles[0], target_wire)
                case "Y":
                    qp.RY(angles[0], target_wire)
                case "Z":
                    qp.RZ(angles[0], target_wire)
            return

        _select_pauli_rot_phase_gradient(
            angles,
            rot_axis,
            control_wires=control_wires,
            target_wire=target_wire,
            angle_wires=angle_wires,
            phase_grad_wires=phase_grad_wires,
            work_wires=work_wires,
        )

    return _decomp_fn

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
import pennylane as qml
from pennylane.decomposition import (
    adjoint_resource_rep,
    change_op_basis_resource_rep,
    controlled_resource_rep,
    resource_rep,
)
from pennylane.ops import Prod

from .rot_to_phase_gradient import _select_pauli_rot_phase_gradient


def make_selectpaulirot_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires):
    r"""
    Custom decomposition rule for :class:`~.SelectPauliRot` gates

    This is a temporary workaround as long as capture does not work, which blocks usage of dynamic allocation.
    Here, we explicitly provide the necessary wires for the `phase gradient decomposition of SelectPauliRot <https://pennylane.ai/compilation/phase-gradient/d-multiplex-rotations>`__.
    This way, this function can be used in a workflow context that explicitly uses those wires to generate this decomposition rule, which can then be used
    as ``alt_decomps`` or ``fixed_decomp`` within :func:`~decompose`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state
        work_wires (Wires): additional work wires for :class:`~SemiAdder` decomposition

    Returns:
        func: decomposition rule to be used within :func:`~decompose`.

    .. seealso:: :func:`~make_rz_to_phase_gradient_decomp`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~RZ` gate using the custom decomposition rule
    that we generate from within the context of the example, where all auxiliary wires exist.

    .. code-block:: python

        import pennylane as qp
        from pennylane.labs.transforms import make_selectpaulirot_to_phase_gradient_decomp
        import numpy as np

        qp.decomposition.enable_graph()

        prec = 3
        angles = np.random.rand(2**3)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)

        @qp.transforms.decompose(
            gate_set={"QROM", "Adjoint(QROM)", "SemiAdder", "CNOT", "X", "Adjoint(X)", "GlobalPhase"},
            fixed_decomps={qp.SelectPauliRot: custom_decomp}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit(angles):
            qp.SelectPauliRot(angles, control_wires=range(3), target_wire=3)
            return qp.state()

        specs = qp.specs(circuit)(angles)["resources"].gate_types

    The resulting circuit corresponds to the `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/d-multiplex-rotations>`__ of RZ,
    containing two CNOT fanouts corresponding to the binary representation of the angle (111 in this case), the :class:`~SemiAdder`, and a :class:`~GlobalPhase`.

    >>> specs
    {'ChangeOpBasis': 1}
    >>> print(qp.draw(circuit, wire_order=[0, 1, 2, 3] + angle_wires + phase_grad_wires + work_wires)(angles))
         0: ─╭QROM(M0)───────────────────────────────────────╭QROM(M0)†─┤  State
         1: ─├QROM(M0)───────────────────────────────────────├QROM(M0)†─┤  State
         2: ─├QROM(M0)───────────────────────────────────────├QROM(M0)†─┤  State
         3: ─│─────────╭●────╭●────╭●───────────────╭●─╭●─╭●─│──────────┤  State
     aux_0: ─├QROM(M0)─│─────│─────│─────╭SemiAdder─│──│──│──├QROM(M0)†─┤  State
     aux_1: ─├QROM(M0)─│─────│─────│─────├SemiAdder─│──│──│──├QROM(M0)†─┤  State
     aux_2: ─╰QROM(M0)─│─────│─────│─────├SemiAdder─│──│──│──╰QROM(M0)†─┤  State
     qft_0: ───────────╰X──X─│─────│─────├SemiAdder─│──│──╰X──X─────────┤  State
     qft_1: ─────────────────╰X──X─│─────├SemiAdder─│──╰X──X────────────┤  State
     qft_2: ───────────────────────╰X──X─├SemiAdder─╰X──X───────────────┤  State
    work_0: ─────────────────────────────├SemiAdder─────────────────────┤  State
    work_1: ─────────────────────────────╰SemiAdder─────────────────────┤  State

    """

    def _resource_fn(num_wires, rot_axis):
        # decomposition costs, using information about angle_wires etc from the outer scope

        num_control_wires = num_wires - 1
        if num_control_wires == 0:
            match rot_axis:
                case "X":
                    return {resource_rep(qml.RX): 1}
                case "Y":
                    return {resource_rep(qml.RY): 1}
                case "Z":
                    return {resource_rep(qml.RZ): 1}

        # 1. QROM compressed rep
        qrom_rep = resource_rep(
            qml.QROM,
            clean=True,
            num_bitstrings=2**num_control_wires,
            num_control_wires=num_control_wires,
            num_target_wires=len(angle_wires),
            num_work_wires=num_control_wires - 1,
        )

        # 2. ctrl(X, control=target_wire, control_values=[0])
        #    -> Controlled X with 1 control, 1 zero-ctrl
        ctrl_x_rep = controlled_resource_rep(
            qml.X, base_params={}, num_control_wires=1, num_zero_control_values=1
        )

        # 3. Prod: MUST be a dict {CompressedResourceOp: count}
        prod_res = {
            qrom_rep: 1,
            ctrl_x_rep: len(phase_grad_wires),
        }
        prod_rep = resource_rep(Prod, resources=prod_res)

        # 4. SemiAdder as the target_op
        semi_adder_rep = resource_rep(qml.SemiAdder, num_y_wires=len(phase_grad_wires))

        # 5. change_op_basis(compute_op, target_op)
        #    compute_op = prod (the QROM + ctrl-X product)
        #    target_op  = SemiAdder
        change_basis_rep = change_op_basis_resource_rep(
            compute_op=prod_rep,
            target_op=semi_adder_rep,
            uncompute_op=prod_rep,
        )

        # 6. Basis adaptation depending on rot_axis
        match rot_axis:
            case "X":
                change_basis_rep_basis_adapted = change_op_basis_resource_rep(
                    resource_rep(qml.Hadamard),
                    change_basis_rep,
                    resource_rep(qml.Hadamard),
                )
            case "Y":
                comp_rep = resource_rep(
                    Prod,
                    resources={
                        resource_rep(qml.Hadamard): 1,
                        adjoint_resource_rep(qml.S): 1,
                    },
                )
                uncomp_rep = resource_rep(
                    Prod,
                    resources={
                        resource_rep(qml.S): 1,
                        resource_rep(qml.Hadamard): 1,
                    },
                )
                change_basis_rep_basis_adapted = change_op_basis_resource_rep(
                    comp_rep, change_basis_rep, uncomp_rep
                )
            case "Z":
                change_basis_rep_basis_adapted = change_basis_rep

        return {change_basis_rep_basis_adapted: 1}

    @qml.register_resources(_resource_fn)
    def _decomp_fn(angles, control_wires, target_wire, rot_axis, **_):

        if len(control_wires) == 0:
            assert len(angles) == 1
            match rot_axis:
                case "X":
                    qml.RX(angles[0], target_wire)
                case "Y":
                    qml.RY(angles[0], target_wire)
                case "Z":
                    qml.RZ(angles[0], target_wire)
            return

        with qml.QueuingManager.stop_recording():
            pg_op = _select_pauli_rot_phase_gradient(
                angles,
                rot_axis,
                control_wires=control_wires,
                target_wire=target_wire,
                angle_wires=angle_wires,
                phase_grad_wires=phase_grad_wires,
                work_wires=work_wires,
            )

        qml.apply(pg_op)

    return _decomp_fn

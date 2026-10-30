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
from pennylane.core.operator import Operator
from pennylane.decomposition._subcircuit import subcircuit
from pennylane.ops.op_math import change_op_basis
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.templates.subroutines.select_pauli_rot import (
    _SelectPauliRotYBasis,
    _SelectPauliRotYBasisInverse,
)
from pennylane.typing import Bool, Wire
from pennylane.wires import WireError, Wires

from .rz_phase_gradient import _precision_dispatcher, validate_phase_gradient_wires


def _select_phase_gradient_compute_resources(
    bitstrings, wires, angle_wires, phase_grad_wires, work_wires
):
    """Resources for loading selected angles and flipping phase-gradient wires."""
    return {
        qp.QROM(bitstrings, wires[:-1], angle_wires, work_wires=work_wires): 1,
        _ctrl_abstract(qp.X, wires[-1:], num_zero_control_values=1): len(phase_grad_wires),
    }


@subcircuit(
    dynamic_argnames=("bitstrings",),
    wire_argnames=("wires", "angle_wires", "phase_grad_wires", "work_wires"),
    arg_specs={
        "bitstrings": Bool[-1, -1],
        "wires": Wire[-1],
        "angle_wires": Wire[-1],
        "phase_grad_wires": Wire[-1],
        "work_wires": Wire[-1],
    },
    grad_method=None,
)
@qp.register_resources(_select_phase_gradient_compute_resources)
def _SelectPauliRotPhaseGradientCompute(
    bitstrings, wires, angle_wires, phase_grad_wires, work_wires
) -> None:
    """Load selected angle bits and flip phase-gradient wires when the target is zero."""
    qp.QROM(bitstrings, wires[:-1], angle_wires, work_wires=work_wires)
    for wire in phase_grad_wires:
        qp.ctrl(qp.X(wire), control=wires[-1], control_values=[0])


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
    binary_int = qp.math.cast(qp.math.binary_decimals(phis, precision, unit=4 * np.pi), bool)

    compute_op = _SelectPauliRotPhaseGradientCompute(
        binary_int,
        Wires(control_wires) + Wires(target_wire),
        angle_wires,
        phase_grad_wires,
        work_wires[: len(control_wires) - 1],  # This forces Λ=1 in Select-Swap
    )
    target_op = qp.SemiAdder(angle_wires, phase_grad_wires, work_wires=work_wires)
    # uncompute=compute is only valid for Λ=1, because QROM creates complex phases otherwise.
    inner_cob = change_op_basis(compute_op, target_op, compute_op)

    match rot_axis:
        case "X":
            return qp.change_op_basis(qp.Hadamard(target_wire), inner_cob, qp.Hadamard(target_wire))

        case "Y":
            return qp.change_op_basis(
                _SelectPauliRotYBasis(target_wire),
                inner_cob,
                _SelectPauliRotYBasisInverse(target_wire),
            )

    return inner_cob


def make_selectpaulirot_to_phase_gradient_decomp(
    angle_wires, phase_grad_wires, work_wires, precision_stats=None
):
    r"""
    Create a custom decomposition rule for :class:`~.SelectPauliRot` gates.

    Here, the necessary auxiliary wires for the `phase gradient decomposition of SelectPauliRot
    <https://pennylane.ai/compilation/phase-gradient/d-multiplex-rotations>`__ are taken as inputs
    and bound into the created decomposition rule.
    This way, this function can be used in a workflow context that explicitly uses those wires
    to generate this decomposition rule, which can then be used as ``alt_decomps`` or
    ``fixed_decomp`` within :func:`~.pennylane.decompose`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state. Should have the same
            length as ``angle_wires``.
        work_wires (Wires): additional work wires for :class:`~.SemiAdder` and :class:`~.QROM`
            decomposition. For the former, at least ``len(angle_wires)-1`` work wires are required.
            For the latter, at least ``len(control_wires)-1`` work wires are required for an
            efficient decomposition, where ``control_wires`` are the control wires of the (largest)
            ``SelectPauliRot`` to be decomposed with the produced decomposition rule.
            Overall, we thus require ``max(len(angle_wires), len(control_wires))-1`` work wires.
        precision_stats (dict[int, int] | None): Optional statistics of the number of bits
            required by the ``SelectPauliRot`` operators, mapping a number of bits to the number
            of operators requiring that many bits. Defaults to ``None``.

    Returns:
        func: decomposition rule to be used within :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~.SelectPauliRot`
    gate using the custom decomposition rule that we generate from within the context of the
    example, where all auxiliary wires exist.

    .. code-block:: python

        import pennylane as qp
        from pennylane.transforms.decompositions import make_selectpaulirot_to_phase_gradient_decomp
        import numpy as np

        qp.decomposition.enable_graph()

        prec = 3
        np.random.seed(35)
        angles = np.random.rand(2**3)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"phg_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
            angle_wires, phase_grad_wires, work_wires
        )

        @qp.transforms._tape_decompose(
            gate_set={"QROM", "Adjoint(QROM)", "SemiAdder", "MultiControlledX", "GlobalPhase"},
            fixed_decomps={qp.SelectPauliRot: custom_decomp}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit(angles):
            qp.SelectPauliRot(angles, control_wires=range(3), target_wire=3)
            return qp.state()

    The resulting circuit corresponds to the
    `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/d-multiplex-rotations>`__
    of ``SelectPauliRot``, containing two static CNOT fanouts that toggle between addition and
    subtraction based on the target wire state, two QROMs that load the binary representation of
    the rotation angles, and the :class:`~.SemiAdder`:

    >>> wire_order = [0, 1, 2, 3] + angle_wires + phase_grad_wires + work_wires
    >>> print(qp.draw(circuit, wire_order=wire_order, show_matrices=False)(angles))
         0: ─╭◑────────────────────────────╭◑─────────────────┤ ╭State
         1: ─├◑────────────────────────────├◑─────────────────┤ ├State
         2: ─├◑────────────────────────────├◑─────────────────┤ ├State
         3: ─│─────────╭○─╭○─╭○────────────│─────────╭○─╭○─╭○─┤ ├State
     aux_0: ─├QROM(M0)─│──│──│──╭SemiAdder─├QROM(M0)─│──│──│──┤ ├State
     aux_1: ─├QROM(M0)─│──│──│──├SemiAdder─├QROM(M0)─│──│──│──┤ ├State
     aux_2: ─├QROM(M0)─│──│──│──├SemiAdder─├QROM(M0)─│──│──│──┤ ├State
     phg_0: ─│─────────╰X─│──│──├SemiAdder─│─────────╰X─│──│──┤ ├State
     phg_1: ─│────────────╰X─│──├SemiAdder─│────────────╰X─│──┤ ├State
     phg_2: ─│───────────────╰X─├SemiAdder─│───────────────╰X─┤ ├State
    work_0: ─├work──────────────├SemiAdder─├work──────────────┤ ├State
    work_1: ─╰work──────────────╰SemiAdder─╰work──────────────┤ ╰State

    .. details::
        :title: Usage details

        **Adaptive precision via ``precision_stats``**

        For angles that do not require the full precision determined by the angle encoding and
        phase gradient wires, we may use a smaller encoding and adder, reducing the cost of the
        phase gradient rotation. However, as rotation angles are not known statically in general,
        this adaptive precision is only supported if additional information about the distribution
        of rotation angles is provided via ``precision_stats``, a dictionary mapping a number
        of bits to the number of rotations requiring that many bits.

        The bits required by an operator are the maximum over all of its angles, where the bits
        required by an angle :math:`\phi` are computed from
        ``qp.math.binary_decimals(phi, len(angle_wires), unit=4 * np.pi)``. If provided,
        the data loading and the :class:`~.SemiAdder` are truncated to the smallest number of
        bits in ``precision_stats`` that suffices for a given angle, using :func:`~.cond`
        branches that are hinted with the corresponding probabilities via :func:`~.hint`.
        If no number of bits in ``precision_stats`` suffices, the full precision is used.
        Keys must lie between ``0`` and ``len(angle_wires)``. Zero counts are ignored, and
        a dictionary without non-zero counts is treated like ``None``. Use
        :func:`~.phase_gradient_precision_stats` to compute the statistics from the angles.

        **Example**

        .. code-block:: python

            from pennylane.transforms.decompositions import phase_gradient_precision_stats

            angles = [
                np.array([0, 1/2, 1/4, 0, 0, 1/2, 0, 1/4]) * 4 * np.pi, # at most 2 bits
                np.array([1/8, 0, 0, 0, 1/2, 0, 0, 0]) * 4 * np.pi, # at most 3 bits
            ]
            precision_stats = phase_gradient_precision_stats(angles, prec, qp.SelectPauliRot)

            truncating_decomp = make_selectpaulirot_to_phase_gradient_decomp(
                angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
            )

            @qp.transforms._tape_decompose(
                gate_set={"QROM", "Adjoint(QROM)", "SemiAdder", "MultiControlledX", "GlobalPhase"},
                fixed_decomps={qp.SelectPauliRot: truncating_decomp}
            )
            @qp.qnode(qp.device("null.qubit"))
            def truncated_circuit(angles):
                qp.SelectPauliRot(angles, control_wires=range(3), target_wire=3)
                return qp.state()

        The first operator only loads two bits per angle and uses two phase gradient wires:

        >>> precision_stats
        {2: 1, 3: 1}
        >>> print(qp.draw(truncated_circuit, wire_order=wire_order, show_matrices=False)(angles[0]))
             0: ─╭◑─────────────────────────╭◑──────────────┤ ╭State
             1: ─├◑─────────────────────────├◑──────────────┤ ├State
             2: ─├◑─────────────────────────├◑──────────────┤ ├State
             3: ─│─────────╭○─╭○────────────│─────────╭○─╭○─┤ ├State
         aux_0: ─├QROM(M0)─│──│──╭SemiAdder─├QROM(M0)─│──│──┤ ├State
         aux_1: ─├QROM(M0)─│──│──├SemiAdder─├QROM(M0)─│──│──┤ ├State
         phg_0: ─│─────────╰X─│──├SemiAdder─│─────────╰X─│──┤ ├State
         phg_1: ─│────────────╰X─├SemiAdder─│────────────╰X─┤ ├State
        work_0: ─├work───────────├SemiAdder─├work───────────┤ ├State
        work_1: ─╰work───────────╰SemiAdder─╰work───────────┤ ╰State

        Finally, keep the following in mind when using ``precision_stats``:

        * The resource function of the rule adds up the cost of all supported precisions, so
          that the graph-based decomposition system accounts for the operators in all
          conditional branches. This over-reports the cost of the rule, so that the
          decomposition graph may prefer another rule over this one when it is passed via
          ``alt_decomps``. Use ``fixed_decomps`` to enforce this rule.
        * Angles that require more bits than the largest key in ``precision_stats`` fall back
          to the full precision. Unless ``len(angle_wires)`` is a key in ``precision_stats``,
          this fallback branch is hinted with probability ``0``, so that resource estimates
          based on the hints do not account for these angles.
        * A ``SelectPauliRot`` without control wires is decomposed into a plain :class:`~.RX`,
          :class:`~.RY` or :class:`~.RZ` rotation. If the resulting :class:`~.RZ` rotations are
          decomposed by a rule from :func:`~.make_rz_to_phase_gradient_decomp` with
          ``precision_stats``, include their angles in the statistics for ``op_type=qp.RZ``,
          which uses the unit :math:`2\pi` instead of :math:`4\pi`.

    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )
    precision = len(angle_wires)
    precisions, dispatch = _precision_dispatcher(precision_stats, precision, "SelectPauliRot")

    # pylint: disable=unused-argument
    def _resource_fn(angles, control_wires, target_wire, rot_axis):
        # decomposition costs, using information about angle_wires etc from the outer scope.
        # Each non-zero precision that may be applied is counted once.

        num_control_wires = len(control_wires)
        if num_control_wires == 0:
            return {{"X": qp.RX, "Y": qp.RY, "Z": qp.RZ}[rot_axis]: 1}

        return {_resource_rep(num_control_wires, rot_axis, p): 1 for p in precisions - {0}}

    def _resource_rep(num_control_wires, rot_axis, p):
        compute_rep = _SelectPauliRotPhaseGradientCompute(
            bitstrings=Bool[2**num_control_wires, p],
            wires=Wire[num_control_wires + 1],
            angle_wires=Wire[p],
            phase_grad_wires=Wire[p],
            work_wires=Wire[num_control_wires - 1],
        )
        semi_adder_rep = qp.SemiAdder(Wire[p], Wire[p], Wire[len(work_wires)])

        change_basis_rep = _change_op_basis_abstract(
            compute_op=compute_rep,
            target_op=semi_adder_rep,
            uncompute_op=compute_rep,
        )

        # Basis adaptation depending on rot_axis
        match rot_axis:
            case "X":
                change_basis_rep_basis_adapted = _change_op_basis_abstract(
                    qp.Hadamard,
                    change_basis_rep,
                    qp.Hadamard,
                )
            case "Y":
                change_basis_rep_basis_adapted = _change_op_basis_abstract(
                    _SelectPauliRotYBasis,
                    change_basis_rep,
                    _SelectPauliRotYBasisInverse,
                )
            case "Z":
                change_basis_rep_basis_adapted = change_basis_rep

        return change_basis_rep_basis_adapted

    # The resources are inexact if conditional branches are used
    @qp.register_resources(_resource_fn, exact=precisions == {precision})
    def _select_pauli_rot_phase_gradient_decomp(angles, control_wires, target_wire, rot_axis, **_):
        if len(control_wires) == 0:
            rot_class = {"X": qp.RX, "Y": qp.RY, "Z": qp.RZ}[rot_axis]
            rot_class(angles[0], target_wire)
            return

        # Validate length of work wires for QROM
        if len(work_wires) < len(control_wires) - 1:
            raise WireError(
                "work_wires need to be at least of size len(control_wires) - 1, "
                f"received {len(work_wires)} but require {len(control_wires) - 1}"
            )

        dispatch(
            angles,
            lambda p: _select_pauli_rot_phase_gradient(
                angles,
                rot_axis,
                control_wires,
                target_wire,
                angle_wires[:p],
                phase_grad_wires[:p],
                work_wires,
            ),
        )

    return _select_pauli_rot_phase_gradient_decomp

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


"""Phase-gradient decomposition rules and circuit helpers for RZ."""

import numpy as np

from pennylane import math, ops, templates
from pennylane.core.operator import Operator
from pennylane.decomposition import register_resources
from pennylane.decomposition_rules.phase_gradient.common import (
    _precision_dispatcher,
    validate_phase_gradient_wires,
)
from pennylane.ops.op_math import change_op_basis
from pennylane.ops.op_math.change_op_basis2 import ChangeOpBasis2
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires


def _rz_phase_gradient(
    phi: float, wire: Wires, angle_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> Operator:
    """Build the RZ phase-gradient circuit without its global phase.

    Precision is determined by the length of ``angle_wires``. Callers add the global phase
    ``phi / 2``.
    """
    precision = len(angle_wires)

    binary_int = math.binary_decimals(phi, precision, unit=2 * np.pi)

    control_wire = Wires(wire)[0]
    compute_op = ops.ctrl(templates.MultiX(binary_int, angle_wires), control=control_wire)
    target_op = templates.SemiAdder(angle_wires, phase_grad_wires, work_wires)

    return change_op_basis(compute_op, target_op, compute_op)


def make_rz_to_phase_gradient_decomp(
    angle_wires, phase_grad_wires, work_wires, precision_stats=None
):
    r"""
    Create a custom decomposition rule for :class:`~.RZ` gates.

    Here, the necessary auxiliary wires for the `phase gradient decomposition of RZ
    <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__ are taken as inputs and
    bound into the created decomposition rule.
    This way, this function can be used in a workflow context that explicitly uses those wires
    to generate this decomposition rule, which can then be registered with
    :func:`~pennylane.decomposition.add_decomps`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state
        work_wires (Wires): additional work wires for :class:`~.SemiAdder` decomposition
        precision_stats (dict[int, int] | None): Optional statistics of the number of bits
            required by the rotation angles, mapping a number of bits to the number of rotations
            requiring that many bits. Defaults to ``None``.

    Returns:
        qp.decomposition.DecompositionRule: decomposition rule to be used within :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_selectpaulirot_to_phase_gradient_decomp`,
        :func:`~.phase_gradient_precision_stats`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~.RZ` gate using the custom decomposition rule
    that we generate from within the context of the example, where all auxiliary wires exist.

    .. code-block:: python

        import pennylane as qp
        from pennylane.decomposition_rules.phase_gradient import (
            make_rz_to_phase_gradient_decomp,
        )
        import numpy as np

        prec = 3
        phi = (1/2 + 1/4 + 1/8) * 2 * np.pi # binary rep is (111)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"phg_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_rz_to_phase_gradient_decomp(
            angle_wires, phase_grad_wires, work_wires
        )

        gate_set = {"CNOT", "SemiAdder", "GlobalPhase", "PauliX"}

        @qp.transforms._tape_decompose(gate_set=gate_set, fixed_decomps={qp.RZ: custom_decomp})
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.RZ(phi, 0)
            return qp.state()

    The resulting circuit corresponds to the `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__ of RZ,
    containing two CNOT fanouts corresponding to the binary representation of the angle (111 in this case), the :class:`~SemiAdder`, and a :class:`~GlobalPhase`.

    >>> print(qp.draw(circuit)())
         0: ─╭GlobalPhase(2.75)─╭●─╭●─╭●────────────╭●─╭●─╭●─┤ ╭State
     aux_0: ─├GlobalPhase(2.75)─╰X─│──│──╭SemiAdder─╰X─│──│──┤ ├State
     aux_1: ─├GlobalPhase(2.75)────╰X─│──├SemiAdder────╰X─│──┤ ├State
     aux_2: ─├GlobalPhase(2.75)───────╰X─├SemiAdder───────╰X─┤ ├State
     phg_0: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
     phg_1: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
     phg_2: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
    work_0: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
    work_1: ─╰GlobalPhase(2.75)──────────╰SemiAdder──────────┤ ╰State

    .. details::
        :title: Usage details

        **Adaptive precision via ``precision_stats``**

        For angles that do not require the full precision determined by the angle encoding and
        phase gradient wires, we may use a smaller encoding and adder, reducing the cost of the
        phase gradient rotation. However, as rotation angles are not known statically in general,
        this adaptive precision is only supported if additional information about the distribution
        of rotation angles is provided via ``precision_stats``, a dictionary mapping a number
        of bits to the number of rotations requiring that many bits.

        The bits required by an angle :math:`\phi` are the
        position of the last set bit in ``qp.math.binary_decimals(phi, len(angle_wires),
        unit=2 * np.pi)``, i.e., the precision without trailing zeros. If provided,
        the data loading and the :class:`~.SemiAdder` are truncated to the smallest number of
        bits in ``precision_stats`` that suffices for a given angle, using :func:`~.cond`
        branches that are hinted with the corresponding probabilities via :func:`~.hint`.
        If no number of bits in ``precision_stats`` suffices, the full precision is used.
        Keys must lie between ``0`` and ``len(angle_wires)``. Zero counts are ignored, and
        a dictionary without non-zero counts is treated like ``None``. Use
        :func:`~.phase_gradient_precision_stats` to compute the statistics from the angles.

        **Example**

        .. code-block:: python

            from pennylane.decomposition_rules.phase_gradient import (
                phase_gradient_precision_stats,
            )

            angles = np.array([1/2, 1/4 + 1/8, 1/2 + 1/4 + 1/8]) * 2 * np.pi # (100), (011), (111)
            precision_stats = phase_gradient_precision_stats(angles, prec, qp.RZ)

            truncating_decomp = make_rz_to_phase_gradient_decomp(
                angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
            )

            @qp.transforms._tape_decompose(
                gate_set=gate_set, fixed_decomps={qp.RZ: truncating_decomp}
            )
            @qp.qnode(qp.device("null.qubit"))
            def truncated_circuit(phi):
                qp.RZ(phi, 0)
                return qp.state()

        One rotation requires a single bit and two rotations require all three bits. The branches
        of the rule are hinted with these frequencies, and the rotation by the first angle only uses
        a single angle wire and phase gradient wire:

        >>> precision_stats
        {1: 1, 3: 2}
        >>> wire_order = [0] + angle_wires + phase_grad_wires + work_wires
        >>> print(qp.draw(truncated_circuit, wire_order=wire_order)(angles[0]))
             0: ─╭GlobalPhase(1.57)─╭●────────────╭●─┤ ╭State
         aux_0: ─├GlobalPhase(1.57)─╰X─╭SemiAdder─╰X─┤ ├State
         phg_0: ─├GlobalPhase(1.57)────├SemiAdder────┤ ├State
        work_0: ─├GlobalPhase(1.57)────├SemiAdder────┤ ├State
        work_1: ─╰GlobalPhase(1.57)────╰SemiAdder────┤ ╰State

        >>> print(qp.draw(truncated_circuit, wire_order=wire_order)(angles[1]))
             0: ─╭GlobalPhase(1.18)─╭●─╭●────────────╭●─╭●─┤ ╭State
         aux_0: ─├GlobalPhase(1.18)─│──│──╭SemiAdder─│──│──┤ ├State
         aux_1: ─├GlobalPhase(1.18)─╰X─│──├SemiAdder─╰X─│──┤ ├State
         aux_2: ─├GlobalPhase(1.18)────╰X─├SemiAdder────╰X─┤ ├State
         phg_0: ─├GlobalPhase(1.18)───────├SemiAdder───────┤ ├State
         phg_1: ─├GlobalPhase(1.18)───────├SemiAdder───────┤ ├State
         phg_2: ─├GlobalPhase(1.18)───────├SemiAdder───────┤ ├State
        work_0: ─├GlobalPhase(1.18)───────├SemiAdder───────┤ ├State
        work_1: ─╰GlobalPhase(1.18)───────╰SemiAdder───────┤ ╰State

        Finally, keep the following in mind when using ``precision_stats``:

        * The resource function of the rule adds up the cost of all supported precisions, so
          that the graph-based decomposition system accounts for the operators in all
          conditional branches. This over-reports the cost of the rule, so that the
          decomposition graph may prefer another rule over this one when it is registered with
          :func:`~pennylane.decomposition.add_decomps`. Use ``fixed_decomps`` to enforce this rule.
        * Angles that require more bits than the largest key in ``precision_stats`` fall back
          to the full precision. Unless ``len(angle_wires)`` is a key in ``precision_stats``,
          this fallback branch is hinted with probability ``0``, so that resource estimates
          based on the hints do not account for these angles.

    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )
    precisions, dispatch = _precision_dispatcher(precision_stats, len(angle_wires), "RZ")

    def _rz_phase_gradient_standard_resources(phi, wires):  # pylint: disable=unused-argument
        # rz decomposition costs, using information about angle_wires etc from the outer scope.
        # Each non-zero precision that may be applied is counted once.
        resources = {ops.GlobalPhase: 1}
        for p in precisions - {0}:
            target_op = templates.SemiAdder(Wire[p], Wire[p], Wire[len(work_wires)])
            fanout = ops.ctrl(templates.MultiX(Bool[p], Wire[p]), control=Wire[1])
            resources[ChangeOpBasis2(fanout, target_op, fanout)] = 1
        return resources

    # MultiX only emits a gate per set bit, so the gate count depends on the concrete angle.
    @register_resources(_rz_phase_gradient_standard_resources, exact=False, name="phase_gradient")
    def _rz_phase_gradient_standard(phi, wires):
        ops.GlobalPhase(phi / 2)
        dispatch(
            phi,
            lambda p: _rz_phase_gradient(
                phi, wires, angle_wires[:p], phase_grad_wires[:p], work_wires
            ),
        )

    return _rz_phase_gradient_standard

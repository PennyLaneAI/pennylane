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
from pennylane.ops.op_math import ChangeOpBasis2, prod
from pennylane.typing import Bool, Wire

from .rz_phase_gradient import _precision_dispatcher, validate_phase_gradient_wires


def _crz_phase_gradient(phi, wires, angle_wires, phase_grad_wires, work_wires):
    """Apply the phase gradient circuit for CRZ. The precision is implicitly defined by the
    length of ``angle_wires``."""
    binary_int = qp.math.binary_decimals(phi, len(angle_wires), unit=4 * np.pi)
    control_wire, target_wire = wires[0], wires[1]

    def _compute_fn():
        qp.ctrl(qp.MultiX(binary_int, angle_wires), control=control_wire)
        # Flip the phase-gradient wires when the target wire is |0> (double-phase trick).
        qp.ctrl(
            qp.MultiX([1] * len(phase_grad_wires), phase_grad_wires),
            control=target_wire,
            control_values=[0],
        )

    target_op = qp.SemiAdder(angle_wires, phase_grad_wires, work_wires=work_wires)
    qp.change_op_basis(_compute_fn, target_op, _compute_fn)


def make_crz_to_phase_gradient_decomp(
    angle_wires, phase_grad_wires, work_wires, precision_stats=None
):
    r"""
    Create a custom decomposition rule for :class:`~.CRZ` gates.

    Here, the necessary auxiliary wires for the `phase gradient decomposition of CRZ
    <https://pennylane.ai/compilation/phase-gradient/c-control-rotations>`__ are taken as inputs
    and bound into the created decomposition rule.
    This way, this function can be used in a workflow context that explicitly uses those wires
    to generate this decomposition rule, which can then be used as ``alt_decomps`` or
    ``fixed_decomp`` within :func:`~.pennylane.decompose`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state. Should have the same
            length as ``angle_wires``.
        work_wires (Wires): additional work wires for :class:`~.SemiAdder` decomposition.
            At least ``len(angle_wires)-1`` work wires are required.
        precision_stats (dict[int, int] | None): Optional statistics of the number of bits
            required by the rotation angles. Defaults to ``None``.

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

        @qp.transforms._tape_decompose(
            gate_set={"CNOT", "SemiAdder", "PauliX"},
            fixed_decomps={qp.CRZ: custom_decomp}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.CRZ(phi, [0, 1])
            return qp.state()


    The resulting circuit corresponds to the phase gradient decomposition of ``CRZ``. The compute
    and uncompute fanouts load the binary representation of the angle (111 in this case) onto the
    ``angle_wires`` with a :class:`~.MultiX` controlled by the control wire (rendering as ``CNOT``,
    since the control is always on |1>), and flip all of the ``phase_grad_wires`` with a single
    :class:`~.MultiX` controlled by the target wire on |0> (rendering as a ``PauliX`` flip of that
    control wire, ``CNOT``\\ s to the ``phase_grad_wires``, and a ``PauliX`` flip back), and
    enclose the :class:`~.SemiAdder`.

    >>> wire_order = [0, 1] + angle_wires + phase_grad_wires + work_wires
    >>> print(qp.draw(circuit, wire_order=wire_order)())
         0: ─╭●─╭●─╭●────────────────────────╭●─╭●─╭●────────────────┤ ╭State
         1: ─│──│──│───X─╭●─╭●─╭●──X─────────│──│──│───X─╭●─╭●─╭●──X─┤ ├State
     aux_0: ─╰X─│──│─────│──│──│──╭SemiAdder─╰X─│──│─────│──│──│─────┤ ├State
     aux_1: ────╰X─│─────│──│──│──├SemiAdder────╰X─│─────│──│──│─────┤ ├State
     aux_2: ───────╰X────│──│──│──├SemiAdder───────╰X────│──│──│─────┤ ├State
     qft_0: ─────────────╰X─│──│──├SemiAdder─────────────╰X─│──│─────┤ ├State
     qft_1: ────────────────╰X─│──├SemiAdder────────────────╰X─│─────┤ ├State
     qft_2: ───────────────────╰X─├SemiAdder───────────────────╰X────┤ ├State
    work_0: ──────────────────────├SemiAdder─────────────────────────┤ ├State
    work_1: ──────────────────────╰SemiAdder─────────────────────────┤ ╰State


    .. details::
        :title: Usage details

        **Adaptive precision via ``precision_stats``**

        For angles that do not require the full precision determined by the angle encoding and
        phase gradient wires, we may use smaller fanouts and adder, reducing the cost of the
        phase gradient rotation. However, as rotation angles are not known statically in general,
        this adaptive precision only is supported if additional information about the destribution
        of rotation angles is provided via ``precision_stats``, a dictionary mapping a number
        of bits to the number of rotations requiring that many bits.

        The bits required by an angle :math:`\phi` are the
        position of the last set bit in ``qp.math.binary_decimals(phi, len(angle_wires),
        unit=4 * np.pi)``, i.e., the precision without trailing zeros. If provided,
        the fanouts and the :class:`~.SemiAdder` are truncated to the smallest number of
        bits in ``precision_stats`` that suffices for a given angle, using :func:`~.cond`
        branches that are hinted with the corresponding probabilities via :func:`~.hint`.
        If no number of bits in ``precision_stats`` suffices, the full precision is used.
        Keys must lie between ``0`` and ``len(angle_wires)``. Zero counts are ignored, and
        a dictionary without non-zero counts is treated like ``None``. Use
        :func:`~.phase_gradient_precision_stats` to compute the statistics from the angles.

        **Example**

        .. code-block:: python

            from pennylane.transforms.decompositions import phase_gradient_precision_stats

            angles = np.array([1/4, 1/2 + 1/8]) * 4 * np.pi # (010) and (101)
            precision_stats = phase_gradient_precision_stats(angles, prec, qp.CRZ)

            truncating_decomp = make_crz_to_phase_gradient_decomp(
                angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
            )

            @qp.transforms._tape_decompose(
                gate_set={"CNOT", "SemiAdder", "PauliX"},
                fixed_decomps={qp.CRZ: truncating_decomp}
            )
            @qp.qnode(qp.device("null.qubit"))
            def truncated_circuit(phi):
                qp.CRZ(phi, [0, 1])
                return qp.state()

        The rotation by the first angle only uses two angle wires and phase gradient wires:

        >>> precision_stats
        {2: 1, 3: 1}
        >>> print(qp.draw(truncated_circuit, wire_order=wire_order)(angles[0]))
             0: ─╭●─────────────────────╭●─────────────┤ ╭State
             1: ─│───X─╭●─╭●──X─────────│───X─╭●─╭●──X─┤ ├State
         aux_0: ─│─────│──│──╭SemiAdder─│─────│──│─────┤ ├State
         aux_1: ─╰X────│──│──├SemiAdder─╰X────│──│─────┤ ├State
         qft_0: ───────╰X─│──├SemiAdder───────╰X─│─────┤ ├State
         qft_1: ──────────╰X─├SemiAdder──────────╰X────┤ ├State
        work_0: ─────────────├SemiAdder────────────────┤ ├State
        work_1: ─────────────╰SemiAdder────────────────┤ ╰State

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


    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )
    precisions, dispatch = _precision_dispatcher(precision_stats, len(angle_wires), "CRZ")

    def _resource_fn(phi, wires):  # pylint: disable=unused-argument
        # Cost from the wires in the outer scope, counting each non-zero precision that may be
        # applied once. The angle-load fanout emits one gate per *set* bit of the (concrete) angle,
        # so this is an upper bound (exact=False below).
        resources = {}
        for p in precisions - {0}:
            target_op = qp.SemiAdder(Wire[p], Wire[p], Wire[len(work_wires)])
            # angle-load and phase-flip fanouts abstractify to the same op (control_values aren't
            # part of the abstract representation), hence the repeated ``fanout`` below.
            fanout = qp.ctrl(qp.MultiX(Bool[p], Wire[p]), control=Wire[1])
            compute_op = uncompute_op = prod(fanout, fanout)
            resources[ChangeOpBasis2(compute_op, target_op, uncompute_op)] = 1
        return resources

    @qp.register_resources(_resource_fn, exact=False)
    def _crz_phase_gradient_decomp(phi, wires):
        dispatch(
            phi,
            lambda p: _crz_phase_gradient(
                phi, wires, angle_wires[:p], phase_grad_wires[:p], work_wires
            ),
        )

    return _crz_phase_gradient_decomp

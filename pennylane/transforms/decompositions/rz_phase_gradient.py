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
Decomposition rule for RZ in terms of `phase gradient states <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__
"""

import numpy as np

import pennylane as qp
from pennylane.ops.op_math.change_op_basis2 import ChangeOpBasis2
from pennylane.ops.op_math.prod2 import Prod2
from pennylane.transforms.rz_phase_gradient import _rz_phase_gradient
from pennylane.typing import Bool, Wire
from pennylane.wires import WireError, Wires


def validate_phase_gradient_wires(angle_wires, phase_grad_wires, work_wires):
    """Validate three wire registers to be valid for phase gradient decompositions.

    Args:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state.
        work_wires (Wires): work wires for :class:`~.SemiAdder` (and possibly :class:`~.QROM`)
            decomposition.

    """
    angle_wires = Wires(angle_wires)
    phase_grad_wires = Wires(phase_grad_wires)
    work_wires = Wires(work_wires)
    if len(angle_wires) != len(phase_grad_wires):
        raise WireError(
            "angle_wires and phase_grad wires must be of same size, but received "
            f"{len(angle_wires)} angle_wires and {len(phase_grad_wires)} phase_grad_wires."
        )
    if len(phase_grad_wires) - 1 > len(work_wires):
        raise WireError(
            f"work_wires need to be at least of size len(phase_grad_wires) - 1, but received "
            f"{len(work_wires)} work_wires and {len(phase_grad_wires)-1=}"
        )
    return angle_wires, phase_grad_wires, work_wires


def make_rz_to_phase_gradient_decomp(
    angle_wires, phase_grad_wires, work_wires, adaptive_precision=True
):
    r"""
    Create a custom decomposition rule for :class:`~.RZ` gates.

    This is a temporary workaround before moving to `capture` as default frontend, which unlocks dynamic wire allocation.
    Here, we explicitly provide the necessary wires for the `phase gradient decomposition of RZ <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__.
    This way, this function can be used in a workflow context that explicitly uses those wires to generate this decomposition rule, which can then be used
    as ``alt_decomps`` or ``fixed_decomp`` within :func:`~.pennylane.decompose` (when using the graph-based decomposition system).

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state
        work_wires (Wires): additional work wires for :class:`~.SemiAdder` decomposition
        adaptive_precision (bool): If ``True`` (default), narrow the ``SemiAdder`` for each concrete
            angle to the bits up to its least-significant set bit (dropping trailing zero bits) and
            skip angles that round to zero. If ``False``, always construct the full
            ``len(angle_wires)``-bit adder.

    Returns:
        qp.decomposition.DecompositionRule: decomposition rule to be used within :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp_double_phase`, :func:`~.make_selectpaulirot_to_phase_gradient_decomp`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~.RZ` gate using the custom decomposition rule
    that we generate from within the context of the example, where all auxiliary wires exist.

    .. code-block:: python

        import pennylane as qp
        from pennylane.transforms.decompositions import make_rz_to_phase_gradient_decomp
        import numpy as np

        qp.decomposition.enable_graph()

        prec = 3
        phi = (1/2 + 1/4 + 1/8) * 2 * np.pi # binary rep is (111)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_rz_to_phase_gradient_decomp(
            angle_wires, phase_grad_wires, work_wires
        )

        gate_set = {"CNOT", "SemiAdder", "GlobalPhase", "PauliX"}

        @qp.transforms.decompose(gate_set=gate_set, fixed_decomps={qp.RZ: custom_decomp})
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.RZ(phi, 0)
            return qp.state()

        specs = qp.specs(circuit)()["resources"].quantum_operations

    The resulting circuit corresponds to the `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__ of RZ,
    containing two CNOT fanouts corresponding to the binary representation of the angle (111 in this case), the :class:`~SemiAdder`, and a :class:`~GlobalPhase`.

    >>> specs
    {'GlobalPhase': 1, 'CNOT': 6, 'SemiAdder': 1}
    >>> print(qp.draw(circuit)())
         0: ─╭GlobalPhase(2.75)─╭●─╭●─╭●────────────╭●─╭●─╭●─┤ ╭State
     aux_0: ─├GlobalPhase(2.75)─╰X─│──│──╭SemiAdder─╰X─│──│──┤ ├State
     aux_1: ─├GlobalPhase(2.75)────╰X─│──├SemiAdder────╰X─│──┤ ├State
     aux_2: ─├GlobalPhase(2.75)───────╰X─├SemiAdder───────╰X─┤ ├State
     qft_0: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
     qft_1: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
     qft_2: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
    work_0: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
    work_1: ─╰GlobalPhase(2.75)──────────╰SemiAdder──────────┤ ╰State

    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )

    def _resource_fn(phi, wires):  # pylint: disable=unused-argument
        # Full-precision cost from the angle_wires etc. in the outer scope. With adaptive_precision
        # the compiled adder can be narrower, so this is an upper bound (exact=False below).
        target_op = qp.SemiAdder(
            Wire[len(angle_wires)],
            Wire[len(phase_grad_wires)],
            Wire[len(work_wires)],
        )
        precision = len(angle_wires)
        fanout = qp.ctrl(qp.MultiX(Bool[precision], Wire[precision]), control=Wire[1])
        change_basis_rep = ChangeOpBasis2(fanout, target_op, fanout)
        return {change_basis_rep: 1, qp.GlobalPhase: 1}

    # MultiX only emits a gate per set bit, and adaptive_precision may narrow the adder further, so
    # the gate count depends on the concrete angle.
    @qp.register_resources(_resource_fn, exact=False)
    def _decomp_fn(phi, wires):
        qp.GlobalPhase(phi / 2)
        _rz_phase_gradient(
            phi, wires, angle_wires, phase_grad_wires, work_wires, adaptive_precision
        )

    return _decomp_fn


def make_rz_to_phase_gradient_decomp_double_phase(
    angle_wires, phase_grad_wires, work_wires, adaptive_precision=True
):
    r"""
    Create a custom decomposition rule for :class:`~.RZ` gates using the double-phase trick.

    Same wiring as :func:`~.make_rz_to_phase_gradient_decomp`, but the angle bits are loaded
    unconditionally (Clifford ``X`` gates) and the unwanted phase on :math:`|0\rangle` is
    cancelled by flipping the phase-gradient register when the target is :math:`|0\rangle`.
    That is the same trick used in :func:`~.make_crz_to_phase_gradient_decomp`.

    Compared to :func:`~.make_rz_to_phase_gradient_decomp`, this spends a fixed ``2p`` CNOTs
    (independent of the angle) plus ``2 \times n_{\mathrm{set}}`` Clifford ``X`` gates, instead
    of ``2 \times n_{\mathrm{set}}`` CNOTs. It is cheaper when the binary angle is dense
    (``n_{\mathrm{set}}`` close to ``p``) and more expensive for sparse angles.

    The angle is encoded in units of :math:`4\pi` (same as :func:`~.make_crz_to_phase_gradient_decomp`),
    so that :math:`+\theta` on :math:`|1\rangle` and :math:`-\theta` on :math:`|0\rangle` realize
    :class:`~.RZ` rather than :class:`~.PhaseShift`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state
        work_wires (Wires): additional work wires for :class:`~.SemiAdder` decomposition
        adaptive_precision (bool): If ``True`` (default), narrow the ``SemiAdder`` and the
            phase-gradient flip for each concrete angle to the bits up to its least-significant
            set bit (dropping trailing zero bits) and skip angles that round to zero. If
            ``False``, always construct the full ``len(angle_wires)``-bit circuit.

    Returns:
        qp.decomposition.DecompositionRule: decomposition rule to be used within :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp`, :func:`~.make_crz_to_phase_gradient_decomp`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~.RZ` gate using the
    double-phase factory. The angle ``111`` loads three unconditional ``X`` gates, and the
    phase-gradient register is flipped with ``p`` CNOTs controlled on :math:`|0\rangle`.

    .. code-block:: python

        import pennylane as qp
        from pennylane.transforms.decompositions import make_rz_to_phase_gradient_decomp_double_phase
        import numpy as np

        qp.decomposition.enable_graph()

        prec = 3
        phi = (1/2 + 1/4 + 1/8) * 4 * np.pi # binary rep is (111)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_rz_to_phase_gradient_decomp_double_phase(
            angle_wires, phase_grad_wires, work_wires
        )

        gate_set = {"CNOT", "SemiAdder", "PauliX"}

        @qp.transforms.decompose(gate_set=gate_set, fixed_decomps={qp.RZ: custom_decomp})
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.RZ(phi, 0)
            return qp.state()

        specs = qp.specs(circuit)()["resources"].quantum_operations

    >>> specs
    {'PauliX': 10, 'CNOT': 6, 'SemiAdder': 1}
    >>> wire_order = [0] + angle_wires + phase_grad_wires + work_wires
    >>> print(qp.draw(circuit, wire_order=wire_order)())
         0: ──X─╭●─╭●─╭●──X──────────X─╭●─╭●─╭●──X─┤ ╭State
     aux_0: ──X─│──│──│──╭SemiAdder──X─│──│──│─────┤ ├State
     aux_1: ──X─│──│──│──├SemiAdder──X─│──│──│─────┤ ├State
     aux_2: ──X─│──│──│──├SemiAdder──X─│──│──│─────┤ ├State
     qft_0: ────╰X─│──│──├SemiAdder────╰X─│──│─────┤ ├State
     qft_1: ───────╰X─│──├SemiAdder───────╰X─│─────┤ ├State
     qft_2: ──────────╰X─├SemiAdder──────────╰X────┤ ├State
    work_0: ─────────────├SemiAdder────────────────┤ ├State
    work_1: ─────────────╰SemiAdder────────────────┤ ╰State

    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )

    def _resource_fn(phi, wires):  # pylint: disable=unused-argument
        # Full-precision cost from the wires in the outer scope. The unconditional angle-load
        # MultiX emits one X per *set* bit, and adaptive_precision may narrow the adder and the
        # phase-gradient flip, so this is an upper bound (exact=False below).
        precision = len(angle_wires)
        target_op = qp.SemiAdder(
            Wire[precision],
            Wire[precision],
            Wire[len(work_wires)],
        )
        angle_load = qp.MultiX(Bool[precision], Wire[precision])
        phg_flip = qp.ctrl(qp.MultiX(Bool[precision], Wire[precision]), control=Wire[1])
        # Prod is right-to-left: apply angle_load, then phg_flip.
        compute_op = uncompute_op = Prod2((phg_flip, angle_load))
        change_basis_rep = ChangeOpBasis2(compute_op, target_op, uncompute_op)
        return {change_basis_rep: 1}

    @qp.register_resources(_resource_fn, exact=False)
    def _decomp_fn(phi, wires):
        precision = len(angle_wires)
        # Double-phase applies +θ on |1⟩ and -θ on |0⟩, i.e. RZ(2θ). Encode φ in units of 4π
        # so that θ = φ/2 (same convention as :func:`~.make_crz_to_phase_gradient_decomp`).
        binary_int = qp.math.binary_decimals(phi, precision, unit=4 * np.pi)
        ang_wires = angle_wires
        phg_wires = phase_grad_wires

        if adaptive_precision and not qp.math.is_abstract(phi):
            width = qp.math.where(binary_int)[0].max(initial=-1) + 1
            if width == 0:
                return
            binary_int = binary_int[:width]
            ang_wires = angle_wires[:width]
            phg_wires = phase_grad_wires[:width]

        wire = Wires(wires)[0]

        def _compute_fn():
            qp.MultiX(binary_int, ang_wires)
            qp.ctrl(
                qp.MultiX([1] * len(phg_wires), phg_wires),
                control=wire,
                control_values=[0],
            )

        target_op = qp.SemiAdder(ang_wires, phg_wires, work_wires=work_wires)
        qp.change_op_basis(_compute_fn, target_op, _compute_fn)

    return _decomp_fn

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

from functools import partial

import numpy as np

import pennylane as qp
from pennylane.ops.op_math.change_op_basis2 import ChangeOpBasis2
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


def _precisions_to_support(precision_stats, max_precision):
    """Validate ``precision_stats`` and return the non-zero precisions that a decomposition rule
    may use, always including the full precision ``max_precision``."""
    if not precision_stats:
        return {max_precision}
    if any(not 0 <= p <= max_precision for p in precision_stats):
        raise ValueError(
            f"precision_stats keys must be between 0 and {max_precision}, but got "
            f"{sorted(precision_stats)}."
        )
    return {max_precision} | {p for p in precision_stats if p > 0}


def _dispatch_precision(binary_int, precision_stats, branch_fn):
    """Apply ``branch_fn(p)`` for the smallest precision ``p`` in ``precision_stats`` that can
    represent ``binary_int`` without its trailing zeros. If there is no such ``p``, the full
    precision is used. The branches are hinted with the probabilities given by
    ``precision_stats``.

    Args:
        binary_int (TensorLike): binary representation(s) of the angle(s), with the precision
            along the last axis. May be traced.
        precision_stats (dict[int, int]): number of rotations for each precision.
        branch_fn (Callable): function that applies the operations for a given precision.
    """
    max_precision = qp.math.shape(binary_int)[-1]
    # Position of the last set bit across all angles, i.e. the required precision
    required = qp.math.max(binary_int * qp.math.arange(1, max_precision + 1))

    total = sum(precision_stats.values())
    precisions = sorted(precision_stats)
    preds = [required <= p for p in precisions]
    branches = [
        qp.hint(partial(branch_fn, p), {"branch-prob": precision_stats[p] / total})
        for p in precisions
    ]
    if precisions[-1] == max_precision:
        preds.pop()
        otherwise = branches.pop()
    else:
        otherwise = partial(branch_fn, max_precision)

    if not branches:
        otherwise()
        return
    qp.cond(preds[0], branches[0], otherwise, elifs=list(zip(preds[1:], branches[1:])))()


def make_rz_to_phase_gradient_decomp(
    angle_wires, phase_grad_wires, work_wires, precision_stats=None
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
        precision_stats (dict[int, int] | None): Optional statistics of the number of bits
            required by the rotation angles, mapping a number of bits to the number of rotations
            requiring that many bits (i.e., the precision without trailing zeros). If provided,
            the data loading and the :class:`~.SemiAdder` are truncated to the smallest number of
            bits in ``precision_stats`` that suffices for a given angle, using :func:`~.cond`
            branches that are hinted with the corresponding probabilities via :func:`~.hint`.
            If no number of bits in ``precision_stats`` suffices, the full precision is used.
            Keys must lie between ``0`` and ``len(angle_wires)``. An empty dictionary is treated
            like ``None``. See the note below for caveats. Defaults to ``None``.

    Returns:
        qp.decomposition.DecompositionRule: decomposition rule to be used within :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_selectpaulirot_to_phase_gradient_decomp`

    .. note::

        When using ``precision_stats``, keep the following in mind:

        * ``precision_stats`` is only expected to work with program capture enabled
          (``qp.capture.enable()``). Without program capture, the angles must be concrete;
          traced angles, e.g., under ``qp.qjit(capture=False)``, raise an error.
        * The resource function of the rule adds up the cost of all supported precisions, so
          that the graph-based decomposition system accounts for the operators in all
          conditional branches. This over-reports the cost of the rule, so that the
          decomposition graph may prefer another rule over this one when it is passed via
          ``alt_decomps``. Use ``fixed_decomps`` to enforce this rule.
        * Angles that require more bits than the largest key in ``precision_stats`` fall back
          to the full precision. Unless ``len(angle_wires)`` is a key in ``precision_stats``,
          this fallback branch is hinted with probability ``0``, so that resource estimates
          based on the hints do not account for these angles.

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

    The resulting circuit corresponds to the `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__ of RZ,
    containing two CNOT fanouts corresponding to the binary representation of the angle (111 in this case), the :class:`~SemiAdder`, and a :class:`~GlobalPhase`.

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
    precision = len(angle_wires)
    precisions = _precisions_to_support(precision_stats, precision)

    def _resource_fn(phi, wires):  # pylint: disable=unused-argument
        # rz decomposition costs, using information about angle_wires etc from the outer scope.
        # With precision_stats, each supported precision is counted once.
        resources = {qp.GlobalPhase: 1}
        for p in precisions:
            target_op = qp.SemiAdder(Wire[p], Wire[p], Wire[len(work_wires)])
            fanout = qp.ctrl(qp.MultiX(Bool[p], Wire[p]), control=Wire[1])
            resources[ChangeOpBasis2(fanout, target_op, fanout)] = 1
        return resources

    # MultiX only emits a gate per set bit, so the gate count depends on the concrete angle.
    @qp.register_resources(_resource_fn, exact=False)
    def _rz_phase_gradient_decomp(phi, wires):
        qp.GlobalPhase(phi / 2)
        if not precision_stats:
            _rz_phase_gradient(phi, wires, angle_wires, phase_grad_wires, work_wires)
            return

        binary_int = qp.math.binary_decimals(phi, precision, unit=2 * np.pi)
        control_wire = Wires(wires)[0]

        def _branch(p):
            if p == 0:
                return

            def compute_fn():
                qp.ctrl(qp.MultiX(binary_int[:p], angle_wires[:p]), control=control_wire)

            target_fn = partial(qp.SemiAdder, angle_wires[:p], phase_grad_wires[:p], work_wires)
            qp.change_op_basis(compute_fn, target_fn, compute_fn)

        _dispatch_precision(binary_int, precision_stats, _branch)

    return _rz_phase_gradient_decomp

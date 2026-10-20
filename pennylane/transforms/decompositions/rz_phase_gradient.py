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

from collections import Counter
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


def _required_precision(binary_int):
    """Position of the last set bit across all binary representations in ``binary_int``, with
    the precision along the last axis. This is the number of bits required to represent all of
    them exactly."""
    return qp.math.max(binary_int * qp.math.arange(1, qp.math.shape(binary_int)[-1] + 1))


def _precision_stats(angles, precision, unit):
    """Compute ``precision_stats`` for a phase gradient decomposition rule from the angles of the
    operators that it will decompose.

    Args:
        angles (Sequence[TensorLike]): angle(s) of each operator. For operators with multiple
            angles, the required precision is the maximum over all of them.
        precision (int): full precision, i.e., the number of angle wires.
        unit (float): unit of the angles, see :func:`~.math.binary_decimals`.

    Returns:
        dict[int, int]: number of operators for each required precision.
    """
    required = (
        int(_required_precision(qp.math.binary_decimals(a, precision, unit=unit))) for a in angles
    )
    return dict(Counter(required))


def _precision_dispatcher(precision_stats, precision, unit):
    """Validate ``precision_stats`` and create a function that applies a phase gradient circuit
    at the smallest precision in ``precision_stats`` that suffices for the given angle(s).

    Args:
        precision_stats (dict[int, int] | None): number of rotations for each required precision.
            ``None`` or an empty dictionary always applies the full ``precision``.
        precision (int): full precision, i.e., the number of angle wires.
        unit (float): unit of the angles, see :func:`~.math.binary_decimals`.

    Returns:
        tuple[set[int], Callable]: The non-zero precisions that may be applied, including the
        full ``precision``, and a function ``dispatch(angles, apply_fn)``, where
        ``apply_fn(p)`` applies the full-precision circuit to the leading ``p`` angle and phase
        gradient wires. ``angles`` may be traced if program capture is enabled.
    """
    precision_stats = precision_stats or {precision: 1}
    if any(not 0 <= p <= precision for p in precision_stats):
        raise ValueError(
            f"precision_stats keys must be between 0 and {precision}, but got "
            f"{sorted(precision_stats)}."
        )
    total = sum(precision_stats.values())
    probs = {p: precision_stats[p] / total for p in sorted(precision_stats)}

    def dispatch(angles, apply_fn):
        def _branch(p):
            if p:
                apply_fn(p)

        if probs.keys() == {precision}:
            _branch(precision)
            return

        # The circuit at precision p reproduces the bits of the angles exactly if p >= required,
        # because the bits beyond p are zero, so that rounding at precision p yields the same
        # bits as rounding at the full precision.
        required = _required_precision(qp.math.binary_decimals(angles, precision, unit=unit))
        branches = [
            (required <= p, qp.hint(partial(_branch, p), {"branch-prob": prob}))
            for p, prob in probs.items()
        ]
        # Without full precision in ``precision_stats``, the fallback is unhinted (probability 0)
        otherwise = branches.pop()[1] if precision in probs else partial(_branch, precision)
        (pred, true_fn), *elifs = branches
        qp.cond(pred, true_fn, otherwise, elifs=elifs)()

    return {precision, *probs} - {0}, dispatch


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
            requiring that many bits. The bits required by an angle :math:`\phi` are the
            position of the last set bit in ``qp.math.binary_decimals(phi, len(angle_wires),
            unit=2 * np.pi)``, i.e., the precision without trailing zeros. If provided,
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

        * With program capture enabled (``qp.capture.enable()``), the angles may be traced.
          Without program capture, the angles must be concrete; traced angles, e.g., under
          ``qp.qjit(capture=False)``, raise an error.
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
     qft_0: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
     qft_1: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
     qft_2: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
    work_0: ─├GlobalPhase(2.75)──────────├SemiAdder──────────┤ ├State
    work_1: ─╰GlobalPhase(2.75)──────────╰SemiAdder──────────┤ ╰State

    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )
    precisions, dispatch = _precision_dispatcher(precision_stats, len(angle_wires), 2 * np.pi)

    def _resource_fn(phi, wires):  # pylint: disable=unused-argument
        # rz decomposition costs, using information about angle_wires etc from the outer scope.
        # Each precision that may be applied is counted once.
        resources = {qp.GlobalPhase: 1}
        for p in precisions:
            target_op = qp.SemiAdder(Wire[p], Wire[p], Wire[len(work_wires)])
            fanout = qp.ctrl(qp.MultiX(Bool[p], Wire[p]), control=Wire[1])
            resources[ChangeOpBasis2(fanout, target_op, fanout)] = 1
        return resources

    # MultiX only emits a gate per set bit, so the gate count depends on the concrete angle.
    # Thus we need to set ``exact=False``
    @qp.register_resources(_resource_fn, exact=False)
    def _rz_phase_gradient_decomp(phi, wires):
        qp.GlobalPhase(phi / 2)
        dispatch(
            phi,
            lambda p: _rz_phase_gradient(
                phi, wires, angle_wires[:p], phase_grad_wires[:p], work_wires
            ),
        )

    return _rz_phase_gradient_decomp

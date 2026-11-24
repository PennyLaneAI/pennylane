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


"""Shared wire validation and precision helpers for phase-gradient rules."""

from collections import Counter
from functools import partial

import numpy as np

from pennylane import capture, math, ops
from pennylane.wires import WireError, Wires


def validate_phase_gradient_wires(angle_wires, phase_grad_wires, work_wires):
    """Validate three wire registers to be valid for phase gradient decompositions.

    Args:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state.
        work_wires (Wires): work wires for :class:`~.SemiAdder` (and possibly :class:`~.QROMLoad`)
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


_ANGLE_UNITS = {"RZ": 2 * np.pi, "CRZ": 4 * np.pi, "SelectPauliRot": 4 * np.pi}


def _angle_unit(op_type):
    """Unit of the angles in the phase gradient decomposition of ``op_type``, which may be an
    operator class or its name."""
    name = getattr(op_type, "__name__", op_type)
    if name not in _ANGLE_UNITS:
        raise ValueError(
            f"op_type must be one of {list(_ANGLE_UNITS)} or the corresponding class, but got "
            f"{op_type}."
        )
    return _ANGLE_UNITS[name]


def _required_precision(binary_int):
    """Position of the last set bit across all binary representations in ``binary_int``, with
    the precision along the last axis. This is the number of bits required to represent all of
    them exactly."""
    return math.max(binary_int * math.arange(1, math.shape(binary_int)[-1] + 1))


def phase_gradient_precision_stats(angles, precision, op_type):
    r"""Compute the ``precision_stats`` argument of the phase gradient decomposition rule
    factories from the rotation angles of the operators that a rule will decompose.

    A phase gradient decomposition with ``precision`` angle wires represents an angle
    :math:`\phi` by the ``precision`` leading binary decimals of :math:`\phi/u` (rounded to
    nearest), where the unit is :math:`u=2\pi` for :class:`~.RZ` and :math:`u=4\pi` for
    :class:`~.CRZ` and :class:`~.SelectPauliRot`. If the trailing decimals are zero, the rotation
    can be realized with fewer bits, i.e., with a smaller :class:`~.SemiAdder` and less data
    loading.
    This function counts how many operators require each number of bits. Passing the result as
    ``precision_stats`` to :func:`~.make_rz_to_phase_gradient_decomp`,
    :func:`~.make_crz_to_phase_gradient_decomp` or
    :func:`~.make_selectpaulirot_to_phase_gradient_decomp` makes the produced rule apply each of
    these operators at exactly the number of bits it requires, and hints the conditional
    branches with the observed frequencies.

    Args:
        angles (Sequence[TensorLike]): The angle(s) of each operator. For operators with
            multiple angles, such as :class:`~.SelectPauliRot`, the bits required by the operator
            are the maximum over all of its angles. The angles must be concrete.
        precision (int): The full precision of the decomposition rule, i.e., the number of angle
            wires.
        op_type (type | str): The type of the operators, i.e., :class:`~.RZ`, :class:`~.CRZ`
            or :class:`~.SelectPauliRot`, or its name. It determines the unit of the angles in
            the respective decomposition rule.

    Returns:
        dict[int, int]: The number of operators requiring each number of bits, sorted by the
        number of bits.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp`,
        :func:`~.make_crz_to_phase_gradient_decomp`,
        :func:`~.make_selectpaulirot_to_phase_gradient_decomp`

    **Example**

    Consider four :class:`~.RZ` rotations that we want to decompose with four angle wires. The
    first angle, :math:`\tfrac{1}{2}\cdot 2\pi`, has the binary representation :math:`(0.1000)_2`
    and thus only requires one bit, whereas :math:`0.1 \cdot 2\pi` is rounded to
    :math:`(0.0010)_2`, requiring three bits:

    >>> from pennylane.decomposition_rules import (
    ...     phase_gradient_precision_stats,
    ... )
    >>> angles = np.array([1 / 2, 1 / 4, 3 / 8, 0.1]) * 2 * np.pi
    >>> phase_gradient_precision_stats(angles, precision=4, op_type=qp.RZ)
    {1: 1, 2: 1, 3: 2}

    See :func:`~.make_rz_to_phase_gradient_decomp` for an example that uses these statistics
    in a decomposition rule.

    For :class:`~.SelectPauliRot`, we pass the angles of each operator. The operator type may
    also be specified by its name:

    >>> ops = [
    ...     qp.SelectPauliRot(np.array([1 / 2, 1 / 4]) * 4 * np.pi, control_wires=[0], target_wire=1),
    ...     qp.SelectPauliRot(np.array([1 / 2, 0]) * 4 * np.pi, control_wires=[0], target_wire=1),
    ... ]
    >>> angles = [op.data[0] for op in ops]
    >>> phase_gradient_precision_stats(angles, precision=4, op_type="SelectPauliRot")
    {1: 1, 2: 1}

    .. note::

        A :class:`~.SelectPauliRot` without control wires is decomposed into a plain
        :class:`~.RX`, :class:`~.RY` or :class:`~.RZ` rotation by
        :func:`~.make_selectpaulirot_to_phase_gradient_decomp`. If the resulting :class:`~.RZ`
        rotations are decomposed by a rule from :func:`~.make_rz_to_phase_gradient_decomp`,
        include their angles in the statistics for ``op_type=qp.RZ``, which uses the unit
        :math:`2\pi` instead of :math:`4\pi`.
    """
    unit = _angle_unit(op_type)
    required = (
        int(_required_precision(math.binary_decimals(a, precision, unit=unit))) for a in angles
    )
    return dict(sorted(Counter(required).items()))


def _precision_dispatcher(precision_stats, precision, op_type):
    """Validate ``precision_stats`` and create a function that applies a phase gradient circuit
    at the smallest precision in ``precision_stats`` that suffices for the given angle(s).

    Args:
        precision_stats (dict[int, int] | None): number of rotations for each required precision.
            Zero counts are ignored. Without non-zero counts, the full ``precision`` is always
            applied.
        precision (int): full precision, i.e., the number of angle wires.
        op_type (type | str): type of the decomposed operators, determining the angle unit.

    Returns:
        tuple[set[int], Callable]: The precisions that may be applied, including the full
        ``precision``, and a function ``dispatch(angles, apply_fn)``, where ``apply_fn(p)``
        applies the full-precision circuit to the leading ``p`` angle and phase gradient wires.
        No conditional is used if the full ``precision`` is the only precision. ``angles`` may
        be traced if program capture is enabled.
    """
    precision_stats = {p: c for p, c in (precision_stats or {}).items() if c} or {precision: 1}
    if any(not 0 <= p <= precision for p in precision_stats):
        raise ValueError(
            f"precision_stats keys must be between 0 and {precision}, but got "
            f"{sorted(precision_stats)}."
        )
    total = sum(precision_stats.values())
    probs = {p: precision_stats[p] / total for p in sorted(precision_stats)}
    precisions = {precision, *probs}
    unit = _angle_unit(op_type)

    def dispatch(angles, apply_fn):
        def _branch(p):
            if p:
                apply_fn(p)

        # If only a single precision or no precision_stats was provided, we just call the branch
        # function for that single precision, without cond branching.
        if precisions == {precision}:
            _branch(precision)
            return

        # The circuit at precision p reproduces the bits of the angles exactly if p >= required,
        # because the bits beyond p are zero, so that rounding at precision p yields the same
        # bits as rounding at the full precision.
        required = _required_precision(math.binary_decimals(angles, precision, unit=unit))
        branches = [
            (required <= p, capture.hint(partial(_branch, p), {"branch-prob": prob}))
            for p, prob in probs.items()
        ]
        # Without full precision in ``precision_stats``, the fallback is unhinted (probability 0)
        otherwise = branches.pop()[1] if precision in probs else partial(_branch, precision)
        (pred, true_fn), *elifs = branches
        ops.cond(pred, true_fn, otherwise, elifs=elifs)()

    return precisions, dispatch

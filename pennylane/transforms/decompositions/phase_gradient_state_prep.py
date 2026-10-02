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
r"""Jones distillation decomposition for phase-gradient state preparation."""

import math

import pennylane as qp
from pennylane.decomposition import register_resources
from pennylane.typing import Wire
from pennylane.wires import WireError, Wires, validate_no_wire_overlaps


def _jones_num_rounds(num_wires):
    """Return a finite-size-safe number of Jones distillation rounds."""
    if num_wires < 3:
        return 0

    numerator = 2 * num_wires - 2 * math.log2(math.pi)
    asymptotic_rounds = max(1, math.ceil(math.log2(numerator / math.log2(9))))

    # Jones's sideband estimate is asymptotic. One extra round closes the finite-size gap
    # for small registers while preserving the O(log(n)) round count.
    return asymptotic_rounds + int(num_wires >= 5)


def _jones_register_sizes(num_wires):
    """Return register sizes from the two-qubit seed through the final round."""
    if num_wires < 3:
        return (num_wires,)

    sizes = [num_wires]
    for _ in range(_jones_num_rounds(num_wires) - 1):
        sizes.append(max(3, (sizes[-1] + 1) // 2))
    return (2, *reversed(sizes))


def _required_workspace(register_sizes):
    """Return the peak number of wires needed by a depth-first local-retry tree."""
    required = register_sizes[0]
    for child_size, output_size in zip(register_sizes, register_sizes[1:]):
        required = max(3 * output_size - 1, child_size + required)
    return required


def _jones_seed(wires):
    """Prepare the negative-phase Clifford seed from Jones, Eq. (4)."""
    for wire in wires:
        qp.Hadamard(wire)
    if wires:
        qp.Z(wires[0])
    if len(wires) > 1:
        qp.adjoint(qp.S(wires[1]))


def _all_zero(measurements):
    """Combine captured measurement outcomes into one Boolean success value."""
    success = qp.math.logical_not(measurements[0])
    for measurement in measurements[1:]:
        success = qp.math.logical_and(success, qp.math.logical_not(measurement))
    return success


def _reset(wires):
    """Measure and reset a register."""
    for wire in wires:
        qp.measure(wire, reset=True)


def _distillation_attempt(
    level, output_wires, scratch_wires, register_sizes, mode
):  # pylint: disable=too-many-arguments
    """Prepare two children and perform one symmetric Jones distillation attempt."""
    output_size = register_sizes[level]
    child_size = register_sizes[level - 1]
    source_wires = scratch_wires[:output_size]
    work_wires = scratch_wires[output_size : 2 * output_size - 1]
    remaining_wires = scratch_wires[2 * output_size - 1 :]

    output_tail = output_wires[child_size:]
    source_tail = source_wires[child_size:]

    _prepare_distilled_state(
        level - 1,
        output_wires[:child_size],
        output_tail + scratch_wires,
        register_sizes,
        mode,
    )
    _prepare_distilled_state(
        level - 1,
        source_wires[:child_size],
        source_tail + output_tail + work_wires + remaining_wires,
        register_sizes,
        mode,
    )

    for wire in output_tail + source_tail:
        qp.Hadamard(wire)

    qp.SemiAdder(source_wires, output_wires, work_wires)

    measurements = []
    for wire in source_wires:
        qp.Hadamard(wire)
        measurements.append(
            qp.measure(wire, reset=True, postselect=0 if mode == "postselect" else None)
        )

    if mode == "postselect":
        return True

    success = _all_zero(measurements)

    @qp.cond(qp.math.logical_not(success))
    def reset_output():
        _reset(output_wires)

    reset_output()
    return success


def _prepare_distilled_state(level, output_wires, scratch_wires, register_sizes, mode):
    """Recursively prepare one node while preserving completed sibling registers."""
    if level == 0:
        _jones_seed(output_wires)
        return

    if mode == "postselect":
        _distillation_attempt(level, output_wires, scratch_wires, register_sizes, mode)
        return

    def not_success(success):
        return qp.math.logical_not(success)

    @qp.while_loop(not_success)
    def retry(_success):
        return _distillation_attempt(level, output_wires, scratch_wires, register_sizes, mode)

    retry(False)


def _accumulate(resources, resource, count):
    resources[resource] = resources.get(resource, 0) + count


def _distillation_resources(num_wires, mode):
    """Resources in the statically captured tree, counting each retry body once."""
    register_sizes = _jones_register_sizes(num_wires)
    resources = {}

    if len(register_sizes) == 1:
        _accumulate(resources, qp.Hadamard, num_wires)
        if num_wires:
            _accumulate(resources, qp.Z, 1)
        if num_wires > 1:
            _accumulate(resources, qp.adjoint(qp.S(Wire[1])), 1)
        return resources

    num_states = 2 ** (len(register_sizes) - 1)
    seed_size = register_sizes[0]
    _accumulate(resources, qp.Hadamard, num_states * seed_size)
    _accumulate(resources, qp.Z, num_states)
    _accumulate(resources, qp.adjoint(qp.S(Wire[1])), num_states)

    for level, (child_size, output_size) in enumerate(
        zip(register_sizes, register_sizes[1:]), start=1
    ):
        num_nodes = 2 ** (len(register_sizes) - level - 1)
        _accumulate(
            resources,
            qp.SemiAdder(Wire[output_size], Wire[output_size], Wire[output_size - 1]),
            num_nodes,
        )
        _accumulate(
            resources,
            qp.Hadamard,
            num_nodes * (2 * (output_size - child_size) + output_size),
        )
        mid_measure = qp.ops.MidMeasure(
            Wire[1], reset=True, postselect=0 if mode == "postselect" else None
        )
        _accumulate(resources, mid_measure, num_nodes * output_size)
        if mode == "repeat-until-success":
            _accumulate(
                resources,
                qp.ops.MidMeasure(Wire[1], reset=True),
                num_nodes * output_size,
            )

    return resources


def make_phase_gradient_distillation_decomp(aux_wires, work_wires, mode="repeat-until-success"):
    r"""Create an opt-in Jones distillation rule for :class:`~.PhaseGradientStatePrep`.

    The rule uses only Clifford gates, :class:`~.SemiAdder`, mid-circuit measurements, and
    classical control. ``mode="repeat-until-success"`` implements local retries with nested
    :func:`~.while_loop` instances. ``mode="postselect"`` represents the successful branch and
    is useful for analytic simulation and fidelity checks.

    The output, auxiliary, and work registers require :math:`n`, :math:`n`, and :math:`n-1`
    wires, respectively. All wires must be distinct and start in :math:`|0\rangle`. The
    decomposition uses the negative-phase convention of :class:`~.PhaseGradientStatePrep`.

    The resources visible to ``specs(..., level="all-mlir")`` describe each static loop body
    once, not the expected number of retries. If :math:`C_{r-1}` is the expected child cost,
    :math:`A_r` the cost of one addition and verification, and :math:`p_r` its success
    probability, the expected cost obeys

    .. math::

        C_r = \frac{2 C_{r-1} + A_r}{p_r}.

    Args:
        aux_wires (WiresLike): auxiliary register with the same size as the output register
        work_wires (WiresLike): at least :math:`n-1` clean wires used by :class:`~.SemiAdder`
        mode (str): ``"repeat-until-success"`` or ``"postselect"``

    Returns:
        pennylane.decomposition.DecompositionRule: the fixed decomposition rule

    .. note::

        Include ``"measure"`` in a Catalyst graph-decomposition gate set. This preserves the
        measurement-assisted uncomputation of :class:`~.TemporaryAND`; omitting it can double
        the :math:`T` count.

        For Catalyst capture, fix the generated rule with the developer-facing ``_fix_decomp``
        function inside a :func:`~.decomposition.local_decomps` context. Keep the context active
        while lazy qjit capture or ``specs`` runs.
    """
    if mode not in {"repeat-until-success", "postselect"}:
        raise ValueError("mode must be 'repeat-until-success' or 'postselect'; " f"got {mode!r}.")

    aux_wires = Wires(aux_wires)
    work_wires = Wires(work_wires)

    def resource_fn(wires):
        return _distillation_resources(len(wires), mode)

    @register_resources(resource_fn, exact=False)
    def phase_gradient_distillation(wires):
        output_wires = Wires(wires)
        num_wires = len(output_wires)

        if len(aux_wires) != num_wires:
            raise WireError(
                "aux_wires must have the same size as the PhaseGradientStatePrep register; "
                f"got {len(aux_wires)} auxiliary wires and {num_wires} output wires."
            )
        if len(work_wires) < max(0, num_wires - 1):
            raise WireError(
                "work_wires must contain at least len(wires) - 1 wires; "
                f"got {len(work_wires)} work wires and {num_wires=}"
            )

        selected_work_wires = work_wires[: max(0, num_wires - 1)]
        validate_no_wire_overlaps(
            {
                "wires": output_wires,
                "aux_wires": aux_wires,
                "work_wires": selected_work_wires,
            }
        )

        register_sizes = _jones_register_sizes(num_wires)
        available_wires = output_wires + aux_wires + selected_work_wires
        required_wires = _required_workspace(register_sizes)
        if required_wires > len(available_wires):
            raise WireError(
                f"Jones schedule requires {required_wires} wires, but only "
                f"{len(available_wires)} were provided."
            )

        _prepare_distilled_state(
            len(register_sizes) - 1,
            list(output_wires),
            list(aux_wires + selected_work_wires),
            register_sizes,
            mode,
        )

    return phase_gradient_distillation

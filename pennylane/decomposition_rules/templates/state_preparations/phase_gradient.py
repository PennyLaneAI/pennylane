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


"""Built-in decomposition rules for templates.state_preparations.phase_gradient."""

import math
from collections import Counter, defaultdict
from functools import lru_cache

from pennylane import capture, control_flow
from pennylane import math as pl_math
from pennylane import ops, templates
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
)
from pennylane.templates.state_preparations.phase_gradient import PhaseGradientStatePrep
from pennylane.templates.state_preparations.phase_gradient_decomp_data import (
    CLIFFORD_T_GLOBAL_PHASES,
    CLIFFORD_T_SEQUENCES,
    GLOBAL_PHASES,
    INITIAL_PPRS,
    PI_OVER_8_PPRS,
)
from pennylane.typing import AbstractWires, Wire
from pennylane.wires import WireError, Wires, WiresLike, validate_no_wire_overlaps


def _ppr_counts(num_wires):
    counts = Counter()
    for init, pprs in zip(INITIAL_PPRS[:num_wires], PI_OVER_8_PPRS[:num_wires], strict=True):
        if init is not None:
            counts[init] += 1
        counts.update((8 if p.isupper() else -8, p.upper()) for p in pprs)
    return counts


def _ppr(denominator, pauli_word, wires):
    op_class = {2: ops.PPR_2, 4: ops.PPR_4, 8: ops.PPR_8}[abs(denominator)]
    return op_class(1 if denominator > 0 else -1, pauli_word, wires=wires)


def _phase_gradient_state_prep_to_ppr_resources(wires: AbstractWires):
    resources = {
        _ppr(denominator, pauli, wires=Wire[1]): count
        for (denominator, pauli), count in _ppr_counts(len(wires)).items()
    }
    resources[ops.GlobalPhase] = 1
    return resources


@register_condition(lambda wires: len(wires) <= len(PI_OVER_8_PPRS))
@register_resources(_phase_gradient_state_prep_to_ppr_resources, name="to_ppr")
def _phase_gradient_state_prep_to_ppr(wires: WiresLike):
    # Wire j is prepared in the state (|0> + exp(-i pi / 2**j)|1>) / sqrt(2) by the PPR
    # ``INITIAL_PPRS[j]`` (angle denominator, Pauli word; no gate if ``None``), followed by the π/8
    # PPRs in ``PI_OVER_8_PPRS[j]``, where upper (lower) case letters denote PPRs with angle π/8
    # (-π/8). Together with the global phase given by summing ``GLOBAL_PHASES`` over the wires,
    # the state prepared on 30 wires deviates from the phase gradient state by
    # less than 1e-12 in the 2-norm.

    num_wires = len(wires)
    for wire, init, pprs in zip(
        wires, INITIAL_PPRS[:num_wires], PI_OVER_8_PPRS[:num_wires], strict=True
    ):
        if init is not None:
            _ppr(*init, wires=wire)
        for p in pprs:
            ops.PPR_8(1 if p.isupper() else -1, p.upper(), wires=wire)
    ops.GlobalPhase(sum(GLOBAL_PHASES[: len(wires)]))


def _phase_gradient_state_prep_to_clifford_t_resources(wires: AbstractWires):
    counts = Counter("".join(CLIFFORD_T_SEQUENCES[: len(wires)]))
    resources = {getattr(ops, gate): count for gate, count in counts.items()}
    resources[ops.GlobalPhase] = 1
    return resources


@register_condition(lambda wires: len(wires) <= len(CLIFFORD_T_SEQUENCES))
@register_resources(_phase_gradient_state_prep_to_clifford_t_resources, name="to_clifford_t")
def _phase_gradient_state_prep_to_clifford_t(wires: WiresLike):
    # Wire j is prepared by the gates in ``CLIFFORD_T_SEQUENCES[j]``, which are obtained from the
    # PPR decomposition and contain one T gate per π/8 PPR. Together with the global phase
    # given by summing ``CLIFFORD_T_GLOBAL_PHASES`` over the wires, the state prepared on 30 wires
    # deviates from the phase gradient state by less than 1e-12 in the 2-norm.

    num_wires = len(wires)
    for wire, sequence in zip(wires, CLIFFORD_T_SEQUENCES[:num_wires], strict=True):
        for gate in sequence:
            getattr(ops, gate)(wire)
    ops.GlobalPhase(sum(CLIFFORD_T_GLOBAL_PHASES[:num_wires]))


add_decomps(
    PhaseGradientStatePrep,
    _phase_gradient_state_prep_to_ppr,
    _phase_gradient_state_prep_to_clifford_t,
)


def _jones_num_rounds(num_wires):
    """Return the number of Jones distillation rounds."""
    if num_wires < 3:
        return 0

    numerator = 2 * num_wires - 2 * math.log2(math.pi)
    return max(1, math.ceil(math.log2(numerator / math.log2(9))))


def _jones_register_sizes(num_wires):
    """Return register sizes from the two-qubit seed through the final round."""
    if num_wires < 3:
        return (num_wires,)

    # Children have floor(s/2) + 1 qubits. Halving even registers exactly (e.g. 7 -> 14) misses
    # the target infidelity at finite sizes.
    sizes = [num_wires]
    for _ in range(_jones_num_rounds(num_wires) - 1):
        sizes.append(max(3, sizes[-1] // 2 + 1))
    return (2, *reversed(sizes))


_MAX_SIMULATED_SIZE = 16


@lru_cache
def _jones_success_probabilities(register_sizes):
    """Return the success probability of each distillation round.

    Rounds are simulated as coefficient squaring in the Fourier basis. Rounds producing more than
    ``_MAX_SIMULATED_SIZE`` wires succeed with probability one up to about ``1e-4`` and are not
    simulated.
    """
    dim = 2 ** register_sizes[0]
    state = pl_math.exp(-2j * math.pi * pl_math.arange(dim, like="numpy") / dim) / math.sqrt(dim)

    probabilities = []
    for child_size, output_size in zip(register_sizes[:-1], register_sizes[1:], strict=True):
        if output_size > _MAX_SIMULATED_SIZE:
            probabilities.append(1.0)
            continue

        dim = 2**output_size
        tail_dim = 2 ** (output_size - child_size)
        tail = pl_math.ones(tail_dim, like="numpy") / math.sqrt(tail_dim)
        squared = (pl_math.fft.ifft(pl_math.kron(state, tail)) * math.sqrt(dim)) ** 2
        probabilities.append(float(pl_math.sum(pl_math.abs(squared) ** 2)))
        state = pl_math.fft.fft(squared / pl_math.linalg.norm(squared)) / math.sqrt(dim)

    return tuple(probabilities)


def _jones_seed(wires):
    """Prepare the Clifford seed from Jones, Eq. (4), conjugated to the negative-phase convention,
    i.e., with S replaced by S†."""
    for wire in wires:
        ops.Hadamard(wire)
    if wires:
        ops.Z(wires[0])
    if len(wires) > 1:
        ops.adjoint(ops.S(wires[1]))


def _reset(wires):
    """Reset a register with Z Pauli measurements and X corrections."""
    for wire in wires:
        ops.cond(ops.pauli_measure("Z", wire), ops.X)(wire)


def _distillation_attempt(level, output_wires, scratch_wires, register_sizes, mode):
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
        ops.Hadamard(wire)

    templates.SemiAdder(source_wires, output_wires, work_wires)

    if mode == "postselect":
        # Catalyst does not honour postselection on pauli_measure yet, and default.qubit cannot
        # execute pauli_measure, so the postselected branch uses standard mid-circuit measurements
        for wire in source_wires:
            ops.Hadamard(wire)
            ops.measure(wire, reset=True, postselect=0)
        return True

    measurements = [ops.pauli_measure("X", wire) for wire in source_wires]
    _reset(source_wires)
    success = pl_math.sum(pl_math.array(measurements, like="jax")) == 0
    success_probability = _jones_success_probabilities(register_sizes)[level - 1]

    @ops.cond(pl_math.logical_not(success))
    @capture.hint({"branch-prob": 1 - success_probability})
    def reset_output():
        _reset(output_wires)

    reset_output()
    return success


def _prepare_distilled_state(level, output_wires, scratch_wires, register_sizes, mode):
    """Recursively prepare one node while preserving completed sibling registers."""
    if level == 0:
        _jones_seed(output_wires)
        return

    def not_success(success):
        return pl_math.logical_not(success)

    expected_attempts = (
        1 / _jones_success_probabilities(register_sizes)[level - 1]
        if mode == "repeat-until-success"
        else 1
    )

    @capture.hint({"num-iters": expected_attempts})
    @control_flow.while_loop(not_success)
    def retry(success):
        # Using the carry keeps MLIR canonicalization from rebuilding the loop and dropping the hint
        return pl_math.logical_or(
            success,
            _distillation_attempt(level, output_wires, scratch_wires, register_sizes, mode),
        )

    retry(False)


def _distillation_resources(num_wires, mode):
    """Resources in the statically captured tree, counting each retry body once."""
    register_sizes = _jones_register_sizes(num_wires)
    resources = defaultdict(int)

    if len(register_sizes) == 1:
        resources[ops.Hadamard] += num_wires
        if num_wires:
            resources[ops.Z] += 1
        if num_wires > 1:
            resources[ops.adjoint(ops.S(Wire[1]))] += 1
        return resources

    num_states = 2 ** (len(register_sizes) - 1)
    seed_size = register_sizes[0]
    resources[ops.Hadamard] += num_states * seed_size
    resources[ops.Z] += num_states
    resources[ops.adjoint(ops.S(Wire[1]))] += num_states

    for level, (child_size, output_size) in enumerate(
        zip(register_sizes[:-1], register_sizes[1:], strict=True), start=1
    ):
        num_nodes = 2 ** (len(register_sizes) - level - 1)
        adder = templates.SemiAdder(Wire[output_size], Wire[output_size], Wire[output_size - 1])
        resources[adder] += num_nodes
        resources[ops.Hadamard] += num_nodes * 2 * (output_size - child_size)
        if mode == "postselect":
            resources[ops.Hadamard] += num_nodes * output_size
            mid_measure = ops.MidMeasure(Wire[1], reset=True, postselect=0)
            resources[mid_measure] += num_nodes * output_size
        else:
            # source measurements, then source and output resets
            resources[ops.PauliMeasure("X", Wire[1])] += num_nodes * output_size
            resources[ops.PauliMeasure("Z", Wire[1])] += 2 * num_nodes * output_size
            resources[ops.X] += 2 * num_nodes * output_size

    return resources


def make_phase_gradient_distillation_decomp(aux_wires, work_wires, mode="repeat-until-success"):
    r"""Create a decomposition rule for :class:`~.PhaseGradientStatePrep` based on Jones distillation.

    The rule implements the Fourier-state distillation protocol from
    `Jones (2013) <https://arxiv.org/abs/1303.3066>`__ and uses only Clifford gates,
    :class:`~.SemiAdder`, (Pauli) mid-circuit measurements, and classical control. The prepared
    state is approximate: the number of distillation rounds grows as :math:`\mathcal{O}(\log n)`
    for an :math:`n`-qubit register and targets an infidelity of at most :math:`\sin^2(\pi/2^n)`.

    The output, auxiliary, and work registers require :math:`n`, :math:`n`, and :math:`n-1`
    wires, respectively. All wires must be distinct and start in :math:`|0\rangle`. The
    decomposition uses the negative-phase convention of :class:`~.PhaseGradientStatePrep`.

    Args:
        aux_wires (WiresLike): auxiliary register with the same size as the output register,
            i.e. the target register of the ``PhaseGradientStatePrep`` operator to be decomposed.
        work_wires (WiresLike): at least :math:`n-1` clean wires used by :class:`~.SemiAdder`
        mode (str): ``"repeat-until-success"`` (default) or ``"postselect"``, see below

    Returns:
        pennylane.decomposition.DecompositionRule: the fixed decomposition rule

    **Example**

    Register the rule and select it by name in :func:`qp.decompose` via ``fixed_decomps``:

    .. code-block:: python

        import pennylane as qp
        from pennylane.decomposition_rules import make_phase_gradient_distillation_decomp

        registers = qp.registers({"grad": 3, "aux": 3, "work": 2})
        rule = make_phase_gradient_distillation_decomp(
            registers["aux"], registers["work"]
        )
        gate_set = {
            "Hadamard", "S", "Adjoint(S)", "T", "Adjoint(T)", "CNOT", "X", "Z",
            "measure", "PauliMeasure",
        }

        qp.add_decomps(qp.PhaseGradientStatePrep, rule)

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=gate_set, fixed_decomps={qp.PhaseGradientStatePrep: rule.name})
        @qp.qnode(qp.device("lightning.qubit", wires=8))
        def circuit():
            qp.PhaseGradientStatePrep(registers["grad"])
            return qp.state()

    Only Clifford+T gates and (Pauli) mid-circuit measurements remain:

    >>> sorted(qp.analyze(circuit)().resources.quantum_operations)
    ['Adjoint(S)', 'Adjoint(T)', 'CNOT', 'Hadamard', 'MidCircuitMeasure', 'PPM-w1', 'PauliX',
     'PauliZ', 'T']
    """

    # ``mode="repeat-until-success"`` implements local retries with nested
    # :func:`~.while_loop`\ s and requires program capture. Each loop is annotated with its
    # expected number of attempts via :func:`~.hint`, such that :func:`~.analyze` reports expected
    # resource counts. ``mode="postselect"`` represents the successful branch and is useful for
    # analytic simulation and fidelity checks.
    if mode not in {"repeat-until-success", "postselect"}:
        raise ValueError("mode must be 'repeat-until-success' or 'postselect'; " f"got {mode!r}.")

    aux_wires = Wires(aux_wires)
    work_wires = Wires(work_wires)
    num_work_wires = max(0, len(aux_wires) - 1)
    if len(work_wires) < num_work_wires:
        raise WireError(
            "work_wires must contain at least len(aux_wires) - 1 wires; "
            f"got {len(work_wires)} work wires and {len(aux_wires)} auxiliary wires."
        )
    work_wires = work_wires[:num_work_wires]

    def _phase_gradient_distillation_resources(wires):
        return _distillation_resources(len(wires), mode)

    @register_resources(_phase_gradient_distillation_resources, exact=False, name="distillation")
    def _phase_gradient_distillation(wires):
        output_wires = Wires(wires)
        num_wires = len(output_wires)

        if len(aux_wires) != num_wires:
            raise WireError(
                "aux_wires must have the same size as the PhaseGradientStatePrep register; "
                f"got {len(aux_wires)} auxiliary wires and {num_wires} output wires."
            )
        validate_no_wire_overlaps(
            {"wires": output_wires, "aux_wires": aux_wires, "work_wires": work_wires}
        )

        register_sizes = _jones_register_sizes(num_wires)
        _prepare_distilled_state(
            len(register_sizes) - 1,
            list(output_wires),
            list(aux_wires + work_wires),
            register_sizes,
            mode,
        )

    return _phase_gradient_distillation

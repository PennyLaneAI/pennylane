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
from collections import defaultdict
from functools import lru_cache

import pennylane as qp
from pennylane.decomposition import register_resources
from pennylane.typing import Wire
from pennylane.wires import WireError, Wires, validate_no_wire_overlaps


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
    state = qp.math.exp(-2j * math.pi * qp.math.arange(dim, like="numpy") / dim) / math.sqrt(dim)

    probabilities = []
    for child_size, output_size in zip(register_sizes[:-1], register_sizes[1:], strict=True):
        if output_size > _MAX_SIMULATED_SIZE:
            probabilities.append(1.0)
            continue

        dim = 2**output_size
        tail_dim = 2 ** (output_size - child_size)
        tail = qp.math.ones(tail_dim, like="numpy") / math.sqrt(tail_dim)
        squared = (qp.math.fft.ifft(qp.math.kron(state, tail)) * math.sqrt(dim)) ** 2
        probabilities.append(float(qp.math.sum(qp.math.abs(squared) ** 2)))
        state = qp.math.fft.fft(squared / qp.math.linalg.norm(squared)) / math.sqrt(dim)

    return tuple(probabilities)


def _jones_seed(wires):
    """Prepare the negative-phase Clifford seed from Jones, Eq. (4)."""
    for wire in wires:
        qp.Hadamard(wire)
    if wires:
        qp.Z(wires[0])
    if len(wires) > 1:
        qp.adjoint(qp.S(wires[1]))


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

    success = qp.math.sum(qp.math.array(measurements, like="jax")) == 0

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

    def not_success(success):
        return qp.math.logical_not(success)

    expected_attempts = (
        1 / _jones_success_probabilities(register_sizes)[level - 1]
        if mode == "repeat-until-success"
        else 1
    )

    @qp.hint({"num-iters": expected_attempts})
    @qp.while_loop(not_success)
    def retry(success):
        # Using the carry keeps MLIR canonicalization from rebuilding the loop and dropping the hint
        return qp.math.logical_or(
            success,
            _distillation_attempt(level, output_wires, scratch_wires, register_sizes, mode),
        )

    retry(False)


def _distillation_resources(num_wires, mode):
    """Resources in the statically captured tree, counting each retry body once."""
    register_sizes = _jones_register_sizes(num_wires)
    resources = defaultdict(int)

    if len(register_sizes) == 1:
        resources[qp.Hadamard] += num_wires
        if num_wires:
            resources[qp.Z] += 1
        if num_wires > 1:
            resources[qp.adjoint(qp.S(Wire[1]))] += 1
        return resources

    num_states = 2 ** (len(register_sizes) - 1)
    seed_size = register_sizes[0]
    resources[qp.Hadamard] += num_states * seed_size
    resources[qp.Z] += num_states
    resources[qp.adjoint(qp.S(Wire[1]))] += num_states

    for level, (child_size, output_size) in enumerate(
        zip(register_sizes[:-1], register_sizes[1:], strict=True), start=1
    ):
        num_nodes = 2 ** (len(register_sizes) - level - 1)
        adder = qp.SemiAdder(Wire[output_size], Wire[output_size], Wire[output_size - 1])
        resources[adder] += num_nodes
        resources[qp.Hadamard] += num_nodes * (2 * (output_size - child_size) + output_size)
        mid_measure = qp.ops.MidMeasure(
            Wire[1], reset=True, postselect=0 if mode == "postselect" else None
        )
        resources[mid_measure] += num_nodes * output_size
        if mode == "repeat-until-success":
            resources[qp.ops.MidMeasure(Wire[1], reset=True)] += num_nodes * output_size

    return resources


def make_phase_gradient_distillation_decomp(aux_wires, work_wires, mode="repeat-until-success"):
    r"""Create an opt-in Jones distillation rule for :class:`~.PhaseGradientStatePrep`.

    The rule implements the Fourier-state distillation protocol from
    `Jones (2013) <https://arxiv.org/abs/1303.3066>`__ and uses only Clifford gates,
    :class:`~.SemiAdder`, mid-circuit measurements, and classical control. The prepared state is
    approximate: the number of distillation rounds grows as :math:`\mathcal{O}(\log n)` for an
    :math:`n`-qubit register and targets an infidelity of at most :math:`\sin^2(\pi/2^n)`.

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

    The rule is passed to :func:`~.decompose` via ``fixed_decomps``:

    .. code-block:: python

        import pennylane as qp

        qp.decomposition.enable_graph()

        registers = qp.registers({"grad": 3, "aux": 3, "work": 2})
        rule = qp.transforms.decompositions.make_phase_gradient_distillation_decomp(
            registers["aux"], registers["work"]
        )
        gate_set = {"Hadamard", "S", "Adjoint(S)", "T", "Adjoint(T)", "CNOT", "X", "Z", "measure"}

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=gate_set, fixed_decomps={qp.PhaseGradientStatePrep: rule})
        @qp.qnode(qp.device("lightning.qubit", wires=8))
        def circuit():
            qp.PhaseGradientStatePrep(registers["grad"])
            return qp.state()

    The auxiliary register is returned to :math:`|0\rangle`, and only Clifford+T gates and
    mid-circuit measurements remain:

    >>> sorted(qp.analyze(circuit)().resources.quantum_operations)
    ['Adjoint(S)', 'Adjoint(T)', 'CNOT', 'Hadamard', 'MidCircuitMeasure', 'PauliX', 'PauliZ', 'T']
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

    return phase_gradient_distillation

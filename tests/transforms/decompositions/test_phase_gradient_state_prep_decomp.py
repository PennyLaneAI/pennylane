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
"""Tests for the Jones PhaseGradientStatePrep decomposition."""

import re
from collections import Counter

import numpy as np
import pytest

import pennylane as qp
from pennylane.decomposition.decomposition_rule import _fix_decomp
from pennylane.exceptions import WireError
from pennylane.transforms.decompositions.phase_gradient_state_prep_decomp import (
    _jones_num_rounds,
    _jones_register_sizes,
    _jones_seed,
    _jones_success_probabilities,
    make_phase_gradient_distillation_decomp,
)


def _required_workspace(register_sizes):
    """Return the peak number of wires needed by a depth-first local-retry tree."""
    required = register_sizes[0]
    for child_size, output_size in zip(register_sizes[:-1], register_sizes[1:], strict=True):
        required = max(3 * output_size - 1, child_size + required)
    return required


def _target_state(num_wires):
    dim = 2**num_wires
    return np.exp(-2j * np.pi * np.arange(dim) / dim) / np.sqrt(dim)


def _seed_state(num_wires):
    state = np.array([1.0 + 0.0j])
    for index in range(num_wires):
        phase = np.exp(-1j * np.pi / 2**index) if index < 2 else 1
        state = np.kron(state, np.array([1, phase]) / np.sqrt(2))
    return state


def _distill(state):
    """Apply coefficient squaring in the Fourier basis and renormalize."""
    dim = len(state)
    squared = (np.sqrt(dim) * np.fft.ifft(state)) ** 2
    return np.fft.fft(squared / np.linalg.norm(squared)) / np.sqrt(dim)


def _simulate_schedule(num_wires):
    sizes = _jones_register_sizes(num_wires)
    state = _seed_state(sizes[0])
    for old_size, new_size in zip(sizes[:-1], sizes[1:], strict=True):
        tail = np.ones(2 ** (new_size - old_size)) / np.sqrt(2 ** (new_size - old_size))
        state = _distill(np.kron(state, tail))
    return state


def test_seed_sign_and_endianness():
    """Test that the seed has Z and S-dagger phases on the two most-significant wires."""

    @qp.qnode(qp.device("default.qubit", wires=["a", "b", "tail"]))
    def circuit():
        _jones_seed(["a", "b", "tail"])
        return qp.state()

    assert np.allclose(circuit(), _seed_state(3))


@pytest.mark.parametrize("num_wires", [3, 4])
def test_postselected_rule_matches_schedule_simulation(num_wires):
    """Test that the postselected circuit prepares the state predicted by the Fourier-space
    model."""
    rule = make_phase_gradient_distillation_decomp(
        aux_wires=range(num_wires, 2 * num_wires),
        work_wires=range(2 * num_wires, 3 * num_wires - 1),
        mode="postselect",
    )

    @qp.qnode(qp.device("default.qubit"), mcm_method="deferred")
    def circuit():
        rule(wires=range(num_wires))
        return qp.density_matrix(range(num_wires))

    rho = circuit()
    expected = _simulate_schedule(num_wires)
    assert np.allclose(rho, np.outer(expected, expected.conj()))


@pytest.mark.parametrize("num_wires", range(3, 21))
def test_finite_schedule_fidelity(num_wires):
    """Test that the schedule meets Jones's target infidelity."""
    state = _simulate_schedule(num_wires)
    fidelity = np.abs(np.vdot(_target_state(num_wires), state)) ** 2
    assert 1 - fidelity <= np.sin(np.pi / 2**num_wires) ** 2


@pytest.mark.parametrize("num_wires", [5, 8])
def test_distilled_state_supports_phase_kickback(num_wires):
    """Test that a modular increment has the expected phase on the approximate distilled state."""
    state = _simulate_schedule(num_wires)
    incremented_state = np.roll(state, 1)
    kickback_phase = np.vdot(state, incremented_state)
    expected_phase = np.exp(2j * np.pi / 2**num_wires)
    error_bound = 2 * np.sin(np.pi / 2**num_wires)
    assert np.abs(kickback_phase - expected_phase) <= error_bound


def test_schedule_fits_three_register_layout():
    """Test that the depth-first retry schedule fits output, auxiliary, and adder-work
    registers."""
    for num_wires in range(3, 200):
        sizes = _jones_register_sizes(num_wires)
        assert sizes[-1] == num_wires
        assert len(sizes) == _jones_num_rounds(num_wires) + 1
        assert _required_workspace(sizes) <= 3 * num_wires - 1


@pytest.mark.parametrize("num_wires", [3, 4])
def test_success_probability_matches_circuit(num_wires):
    """Test that the simulated success probability is the all-zero probability of the source
    register."""
    output = list(range(num_wires))
    source = list(range(num_wires, 2 * num_wires))
    work = list(range(2 * num_wires, 3 * num_wires - 1))

    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        _jones_seed(output[:2])
        _jones_seed(source[:2])
        for wire in output[2:] + source[2:]:
            qp.Hadamard(wire)
        qp.SemiAdder(source, output, work)
        for wire in source:
            qp.Hadamard(wire)
        return qp.probs(source)

    probabilities = _jones_success_probabilities(_jones_register_sizes(num_wires))
    assert circuit()[0] == pytest.approx(probabilities[0])


def test_success_probabilities_skip_large_rounds():
    """Test that rounds above the simulation threshold are treated as always successful."""
    sizes = _jones_register_sizes(40)
    probabilities = _jones_success_probabilities(sizes)

    assert len(probabilities) == len(sizes) - 1
    for size, probability in zip(sizes[1:], probabilities, strict=True):
        assert probability == 1.0 if size > 16 else 0.5 < probability < 1


def test_postselected_rule_queues_expected_operations():
    """Test that the successful branch contains one adder and postselected source
    measurements."""
    num_wires = 3
    rule = make_phase_gradient_distillation_decomp(
        aux_wires=["a0", "a1", "a2"],
        work_wires=["w0", "w1"],
        mode="postselect",
    )
    with qp.queuing.AnnotatedQueue() as queue:
        rule(wires=["g0", "g1", "g2"])

    operations = list(queue.queue)
    assert sum(isinstance(op, qp.SemiAdder) for op in operations) == 1
    measurements = [op for op in operations if isinstance(op, qp.ops.MidMeasure)]
    assert len(measurements) == num_wires
    assert all(op.postselect == 0 and op.reset for op in measurements)
    assert not any(isinstance(op, (qp.PhaseShift, qp.RZ)) for op in operations)


def test_repeat_rule_captures_nested_local_loops():
    """Test that a multi-round rule captures separate retry loops hinted with their expected
    attempts."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    num_wires = 5
    rule = make_phase_gradient_distillation_decomp(
        aux_wires=range(num_wires, 2 * num_wires),
        work_wires=range(2 * num_wires, 3 * num_wires - 1),
    )

    qp.capture.enable()
    try:
        plxpr = qp.capture.make_plxpr(rule, autograph=False)(wires=jnp.arange(num_wires))
    finally:
        qp.capture.disable()

    assert str(plxpr).count("while_loop[") > 1

    hints = {float(h) for h in re.findall(r"estimated_iterations=([\d.]+)", str(plxpr))}
    probabilities = _jones_success_probabilities(_jones_register_sizes(num_wires))
    assert sorted(hints) == pytest.approx(sorted({1 / p for p in probabilities}))


def test_resources_are_structural_one_attempt_counts():
    """Test that the registered resources count the static retry body, not expected
    repetitions."""
    rule = make_phase_gradient_distillation_decomp(aux_wires=range(3, 6), work_wires=range(6, 8))
    resources = rule.compute_resources(qp.typing.Wire[3])
    counts = Counter()
    for resource, count in resources.gate_counts.items():
        counts[resource.name] += count

    # PauliMeasure combines three X measurements with three source and three output resets
    assert counts == {
        "Hadamard": 6,
        "PauliZ": 2,
        "Adjoint(S)": 2,
        "SemiAdder": 1,
        "PauliMeasure": 9,
        "PauliX": 6,
    }


@pytest.mark.parametrize(
    ("num_wires", "expected"),
    [
        (0, {}),
        (1, {"Hadamard": 1, "PauliZ": 1}),
        (2, {"Hadamard": 2, "PauliZ": 1, "Adjoint(S)": 1}),
    ],
)
def test_small_register_resources(num_wires, expected):
    """Test that registers below the distillation threshold use only the Clifford seed."""
    assert _jones_num_rounds(num_wires) == 0

    rule = make_phase_gradient_distillation_decomp(
        aux_wires=range(num_wires), work_wires=range(max(0, num_wires - 1))
    )
    resources = rule.compute_resources(qp.typing.Wire[num_wires])
    counts = {resource.name: count for resource, count in resources.gate_counts.items()}

    assert counts == expected


@pytest.mark.parametrize(
    "aux, work, match",
    [
        ([3, 4], [6, 7], "aux_wires must have the same size"),
        ([3, 4, 5], [6], "work_wires must contain at least"),
        ([0, 4, 5], [6, 7], "must not overlap"),
        ([3, 4, 5], [5, 7], "must not overlap"),
    ],
)
def test_wire_validation(aux, work, match):
    """Test that an error is raised if the registers have the wrong size or overlap."""
    with pytest.raises((WireError, ValueError), match=match):
        rule = make_phase_gradient_distillation_decomp(aux, work, mode="postselect")
        rule(wires=[0, 1, 2])


def test_mode_validation():
    """Test that an error is raised for an unknown execution mode."""
    with pytest.raises(ValueError, match="mode must be"):
        make_phase_gradient_distillation_decomp([], [], mode="unknown")


@pytest.mark.catalyst
@pytest.mark.usefixtures("enable_graph_decomposition")
def test_qjit_graph_decomposition_and_analyze():
    """Test that the retry rule survives capture and graph decomposition to Clifford+T and
    MCM."""
    catalyst = pytest.importorskip("catalyst")
    num_wires = 3
    registers = qp.registers({"grad": num_wires, "aux": num_wires, "work": num_wires - 1})
    rule = make_phase_gradient_distillation_decomp(registers["aux"], registers["work"])
    gate_set = {
        "X",
        "Y",
        "Z",
        "T",
        "Adjoint(T)",
        "S",
        "Adjoint(S)",
        "CNOT",
        "GlobalPhase",
        "Hadamard",
        "measure",
        "PauliMeasure",
    }

    with qp.decomposition.local_decomps():
        _fix_decomp(qp.PhaseGradientStatePrep, rule)

        @qp.qjit(capture=True, target="mlir")
        @catalyst.passes.graph_decomposition(gate_set=gate_set)
        @qp.qnode(qp.device("null.qubit", wires=3 * num_wires - 1))
        def circuit():
            qp.PhaseGradientStatePrep(registers["grad"])
            return qp.state()

        analysis = qp.analyze(circuit, level="all")()

    final_resources = analysis.resources[max(analysis.resources)]
    gate_types = final_resources.quantum_operations

    assert gate_types.get("SemiAdder", 0) == 0
    assert gate_types.get("TemporaryAND", 0) == 0
    assert gate_types.get("PhaseShift", 0) == 0
    assert gate_types.get("RZ", 0) == 0
    assert gate_types["MidCircuitMeasure"] > 0
    assert gate_types["PPM-w1"] > 0
    # 8 T gates per attempt (16 without measurement-based uncomputation), scaled by the expected
    # 4/3 attempts if graph decomposition keeps the loop hint
    assert 8 <= gate_types["T"] + gate_types["Adjoint(T)"] < 16


@pytest.mark.catalyst
def test_qjit_analyze_counts_expected_attempts():
    """Test that the retry and reset hints survive compilation and weight the reported
    resources."""
    pytest.importorskip("catalyst")
    rule = make_phase_gradient_distillation_decomp(aux_wires=range(3, 6), work_wires=range(6, 8))

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=8))
    def circuit():
        rule(wires=range(3))
        return qp.state()

    gate_types = qp.analyze(circuit, level=0)().resources.quantum_operations
    assert gate_types["Hadamard"] == 6 / 0.75
    # three X measurements and three source resets per attempt, three output resets per failed
    # attempt
    assert gate_types["PPM-w1"] == 6 / 0.75 + 3 * 0.25 / 0.75


@pytest.mark.catalyst
@pytest.mark.parametrize("num_wires", [3, 5])
def test_repeat_rule_prepares_distilled_state(num_wires, monkeypatch):
    """Test that the repeat-until-success rule prepares the distilled state and resets all
    scratch wires."""
    pytest.importorskip("catalyst")
    # Catalyst's lowering of the SemiAdder decomposition leaves a work wire dirty, so the rule is
    # traced by PennyLane instead
    semi_adder_rule = qp.list_decomps(qp.SemiAdder)[0]
    monkeypatch.setattr(
        qp, "SemiAdder", lambda x, y, w: semi_adder_rule(x_wires=x, y_wires=y, work_wires=w)
    )
    rule = make_phase_gradient_distillation_decomp(
        aux_wires=range(num_wires, 2 * num_wires),
        work_wires=range(2 * num_wires, 3 * num_wires - 1),
    )
    gate_set = {"Toffoli", "CNOT", "X", "Z", "Hadamard", "Adjoint(S)", "S", "T", "Adjoint(T)"}
    gate_set |= {"CZ", "PauliMeasure", "measure"}

    @qp.qjit(capture=True, seed=42)
    @qp.decompose(gate_set=gate_set)
    @qp.qnode(qp.device("lightning.qubit", wires=3 * num_wires - 1))
    def circuit():
        rule(wires=range(num_wires))
        return qp.state()

    clean_scratch = np.eye(1, 2 ** (2 * num_wires - 1))[0]
    expected = np.kron(_simulate_schedule(num_wires), clean_scratch)
    assert np.abs(np.vdot(expected, circuit())) ** 2 == pytest.approx(1)

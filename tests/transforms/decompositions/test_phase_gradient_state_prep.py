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

from importlib import import_module

import numpy as np
import pytest

import pennylane as qp
from pennylane.exceptions import WireError

phase_gradient_decomp = import_module(
    "pennylane.transforms.decompositions.phase_gradient_state_prep"
)
make_phase_gradient_distillation_decomp = (
    phase_gradient_decomp.make_phase_gradient_distillation_decomp
)
_jones_num_rounds = getattr(phase_gradient_decomp, "_jones_num_rounds")
_jones_register_sizes = getattr(phase_gradient_decomp, "_jones_register_sizes")
_jones_seed = getattr(phase_gradient_decomp, "_jones_seed")
_required_workspace = getattr(phase_gradient_decomp, "_required_workspace")


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
    """Apply coefficient squaring in the Fourier basis and return state and success."""
    dim = len(state)
    coefficients = np.sqrt(dim) * np.fft.ifft(state)
    squared = coefficients**2
    success = np.sum(np.abs(squared) ** 2)
    return np.fft.fft(squared / np.sqrt(success)) / np.sqrt(dim), success


def _simulate_schedule(num_wires):
    sizes = _jones_register_sizes(num_wires)
    state = _seed_state(sizes[0])
    for old_size, new_size in zip(sizes, sizes[1:]):
        tail = np.ones(2 ** (new_size - old_size)) / np.sqrt(2 ** (new_size - old_size))
        state, _ = _distill(np.kron(state, tail))
    return state


def test_seed_sign_and_endianness():
    """The seed has Z and S-dagger phases on the two most-significant wires."""

    @qp.qnode(qp.device("default.qubit", wires=["a", "b", "tail"]))
    def circuit():
        _jones_seed(["a", "b", "tail"])
        return qp.state()

    assert np.allclose(circuit(), _seed_state(3))


def test_one_round_squares_coefficients_and_success():
    """One symmetric round squares Fourier coefficients and succeeds near 0.67."""
    state = _seed_state(8)
    coefficients = np.sqrt(len(state)) * np.fft.ifft(state)
    result, success = _distill(state)
    result_coefficients = np.sqrt(len(result)) * np.fft.ifft(result)

    expected = coefficients**2 / np.sqrt(np.sum(np.abs(coefficients) ** 4))
    assert np.allclose(result_coefficients, expected)
    assert success == pytest.approx(0.669, abs=0.01)


@pytest.mark.parametrize("num_wires", [3, 5, 8, 11, 14])
def test_finite_schedule_fidelity(num_wires):
    """Representative power-of-two and non-power-of-two widths meet Jones's target."""
    state = _simulate_schedule(num_wires)
    fidelity = np.abs(np.vdot(_target_state(num_wires), state)) ** 2
    assert 1 - fidelity <= np.sin(np.pi / 2**num_wires) ** 2


@pytest.mark.parametrize("num_wires", [5, 8])
def test_distilled_state_supports_phase_kickback(num_wires):
    """Modular increment has the expected phase on the approximate distilled state."""
    state = _simulate_schedule(num_wires)
    incremented_state = np.roll(state, 1)
    kickback_phase = np.vdot(state, incremented_state)
    expected_phase = np.exp(2j * np.pi / 2**num_wires)
    error_bound = 2 * np.sin(np.pi / 2**num_wires)
    assert np.abs(kickback_phase - expected_phase) <= error_bound


@pytest.mark.parametrize("num_wires", [3, 5, 8, 14, 20])
def test_schedule_fits_three_register_layout(num_wires):
    """The depth-first retry schedule fits output, auxiliary, and adder-work registers."""
    sizes = _jones_register_sizes(num_wires)
    assert sizes[-1] == num_wires
    assert len(sizes) == _jones_num_rounds(num_wires) + 1
    assert _required_workspace(sizes) <= 3 * num_wires - 1


def test_postselected_rule_queues_expected_operations():
    """The successful branch contains one adder and postselected source measurements."""
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
    """A multi-round rule captures separate retry loops for its subtrees."""
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


def test_resources_are_structural_one_attempt_counts():
    """Registered resources count the static retry body, not expected repetitions."""
    rule = make_phase_gradient_distillation_decomp(aux_wires=range(3, 6), work_wires=range(6, 8))
    resources = rule.compute_resources(qp.typing.Wire[3])
    counts = {resource.name: count for resource, count in resources.gate_counts.items()}

    assert counts == {
        "Hadamard": 9,
        "PauliZ": 2,
        "Adjoint(S)": 2,
        "SemiAdder": 1,
        "MidMeasureMP": 6,
    }


def test_fixed_rule_is_opt_in():
    """The default remains exact while fixed_decomps explicitly selects distillation."""
    qp.decomposition.enable_graph()
    num_wires = 3
    output = range(num_wires)
    aux = range(num_wires, 2 * num_wires)
    work = range(2 * num_wires, 3 * num_wires - 1)
    rule = make_phase_gradient_distillation_decomp(aux, work, mode="postselect")

    default_ops = qp.PhaseGradientStatePrep(output).decomposition()
    assert any(op.name == "Adjoint(T)" for op in default_ops)

    @qp.transforms.decompose(
        gate_set={"Hadamard", "PauliZ", "Adjoint(S)", "SemiAdder", "MidMeasureMP"},
        fixed_decomps={qp.PhaseGradientStatePrep: rule},
    )
    @qp.qnode(qp.device("null.qubit", wires=3 * num_wires - 1))
    def circuit():
        qp.PhaseGradientStatePrep(output)
        return qp.state()

    tape = qp.workflow.construct_tape(circuit)()
    assert not any(isinstance(op, qp.PhaseGradientStatePrep) for op in tape.operations)
    assert any(isinstance(op, qp.SemiAdder) for op in tape.operations)


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
    """Auxiliary registers must have the correct size and be disjoint."""
    rule = make_phase_gradient_distillation_decomp(aux, work, mode="postselect")
    with pytest.raises((WireError, ValueError), match=match):
        rule(wires=[0, 1, 2])


def test_mode_validation():
    """Only the two documented execution modes are accepted."""
    with pytest.raises(ValueError, match="mode must be"):
        make_phase_gradient_distillation_decomp([], [], mode="unknown")


@pytest.mark.catalyst
def test_qjit_graph_decomposition_and_all_mlir_specs():
    """The retry rule survives capture and graph decomposition to Clifford+T and MCM."""
    catalyst = pytest.importorskip("catalyst")
    qp.decomposition.enable_graph()
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
    }

    @qp.qjit(capture=True, target="mlir")
    @catalyst.passes.graph_decomposition(gate_set=gate_set)
    @qp.qnode(qp.device("null.qubit", wires=3 * num_wires - 1))
    def circuit():
        # Calling the rule directly is temporary. Catalyst cannot yet capture a local fixed
        # decomposition for an operator absent from COMPILER_OPS_FOR_DECOMPOSITION.
        rule(wires=registers["grad"])
        return qp.state()

    specs = qp.specs(circuit, level="all-mlir")()
    final_resources = specs.resources[max(specs.resources)]
    gate_types = final_resources.quantum_operations

    assert gate_types.get("SemiAdder", 0) == 0
    assert gate_types.get("TemporaryAND", 0) == 0
    assert gate_types.get("PhaseShift", 0) == 0
    assert gate_types.get("RZ", 0) == 0
    assert gate_types["MidCircuitMeasure"] > 0
    assert gate_types["T"] + gate_types["Adjoint(T)"] == 8

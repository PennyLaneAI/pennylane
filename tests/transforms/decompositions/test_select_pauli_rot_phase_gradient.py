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

"""Tests for the decomposition rule qp.transforms.decompositions.make_selectpaulirot_to_phase_gradient_decomp"""

import warnings

import hint_helpers
import numpy as np

# pylint: disable=no-value-for-parameter
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.ops.op_math.composite2 import CompositeOp2
from pennylane.transforms.decompositions import (
    make_selectpaulirot_to_phase_gradient_decomp,
)
from pennylane.wires import WireError


def test_wires_error_decomp_fun():
    """Test WireError is raised correctly when calling the decomposition function on a large
    SelectPauliRot that needs more work wires for its QROM than are available."""
    registers = {
        "angle_wires": 3,
        "phase_grad_wires": 3,
        "work_wires": 2,
        "control_wires": 4,
        "target_wires": 1,
    }
    registers = qp.registers(registers)
    control_wires = registers.pop("control_wires")
    target_wire = registers.pop("target_wires")[0]
    rule = make_selectpaulirot_to_phase_gradient_decomp(**registers)
    angles = np.random.random(2**4)
    with pytest.raises(WireError, match=r"work_wires need to be at least of size len\(control"):
        rule(angles, control_wires, target_wire, "X")


def test_decomp_has_a_name():
    """Test that the decomposition rule from make_selectpaulirot_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources"""

    prec = 3
    num_controls = 2

    first_aux = num_controls + 1

    angle_wires = list(range(first_aux, first_aux + prec))
    phase_grad_wires = list(range(first_aux + prec, first_aux + 2 * prec))
    num_work_wires = max(prec, num_controls + 1) - 1
    work_wires = list(range(first_aux + 2 * prec, first_aux + 2 * prec + num_work_wires))

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )
    assert custom_decomp.name == "_select_pauli_rot_phase_gradient_decomp"


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("prec", [2, 3, 5])
@pytest.mark.parametrize("num_controls", [1, 2])
def test_valid_decomp(prec, num_controls):
    """Test that the decomposition rule from make_selectpaulirot_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources"""

    angles = (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        @ np.array([1 / 2, 1 / 4, 1 / 8])
        * 4
        * np.pi
    )[: 2**num_controls]

    # If precision is very low, the number of control wires of the multiplexer dictate the
    # required number of work wires.
    num_work_wires = max(prec, num_controls + 1) - 1

    wire_idx = 0
    angle_wires = qp.wires.Wires([wire_idx + i for i in range(prec)])
    wire_idx += len(angle_wires)
    phase_grad_wires = qp.wires.Wires([wire_idx + i for i in range(prec)])
    wire_idx += len(phase_grad_wires)
    work_wires = qp.wires.Wires([wire_idx + i for i in range(num_work_wires)])
    wire_idx += len(work_wires)

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    control_wires = qp.wires.Wires([wire_idx + i for i in range(num_controls)])
    wire_idx += len(control_wires)
    op = qp.SelectPauliRot(angles, control_wires=control_wires, target_wire=wire_idx)
    _test_decomposition_rule(op, custom_decomp)


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("prec", [2, 3, 5])
@pytest.mark.parametrize("num_controls", [1, 2])
def test_as_fixed_decomps(prec, num_controls):
    """Test that the decomposition rule from make_selectpaulirot_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources"""
    angles = (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        @ np.array([1 / 2, 1 / 4, 1 / 8])
        * 4
        * np.pi
    )[: 2**num_controls]

    # If precision is very low, the number of control wires of the multiplexer dictate the
    # required number of work wires.
    num_work_wires = max(prec, num_controls + 1) - 1

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(num_work_wires)])

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    @qp.transforms.decompose(
        gate_set={
            "QROM",
            "SemiAdder",
            "CNOT",
            "X",
            "Adjoint(X)",
            "GlobalPhase",
        },
        fixed_decomps={qp.SelectPauliRot: custom_decomp},
    )
    @qp.qnode(qp.device("null.qubit"))
    def circuit(angles):
        qp.SelectPauliRot(angles, control_wires=range(num_controls), target_wire=num_controls)
        return qp.state()

    tape = qp.workflow.construct_tape(circuit)(angles)
    specs = qp.resource.resources_from_tape(tape).quantum_operations
    expected_specs = {
        "QROM": 2,
        "CNOT": 2 * prec,
        "PauliX": 2 * prec,
        "SemiAdder": 1,
    }
    assert expected_specs == specs


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("rot_axis", ["X", "Y", "Z"])
def test_integration_multi_wire(rot_axis, seed):
    """Numerically verify the decomposition reproduces SelectPauliRot's
    action on the system wires.

    Mirrors ``test_integration_multi_wire`` in test_rz_phase_gradient.py.
    """
    prec = 3
    num_controls = 2
    angles = np.array([0.5, 1.5, 2.0, 2.5]) * np.pi  # 2**num_controls entries

    ctrl_wires = list(range(num_controls))
    target_wire = num_controls
    sys_wires = ctrl_wires + [target_wire]

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
    num_work = max(prec, num_controls + 1) - 1
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(num_work)])

    all_wires = angle_wires + phase_grad_wires + work_wires + qp.wires.Wires(sys_wires)

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    gs = {
        "QROM",
        "SemiAdder",
        "CNOT",
        "PauliX",
        "GlobalPhase",
        "PhaseGradientStatePrep",
        "Adjoint(PhaseGradientStatePrep)",
        "StatePrep",
    }

    # Depending on the rot_axis, additional operators
    # are expected in the decomposition
    if rot_axis == "Y":
        gs |= {
            "S",
            "Adjoint(S)",
        }
    if rot_axis in ("X", "Y"):
        gs |= {"Hadamard"}

    @qp.decompose(
        gate_set=gs,
        fixed_decomps={qp.SelectPauliRot: custom_decomp},
    )
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(in_state):
        # input state
        qp.StatePrep(in_state, wires=sys_wires)
        # phase gradient state
        qp.PhaseGradientStatePrep(wires=phase_grad_wires)
        qp.SelectPauliRot(
            angles, control_wires=ctrl_wires, target_wire=target_wire, rot_axis=rot_axis
        )
        # uncompute phase gradient state
        qp.adjoint(qp.PhaseGradientStatePrep(wires=phase_grad_wires))
        return qp.state()

    # random input state
    rng = np.random.default_rng(seed)
    in_state = rng.random(2 ** len(sys_wires)) + 1j * rng.random(2 ** len(sys_wires))
    in_state /= np.linalg.norm(in_state)

    # returned output state

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out_state = circuit(in_state)

    # Assert no warnings are raised from the decomposition
    assert len(record) == 0

    # expected: SelectPauliRot is applied on the system wires
    out_state_expected = (
        qp.matrix(
            qp.SelectPauliRot(
                angles, control_wires=ctrl_wires, target_wire=target_wire, rot_axis=rot_axis
            ),
            wire_order=sys_wires,
        )
        @ in_state
    )
    n_aux = len(angle_wires) + len(phase_grad_wires) + len(work_wires)
    # and aux registers back in |0...0>
    zeros = np.eye(2**n_aux, 1)[:, 0]  # |0...0> on all aux wires
    expected = np.kron(zeros, out_state_expected)

    assert np.allclose(out_state, expected), f"decomposition wrong for rot_axis={rot_axis}"


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_capture_compatibility():
    """Ensures capture compatibility."""

    prec = 3
    num_controls = 2
    control_wires = list(range(num_controls))
    target_wire = num_controls
    first_aux = num_controls + 1

    angle_wires = list(range(first_aux, first_aux + prec))
    phase_grad_wires = list(range(first_aux + prec, first_aux + 2 * prec))
    num_work_wires = max(prec, num_controls + 1) - 1
    work_wires = list(range(first_aux + 2 * prec, first_aux + 2 * prec + num_work_wires))

    angles = np.array(
        [
            (1 / 2 + 1 / 4 + 1 / 8) * 4 * np.pi,
            (1 / 2 + 1 / 4 + 0 / 8) * 4 * np.pi,
            (1 / 2 + 0 / 4 + 1 / 8) * 4 * np.pi,
            (0 / 2 + 1 / 4 + 1 / 8) * 4 * np.pi,
        ]
    )

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    op = qp.SelectPauliRot(angles, control_wires=control_wires, target_wire=target_wire)
    _test_decomposition_rule(op, custom_decomp)


@pytest.mark.parametrize(
    "rot_axis, expected_op",
    [
        ("X", qp.RX),
        ("Y", qp.RY),
        ("Z", qp.RZ),
    ],
)
@pytest.mark.usefixtures("enable_and_disable_capture")
def test_rot_axis_zero_controls(rot_axis, expected_op):
    """Test the 0-control-wire edge case for all rotation axes."""
    angle_wires = qp.wires.Wires(["aux_0"])
    phase_grad_wires = qp.wires.Wires(["qft_0"])
    work_wires = qp.wires.Wires(["work_0"])

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    angles = np.array([1.23])
    op = qp.SelectPauliRot(angles, control_wires=[], target_wire=0, rot_axis=rot_axis)

    _test_decomposition_rule(op, custom_decomp)

    with qp.queuing.AnnotatedQueue() as q:
        custom_decomp(angles, [], 0, rot_axis=rot_axis)

    assert len(q.queue) == 1
    assert isinstance(q.queue[0], expected_op)


def _leaf_ops(ops):
    """Recursively unpack composite operators (change of basis, products)."""
    for op in ops:
        if isinstance(op, CompositeOp2):
            yield from _leaf_ops(op.decomposition())
        else:
            yield op


def _bits_to_angles(bits):
    bits = np.array(bits)
    return bits @ (2.0 ** -np.arange(1, bits.shape[1] + 1)) * 4 * np.pi


# Cases for a 4-bit angle register: (binary angles, precision_stats, expected truncated width)
PRECISION_STATS_CASES = [
    ([[1, 0, 1, 0], [0, 1, 0, 0]], {3: 5}, 3),
    ([[1, 0, 0, 0], [0, 1, 0, 0]], {1: 1, 2: 3}, 2),
    ([[1, 0, 0, 1], [0, 0, 0, 0]], {2: 1, 3: 1}, 4),
    ([[0, 0, 0, 0], [0, 0, 0, 0]], {0: 1, 4: 1}, 0),
    ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]], {2: 1, 4: 1}, 2),
]


def test_precision_stats_invalid_key():
    """Test that precision_stats keys outside of 0..len(angle_wires) raise an error."""
    with pytest.raises(ValueError, match="precision_stats keys must be between 0 and 3"):
        make_selectpaulirot_to_phase_gradient_decomp(
            [2, 3, 4], [5, 6, 7], [8, 9], precision_stats={4: 1}
        )


@pytest.mark.parametrize("rot_axis", ["X", "Y", "Z"])
@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_truncates(bits, precision_stats, width, rot_axis):
    """Test that the QROM, the phase-gradient flips and the SemiAdder are truncated to the
    smallest precision in ``precision_stats`` that suffices for all angles, or to full precision
    otherwise."""
    num_controls = int(np.log2(len(bits)))
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(4)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(4)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(3)])
    rule = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
    )

    with qp.queuing.AnnotatedQueue() as q:
        rule(_bits_to_angles(bits), list(range(num_controls)), num_controls, rot_axis)
    ops = list(_leaf_ops(q.queue))

    if width == 0:
        assert not ops
        return
    adders = [op for op in ops if isinstance(op, qp.SemiAdder)]
    qroms = [op for op in ops if isinstance(op, qp.QROM)]
    flips = [op for op in ops if isinstance(op, qp.MultiControlledX)]
    assert len(adders) == 1 and len(qroms) == 2 and len(flips) == 2 * width
    assert adders[0].x_wires == angle_wires[:width]
    assert adders[0].y_wires == phase_grad_wires[:width]
    for qrom in qroms:
        assert qrom.target_wires == angle_wires[:width]
        assert np.array_equal(qrom.bitstrings, np.array(bits)[:, :width])
    assert {op.target_wires[0] for op in flips} == set(phase_grad_wires[:width])
    # The uncomputation of the basis change is a single adjoint operator
    num_basis_ops = {"X": 2, "Y": 3, "Z": 0}[rot_axis]
    assert len(ops) == 3 + 2 * width + num_basis_ops


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("rot_axis", ["X", "Y", "Z"])
@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_valid_decomp(bits, precision_stats, width, rot_axis):
    """Test that the rule with ``precision_stats`` yields a valid decomposition."""
    # pylint: disable=unused-argument
    if qp.capture.enabled() and rot_axis != "Z":
        pytest.xfail("The nested change of basis for X and Y is flattened with capture enabled.")
    num_controls = int(np.log2(len(bits)))
    rule = make_selectpaulirot_to_phase_gradient_decomp(
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], precision_stats=precision_stats
    )
    op = qp.SelectPauliRot(
        _bits_to_angles(bits),
        control_wires=list(range(11, 11 + num_controls)),
        target_wire=11 + num_controls,
        rot_axis=rot_axis,
    )
    _test_decomposition_rule(op, rule)


@pytest.mark.capture
@pytest.mark.parametrize(
    "precision_stats, expected",
    [
        ({1: 1, 3: 3}, (0.25, 0.75, 0.0)),
        ({2: 1, 4: 3}, (0.25, 0.75)),
        ({0: 2, 4: 2}, (0.5, 0.5)),
    ],
)
def test_precision_stats_hints(precision_stats, expected):
    """Test that the conditional branches are hinted with the probabilities derived from
    ``precision_stats``."""
    import jax

    rule = make_selectpaulirot_to_phase_gradient_decomp(
        [2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12], precision_stats=precision_stats
    )
    jaxpr = jax.make_jaxpr(lambda angles: rule(angles, [0], 1, "Z"))(np.array([0.4, 0.2]))
    assert hint_helpers.cond_hints(jaxpr.jaxpr, skip_none=True) == [pytest.approx(expected)]


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("rot_axis", ["X", "Y", "Z"])
@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_integration(bits, precision_stats, width, rot_axis, seed):
    """Test that the decomposition with ``precision_stats`` realizes SelectPauliRot."""
    # pylint: disable=unused-argument
    angles = _bits_to_angles(bits)
    num_controls = int(np.log2(len(bits)))
    ctrl_wires = list(range(num_controls))
    target_wire = num_controls
    sys_wires = ctrl_wires + [target_wire]

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(4)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(4)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(3)])
    all_wires = angle_wires + phase_grad_wires + work_wires + qp.wires.Wires(sys_wires)

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
    )

    gate_set = {
        "QROM",
        "SemiAdder",
        "CNOT",
        "PauliX",
        "Hadamard",
        "S",
        "Adjoint(S)",
        "PhaseGradientStatePrep",
        "Adjoint(PhaseGradientStatePrep)",
        "StatePrep",
    }

    @qp.decompose(gate_set=gate_set, fixed_decomps={qp.SelectPauliRot: custom_decomp})
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(in_state):
        qp.StatePrep(in_state, wires=sys_wires)
        qp.PhaseGradientStatePrep(wires=phase_grad_wires)
        qp.SelectPauliRot(
            angles, control_wires=ctrl_wires, target_wire=target_wire, rot_axis=rot_axis
        )
        qp.adjoint(qp.PhaseGradientStatePrep(wires=phase_grad_wires))
        return qp.state()

    rng = np.random.default_rng(seed)
    in_state = rng.random(2 ** len(sys_wires)) + 1j * rng.random(2 ** len(sys_wires))
    in_state /= np.linalg.norm(in_state)

    out_state = circuit(in_state)

    op = qp.SelectPauliRot(
        angles, control_wires=ctrl_wires, target_wire=target_wire, rot_axis=rot_axis
    )
    zeros = np.eye(2 ** (len(all_wires) - len(sys_wires)), 1)[:, 0]
    expected = np.kron(zeros, qp.matrix(op, wire_order=sys_wires) @ in_state)
    assert np.allclose(out_state, expected)

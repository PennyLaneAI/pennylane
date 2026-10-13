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

"""Tests for ``qp.transforms.decompositions.make_rz_to_phase_gradient_decomp``"""

# pylint: disable=no-value-for-parameter,disable=too-many-arguments

import hint_helpers
import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.ops.op_math.composite2 import CompositeOp2
from pennylane.transforms.decompositions import (
    make_rz_to_phase_gradient_decomp,
    validate_phase_gradient_wires,
)
from pennylane.wires import WireError


def _num_set_bits(phi, p):
    """Number of set bits in the ``p``-bit binary representation of ``phi`` (in units of 2*pi)."""
    bits = [int(b) for b in qp.math.binary_decimals(phi, p, unit=2 * np.pi)]
    return sum(bits)


def _expected_rz_specs(phi, p):
    """Expected specs for a single ``RZ``.

    The compute/uncompute fanout is a ``MultiX`` (one ``X`` per set bit) controlled by the ``RZ``
    target wire, realizing one ``CNOT`` per set bit on each of the two (compute and uncompute)
    passes (a single-control ``MultiX`` fanout that's always controlled on |1>, so it renders
    directly as ``CNOT``), so the count is ``2 * num_set_bits``. Trailing zero bits emit no gate,
    hence ``CNOT`` is absent for an angle that rounds to zero."""
    specs = {"GlobalPhase": 1, "SemiAdder": 1}
    n = _num_set_bits(phi, p)
    if n:
        specs["CNOT"] = 2 * n
    return specs


@pytest.mark.parametrize(
    "n_angle_wires, n_phase_grad_wires, n_work_wires, msg_match",
    [
        [5, 3, 2, "angle_wires and phase_grad wires must be of same size"],
        [3, 4, 2, "angle_wires and phase_grad wires must be of same size"],
        [4, 4, 2, r"work_wires need to be at least of size len\(phase_grad_wires\) - 1"],
    ],
)
def test_validate_phase_gradient_wires(n_angle_wires, n_phase_grad_wires, n_work_wires, msg_match):
    """Test that a WireError is raised correctly for wrongly-sized registers"""
    angle_wires = qp.wires.Wires([f"ang_{i}" for i in range(n_angle_wires)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(n_phase_grad_wires)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(n_work_wires)])

    with pytest.raises(WireError, match=msg_match):
        _ = validate_phase_gradient_wires(angle_wires, phase_grad_wires, work_wires)


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_decomp_has_name():
    """Test that ``make_rz_to_phase_gradient_decomp`` yields a decomposition with a name property."""
    p = 1

    first_free = 1
    angle_wires = list(range(first_free, first_free + p))
    phase_grad_wires = list(range(first_free + p, first_free + 2 * p))
    work_wires = list(range(first_free + 2 * p, first_free + 3 * p - 1))

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_rz_to_phase_gradient_decomp(**kwargs)
    assert custom_decomp.name == "_rz_phase_gradient_decomp"


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("p", [2, 3, 4])
def test_valid_decomp(p):
    """Test that ``make_rz_to_phase_gradient_decomp`` yields a valid decomposition, with capture
    both disabled (concrete angle) and enabled (abstract angle).

    ``_test_decomposition_rule`` checks the emitted circuit against the rule's full-precision
    resource estimate. We use an all-ones angle so the concrete decomposition saturates that
    estimate (every bit is set); other angles emit a smaller circuit than the (upper-bound)
    estimate and are instead covered by ``test_as_fixed_decomps``/``test_as_alt_decomps``."""

    phi = (1 - 2.0**-p) * 2 * np.pi  # binary all-ones at p bits
    first_free = 1
    angle_wires = list(range(first_free, first_free + p))
    phase_grad_wires = list(range(first_free + p, first_free + 2 * p))
    work_wires = list(range(first_free + 2 * p, first_free + 3 * p - 1))

    custom_decomp = make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)
    _test_decomposition_rule(qp.RZ(phi, 0), custom_decomp, skip_decomp_matrix_check=True)


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_fixed_decomps(phi, p):
    """Test that the decomposition rule from make_rz_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources."""
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_rz_to_phase_gradient_decomp(**kwargs)
    gate_set = {"SemiAdder", "CNOT", "PauliX", "GlobalPhase"}

    @qp.transforms.decompose(gate_set=gate_set, fixed_decomps={qp.RZ: custom_decomp})
    @qp.qnode(qp.device("null.qubit"))
    def circuit():
        qp.RZ(phi, 0)
        return qp.state()

    tape = qp.workflow.construct_tape(circuit)()
    specs = qp.resource.resources_from_tape(tape).quantum_operations
    assert specs == _expected_rz_specs(phi, p)


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_alt_decomps(phi, p):
    """Test that the decomposition rule from ``make_rz_to_phase_gradient_decomp works`` as
    expected as an alternative decomposition and yields the correct resources."""
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_rz_to_phase_gradient_decomp(**kwargs)
    gate_set = {"SemiAdder", "CNOT", "PauliX", "GlobalPhase"}

    @qp.transforms.decompose(gate_set=gate_set, alt_decomps={qp.RZ: [custom_decomp]})
    @qp.qnode(qp.device("null.qubit"))
    def circuit():
        qp.RZ(phi, 0)
        return qp.state()

    tape = qp.workflow.construct_tape(circuit)()
    specs = qp.resource.resources_from_tape(tape).quantum_operations
    assert specs == _expected_rz_specs(phi, p)


@pytest.mark.usefixtures("enable_graph_decomposition")
def test_integration_multi_wire(seed):
    """
    Tests that the decomposition correctly realizes the phase gradient decomposition of RZ as described in
    https://pennylane.ai/compilation/phase-gradient/b-rotations
    """

    prec = 3

    phi = (1 / 2 + 0 / 4 + 1 / 8) * 4 * np.pi
    wires = [0]

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

    all_wires = angle_wires + phase_grad_wires + work_wires + wires

    custom_decomp = make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)

    @qp.transforms.decompose(
        gate_set={
            "StatePrep",
            "PhaseGradientStatePrep",
            "Adjoint(PhaseGradientStatePrep)",
            "SemiAdder",
            "CNOT",
            "PauliX",
            "GlobalPhase",
        },
        fixed_decomps={qp.RZ: custom_decomp},
    )
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(phi, in_state):
        qp.StatePrep(in_state, wires=wires)  # input state
        qp.PhaseGradientStatePrep(phase_grad_wires)  # phase gradient state
        qp.RZ(phi, wires)
        qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires))  # uncompute phase gradient state
        return qp.state()

    # random input state
    rng = np.random.default_rng(seed=seed)
    in_state = rng.random(2 ** len(wires)) + 1j * rng.random(2 ** len(wires))
    in_state /= np.linalg.norm(in_state)

    # returned output state
    out_state = circuit(phi, in_state)

    # expected output state
    zeros = np.eye(2 ** (prec * 3 - 1), 1)[:, 0]  # |000> on all the aux wires
    out_state_expected = qp.matrix(qp.RZ(phi, wires)) @ in_state
    out_state_expected = np.kron(zeros, out_state_expected)

    assert np.allclose(out_state, out_state_expected)


def _leaf_ops(ops):
    """Recursively unpack composite operators (change of basis, products)."""
    for op in ops:
        if isinstance(op, CompositeOp2):
            yield from _leaf_ops(op.decomposition())
        else:
            yield op


# Cases for a 4-bit angle register: (binary angle, precision_stats, expected truncated width)
PRECISION_STATS_CASES = [
    ([1, 0, 1, 0], {3: 5}, 3),
    ([1, 0, 1, 0], {1: 2, 3: 5, 4: 1}, 3),
    ([1, 0, 1, 0], {2: 1, 4: 1}, 4),
    ([1, 0, 1, 0], {1: 1, 2: 1}, 4),
    ([1, 0, 0, 0], {2: 1}, 2),
    ([0, 0, 0, 0], {0: 1, 3: 1}, 0),
    ([0, 0, 0, 0], {2: 1}, 2),
    ([1, 0, 1, 0], {4: 3}, 4),
]


def _bits_to_angle(bits, unit):
    return sum(b * 2.0 ** -(i + 1) for i, b in enumerate(bits)) * unit


@pytest.mark.parametrize("key", [-1, 5])
def test_precision_stats_invalid_key(key):
    """Test that precision_stats keys outside of 0..len(angle_wires) raise an error."""
    with pytest.raises(ValueError, match="precision_stats keys must be between 0 and 4"):
        make_rz_to_phase_gradient_decomp(
            [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11], precision_stats={key: 1}
        )


@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_truncates(bits, precision_stats, width):
    """Test that the data loading and the SemiAdder are truncated to the smallest precision in
    ``precision_stats`` that suffices for the angle, or to full precision otherwise."""
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(4)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(4)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(3)])
    rule = make_rz_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
    )

    with qp.queuing.AnnotatedQueue() as q:
        rule(_bits_to_angle(bits, 2 * np.pi), wires=[0])
    ops = list(_leaf_ops(q.queue))

    assert isinstance(ops[0], qp.GlobalPhase)
    adders = [op for op in ops if isinstance(op, qp.SemiAdder)]
    fanouts = [op for op in ops if isinstance(op, qp.ops.Controlled)]
    if width == 0:
        assert len(ops) == 1
        return
    assert len(adders) == 1 and len(fanouts) == 2
    assert adders[0].x_wires == angle_wires[:width]
    assert adders[0].y_wires == phase_grad_wires[:width]
    for fanout in fanouts:
        assert isinstance(fanout.base, qp.MultiX)
        assert fanout.base.wires == angle_wires[:width]
        assert list(fanout.base.bitstring) == bits[:width]


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_valid_decomp(bits, precision_stats, width):
    """Test that the rule with ``precision_stats`` yields a valid decomposition."""
    # pylint: disable=unused-argument
    rule = make_rz_to_phase_gradient_decomp(
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11], precision_stats=precision_stats
    )
    op = qp.RZ(_bits_to_angle(bits, 2 * np.pi), 0)
    _test_decomposition_rule(op, rule, skip_decomp_matrix_check=True)


@pytest.mark.capture
@pytest.mark.parametrize(
    "precision_stats, expected",
    [
        ({1: 1, 3: 3}, (0.25, 0.75, 0.0)),
        ({2: 1, 4: 3}, (0.25, 0.75)),
        ({0: 2, 4: 2}, (0.5, 0.5)),
        ({3: 7}, (1.0, 0.0)),
    ],
)
def test_precision_stats_hints(precision_stats, expected):
    """Test that the conditional branches are hinted with the probabilities derived from
    ``precision_stats``, and that an unhinted full-precision fallback is only added if the full
    precision is not part of ``precision_stats``."""
    import jax

    rule = make_rz_to_phase_gradient_decomp(
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11], precision_stats=precision_stats
    )
    jaxpr = jax.make_jaxpr(lambda phi: rule(phi, wires=[0]))(0.4)
    assert hint_helpers.cond_hints(jaxpr.jaxpr) == [pytest.approx(expected)]


@pytest.mark.capture
def test_precision_stats_full_precision_only_has_no_cond():
    """Test that no conditional is used if ``precision_stats`` only contains the full precision,
    in which case the full precision is applied directly."""
    import jax

    rule = make_rz_to_phase_gradient_decomp(
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11], precision_stats={4: 3}
    )
    jaxpr = jax.make_jaxpr(lambda phi: rule(phi, wires=[0]))(0.4)
    assert not hint_helpers.cond_hints(jaxpr.jaxpr)

    ops = list(_leaf_ops(qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 0.4).operations))
    adders = [op for op in ops if isinstance(op, qp.SemiAdder)]
    assert len(adders) == 1 and adders[0].x_wires == qp.wires.Wires([1, 2, 3, 4])


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_integration(bits, precision_stats, width, seed):
    """Test that the decomposition with ``precision_stats`` realizes RZ."""
    # pylint: disable=unused-argument
    phi = _bits_to_angle(bits, 2 * np.pi)
    wires = [0]
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(4)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(4)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(3)])
    all_wires = angle_wires + phase_grad_wires + work_wires + wires

    custom_decomp = make_rz_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
    )

    @qp.transforms.decompose(
        gate_set={
            "StatePrep",
            "PhaseGradientStatePrep",
            "Adjoint(PhaseGradientStatePrep)",
            "SemiAdder",
            "CNOT",
            "PauliX",
            "GlobalPhase",
        },
        fixed_decomps={qp.RZ: custom_decomp},
    )
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(phi, in_state):
        qp.StatePrep(in_state, wires=wires)
        qp.PhaseGradientStatePrep(phase_grad_wires)
        qp.RZ(phi, wires)
        qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires))
        return qp.state()

    rng = np.random.default_rng(seed=seed)
    in_state = rng.random(2) + 1j * rng.random(2)
    in_state /= np.linalg.norm(in_state)

    out_state = circuit(phi, in_state)

    zeros = np.eye(2 ** (len(all_wires) - 1), 1)[:, 0]
    out_state_expected = np.kron(zeros, qp.matrix(qp.RZ(phi, wires)) @ in_state)
    assert np.allclose(out_state, out_state_expected)

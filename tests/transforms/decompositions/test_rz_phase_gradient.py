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

from collections import Counter

import hint_helpers
import numpy as np
import phase_gradient_helpers
import pytest

import pennylane as qp
from pennylane.testing import assert_valid_decomp_rule
from pennylane.transforms.decompositions import (
    make_rz_to_phase_gradient_decomp,
    phase_gradient_precision_stats,
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

    ``assert_valid_decomp_rule`` checks the emitted circuit against the rule's full-precision
    resource estimate. We use an all-ones angle so the concrete decomposition saturates that
    estimate (every bit is set); other angles emit a smaller circuit than the (upper-bound)
    estimate and are instead covered by ``test_as_fixed_decomps``/``test_as_alt_decomps``."""

    phi = (1 - 2.0**-p) * 2 * np.pi  # binary all-ones at p bits
    first_free = 1
    angle_wires = list(range(first_free, first_free + p))
    phase_grad_wires = list(range(first_free + p, first_free + 2 * p))
    work_wires = list(range(first_free + 2 * p, first_free + 3 * p - 1))

    custom_decomp = make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)
    assert_valid_decomp_rule(qp.RZ(phi, 0), custom_decomp, check_matrix=False)


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

    # pylint: disable-next=protected-access
    @qp.transforms._tape_decompose(gate_set=gate_set, fixed_decomps={qp.RZ: custom_decomp})
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

    # pylint: disable-next=protected-access
    @qp.transforms._tape_decompose(gate_set=gate_set, alt_decomps={qp.RZ: [custom_decomp]})
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

    # pylint: disable-next=protected-access
    @qp.transforms._tape_decompose(
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


# Cases for a 4-bit angle register: (binary angle, precision_stats, expected truncated width)
PRECISION_STATS_CASES = [
    ([1, 0, 1, 0], {1: 2, 3: 5, 4: 1}, 3),  # smallest sufficient precision
    ([1, 0, 0, 0], {2: 1}, 2),  # larger precision than required
    ([1, 0, 1, 0], {1: 1, 2: 1}, 4),  # unhinted fallback to full precision
    ([0, 0, 0, 0], {0: 1, 3: 1}, 0),  # no rotation required
    ([1, 0, 1, 0], {}, 4),  # treated like None
    ([1, 0, 1, 0], {2: 0, 4: 0}, 4),  # zero counts are ignored, so treated like None
]


@pytest.mark.parametrize("key", [-1, 5])
def test_precision_stats_invalid_key(key):
    """Test that precision_stats keys outside of 0..len(angle_wires) raise an error."""
    with pytest.raises(ValueError, match="precision_stats keys must be between 0 and 4"):
        make_rz_to_phase_gradient_decomp(
            [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11], precision_stats={key: 1}
        )


def test_precision_stats_from_angles():
    """Test that ``phase_gradient_precision_stats`` counts the bits required by each operator,
    rounding at the full precision and maximizing over the angles of multi-angle operators."""
    bits = [
        [1, 0, 1, 0],  # 3 bits
        [0, 1, 0, 0],  # 2 bits
        [0, 0, 0, 0],  # 0 bits
        [0, 0, 1, 1, 1],  # rounds to 0100, i.e., 2 bits
        [1, 1, 1, 1, 1],  # rounds to 10000, i.e., 0 bits
        [[1, 0, 0, 0], [0, 0, 1, 0]],  # 3 bits for the second angle
    ]
    angles = [phase_gradient_helpers.bits_to_angles(b, 2 * np.pi) for b in bits]
    assert phase_gradient_precision_stats(angles, 4, qp.RZ) == {3: 2, 2: 2, 0: 2}


def test_precision_stats_from_angles_op_type():
    """Test that ``phase_gradient_precision_stats`` infers the unit of the angles from the
    operator type, given as class or name, and raises an error for unsupported types."""
    angles = [np.pi / 2, np.pi]  # (0.0100)_2 and (0.1000)_2 in units of 2 pi
    assert phase_gradient_precision_stats(angles, 4, "RZ") == {1: 1, 2: 1}
    # In units of 4 pi, the angles are (0.0010)_2 and (0.0100)_2
    for op_type in (qp.CRZ, "CRZ", qp.SelectPauliRot, "SelectPauliRot"):
        assert phase_gradient_precision_stats(angles, 4, op_type) == {2: 1, 3: 1}
    with pytest.raises(ValueError, match="op_type must be one of"):
        phase_gradient_precision_stats(angles, 4, qp.RX)


def test_precision_stats_from_angles_dispatches_exactly():
    """Test that each angle is applied at exactly its required precision if ``precision_stats``
    is computed from the angles with ``phase_gradient_precision_stats``."""
    bits = [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]]
    angles = phase_gradient_helpers.bits_to_angles(bits, 2 * np.pi)
    rule = make_rz_to_phase_gradient_decomp(
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11],
        phase_gradient_precision_stats(angles, 4, qp.RZ),
    )
    for b, phi in zip(bits, angles):
        with qp.queuing.AnnotatedQueue() as q:
            rule(phi, wires=[0])
        ops = phase_gradient_helpers.leaf_ops(q.queue)
        widths = [len(o.x_wires) for o in ops if isinstance(o, qp.SemiAdder)]
        required = max((i + 1 for i, bit in enumerate(b) if bit), default=0)
        assert widths == ([required] if required else [])


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_decomp(bits, precision_stats, width, seed):
    """Test that the rule with ``precision_stats`` truncates the SemiAdder to the smallest
    precision in ``precision_stats`` that suffices for the angle (or to full precision otherwise),
    accounts for all gates in its resources, and realizes RZ."""
    angle_wires, phase_grad_wires = qp.wires.Wires([1, 2, 3, 4]), qp.wires.Wires([5, 6, 7, 8])
    rule = make_rz_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, [9, 10, 11], precision_stats=precision_stats
    )
    op = qp.RZ(phase_gradient_helpers.bits_to_angles(bits, 2 * np.pi), 0)
    _test_decomposition_rule(op, rule, skip_decomp_matrix_check=True)

    tape = phase_gradient_helpers.decomp_to_tape(op, rule)
    adders = [
        o for o in phase_gradient_helpers.leaf_ops(tape.operations) if isinstance(o, qp.SemiAdder)
    ]
    assert [(a.x_wires, a.y_wires) for a in adders] == (
        [(angle_wires[:width], phase_grad_wires[:width])] if width else []
    )
    phase_gradient_helpers.assert_decomp_realizes_op(op, tape, phase_grad_wires, seed)


@pytest.mark.capture
@pytest.mark.parametrize(
    "precision_stats, expected",
    [
        ({1: 1, 3: 3}, [(0.25, 0.75, 0.0)]),
        ({2: 1, 4: 3}, [(0.25, 0.75)]),
        ({0: 2, 4: 2}, [(0.5, 0.5)]),
        ({3: 7}, [(1.0, 0.0)]),
        ({1: 0, 3: 2, 4: 0}, [(1.0, 0.0)]),
        ({4: 3}, []),
        (None, []),
    ],
)
def test_precision_stats_hints(precision_stats, expected):
    """Test that the conditional branches are hinted with the probabilities derived from
    ``precision_stats``, that an unhinted full-precision fallback is only added if the full
    precision is not part of ``precision_stats``, that no conditional is used if only the full
    precision is required, and that zero counts are ignored."""
    import jax

    rule = make_rz_to_phase_gradient_decomp(
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11], precision_stats=precision_stats
    )
    jaxpr = jax.make_jaxpr(lambda phi: rule(phi, wires=[0]))(0.4)
    assert hint_helpers.cond_hints(jaxpr.jaxpr) == [pytest.approx(e) for e in expected]


@pytest.mark.catalyst
def test_precision_stats_hint_weighted_resources():
    """Test that the resources estimated from the hinted branches for a batch of rotations match
    the decompositions of the individual rotations if ``precision_stats`` is computed from the
    angles."""
    pytest.importorskip("catalyst")
    bits = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 1, 1]]
    angles = phase_gradient_helpers.bits_to_angles(bits, 2 * np.pi)
    rule = make_rz_to_phase_gradient_decomp(
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11],
        phase_gradient_precision_stats(angles, 4, qp.RZ),
    )

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("null.qubit", wires=12))
    def circuit(angles):
        qp.for_loop(len(angles))(lambda i: rule(angles[i], wires=[0]))()
        return qp.expval(qp.Z(0))

    expected = Counter()
    for phi in angles:
        with qp.queuing.AnnotatedQueue() as q:
            rule(phi, wires=[0])
        expected.update(op.name for op in phase_gradient_helpers.leaf_ops(q.queue))
    assert qp.specs(circuit, level=0)(angles).resources.counts == expected

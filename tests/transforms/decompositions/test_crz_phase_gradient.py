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

"""Tests for ``qp.transforms.decompositions.make_crz_to_phase_gradient_decomp``"""

# pylint: disable=no-value-for-parameter, disable=too-many-arguments

import hint_helpers
import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.ops.op_math.composite2 import CompositeOp2
from pennylane.transforms.decompositions import make_crz_to_phase_gradient_decomp


def _num_set_bits(phi, p):
    """Number of set bits in the ``p``-bit binary representation of ``phi`` (in units of 4*pi)."""
    bits = [int(b) for b in qp.math.binary_decimals(phi, p, unit=4 * np.pi)]
    return sum(bits)


def _expected_crz_specs(phi, p):
    """Expected specs for a single ``CRZ``.

    The compute/uncompute fanout loads the angle bits onto the angle wires with a controlled
    ``MultiX`` (one single-control ``CNOT`` per *set* bit, so ``2 * num_set_bits`` across both
    passes, since it's always controlled on |1>) and flips all ``p`` phase-gradient wires with a
    single controlled ``MultiX`` controlled on |0> (``p`` ``CNOT``\\ s plus one ``PauliX`` flip of
    that control wire, independent of the angle, per pass)."""
    n = _num_set_bits(phi, p)
    return {"SemiAdder": 1, "CNOT": 2 * p + 2 * n, "PauliX": 4}


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_valid_decomp(p):
    """Test that ``make_crz_to_phase_gradient_decomp`` yields a valid decomposition, with capture
    both disabled (concrete angle) and enabled (abstract angle). We use an all-ones angle so the
    concrete decomposition saturates the rule's full-precision resource estimate."""

    phi = (1 - 2.0**-p) * 4 * np.pi  # binary all-ones at p bits (CRZ uses units of 4*pi)
    first_free = 2
    angle_wires = list(range(first_free, first_free + p))
    phase_grad_wires = list(range(first_free + p, first_free + 2 * p))
    work_wires = list(range(first_free + 2 * p, first_free + 3 * p - 1))

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_crz_to_phase_gradient_decomp(**kwargs)
    op = qp.CRZ(phi, [0, 1])
    _test_decomposition_rule(op, custom_decomp, skip_decomp_matrix_check=True)


@pytest.mark.usefixtures("enable_graph_decomposition")
def test_decomp_has_name():
    """Ensures the decomp rule has a name property."""
    p = 2

    first_free = 2
    angle_wires = list(range(first_free, first_free + p))
    phase_grad_wires = list(range(first_free + p, first_free + 2 * p))
    work_wires = list(range(first_free + 2 * p, first_free + 3 * p - 1))

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_crz_to_phase_gradient_decomp(**kwargs)
    assert custom_decomp.name == "_crz_phase_gradient_decomp"


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_fixed_decomps(phi, p):
    """Test that the decomposition rule from ``make_crz_to_phase_gradient_decomp`` works as
    expected as a fixed decomposition and yields the correct resources"""
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_crz_to_phase_gradient_decomp(**kwargs)
    gate_set = {"SemiAdder", "CNOT", "PauliX"}

    @qp.transforms.decompose(gate_set=gate_set, fixed_decomps={qp.CRZ: custom_decomp})
    @qp.qnode(qp.device("null.qubit"))
    def circuit():
        qp.CRZ(phi, [0, 1])
        return qp.state()

    tape = qp.workflow.construct_tape(circuit)()
    specs = qp.resource.resources_from_tape(tape).quantum_operations
    assert specs == _expected_crz_specs(phi, p)


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_alt_decomps(phi, p):
    """Test that the decomposition rule from ``make_crz_to_phase_gradient_decomp`` works as
    expected as an alternative decomposition and yields the correct resources"""
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_crz_to_phase_gradient_decomp(**kwargs)
    gate_set = {"SemiAdder", "CNOT", "PauliX"}

    @qp.transforms.decompose(gate_set=gate_set, alt_decomps={qp.CRZ: [custom_decomp]})
    @qp.qnode(qp.device("null.qubit"))
    def circuit():
        qp.CRZ(phi, [0, 1])
        return qp.state()

    tape = qp.workflow.construct_tape(circuit)()
    specs = qp.resource.resources_from_tape(tape).quantum_operations
    assert specs == _expected_crz_specs(phi, p)


@pytest.mark.usefixtures("enable_graph_decomposition")
def test_integration_multi_wire(seed):
    """
    Tests that the decomposition correctly realizes the phase gradient decomposition of CRZ as
    described in https://pennylane.ai/compilation/phase-gradient/c-control-rotations
    """

    prec = 3

    phi = (1 / 2 + 0 / 4 + 1 / 8) * 4 * np.pi
    wires = [0, 1]

    angle_wires = [f"aux_{i}" for i in range(prec)]
    phase_grad_wires = [f"qft_{i}" for i in range(prec)]
    work_wires = [f"work_{i}" for i in range(prec - 1)]

    all_wires = angle_wires + phase_grad_wires + work_wires + wires

    custom_decomp = make_crz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)

    @qp.transforms.decompose(
        gate_set={
            "StatePrep",
            "PhaseGradientStatePrep",
            "Adjoint(PhaseGradientStatePrep)",
            "CNOT",
            "PauliX",
            "SemiAdder",
        },
        fixed_decomps={qp.CRZ: custom_decomp},
    )
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(phi, in_state):
        # Prepare input state
        qp.StatePrep(in_state, wires=wires)
        # Prepare phase gradient state
        qp.PhaseGradientStatePrep(wires=phase_grad_wires)
        qp.CRZ(phi, wires)
        # uncompute phase gradient state
        qp.adjoint(qp.PhaseGradientStatePrep(wires=phase_grad_wires))
        return qp.state()

    # random input state
    rng = np.random.default_rng(seed=seed)
    in_state = rng.random(2 ** len(wires))
    in_state /= np.linalg.norm(in_state)

    # returned output state
    out_state = circuit(phi, in_state)

    # expected output state
    zeros = np.eye(2 ** (prec * 3 - 1), 1)[:, 0]  # |000> on all the aux wires
    out_state_expected = qp.matrix(qp.CRZ(phi, wires)) @ in_state
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
    ([1, 0, 1, 0], {}, 4),
]


def _bits_to_angle(bits, unit):
    return sum(b * 2.0 ** -(i + 1) for i, b in enumerate(bits)) * unit


@pytest.mark.parametrize("key", [-1, 5])
def test_precision_stats_invalid_key(key):
    """Test that precision_stats keys outside of 0..len(angle_wires) raise an error."""
    with pytest.raises(ValueError, match="precision_stats keys must be between 0 and 4"):
        make_crz_to_phase_gradient_decomp(
            [2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12], precision_stats={key: 1}
        )


@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_truncates(bits, precision_stats, width):
    """Test that the data loading, the phase-gradient flips and the SemiAdder are truncated to
    the smallest precision in ``precision_stats`` that suffices for the angle, or to full
    precision otherwise."""
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(4)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(4)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(3)])
    rule = make_crz_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
    )

    with qp.queuing.AnnotatedQueue() as q:
        rule(_bits_to_angle(bits, 4 * np.pi), wires=[0, 1])
    ops = list(_leaf_ops(q.queue))

    if width == 0:
        assert not ops
        return
    adders = [op for op in ops if isinstance(op, qp.SemiAdder)]
    fanouts = [op for op in ops if isinstance(op, qp.ops.Controlled)]
    assert len(adders) == 1 and len(fanouts) == 4
    assert adders[0].x_wires == angle_wires[:width]
    assert adders[0].y_wires == phase_grad_wires[:width]
    angle_fanouts = [op for op in fanouts if op.base.wires == angle_wires[:width]]
    flip_fanouts = [op for op in fanouts if op.base.wires == phase_grad_wires[:width]]
    assert len(angle_fanouts) == len(flip_fanouts) == 2
    for fanout in angle_fanouts:
        assert list(fanout.base.bitstring) == bits[:width]


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_valid_decomp(bits, precision_stats, width):
    """Test that the rule with ``precision_stats`` yields a valid decomposition."""
    # pylint: disable=unused-argument
    rule = make_crz_to_phase_gradient_decomp(
        [2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12], precision_stats=precision_stats
    )
    op = qp.CRZ(_bits_to_angle(bits, 4 * np.pi), [0, 1])
    _test_decomposition_rule(op, rule, skip_decomp_matrix_check=True)


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

    rule = make_crz_to_phase_gradient_decomp(
        [2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12], precision_stats=precision_stats
    )
    jaxpr = jax.make_jaxpr(lambda phi: rule(phi, wires=[0, 1]))(0.4)
    assert hint_helpers.cond_hints(jaxpr.jaxpr) == [pytest.approx(expected)]


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("bits, precision_stats, width", PRECISION_STATS_CASES)
def test_precision_stats_integration(bits, precision_stats, width, seed):
    """Test that the decomposition with ``precision_stats`` realizes CRZ."""
    # pylint: disable=unused-argument
    phi = _bits_to_angle(bits, 4 * np.pi)
    wires = [0, 1]
    angle_wires = [f"aux_{i}" for i in range(4)]
    phase_grad_wires = [f"qft_{i}" for i in range(4)]
    work_wires = [f"work_{i}" for i in range(3)]
    all_wires = angle_wires + phase_grad_wires + work_wires + wires

    custom_decomp = make_crz_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
    )

    @qp.transforms.decompose(
        gate_set={
            "StatePrep",
            "PhaseGradientStatePrep",
            "Adjoint(PhaseGradientStatePrep)",
            "CNOT",
            "PauliX",
            "SemiAdder",
        },
        fixed_decomps={qp.CRZ: custom_decomp},
    )
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(phi, in_state):
        qp.StatePrep(in_state, wires=wires)
        qp.PhaseGradientStatePrep(wires=phase_grad_wires)
        qp.CRZ(phi, wires)
        qp.adjoint(qp.PhaseGradientStatePrep(wires=phase_grad_wires))
        return qp.state()

    rng = np.random.default_rng(seed=seed)
    in_state = rng.random(4) + 1j * rng.random(4)
    in_state /= np.linalg.norm(in_state)

    out_state = circuit(phi, in_state)

    zeros = np.eye(2 ** (len(all_wires) - 2), 1)[:, 0]
    out_state_expected = np.kron(zeros, qp.matrix(qp.CRZ(phi, wires)) @ in_state)
    assert np.allclose(out_state, out_state_expected)

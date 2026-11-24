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
"""Tests for ``qp.transforms.decompositions.make_rzlayer_to_phase_gradient_decomp``."""

from collections import Counter

import numpy as np
import pytest
from phase_gradient_helpers import bits_to_angles

import pennylane as qp
from pennylane.testing import assert_valid_decomp_rule
from pennylane.transforms import _tape_decompose
from pennylane.transforms.decompositions import make_rzlayer_to_phase_gradient_decomp
from pennylane.transforms.decompositions.rz_layer_phase_gradient import (
    _Monomials,
    _RZLayerPhaseGradientCompute,
)
from pennylane.typing import Float, Wire

# Bits of the rotation angles, in units of 2π. Batches of distinct angles use one more bit.
NUM_BITS = 2

GATE_SET = {"CNOT", "PauliX", "TemporaryAND", "Adjoint(TemporaryAND)", "SemiAdder", "GlobalPhase"}

# (num_angles, block_size, batch_size, num_work_wires): (num_adders, num_ands), where the RZLayer
# provides the work wires the rule needs, and num_ands counts the ANDs on top of the adders. Each
# of them targets a work wire: 0, 2 and 7 per block of 2, 4 and 6 equal angles, and 0, 1 and 4 per
# batch of 2, 3 and 4 distinct angles. The cases cover even blocks, batches with a smaller last
# batch and with fewer angles than ``batch_size``, and the RZ rule on every wire for odd blocks,
# also with ``batch_size``, and for single rotations without ``batch_size``.
CASES = {
    (2, 2, None, 0): (2, 0),
    (2, 4, None, 2): (2, 4),
    (1, 6, None, 7): (1, 7),
    (5, 1, 2, 0): (3, 0),
    (3, 1, 4, 1): (1, 1),
    (5, 1, 4, 4): (2, 4),
    (2, 3, 2, 0): (6, 0),
    (3, 1, None, 0): (3, 0),
}


def _rule_and_op(num_angles, block_size, batch_size, num_work_wires, seed):
    """A rule and an RZLayer with random angles on the wires ``0, 1, ...``, followed by its work
    wires and by the angle, phase gradient and adder work registers of the rule. The angles have
    ``NUM_BITS + 1`` binary decimals in units of 4π, so that all branches of the rule realize them
    exactly."""
    num_wires = num_angles * block_size
    bits = np.random.default_rng(seed).integers(0, 2, size=(num_angles, NUM_BITS + 1))
    op = qp.RZLayer(
        bits_to_angles(bits, 4 * np.pi),
        wires=range(num_wires),
        work_wires=range(num_wires, num_wires + num_work_wires),
    )
    b = NUM_BITS + (batch_size is not None)
    start = num_wires + num_work_wires
    registers = [range(start, start + b), range(start + b, start + 2 * b)]
    registers.append(range(start + 2 * b, start + 3 * b - 1))
    rule = make_rzlayer_to_phase_gradient_decomp(*registers, batch_size=batch_size)
    return rule, op, registers


@pytest.mark.parametrize("case", CASES)
def test_state(case, seed):
    """Test that the rule realizes the rotations including their global phase on a random state
    if a phase gradient state is prepared, and that it returns the work wires and the registers to
    their initial state."""
    rule, op, registers = _rule_and_op(*case, seed)
    wires, phase_grad_wires = op.arguments["wires"], registers[1]
    num_aux_wires = len(op.work_wires) + 3 * len(phase_grad_wires) - 1
    state = np.array([1, 1j]) @ np.random.default_rng(seed).normal(size=(2, 2 ** len(wires)))
    state /= np.linalg.norm(state)

    @qp.qnode(qp.device("default.qubit", wires=len(wires) + num_aux_wires))
    def circuit():
        qp.StatePrep(state, wires=wires)
        qp.PhaseGradientStatePrep(phase_grad_wires)
        rule(op.angles, wires, op.work_wires)
        qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires))
        return qp.state()

    zeros = np.eye(2**num_aux_wires, 1)[:, 0]
    expected = np.kron(qp.matrix(qp.RZLayer(op.angles, wires)) @ state, zeros)
    assert np.allclose(circuit(), expected)


@pytest.mark.parametrize("num_angles, block_size", [(2, 4), (5, 1)])
def test_allocates_missing_work_wires(num_angles, block_size, seed):
    """Test that the rule allocates the work wires that the RZLayer does not provide and realizes
    the rotations, for equal angles and for batches of four distinct angles. On ``|+>``, ``RZ(a)``
    gives ``<X> = cos(a)`` and ``<Y> = sin(a)``."""
    rule, op, registers = _rule_and_op(num_angles, block_size, 4, 0, seed)
    wires, phase_grad_wires = op.arguments["wires"], registers[1]
    assert rule.get_work_wire_spec(**op.arguments).total > 0

    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        for wire in wires:
            qp.H(wire)
        qp.PhaseGradientStatePrep(phase_grad_wires)
        rule(op.angles, wires, op.work_wires)
        qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires))
        return [qp.expval(qp.X(w)) for w in wires] + [qp.expval(qp.Y(w)) for w in wires]

    expected = np.repeat(op.angles, block_size)
    assert np.allclose(circuit(), np.concatenate([np.cos(expected), np.sin(expected)]))


def test_uses_spare_work_wires(seed):
    """Test that the TemporaryAND gates use the work wires of the rule that the adder does not
    need, together with those of the RZLayer, and that the rule realizes the rotations without
    requesting further work wires. On ``|+>``, ``RZ(a)`` gives ``<X> = cos(a)`` and
    ``<Y> = sin(a)``."""
    _, op, registers = _rule_and_op(2, 4, None, 1, seed)
    wires, phase_grad_wires = op.arguments["wires"], registers[1]
    spare_wire = registers[2][-1] + 1
    rule = make_rzlayer_to_phase_gradient_decomp(*registers[:2], [*registers[2], spare_wire])
    assert rule.get_work_wire_spec(**op.arguments).total == 0

    @qp.qnode(qp.device("default.qubit", wires=spare_wire + 1))
    def circuit():
        for wire in wires:
            qp.H(wire)
        qp.PhaseGradientStatePrep(phase_grad_wires)
        rule(op.angles, wires, op.work_wires)
        qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires))
        return [qp.expval(qp.X(w)) for w in wires] + [qp.expval(qp.Y(w)) for w in wires]

    expected = np.repeat(op.angles, 4)
    assert np.allclose(circuit(), np.concatenate([np.cos(expected), np.sin(expected)]))
    tape = qp.workflow.construct_tape(circuit)()
    assert spare_wire in tape.wires


class TestRule:
    """Tests for the resources and work wires of the rule."""

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("case", CASES)
    def test_valid_decomp_rule(self, case, seed):
        """Test that the resources of the rule match the emitted operators and can be computed
        from the abstract shapes ``Float[k]``, ``Wire[k*m]`` and ``Wire[w]``. Under program
        capture, the angles are traced."""
        rule, op, _ = _rule_and_op(*case, seed)
        num_angles, num_wires = len(op.angles), len(op.arguments["wires"])

        assert_valid_decomp_rule(op, rule, check_matrix=False)
        resources = rule.compute_resources(
            angles=Float[num_angles], wires=Wire[num_wires], work_wires=Wire[len(op.work_wires)]
        )
        assert resources.gate_counts == rule.compute_resources(**op.arguments).gate_counts

    @pytest.mark.parametrize("case, num_adders, num_ands", [(c, *n) for c, n in CASES.items()])
    def test_counts(self, case, num_adders, num_ands, seed):
        """Test the number of adders and of ANDs on top of them, and that the rule only uses the
        wires of the RZLayer and the registers if the RZLayer provides enough work wires. With
        fewer work wires, it requests the missing ones."""
        rule, op, registers = _rule_and_op(*case, seed)

        [tape], _ = _tape_decompose(
            qp.tape.QuantumScript([op]), gate_set=GATE_SET, fixed_decomps={qp.RZLayer: rule}
        )

        counts = Counter(gate.name for gate in tape.operations)
        assert set(counts) <= GATE_SET
        assert counts["SemiAdder"] == num_adders
        assert counts["TemporaryAND"] == counts["Adjoint(TemporaryAND)"] == num_ands
        assert set(tape.wires) <= set(op.wires + qp.wires.Wires.all_wires(registers))
        for num_missing in range(len(op.work_wires) + 1):
            fewer = qp.RZLayer(op.angles, op.arguments["wires"], op.work_wires[num_missing:])
            assert rule.get_work_wire_spec(**fewer.arguments).total == num_missing

    @pytest.mark.parametrize(
        "op",
        [
            _Monomials([0, 1, 2], [3, 4, 5, 6]),
            _RZLayerPhaseGradientCompute(
                np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=bool),
                [0, 1, 2],
                [3],
                [4, 5],
                [6, 7],
            ),
        ],
    )
    def test_subcircuits(self, op):
        """Test that the resources of the subcircuits match their decompositions."""
        qp.testing.assert_decomposable(op, check_matrix=False)


@pytest.mark.catalyst
@pytest.mark.parametrize(
    "case, num_ands",
    [((2, 4, None, 2), 4 + 2), ((5, 1, 4, 4), 4 + 2 * 2), ((2, 3, None, 0), 6)],
    ids=["equal_angles", "batches", "per_wire"],
)
def test_qjit(case, num_ands, seed):
    """Test that, compiled with qjit, the rule realizes the rotations. The RZLayer provides all
    work wires the rule needs, as Catalyst does not support the dynamic allocation of missing
    ones. On ``|+>``, ``RZ(a)`` gives ``<X> = cos(a)`` and ``<Y> = sin(a)``. On the device, the
    ``num_ands`` ANDs, including those of the adders, and their adjoints become Toffoli gates."""
    pytest.importorskip("catalyst")
    rule, op, registers = _rule_and_op(*case, seed)
    wires, phase_grad_wires = op.arguments["wires"], registers[1]
    block_size = len(wires) // len(op.angles)
    qp.add_decomps(qp.RZLayer, rule)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET | {"Hadamard", "PhaseShift"}, fixed_decomps={qp.RZLayer: rule})
    @qp.qnode(qp.device("lightning.qubit", wires=registers[2][-1] + 1))
    def circuit(angles):
        for wire in wires:
            qp.H(wire)
        qp.PhaseGradientStatePrep(phase_grad_wires)
        qp.RZLayer(angles, wires, work_wires=op.work_wires)
        qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires))
        return [qp.expval(qp.X(w)) for w in wires] + [qp.expval(qp.Y(w)) for w in wires]

    expected = np.repeat(op.angles, block_size)
    assert np.allclose(circuit(op.angles), np.concatenate([np.cos(expected), np.sin(expected)]))
    counts = qp.analyze(circuit, level="device")(op.angles).resources.counts
    assert counts["Toffoli"] == 2 * num_ands

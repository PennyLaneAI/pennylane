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
    _signed_add,
)
from pennylane.typing import Float, Wire

# Bits of the rotation angle, in units of 2π. With two bits, all lookup tables are linear in the
# bits of the Hamming weight, so that tests of their products use three.
NUM_BITS = 2

GATE_SET = {"CNOT", "PauliX", "TemporaryAND", "Adjoint(TemporaryAND)", "SemiAdder", "GlobalPhase"}

# (num_wires, num_work_wires): num_ands, where the RZLayer provides the work wires the rule needs,
# and num_ands counts the ANDs on top of the adder. Each of them targets a work wire: 0, 2, 2, 5 and
# 5 for 2 to 6 wires. The cases cover even and odd numbers of wires, and the RZ rule for one wire.
CASES = {(2, 0): 0, (4, 2): 2, (6, 5): 5, (3, 2): 2, (5, 5): 5, (1, 0): 0}


def _rule_and_op(num_wires, num_work_wires, seed, b=NUM_BITS):
    """A rule and an RZLayer with a random angle on the wires ``0, 1, ...``, followed by its work
    wires and by the angle, phase gradient and adder work registers of ``b``, ``b`` and ``b - 1``
    wires. The angle lies in ``[0, 4π)`` and has ``b`` binary decimals in units of 2π, so that the
    rule realizes it exactly."""
    bits = np.random.default_rng(seed).integers(0, 2, size=b + 1)
    op = qp.RZLayer(
        bits_to_angles(bits, 4 * np.pi),
        wires=range(num_wires),
        work_wires=range(num_wires, num_wires + num_work_wires),
    )
    start = num_wires + num_work_wires
    registers = [range(start, start + b), range(start + b, start + 2 * b)]
    registers.append(range(start + 2 * b, start + 3 * b - 1))
    rule = make_rzlayer_to_phase_gradient_decomp(*registers)
    return rule, op, registers


def _check_state(rule, op, registers, seed):
    """Check that the rule realizes the rotations including their global phase on a random state
    if a phase gradient state is prepared, and that it returns the work wires and the registers to
    their initial state."""
    wires, phase_grad_wires = op.arguments["wires"], registers[1]
    num_aux_wires = len(op.work_wires) + 3 * len(phase_grad_wires) - 1
    state = np.array([1, 1j]) @ np.random.default_rng(seed).normal(size=(2, 2 ** len(wires)))
    state /= np.linalg.norm(state)

    @qp.qnode(qp.device("default.qubit", wires=len(wires) + num_aux_wires))
    def circuit():
        qp.StatePrep(state, wires=wires)
        qp.PhaseGradientStatePrep(phase_grad_wires)
        rule(op.angle, wires, op.work_wires)
        qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires))
        return qp.state()

    zeros = np.eye(2**num_aux_wires, 1)[:, 0]
    expected = np.kron(qp.matrix(qp.RZLayer(op.angle, wires)) @ state, zeros)
    assert np.allclose(circuit(), expected)


@pytest.mark.parametrize("case", CASES)
def test_state(case, seed):
    """Test that the rule realizes the rotations on a random state, see ``_check_state``."""
    _check_state(*_rule_and_op(*case, seed), seed)


@pytest.mark.parametrize(
    "num_wires, num_work_wires, angle",
    [(3, 2, 11 * np.pi / 4), (4, 2, 11 * np.pi / 4), (5, 5, 11 * np.pi / 4), (6, 5, 9 * np.pi / 4)],
)
def test_state_with_products(num_wires, num_work_wires, angle, seed):
    """Test that the rule realizes rotations whose lookup table contains a product of the bits of
    the Hamming weight, see ``_check_state``. With three bits, the angles ``11π/4`` for three to
    five wires and ``9π/4`` for six wires, which also lie in ``[2π, 4π)``, have such a table. For
    five and six wires, the rule skips some of the monomials."""
    rule, op, registers = _rule_and_op(num_wires, num_work_wires, seed, b=3)
    op = qp.RZLayer(angle, op.arguments["wires"], op.work_wires)
    _check_state(rule, op, registers, seed)


@pytest.mark.parametrize("max_input", [5, 6, 7])
def test_signed_add_table(max_input, seed):
    """Test that ``_signed_add`` adds the entry of a random lookup table over three bits at each
    input up to ``max_input`` to the phase gradient register in the computational basis, also if it
    skips the monomials that vanish for all these inputs."""
    table = np.random.default_rng(seed).integers(0, 8, size=8)
    var_wires, mono_wires = [0, 1, 2], [3, 4, 5, 6][: max_input - 3]
    registers = ([7, 8, 9], [10, 11, 12], [13, 14])

    @qp.qnode(qp.device("default.qubit"))
    def circuit(bits):
        qp.BasisState(bits, var_wires)
        _signed_add(table, var_wires, mono_wires, [], registers)
        return qp.probs(wires=registers[1])

    for v in range(max_input + 1):
        assert np.isclose(circuit([(v >> j) & 1 for j in range(3)])[table[v]], 1)


@pytest.mark.parametrize("num_wires", [3, 4])
def test_allocates_missing_work_wires(num_wires, seed):
    """Test that the rule allocates the work wires that the RZLayer does not provide and realizes
    the rotations. On ``|+>``, ``RZ(a)`` gives ``<X> = cos(a)`` and ``<Y> = sin(a)``."""
    rule, op, registers = _rule_and_op(num_wires, 0, seed)
    wires, phase_grad_wires = op.arguments["wires"], registers[1]
    assert rule.get_work_wire_spec(**op.arguments).total > 0

    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        for wire in wires:
            qp.H(wire)
        qp.PhaseGradientStatePrep(phase_grad_wires)
        rule(op.angle, wires, op.work_wires)
        qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires))
        return [qp.expval(qp.X(w)) for w in wires] + [qp.expval(qp.Y(w)) for w in wires]

    assert np.allclose(circuit(), np.repeat([np.cos(op.angle), np.sin(op.angle)], num_wires))


def test_uses_spare_work_wires(seed):
    """Test that the TemporaryAND gates use the work wires of the rule that the adder does not
    need, together with those of the RZLayer, and that the rule realizes the rotations without
    requesting further work wires. On ``|+>``, ``RZ(a)`` gives ``<X> = cos(a)`` and
    ``<Y> = sin(a)``."""
    _, op, registers = _rule_and_op(4, 1, seed)
    wires, phase_grad_wires = op.arguments["wires"], registers[1]
    spare_wire = registers[2][-1] + 1
    rule = make_rzlayer_to_phase_gradient_decomp(*registers[:2], [*registers[2], spare_wire])
    assert rule.get_work_wire_spec(**op.arguments).total == 0

    @qp.qnode(qp.device("default.qubit", wires=spare_wire + 1))
    def circuit():
        for wire in wires:
            qp.H(wire)
        qp.PhaseGradientStatePrep(phase_grad_wires)
        rule(op.angle, wires, op.work_wires)
        qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires))
        return [qp.expval(qp.X(w)) for w in wires] + [qp.expval(qp.Y(w)) for w in wires]

    assert np.allclose(circuit(), np.repeat([np.cos(op.angle), np.sin(op.angle)], 4))
    tape = qp.workflow.construct_tape(circuit)()
    assert spare_wire in tape.wires


class TestRule:
    """Tests for the resources and work wires of the rule."""

    def test_name(self):
        """Test the name of the rule, which selects it in ``fixed_decomps``."""
        assert make_rzlayer_to_phase_gradient_decomp([0, 1], [2, 3], [4]).name == "phase_gradient"

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("case", CASES)
    def test_valid_decomp_rule(self, case, seed):
        """Test that the resources of the rule match the emitted operators and can be computed
        from the abstract shapes ``Float``, ``Wire[m]`` and ``Wire[w]``. Under program capture,
        the angle is traced."""
        rule, op, _ = _rule_and_op(*case, seed)
        num_wires, num_work_wires = case

        assert_valid_decomp_rule(op, rule, check_matrix=False)
        resources = rule.compute_resources(
            angle=Float, wires=Wire[num_wires], work_wires=Wire[num_work_wires]
        )
        assert resources.gate_counts == rule.compute_resources(**op.arguments).gate_counts

    @pytest.mark.parametrize("case, num_ands", CASES.items())
    def test_counts(self, case, num_ands, seed):
        """Test that the rule uses one adder, the number of ANDs on top of it, and that the rule
        only uses the wires of the RZLayer and the registers if the RZLayer provides enough work
        wires. With fewer work wires, it requests the missing ones."""
        rule, op, registers = _rule_and_op(*case, seed)

        [tape], _ = _tape_decompose(
            qp.tape.QuantumScript([op]), gate_set=GATE_SET, fixed_decomps={qp.RZLayer: rule}
        )

        counts = Counter(gate.name for gate in tape.operations)
        assert set(counts) <= GATE_SET
        assert counts["SemiAdder"] == 1
        assert counts["TemporaryAND"] == counts["Adjoint(TemporaryAND)"] == num_ands
        assert set(tape.wires) <= set(op.wires + qp.wires.Wires.all_wires(registers))
        for num_missing in range(len(op.work_wires) + 1):
            fewer = qp.RZLayer(op.angle, op.arguments["wires"], op.work_wires[num_missing:])
            assert rule.get_work_wire_spec(**fewer.arguments).total == num_missing

    @pytest.mark.parametrize(
        "op",
        [
            _Monomials([0, 1, 2], [3, 4, 5, 6]),
            _Monomials([0, 1, 2], [3, 4]),
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
def test_qjit(seed):
    """Test that, compiled with qjit, the rule realizes the rotations on an even and an odd number
    of wires and a single rotation. The RZLayers get all work wires the rule needs, as Catalyst
    does not support the dynamic allocation of missing ones. On ``|+>``, ``RZ(a)`` gives
    ``<X> = cos(a)`` and ``<Y> = sin(a)``. On the device, the ANDs, including those of the adders,
    and their adjoints become Toffoli gates: three ANDs for each of the first two RZLayers and one
    for the single rotation."""
    bits = np.random.default_rng(seed).integers(0, 2, size=(3, NUM_BITS + 1))
    angles = bits_to_angles(bits, 4 * np.pi)
    work_wires, registers = [8, 9], [[10, 11], [12, 13], [14]]
    rule = make_rzlayer_to_phase_gradient_decomp(*registers)
    qp.add_decomps(qp.RZLayer, rule)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET | {"Hadamard", "PhaseShift"}, fixed_decomps={qp.RZLayer: rule})
    @qp.qnode(qp.device("lightning.qubit", wires=15))
    def circuit(angles):
        for wire in range(8):
            qp.H(wire)
        qp.PhaseGradientStatePrep(registers[1])
        qp.RZLayer(angles[0], [0, 1, 2, 3], work_wires=work_wires)
        qp.RZLayer(angles[1], [4, 5, 6], work_wires=work_wires)
        qp.RZLayer(angles[2], [7], work_wires=work_wires)
        qp.adjoint(qp.PhaseGradientStatePrep(registers[1]))
        return [qp.expval(qp.X(w)) for w in range(8)] + [qp.expval(qp.Y(w)) for w in range(8)]

    expected = np.repeat(angles, [4, 3, 1])
    assert np.allclose(circuit(angles), np.concatenate([np.cos(expected), np.sin(expected)]))
    counts = qp.analyze(circuit, level="device")(angles).resources.counts
    assert counts["Toffoli"] == 2 * (3 + 3 + 1)

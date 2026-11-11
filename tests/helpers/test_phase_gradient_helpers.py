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
"""Tests for the pytest helper functions in ``phase_gradient_helpers.py``."""

from collections import Counter

import numpy as np
import phase_gradient_helpers
import pytest

import pennylane as qp
from pennylane.ops.op_math import Adjoint2, Prod2
from pennylane.transforms.decompositions import (
    make_crz_to_phase_gradient_decomp,
    make_rz_to_phase_gradient_decomp,
)

SEMI_ADDER_RULE = qp.list_decomps(qp.SemiAdder)["_semi_adder"]


def _phase_grad_registers(p):
    """Angle, phase gradient and work wires of a ``p``-bit phase gradient rule, after wires 0, 1."""
    return list(range(2, 2 + p)), list(range(2 + p, 2 + 2 * p)), list(range(2 + 2 * p, 1 + 3 * p))


def _rz_phase_gradient_tape():
    """An ``RZ`` with an exactly representable angle, its decomposition by the 4-bit phase gradient
    rule, and the phase gradient wires."""
    angle_wires, phase_grad_wires, work_wires = _phase_grad_registers(4)
    rule = make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)
    op = qp.RZ(phase_gradient_helpers.bits_to_angles([1, 0, 1, 1], 2 * np.pi), 0)
    return op, qp.testing.decomp_rule_to_tape(op, rule), phase_grad_wires


def _fail_on_wide_matrix(matrix):
    """Wrap a ``matrix`` method to fail at once for more than 12 wires, rather than exhausting the
    memory with a dense matrix."""

    def guarded_matrix(op, wire_order=None):
        if len(op.wires) > 12:
            pytest.fail(f"{op.name} on {len(op.wires)} wires is simulated as a dense matrix")
        return matrix(op, wire_order=wire_order)

    return guarded_matrix


def _compute_and_of_three():
    """Compute the AND of wires 0, 1 and 2 into wire 4, with a temporary AND on wire 3."""
    qp.TemporaryAND([0, 1, 3])
    qp.TemporaryAND([3, 2, 4])
    qp.adjoint(qp.TemporaryAND([0, 1, 3]))


def test_leaf_ops_unpacks_default_uncompute_block():
    """Test that ``leaf_ops`` unpacks the default uncompute block of a change of basis, which is the
    adjoint of the compute block, and that ``count_ands`` counts the AND that it recomputes."""
    op = qp.change_op_basis(_compute_and_of_three, qp.SemiAdder([4], [5, 6, 7], [8, 9]))
    assert [o.name for o in phase_gradient_helpers.leaf_ops([op])] == [
        "TemporaryAND",
        "TemporaryAND",
        "Adjoint(TemporaryAND)",
        "SemiAdder",
        "TemporaryAND",
        "Adjoint(TemporaryAND)",
        "Adjoint(TemporaryAND)",
    ]
    assert phase_gradient_helpers.count_ands([op]) == (3, 2)


@pytest.mark.catalyst
def test_leaf_ops_matches_qjit_resources():
    """Test that ``leaf_ops`` unpacks a change of basis with the default uncompute block into the
    operators of the circuit compiled with ``qjit``, so that tests can compare the two."""
    pytest.importorskip("catalyst")

    def qfunc():
        qp.change_op_basis(_compute_and_of_three, qp.SemiAdder([4], [5, 6, 7], [8, 9]))

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("null.qubit", wires=10))
    def circuit():
        qfunc()
        return qp.expval(qp.Z(0))

    ops = phase_gradient_helpers.record(qfunc)
    expected = Counter(op.name for op in phase_gradient_helpers.leaf_ops(ops))
    assert qp.analyze(circuit, level=0)().resources.counts == expected


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("num_y_wires", [2, 4])
def test_count_ands_semi_adder(num_y_wires):
    """Test that ``count_ands`` counts ``len(y_wires) - 1`` ANDs for a ``SemiAdder``, and as many
    ``TemporaryAND`` gates in its decomposition, whose adjoint ``TemporaryAND`` gates are
    measurement-based and not counted."""
    x_wires = list(range(num_y_wires))
    y_wires = list(range(num_y_wires, 2 * num_y_wires))
    work_wires = list(range(2 * num_y_wires, 3 * num_y_wires - 1))
    op = qp.SemiAdder(x_wires, y_wires, work_wires)
    assert phase_gradient_helpers.count_ands([op]) == (0, num_y_wires - 1)

    tape = qp.testing.decomp_rule_to_tape(op, SEMI_ADDER_RULE)
    assert phase_gradient_helpers.count_ands(tape.operations) == (num_y_wires - 1, 0)


def test_count_ands_adjoint_semi_adder():
    """Test that ``count_ands`` counts ``len(y_wires) - 1`` ANDs for an adjoint ``SemiAdder``, a
    subtractor, as many as the ``TemporaryAND`` gates in its decomposition. ``leaf_ops`` produces
    it from the adjoint of a change of basis around an adder."""
    adder = qp.SemiAdder([4], [5, 6, 7], [8, 9])
    op = qp.adjoint(qp.change_op_basis(_compute_and_of_three, adder))
    assert phase_gradient_helpers.count_ands([op]) == (3, 2)
    assert phase_gradient_helpers.count_ands(qp.adjoint(adder).decomposition()) == (2, 0)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize(
    "op_type, wires, make_rule",
    [
        (qp.RZ, [0], make_rz_to_phase_gradient_decomp),
        (qp.CRZ, [0, 1], make_crz_to_phase_gradient_decomp),
    ],
)
def test_count_ands_phase_gradient_rules(op_type, wires, make_rule):
    """Test that ``count_ands`` finds the adder in the change of basis of the RZ and CRZ phase
    gradient rules."""
    p = 4
    rule = make_rule(*_phase_grad_registers(p))
    tape = qp.testing.decomp_rule_to_tape(op_type(0.3, wires), rule)
    assert phase_gradient_helpers.count_ands(tape.operations) == (0, p - 1)


@pytest.mark.parametrize("num_work_wires, expected", [(3, 0), (4, 0), (2, 1), (0, 3)])
def test_work_wire_spec_total_semi_adder(num_work_wires, expected):
    """Test that ``work_wire_spec_total`` returns the number of work wires that the ``SemiAdder``
    rule allocates: none if the operator has ``len(y_wires) - 1`` work wires, and the missing ones
    otherwise, in which case Catalyst excludes the rule."""
    op = qp.SemiAdder([0, 1, 2, 3], [4, 5, 6, 7], list(range(8, 8 + num_work_wires)))
    assert phase_gradient_helpers.work_wire_spec_total(SEMI_ADDER_RULE, op) == expected


def test_realizes_op_with_zeroed_wires(seed):
    """Test that the ``zeroed_wires`` of the operator start in zero, so that a decomposition may
    use them as work wires. The ``SemiAdder`` decomposition with its work wires in reverse order
    realizes the operator only if they start in zero."""
    op = qp.SemiAdder([0, 1], [2, 3, 4], [5, 6])
    tape = qp.testing.decomp_rule_to_tape(qp.SemiAdder([0, 1], [2, 3, 4], [6, 5]), SEMI_ADDER_RULE)
    phase_gradient_helpers.assert_decomp_realizes_op(op, tape, [], seed, zeroed_wires=[5, 6])
    with pytest.raises(AssertionError):
        phase_gradient_helpers.assert_decomp_realizes_op(op, tape, [], seed)


def test_realizes_op_zeroed_wires_return_to_zero(seed):
    """Test that the ``zeroed_wires`` of the operator must return to zero."""
    op = qp.SemiAdder([0, 1], [2, 3, 4], [5, 6])
    tape = qp.testing.decomp_rule_to_tape(op, SEMI_ADDER_RULE)
    dirty_tape = qp.tape.QuantumScript([*tape.operations, qp.X(6)])
    with pytest.raises(AssertionError):
        phase_gradient_helpers.assert_decomp_realizes_op(
            op, dirty_tape, [], seed, zeroed_wires=[5, 6]
        )


def test_realizes_op_checks_global_phase(seed):
    """Test that a global phase error of 1e-6, which ``np.allclose`` misses, fails the check."""
    op, tape, phase_grad_wires = _rz_phase_gradient_tape()
    phase_gradient_helpers.assert_decomp_realizes_op(op, tape, phase_grad_wires, seed)
    tape = qp.tape.QuantumScript([*tape.operations, qp.GlobalPhase(1e-6)])
    with pytest.raises(AssertionError):
        phase_gradient_helpers.assert_decomp_realizes_op(op, tape, phase_grad_wires, seed)


def test_realizes_op_checks_phase_gradient_restored(seed):
    """Test that the phase gradient register must return to the phase gradient state."""
    op, tape, phase_grad_wires = _rz_phase_gradient_tape()
    tape = qp.tape.QuantumScript([*tape.operations, qp.Z(phase_grad_wires[0])])
    with pytest.raises(AssertionError):
        phase_gradient_helpers.assert_decomp_realizes_op(op, tape, phase_grad_wires, seed)


@pytest.mark.parametrize("default_uncompute", [False, True])
def test_realizes_op_flattens_change_of_basis(default_uncompute, seed, monkeypatch):
    """Test that ``assert_decomp_realizes_op`` simulates the gates in a change of basis rather than
    its blocks. At ``p = 6``, the compute block of the CRZ phase gradient rule acts on 14 of the 19
    wires, too many to simulate it as a dense matrix. So does the default uncompute block, the
    adjoint of the compute block, which equals the rule's uncompute block."""
    monkeypatch.setattr(Prod2, "matrix", _fail_on_wide_matrix(Prod2.matrix))
    monkeypatch.setattr(Adjoint2, "matrix", _fail_on_wide_matrix(Adjoint2.matrix))
    p = 6
    angle_wires, phase_grad_wires, work_wires = _phase_grad_registers(p)
    rule = make_crz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)
    phi = phase_gradient_helpers.bits_to_angles([1, 0, 1, 1, 0, 1], 4 * np.pi)
    op = qp.CRZ(phi, [0, 1])
    tape = qp.testing.decomp_rule_to_tape(op, rule)
    if default_uncompute:
        [cob] = tape.operations
        tape = qp.tape.QuantumScript([qp.change_op_basis(cob.compute_op, cob.target_op)])
    phase_gradient_helpers.assert_decomp_realizes_op(op, tape, phase_grad_wires, seed)


@pytest.mark.parametrize(
    "x, width, expected", [(1, 3, [0, 0, 1]), (6, 3, [1, 1, 0]), (5, 4, [0, 1, 0, 1])]
)
def test_bits(x, width, expected):
    """Test that ``bits`` returns the big-endian bits of an integer, padded to ``width``."""
    assert phase_gradient_helpers.bits(x, width) == expected


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_record():
    """Test that ``record`` returns the operations of ``qfunc(*args)``. Under program capture, the
    arguments are traced as jax arrays, so that a ``for_loop`` can index them with its index."""

    def qfunc(angles, wires):
        @qp.for_loop(len(wires))
        def _rz(i):
            qp.RZ(angles[i], wires[i])

        _rz()

    ops = phase_gradient_helpers.record(qfunc, [0.3, 0.5], [2, 7])
    for op, expected_op in zip(ops, [qp.RZ(0.3, 2), qp.RZ(0.5, 7)], strict=True):
        qp.assert_equal(op, expected_op, check_interface=False)


def test_basis_outputs():
    """Test that ``basis_outputs`` returns the big-endian bits of the basis states that the
    operations map the inputs to, and fails for an output with a phase other than 1."""
    ops = [qp.X(0), qp.CNOT([0, 2])]
    inputs = [[0, 0, 0], [1, 1, 0]]
    assert phase_gradient_helpers.basis_outputs(ops, 3, inputs) == [[1, 0, 1], [0, 1, 0]]
    with pytest.raises(AssertionError):
        phase_gradient_helpers.basis_outputs([*ops, qp.Z(0)], 3, inputs)

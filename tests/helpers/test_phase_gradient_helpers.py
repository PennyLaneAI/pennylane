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
from pennylane.transforms.decompositions import make_rz_to_phase_gradient_decomp

SEMI_ADDER_RULE = qp.list_decomps(qp.SemiAdder)["_semi_adder"]


def _rz_phase_gradient_tape():
    """An ``RZ`` with an exactly representable angle, its decomposition by the 4-bit phase gradient
    rule, and the phase gradient wires."""
    angle_wires, phase_grad_wires, work_wires = [2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12]
    rule = make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)
    op = qp.RZ(phase_gradient_helpers.bits_to_angles([1, 0, 1, 1], 2 * np.pi), 0)
    return op, qp.testing.decomp_rule_to_tape(op, rule), phase_grad_wires


def _compute_and_of_three():
    """Compute the AND of wires 0, 1 and 2 into wire 4, with a temporary AND on wire 3."""
    qp.TemporaryAND([0, 1, 3])
    qp.TemporaryAND([3, 2, 4])
    qp.adjoint(qp.TemporaryAND([0, 1, 3]))


def _change_of_basis_around_adder():
    """A change of basis around an adder whose uncompute block is the default one, the adjoint of
    the compute block."""
    return qp.change_op_basis(_compute_and_of_three, qp.SemiAdder([4], [5, 6, 7], [8, 9]))


def test_leaf_ops_and_count_ands():
    """Test that ``leaf_ops`` unpacks a change of basis with the default uncompute block, and that
    ``count_ands`` counts the ANDs that it computes and recomputes and those of the adder, also for
    the adjoint of the change of basis."""
    op = _change_of_basis_around_adder()
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
    assert phase_gradient_helpers.count_ands([qp.adjoint(op)]) == (3, 2)


@pytest.mark.catalyst
def test_leaf_ops_matches_qjit_resources():
    """Test that ``leaf_ops`` unpacks a change of basis with the default uncompute block into the
    operators of the circuit compiled with ``qjit``, so that tests can compare the two."""
    pytest.importorskip("catalyst")

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("null.qubit", wires=10))
    def circuit():
        _change_of_basis_around_adder()
        return qp.expval(qp.Z(0))

    ops = phase_gradient_helpers.record(_change_of_basis_around_adder)
    expected = Counter(op.name for op in phase_gradient_helpers.leaf_ops(ops))
    assert qp.analyze(circuit, level=0)().resources.counts == expected


@pytest.mark.parametrize("num_work_wires, expected", [(3, 0), (2, 1)])
def test_work_wire_spec_total_semi_adder(num_work_wires, expected):
    """Test that ``work_wire_spec_total`` returns the number of work wires that the ``SemiAdder``
    rule allocates: none if the operator has ``len(y_wires) - 1`` work wires, and the missing ones
    otherwise, in which case Catalyst excludes the rule."""
    op = qp.SemiAdder([0, 1, 2, 3], [4, 5, 6, 7], list(range(8, 8 + num_work_wires)))
    assert phase_gradient_helpers.work_wire_spec_total(SEMI_ADDER_RULE, op) == expected


def test_realizes_op_with_zeroed_wires(seed):
    """Test that the ``zeroed_wires`` of the operator start in zero, so that a decomposition may
    use them as work wires, and must return to zero. The ``SemiAdder`` decomposition with its work
    wires in reverse order realizes the operator only if they start in zero."""
    op = qp.SemiAdder([0, 1], [2, 3, 4], [5, 6])
    tape = qp.testing.decomp_rule_to_tape(qp.SemiAdder([0, 1], [2, 3, 4], [6, 5]), SEMI_ADDER_RULE)
    phase_gradient_helpers.assert_decomp_realizes_op(op, tape, [], seed, zeroed_wires=[5, 6])
    with pytest.raises(AssertionError):
        phase_gradient_helpers.assert_decomp_realizes_op(op, tape, [], seed)

    dirty_tape = qp.tape.QuantumScript([*tape.operations, qp.X(6)])
    with pytest.raises(AssertionError):
        phase_gradient_helpers.assert_decomp_realizes_op(
            op, dirty_tape, [], seed, zeroed_wires=[5, 6]
        )


def test_realizes_op_checks_phases(seed):
    """Test that the check passes for a phase gradient rule, and fails for a global phase error of
    1e-6, which ``np.allclose`` misses, and for a phase gradient register that is not restored."""
    op, tape, phase_grad_wires = _rz_phase_gradient_tape()
    phase_gradient_helpers.assert_decomp_realizes_op(op, tape, phase_grad_wires, seed)
    for error in [qp.GlobalPhase(1e-6), qp.Z(phase_grad_wires[0])]:
        wrong_tape = qp.tape.QuantumScript([*tape.operations, error])
        with pytest.raises(AssertionError):
            phase_gradient_helpers.assert_decomp_realizes_op(op, wrong_tape, phase_grad_wires, seed)


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
    outputs = phase_gradient_helpers.basis_outputs(ops, 3, inputs)
    assert np.array_equal(outputs, [[1, 0, 1], [0, 1, 0]])
    with pytest.raises(AssertionError):
        phase_gradient_helpers.basis_outputs([*ops, qp.Z(0)], 3, inputs)

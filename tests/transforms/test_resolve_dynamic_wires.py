# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the `resolve_dynamic_wires` transform.
"""
import pytest

import pennylane as qml
from pennylane.allocation import AllocateState


def test_error_if_no_available_wires():
    """Test that an error is produced if there are no available wires."""

    tape = qml.tape.QuantumScript([qml.allocation.Allocate.from_num_wires(2)])

    with pytest.raises(qml.exceptions.AllocationError, match="no wires left to allocate"):
        qml.transforms.resolve_dynamic_wires(tape)


def test_error_if_use_deallocated_wire():
    """Test an error is raised if we use a deallocated wire."""

    op = qml.allocation.Allocate.from_num_wires(1)
    tape = qml.tape.QuantumScript([op, qml.allocation.Deallocate(op.wires), qml.X(op.wires)])
    with pytest.raises(qml.exceptions.AllocationError, match="Encountered deallocated wires"):
        qml.transforms.resolve_dynamic_wires(tape, min_int=0)


def test_deallocated_wire_in_measurement():
    """Test that an error is raised if a deallocated wire is used in a measurement."""

    op = qml.allocation.Allocate.from_num_wires(1)
    tape = qml.tape.QuantumScript(
        [op, qml.allocation.Deallocate(op.wires)], [qml.probs(wires=op.wires)]
    )
    with pytest.raises(qml.exceptions.AllocationError, match="Encountered deallocated wires"):
        qml.transforms.resolve_dynamic_wires(tape, min_int=0)


def test_highly_nested_min_int():
    """Test that min integer will keep incrementing to higher numbers."""

    allocations = [qml.allocation.Allocate.from_num_wires(1) for _ in range(10)]
    ops = [qml.X(op.wires) for op in allocations]
    tape = qml.tape.QuantumScript(allocations + ops)

    [new_tape], fn = qml.transforms.resolve_dynamic_wires(tape, min_int=5)

    assert fn(("a",)) == "a"
    for op, wire in zip(new_tape.operations, range(5, 15)):
        qml.assert_equal(op, qml.X(wire))


def test_map_measurement_on_dynamic_wires():
    """Test that we can include dynamic wires in measurements and still have them mapped."""

    alloc_op = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ZERO)
    tape = qml.tape.QuantumScript([alloc_op], [qml.probs(wires=alloc_op.wires)])

    [new_tape], fn = qml.transforms.resolve_dynamic_wires(tape, zeroed=("a",))
    assert fn(("a",)) == "a"

    expected = qml.tape.QuantumScript([], [qml.probs(wires="a")])
    qml.assert_equal(new_tape, expected)


def test_shots_preserved():
    """Test that shots are preserved by the transform."""

    op = qml.allocation.Allocate.from_num_wires(1)

    tape = qml.tape.QuantumScript(
        [op, qml.allocation.Deallocate(op.wires)], [qml.state()], shots=500
    )
    [new_tape], _ = qml.transforms.resolve_dynamic_wires(tape, min_int=0)
    assert new_tape.shots == qml.measurements.Shots(500)


class TestFirstExtraction:

    def test_use_zeroed_wire_as_zeroed(self):
        """Test that a zeroed wire is used when one is requested and available."""

        alloc_op = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ZERO)
        tape = qml.tape.QuantumScript(
            [alloc_op, qml.X(alloc_op.wires), qml.allocation.Deallocate(alloc_op.wires)]
        )

        [new_tape], _ = qml.transforms.resolve_dynamic_wires(tape, zeroed=("a",), any_state=("b",))

        assert len(new_tape.operations) == 1
        qml.assert_equal(new_tape[0], qml.X("a"))

    def test_use_any_state_as_any_state(self):
        """Test that an any_state wire is used when one is requested and available."""

        alloc_op = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ANY)
        tape = qml.tape.QuantumScript(
            [alloc_op, qml.X(alloc_op.wires), qml.allocation.Deallocate(alloc_op.wires)]
        )

        [new_tape], _ = qml.transforms.resolve_dynamic_wires(tape, zeroed=("a",), any_state=("b",))

        assert len(new_tape.operations) == 1
        qml.assert_equal(new_tape[0], qml.X("b"))

    def test_use_zeroed_if_no_any_state_available(self):
        """Test that a zeroed wire is provided if no any_state wire is available"""

        alloc_op = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ANY)
        tape = qml.tape.QuantumScript(
            [alloc_op, qml.X(alloc_op.wires), qml.allocation.Deallocate(alloc_op.wires)]
        )

        [new_tape], _ = qml.transforms.resolve_dynamic_wires(tape, zeroed=("a",))

        assert len(new_tape.operations) == 1
        qml.assert_equal(new_tape[0], qml.X("a"))

    def test_reset_any_state_if_no_zeroed_available(self):
        """Test that an any_state wire is reset and provided if no zeroed wire is available."""

        alloc_op = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ZERO)
        tape = qml.tape.QuantumScript(
            [alloc_op, qml.X(alloc_op.wires), qml.allocation.Deallocate(alloc_op.wires)]
        )

        [new_tape], _ = qml.transforms.resolve_dynamic_wires(tape, any_state=("a",))

        assert len(new_tape) == 2
        assert isinstance(new_tape[0], qml.measurements.MidMeasureMP)
        assert new_tape[0].wires == qml.wires.Wires(("a",))
        assert new_tape[0].reset

    def test_no_reset_if_forbidden(self):
        """Test that no reset operations occur if resets are forbidden."""

        alloc_op = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ZERO)
        tape = qml.tape.QuantumScript(
            [alloc_op, qml.X(alloc_op.wires), qml.allocation.Deallocate(alloc_op.wires)]
        )

        with pytest.raises(qml.exceptions.AllocationError, match="no wires left to allocate"):
            qml.transforms.resolve_dynamic_wires(tape, any_state=("a",), allow_resets=False)


class TestReuse:

    def test_reuse_zeroed_qubit(self):
        """Test that zeroed qubits can be reused without reset."""

        alloc1 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ZERO, restored=True)
        dealloc1 = qml.allocation.Deallocate(alloc1.wires)
        alloc2 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ZERO, restored=True)
        dealloc2 = qml.allocation.Deallocate(alloc2.wires)

        tape = qml.tape.QuantumScript(
            [alloc1, qml.X(alloc1.wires), dealloc1, alloc2, qml.Y(alloc2.wires), dealloc2]
        )

        [new_tape], _ = qml.transforms.resolve_dynamic_wires(tape, zeroed=("a",))

        expected = qml.tape.QuantumScript([qml.X("a"), qml.Y("a")])

        qml.assert_equal(new_tape, expected)

    def test_reuse_garbage_qubit(self):
        """Test that garbage qubits can be reused with `state=AllocateState.ANY` without reset."""

        alloc1 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ANY, restored=False)
        dealloc1 = qml.allocation.Deallocate(alloc1.wires)
        alloc2 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ANY, restored=False)
        dealloc2 = qml.allocation.Deallocate(alloc2.wires)

        tape = qml.tape.QuantumScript(
            [alloc1, qml.X(alloc1.wires), dealloc1, alloc2, qml.Y(alloc2.wires), dealloc2]
        )

        [new_tape], _ = qml.transforms.resolve_dynamic_wires(tape, zeroed=("a",))

        expected = qml.tape.QuantumScript([qml.X("a"), qml.Y("a")])

        qml.assert_equal(new_tape, expected)

    def test_reused_garbage_qubit_as_zeroed(self):
        """Test that a garbage qubit can be reused in conjunction with a reset."""

        alloc1 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ANY, restored=False)
        dealloc1 = qml.allocation.Deallocate(alloc1.wires)
        alloc2 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ZERO, restored=False)
        dealloc2 = qml.allocation.Deallocate(alloc2.wires)

        tape = qml.tape.QuantumScript(
            [alloc1, qml.X(alloc1.wires), dealloc1, alloc2, qml.Y(alloc2.wires), dealloc2]
        )

        [new_tape], _ = qml.transforms.resolve_dynamic_wires(tape, zeroed=("a",))

        assert len(new_tape) == 3
        qml.assert_equal(new_tape[0], qml.X("a"))
        qml.assert_equal(new_tape[2], qml.Y("a"))

        assert isinstance(new_tape[1], qml.measurements.MidMeasureMP)
        assert new_tape[1].wires == qml.wires.Wires(("a",))
        assert new_tape[1].reset

    def test_error_if_no_zeroed_wires_left_no_reset(self):
        """Test that an error is raised if all zeroed wires have been used and cant reset."""

        alloc1 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ANY, restored=False)
        dealloc1 = qml.allocation.Deallocate(alloc1.wires)
        alloc2 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ZERO, restored=False)
        dealloc2 = qml.allocation.Deallocate(alloc2.wires)

        tape = qml.tape.QuantumScript(
            [alloc1, qml.X(alloc1.wires), dealloc1, alloc2, qml.Y(alloc2.wires), dealloc2]
        )

        with pytest.raises(qml.exceptions.AllocationError, match="no wires left to allocate."):
            qml.transforms.resolve_dynamic_wires(tape, zeroed=("a",), allow_resets=False)

    def test_no_resets_min_int(self):
        """Test that if a min_int is specified along with allow_resets=False, then fresh wires keep getting added."""

        alloc1 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ANY, restored=False)
        dealloc1 = qml.allocation.Deallocate(alloc1.wires)
        alloc2 = qml.allocation.Allocate.from_num_wires(1, state=AllocateState.ZERO, restored=False)
        dealloc2 = qml.allocation.Deallocate(alloc2.wires)

        tape = qml.tape.QuantumScript(
            [alloc1, qml.X(alloc1.wires), dealloc1, alloc2, qml.Y(alloc2.wires), dealloc2]
        )

        [new_tape], _ = qml.transforms.resolve_dynamic_wires(tape, allow_resets=False, min_int=2)

        expected = qml.tape.QuantumScript([qml.X(2), qml.Y(3)])
        qml.assert_equal(expected, new_tape)

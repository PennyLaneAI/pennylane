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
"""Unit tests for the :func:`pennylane.apply` utility."""

import pytest

import pennylane as qp
from pennylane.core.apply import apply

test_observables = [
    qp.PauliZ(0) @ qp.PauliZ(1),
    qp.Hamiltonian([0.1, 0.2, 0.3], [qp.PauliZ(0) @ qp.PauliZ(1), qp.PauliY(1), qp.Identity(2)]),
]


class TestApplyOp:
    """Tests for the apply function."""

    def test_public_imports(self):
        """Test that the public and legacy imports expose the standalone utility."""
        assert qp.apply is apply
        assert qp.queuing.apply is apply

    def test_error(self):
        """Test that applying an operation without an active context raises an error."""
        with pytest.raises(RuntimeError, match="No queuing context"):
            apply(qp.PauliZ(0))

    def test_default_queue_operation_inside(self):
        """Test applying an operation instantiated within the queuing
        context to the existing active queue"""
        with qp.queuing.AnnotatedQueue() as q:
            op1 = qp.PauliZ(0)
            op2 = apply(op1)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert tape.operations == [op1, op2]

    def test_default_queue_operation_outside(self):
        """Test applying an operation instantiated outside a queuing context
        to an existing active queue"""
        op = qp.PauliZ(0)

        with qp.queuing.AnnotatedQueue() as q:
            apply(op)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert tape.operations == [op]

    @pytest.mark.parametrize("obs", test_observables)
    def test_default_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated outside a queuing context
        to an existing active queue"""
        op = qp.expval(obs)

        with qp.queuing.AnnotatedQueue() as q:
            apply(op)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert tape.measurements == [op]

    @pytest.mark.parametrize("obs", test_observables)
    def test_default_queue_measurements_inside(self, obs):
        """Test applying a measurement instantiated inside a queuing context
        to an existing active queue"""

        with qp.queuing.AnnotatedQueue() as q:
            op1 = qp.expval(obs)
            op2 = apply(op1)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert tape.measurements == [op1, op2]

    def test_different_queue_operation_inside(self):
        """Test applying an operation instantiated within the queuing
        context to a specified queuing context"""
        with qp.queuing.AnnotatedQueue() as q1:
            with qp.queuing.AnnotatedQueue() as q2:
                op1 = qp.PauliZ(0)
                op2 = apply(op1, q1)

            tape2 = qp.tape.QuantumScript.from_queue(q2)
        tape1 = qp.tape.QuantumScript.from_queue(q1)
        assert tape1.operations == [op2]
        assert tape2.operations == [op1]

    def test_different_queue_operation_outside(self):
        """Test applying an operation instantiated outside a queuing context
        to a specified queuing context"""
        op = qp.PauliZ(0)

        with qp.queuing.AnnotatedQueue() as q1:
            with qp.queuing.AnnotatedQueue() as q2:
                apply(op, q1)

            tape2 = qp.tape.QuantumScript.from_queue(q2)
        tape1 = qp.tape.QuantumScript.from_queue(q1)
        assert tape1.operations == [op]
        assert tape2.operations == []

    @pytest.mark.parametrize("obs", test_observables)
    def test_different_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated outside a queuing context
        to a specified queuing context"""
        op = qp.expval(obs)

        with qp.queuing.AnnotatedQueue() as q1:
            with qp.queuing.AnnotatedQueue() as q2:
                apply(op, q1)

            tape2 = qp.tape.QuantumScript.from_queue(q2)
        tape1 = qp.tape.QuantumScript.from_queue(q1)
        assert tape1.measurements == [op]
        assert tape2.measurements == []

    @pytest.mark.parametrize("obs", test_observables)
    def test_different_queue_measurements_inside(self, obs):
        """Test applying a measurement instantiated inside a queuing context
        to a specified queuing context"""

        with qp.queuing.AnnotatedQueue() as q1:
            with qp.queuing.AnnotatedQueue() as q2:
                op1 = qp.expval(obs)
                op2 = apply(op1, q1)

            tape2 = qp.tape.QuantumScript.from_queue(q2)
        tape1 = qp.tape.QuantumScript.from_queue(q1)
        assert tape1.measurements == [op2]
        assert tape2.measurements == [op1]

    def test_apply_no_queue_method(self):
        """Test that an object with no queue method is still added to the queuing context."""
        with qp.queuing.AnnotatedQueue() as q1:
            with qp.queuing.AnnotatedQueue() as q2:
                op1 = apply(5)
                op2 = apply(6, q1)

        assert q1.queue == [op2]
        assert q2.queue == [op1]

    def test_apply_plus_dequeuing(self):
        """Test that operations queued with apply don't get dequeued by subsequent ops."""

        h = qp.H(0)
        with qp.queuing.AnnotatedQueue() as q1:
            op1 = apply(h)
            op2 = qp.adjoint(h)

        assert q1.queue == [op1, op2]

# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the :mod:`pennylane` :class:`QueuingContext` class.
"""
import contextlib

import pytest
import pennylane as qml
import numpy as np

from pennylane.queuing import AnnotatedQueue, AnnotatedQueue, QueuingContext, QueuingError
from pennylane.tape import OperationRecorder


@pytest.fixture(scope="function")
def mock_queuing_context(monkeypatch):
    """A mock instance of the abstract QueuingContext class."""
    with monkeypatch.context() as m:
        m.setattr(QueuingContext, "__abstractmethods__", frozenset())
        m.setattr(
            QueuingContext,
            "_append",
            lambda self, operator: self.queue.append(operator),
        )
        m.setattr(
            QueuingContext,
            "_remove",
            lambda self, operator: self.queue.remove(operator),
        )
        context = QueuingContext()
        context.queue = []

        yield context


@pytest.fixture(scope="function")
def three_mock_queuing_contexts(monkeypatch):
    """A list of three mock instances of the abstract QueuingContext class."""
    with monkeypatch.context() as m:
        m.setattr(QueuingContext, "__abstractmethods__", frozenset())
        m.setattr(
            QueuingContext,
            "_append",
            lambda self, operator: self.queue.append(operator),
        )
        m.setattr(
            QueuingContext,
            "_remove",
            lambda self, operator: self.queue.remove(operator),
        )

        contexts = [QueuingContext() for _ in range(3)]
        for context in contexts:
            context.queue = []

        yield contexts


class TestQueuingContext:
    """Test the logic associated with the QueuingContext class."""

    def test_context_activation(self, mock_queuing_context):
        """Test that the QueuingContext is properly activated and deactivated."""

        # Assert that the list of active contexts is empty
        assert not QueuingContext._active_contexts

        with mock_queuing_context:
            assert len(QueuingContext._active_contexts) == 1
            assert mock_queuing_context in QueuingContext._active_contexts

        assert not QueuingContext._active_contexts

    def test_multiple_context_activation(self, three_mock_queuing_contexts):
        """Test that multiple QueuingContexts are properly activated and deactivated."""

        # Assert that the list of active contexts is empty
        assert not QueuingContext._active_contexts

        with three_mock_queuing_contexts[0]:
            with three_mock_queuing_contexts[1]:
                with three_mock_queuing_contexts[2]:
                    assert len(QueuingContext._active_contexts) == 3
                    assert three_mock_queuing_contexts[0] in QueuingContext._active_contexts
                    assert three_mock_queuing_contexts[1] in QueuingContext._active_contexts
                    assert three_mock_queuing_contexts[2] in QueuingContext._active_contexts

        assert not QueuingContext._active_contexts

    def test_append_no_context(self):
        """Test that append does not fail when no context is present."""

        QueuingContext.append(qml.PauliZ(0))

    def test_remove_no_context(self):
        """Test that remove does not fail when no context is present."""

        QueuingContext.remove(qml.PauliZ(0))

    def test_no_active_context(self, mock_queuing_context):
        """Test that if there are no active contexts, active_context() returns None"""
        assert mock_queuing_context.active_context() is None


class TestAnnotatedQueue:
    """Tests for the annotated queue class"""

    def test_remove_not_in_queue(self):
        """Test that remove fails when the object to be removed is not in the queue."""

        with AnnotatedQueue() as q1:
            op1 = qml.PauliZ(0)
            op2 = qml.PauliZ(1)
            q1.append(op1)
            q1.append(op2)

        with AnnotatedQueue() as q2:
            q2.append(op1)
            with pytest.raises(KeyError):
                q2.remove(op2)

    def test_append_qubit_gates(self):
        """Test that gates are successfully appended to the queue."""
        with AnnotatedQueue() as q:
            ops = [
                qml.RX(0.5, wires=0),
                qml.RY(-10.1, wires=1),
                qml.CNOT(wires=[0, 1]),
                qml.PhaseShift(-1.1, wires=18),
                qml.T(wires=99),
            ]
        assert q.queue == ops

    def test_append_qubit_observables(self):
        """Test that ops that are also observables are successfully
        appended to the queue."""
        with AnnotatedQueue() as q:
            # wire repetition is deliberate, Queue contains no checks/logic
            # for circuits
            ops = [
                qml.Hadamard(wires=0),
                qml.PauliX(wires=1),
                qml.PauliY(wires=1),
                qml.Hermitian(np.ones([2, 2]), wires=7),
            ]
        assert q.queue == ops

    def test_append_tensor_ops(self):
        """Test that ops which are used as inputs to `Tensor`
        are successfully added to the queue, as well as the `Tensor` object."""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = qml.operation.Tensor(A, B)
        assert q.queue == [A, B, tensor_op]
        assert tensor_op.obs == [A, B]

    def test_append_tensor_ops_overloaded(self):
        """Test that Tensor ops created using `@`
        are successfully added to the queue, as well as the `Tensor` object."""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = A @ B
        assert q.queue == [A, B, tensor_op]
        assert tensor_op.obs == [A, B]

    def test_get_info(self):
        """Test that get_info correctly returns an annotation"""
        A = qml.RZ(0.5, wires=1)

        with AnnotatedQueue() as q:
            q.append(A, inv=True)

        assert q._get_info(A) == {"inv": True}

    def test_get_info_error(self):
        """Test that an exception is raised if get_info is called
        for a non-existent object"""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)

        B = qml.PauliY(1)

        with pytest.raises(QueuingError, match="not in the queue"):
            q._get_info(B)

    def test_get_info_none(self):
        """Test that get_info returns None if there is no active queuing context"""
        A = qml.RZ(0.5, wires=1)

        with AnnotatedQueue() as q:
            q.append(A, inv=True)

        assert QueuingContext.get_info(A) is None

    def test_update_info(self):
        """Test that update_info correctly updates an annotation"""
        A = qml.RZ(0.5, wires=1)

        with AnnotatedQueue() as q:
            q.append(A, inv=True)
            assert QueuingContext.get_info(A) == {"inv": True}

            qml.QueuingContext.update_info(A, key="value1")

        # should pass silently because no longer recording
        qml.QueuingContext.update_info(A, key="value2")

        assert q._get_info(A) == {"inv": True, "key": "value1"}

        q._update_info(A, inv=False, owner=None)
        assert q._get_info(A) == {"inv": False, "owner": None, "key": "value1"}

    def test_update_error(self):
        """Test that an exception is raised if get_info is called
        for a non-existent object"""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)

        B = qml.PauliY(1)

        with pytest.raises(QueuingError, match="not in the queue"):
            q._update_info(B, inv=True)

    def test_safe_update_info_queued(self):
        """Test the `safe_update_info` method if the object is already queued."""
        op = qml.RX(0.5, wires=1)

        with AnnotatedQueue() as q:
            q.append(op, key="value1")
            assert q.get_info(op) == {"key": "value1"}
            qml.QueuingContext.safe_update_info(op, key="value2")

        qml.QueuingContext.safe_update_info(op, key="no changes here")
        assert q.get_info(op) == {"key": "value2"}

        q.safe_update_info(op, key="value3")
        assert q.get_info(op) == {"key": "value3"}

        q._safe_update_info(op, key="value4")
        assert q.get_info(op) == {"key": "value4"}

    def test_safe_update_info_not_queued(self):
        """Tests the safe_update_info method passes silently if the object is
        not already queued."""
        op = qml.RX(0.5, wires=1)

        with AnnotatedQueue() as q:
            qml.QueuingContext.safe_update_info(op, key="value2")
        qml.QueuingContext.safe_update_info(op, key="no changes here")

        assert len(q.queue) == 0

        q.safe_update_info(op, key="value3")
        assert len(q.queue) == 0

        q._safe_update_info(op, key="value4")
        assert len(q.queue) == 0

    def test_append_annotating_object(self):
        """Test appending an object that writes annotations when queuing itself"""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = qml.operation.Tensor(A, B)

        assert q.queue == [A, B, tensor_op]
        assert q._get_info(A) == {"owner": tensor_op}
        assert q._get_info(B) == {"owner": tensor_op}
        assert q._get_info(tensor_op) == {"owns": (A, B)}


class TestOperationRecorder:
    """Test the OperationRecorder class."""

    def test_circuit_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "PauliY(wires=[0])\n"
            + "PauliY(wires=[1])\n"
            + "RZ(0.4, wires=[0])\n"
            + "RZ(0.4, wires=[1])\n"
            + "CNOT(wires=[0, 1])\n"
            + "\n"
            + "Observables\n"
            + "===========\n"
        )

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)

            with qml.tape.OperationRecorder() as recorder:
                ops = [
                    qml.PauliY(0),
                    qml.PauliY(1),
                    qml.RZ(c, wires=0),
                    qml.RZ(c, wires=1),
                    qml.CNOT(wires=[0, 1]),
                ]

            assert str(recorder) == expected_output
            assert recorder.queue == ops

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit(0.1, 0.2, 0.4)

    def test_template_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "RZ(0, wires=[0])\n"
            + "RZ(3, wires=[0])\n"
            + "RZ(6, wires=[0])\n"
            + "RZ(9, wires=[0])\n"
            + "RZ(12, wires=[0])\n"
            + "\n"
            + "Observables\n"
            + "===========\n"
        )

        def template(x):
            for i in range(5):
                qml.RZ(i * x, wires=0)

        with qml.tape.OperationRecorder() as recorder:
            template(3)

        assert str(recorder) == expected_output

    def test_template_with_return_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "RZ(0, wires=[0])\n"
            + "RZ(3, wires=[0])\n"
            + "RZ(6, wires=[0])\n"
            + "RZ(9, wires=[0])\n"
            + "RZ(12, wires=[0])\n"
            + "\n"
            + "Observables\n"
            + "===========\n"
            + "var(PauliZ(wires=[0]))\n"
            + "sample(PauliX(wires=[1]))\n"
        )

        def template(x):
            for i in range(5):
                qml.RZ(i * x, wires=0)

            return qml.var(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        with qml.tape.OperationRecorder() as recorder:
            template(3)

        assert str(recorder) == expected_output


test_observables = [
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.operation.Tensor(qml.PauliZ(0), qml.PauliX(1)),
    qml.operation.Tensor(qml.PauliZ(0), qml.PauliX(1)) @ qml.Hadamard(2),
    qml.Hamiltonian(
        [0.1, 0.2, 0.3], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliY(1), qml.Identity(2)]
    ),
]


class TestApplyOp:
    """Tests for the apply function"""

    def test_error(self):
        """Test that applying an operation without an active
        context raises an error"""
        with pytest.raises(RuntimeError, match="No queuing context"):
            qml.apply(qml.PauliZ(0))

    def test_default_queue_operation_inside(self):
        """Test applying an operation instantiated within the queuing
        context to the existing active queue"""
        with qml.tape.QuantumTape() as tape:
            op1 = qml.PauliZ(0)
            op2 = qml.apply(op1)

        assert tape.operations == [op1, op2]

    def test_default_queue_operation_outside(self):
        """Test applying an operation instantiated outside a queuing context
        to an existing active queue"""
        op = qml.PauliZ(0)

        with qml.tape.QuantumTape() as tape:
            qml.apply(op)

        assert tape.operations == [op]

    @pytest.mark.parametrize("obs", test_observables)
    def test_default_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated outside a queuing context
        to an existing active queue"""
        op = qml.expval(obs)

        with qml.tape.QuantumTape() as tape:
            qml.apply(op)

        assert tape.measurements == [op]

    @pytest.mark.parametrize("obs", test_observables)
    def test_default_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated inside a queuing context
        to an existing active queue"""

        with qml.tape.QuantumTape() as tape:
            op1 = qml.expval(obs)
            op2 = qml.apply(op1)

        assert tape.measurements == [op1, op2]

    def test_different_queue_operation_inside(self):
        """Test applying an operation instantiated within the queuing
        context to a specfied queuing context"""
        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                op1 = qml.PauliZ(0)
                op2 = qml.apply(op1, tape1)

        assert tape1.operations == [tape2, op2]
        assert tape2.operations == [op1]

    def test_different_queue_operation_outside(self):
        """Test applying an operation instantiated outside a queuing context
        to a specfied queuing context"""
        op = qml.PauliZ(0)

        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                qml.apply(op, tape1)

        assert tape1.operations == [tape2, op]
        assert tape2.operations == []

    @pytest.mark.parametrize("obs", test_observables)
    def test_different_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated outside a queuing context
        to a specfied queuing context"""
        op = qml.expval(obs)

        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                qml.apply(op, tape1)

        assert tape1.measurements == [op]
        assert tape2.measurements == []

    @pytest.mark.parametrize("obs", test_observables)
    def test_different_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated inside a queuing context
        to a specfied queuing context"""

        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                op1 = qml.expval(obs)
                op2 = qml.apply(op1, tape1)

        assert tape1.measurements == [op2]
        assert tape2.measurements == [op1]

    def test_apply_no_queue_method(self):
        """Test that an object with no queue method is still
        added to the queuing context"""
        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                op1 = qml.apply(5)
                op2 = qml.apply(6, tape1)

        assert tape1.queue == [tape2, op2]
        assert tape2.queue == [op1]

        # note that tapes don't know how to process integers,
        # so they are not included after queue processing
        assert tape1.operations == [tape2]
        assert tape2.operations == []

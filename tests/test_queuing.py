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

import pytest
import pennylane as qml
import numpy as np
from pennylane import QueuingContext


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
                    assert (
                        three_mock_queuing_contexts[0]
                        in QueuingContext._active_contexts
                    )
                    assert (
                        three_mock_queuing_contexts[1]
                        in QueuingContext._active_contexts
                    )
                    assert (
                        three_mock_queuing_contexts[2]
                        in QueuingContext._active_contexts
                    )

        assert not QueuingContext._active_contexts

    def test_append_no_context(self):
        """Test that append does not fail when no context is present."""

        QueuingContext.append(qml.PauliZ(0))

    def test_remove_operator_no_context(self):
        """Test that remove_operator does not fail when no context is present."""

        QueuingContext.remove(qml.PauliZ(0))

    def test_append(self, mock_queuing_context):
        """Test that append appends the operator to the queue."""

        op = qml.PauliZ(0)
        assert not mock_queuing_context.queue

        with mock_queuing_context:
            QueuingContext.append(op)

        assert len(mock_queuing_context.queue) == 1
        assert op in mock_queuing_context.queue

    def test_remove_operator(self, mock_queuing_context):
        """Test that remove_operator removes the operator from the queue."""

        op = qml.PauliZ(0)
        assert not mock_queuing_context.queue

        with mock_queuing_context:
            QueuingContext.append(op)

            assert len(mock_queuing_context.queue) == 1
            assert op in mock_queuing_context.queue

            QueuingContext.remove(op)

        assert not mock_queuing_context.queue

    def test_remove_operator_not_in_list(self, mock_queuing_context):
        """Test that remove_operator does not fail when the operator to be removed is not in the queue."""

        op1 = qml.PauliZ(0)
        op2 = qml.PauliZ(1)
        assert not mock_queuing_context.queue

        with mock_queuing_context:
            QueuingContext.append(op1)

            assert len(mock_queuing_context.queue) == 1
            assert op1 in mock_queuing_context.queue

            QueuingContext.remove(op2)

        assert len(mock_queuing_context.queue) == 1
        assert op1 in mock_queuing_context.queue

    def test_append_multiple_queues(self, three_mock_queuing_contexts):
        """Test that append appends the operator to multiple queues."""

        op = qml.PauliZ(0)
        assert not three_mock_queuing_contexts[0].queue
        assert not three_mock_queuing_contexts[1].queue
        assert not three_mock_queuing_contexts[2].queue

        with three_mock_queuing_contexts[0]:
            with three_mock_queuing_contexts[1]:
                with three_mock_queuing_contexts[2]:
                    QueuingContext.append(op)

        assert len(three_mock_queuing_contexts[0].queue) == 1
        assert op in three_mock_queuing_contexts[0].queue

        assert len(three_mock_queuing_contexts[1].queue) == 1
        assert op in three_mock_queuing_contexts[1].queue

        assert len(three_mock_queuing_contexts[1].queue) == 1
        assert op in three_mock_queuing_contexts[1].queue

    def test_remove_operator_multiple_queues(self, three_mock_queuing_contexts):
        """Test that remove_operator removes the operator from the queue."""

        op = qml.PauliZ(0)
        assert not three_mock_queuing_contexts[0].queue
        assert not three_mock_queuing_contexts[1].queue
        assert not three_mock_queuing_contexts[2].queue

        with three_mock_queuing_contexts[0]:
            with three_mock_queuing_contexts[1]:
                with three_mock_queuing_contexts[2]:
                    QueuingContext.append(op)

                    assert len(three_mock_queuing_contexts[0].queue) == 1
                    assert op in three_mock_queuing_contexts[0].queue

                    assert len(three_mock_queuing_contexts[1].queue) == 1
                    assert op in three_mock_queuing_contexts[1].queue

                    assert len(three_mock_queuing_contexts[2].queue) == 1
                    assert op in three_mock_queuing_contexts[2].queue

                    QueuingContext.remove(op)

        assert not three_mock_queuing_contexts[0].queue
        assert not three_mock_queuing_contexts[1].queue
        assert not three_mock_queuing_contexts[2].queue


class TestQueue:
    """Test the Queue class."""

    def test_arbitrary_obj(self):
        """Tests that arbitrary objects can be appended to and removed from the queue."""

        objs = [5, "hi", 1.2, np.einsum, lambda x: x + 1]
        with qml._queuing.Queue() as q:
            for obj in objs:
                q.append(obj)
        assert q.queue == objs

        with q:
            for _ in range(len(objs)):
                obj = objs.pop()
                q.remove(obj)
                assert q.queue == objs

    def test_append_qubit_gates(self):
        """Test that gates are successfully appended to the queue."""
        with qml._queuing.Queue() as q:
            ops = [qml.RX(0.5, wires=0),
                   qml.RY(-10.1, wires=1),
                   qml.CNOT(wires=[0,1]),
                   qml.PhaseShift(-1.1, wires=18),
                   qml.T(wires=99)]
        assert q.queue == ops

    def test_append_qubit_observables(self):
        """Test that ops that are also observables are successfully
        appended to the queue."""
        with qml._queuing.Queue() as q:
            # wire repetition is deliberate, Queue contains no checks/logic
            # for circuits
            ops = [qml.Hadamard(wires=0),
                   qml.PauliX(wires=1),
                   qml.PauliY(wires=1),
                   qml.Hermitian(np.ones([2,2]), wires=7)
                   ]
        assert q.queue == ops

    def test_append_tensor_ops(self):
        """Test that ops which are used as inputs to `Tensor`
        are successfully added to the queue, but no `Tensor` object is."""

        with qml._queuing.Queue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = qml.operation.Tensor(A, B)
        assert q.queue == [A, B]
        assert tensor_op.obs == [A, B]
        assert all(not isinstance(op, qml.operation.Tensor) for op in q.queue)

    def test_append_tensor_ops_overloaded(self):
        """Test that Tensor ops created using `@`
        are successfully added to the queue, but no `Tensor` object is."""

        with qml._queuing.Queue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = A @ B
        assert q.queue == [A, B]
        assert tensor_op.obs == [A, B]
        assert all(not isinstance(op, qml.operation.Tensor) for op in q.queue)


class TestOperationRecorder:
    """Test the OperationRecorder class."""

    def test_context_adding(self, monkeypatch):
        """Test that the OperationRecorder is added to the list of contexts."""
        with qml._queuing.OperationRecorder() as recorder:
            assert recorder in qml.QueuingContext._active_contexts

        assert recorder not in qml.QueuingContext._active_contexts

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
            + "==========\n"
        )

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)

            with qml._queuing.OperationRecorder() as recorder:
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
            + "==========\n"
        )

        def template(x):
            for i in range(5):
                qml.RZ(i * x, wires=0)

        with qml._queuing.OperationRecorder() as recorder:
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
            + "==========\n"
            + "var(PauliZ(wires=[0]))\n"
            + "sample(PauliX(wires=[1]))\n"
        )

        def template(x):
            for i in range(5):
                qml.RZ(i * x, wires=0)

            return qml.var(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        with qml._queuing.OperationRecorder() as recorder:
            template(3)

        assert str(recorder) == expected_output

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
            QueuingContext, "_append", lambda self, operator: self.queue.append(operator),
        )
        m.setattr(
            QueuingContext, "_remove", lambda self, operator: self.queue.remove(operator),
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
            QueuingContext, "_append", lambda self, operator: self.queue.append(operator),
        )
        m.setattr(
            QueuingContext, "_remove", lambda self, operator: self.queue.remove(operator),
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

    def test_remove_not_in_queue(self):
        """Test that remove does not fail when the object to be removed is not in the queue."""

        with qml._queuing.Queue() as q1:
            op1 = qml.PauliZ(0)
            op2 = qml.PauliZ(1)
            q1.append(op1)
            q1.append(op2)

        with qml._queuing.Queue() as q2:
            q2.append(op1)
            q2.remove(op2)

    def test_append_qubit_gates(self):
        """Test that gates are successfully appended to the queue."""
        with qml._queuing.Queue() as q:
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
        with qml._queuing.Queue() as q:
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


class AnnotatingTensor(qml.operation.Tensor):
    """Dummy tensor class that queues itself on initialization
    to an annotating queue."""

    def __init__(self, *args):
        super().__init__(*args)
        self.queue()

    def queue(self):
        qml.QueuingContext.append(self)

        for o in self.obs:
            try:
                qml.QueuingContext.active_context().update_info(o, owner=self)
            except AttributeError:
                pass

        return self


class TestAnnotatedQueue:
    """Tests for the annotated queue class"""

    def test_append_operation(self):
        """Test appending arbitrary operations to the queue"""

        with qml._queuing.AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = AnnotatingTensor(A, B)

        assert q.objects() == [A, B, tensor_op]
        assert q.get_info(A) == {"owner": tensor_op}
        assert q.get_info(B) == {"owner": tensor_op}

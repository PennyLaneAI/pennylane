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
from pennylane import QueuingContext


@pytest.fixture(scope="function")
def mock_queuing_context(monkeypatch):
    """A mock instance of the abstract QueuingContext class."""
    with monkeypatch.context() as m:
        m.setattr(QueuingContext, "__abstractmethods__", frozenset())
        m.setattr(
            QueuingContext, "_append_operator", lambda self, operator: self.queue.append(operator)
        )
        m.setattr(
            QueuingContext, "_remove_operator", lambda self, operator: self.queue.remove(operator)
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
            QueuingContext, "_append_operator", lambda self, operator: self.queue.append(operator)
        )
        m.setattr(
            QueuingContext, "_remove_operator", lambda self, operator: self.queue.remove(operator)
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

    def test_append_operator_no_context(self):
        """Test that append_operator does not fail when no context is present."""

        QueuingContext.append_operator(qml.PauliZ(0))

    def test_remove_operator_no_context(self):
        """Test that remove_operator does not fail when no context is present."""

        QueuingContext.remove_operator(qml.PauliZ(0))

    def test_append_operator(self, mock_queuing_context):
        """Test that append_operator appends the operator to the queue."""

        op = qml.PauliZ(0)
        assert not mock_queuing_context.queue

        with mock_queuing_context:
            QueuingContext.append_operator(op)

        assert len(mock_queuing_context.queue) == 1
        assert op in mock_queuing_context.queue

    def test_remove_operator(self, mock_queuing_context):
        """Test that remove_operator removes the operator from the queue."""

        op = qml.PauliZ(0)
        assert not mock_queuing_context.queue

        with mock_queuing_context:
            QueuingContext.append_operator(op)

            assert len(mock_queuing_context.queue) == 1
            assert op in mock_queuing_context.queue

            QueuingContext.remove_operator(op)

        assert not mock_queuing_context.queue

    def test_remove_operator_not_in_list(self, mock_queuing_context):
        """Test that remove_operator does not fail when the operator to be removed is not in the queue."""

        op1 = qml.PauliZ(0)
        op2 = qml.PauliZ(1)
        assert not mock_queuing_context.queue

        with mock_queuing_context:
            QueuingContext.append_operator(op1)

            assert len(mock_queuing_context.queue) == 1
            assert op1 in mock_queuing_context.queue

            QueuingContext.remove_operator(op2)

        assert len(mock_queuing_context.queue) == 1
        assert op1 in mock_queuing_context.queue

    def test_append_operator_multiple_queues(self, three_mock_queuing_contexts):
        """Test that append_operator appends the operator to multiple queues."""

        op = qml.PauliZ(0)
        assert not three_mock_queuing_contexts[0].queue
        assert not three_mock_queuing_contexts[1].queue
        assert not three_mock_queuing_contexts[2].queue

        with three_mock_queuing_contexts[0]:
            with three_mock_queuing_contexts[1]:
                with three_mock_queuing_contexts[2]:
                    QueuingContext.append_operator(op)

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
                    QueuingContext.append_operator(op)

                    assert len(three_mock_queuing_contexts[0].queue) == 1
                    assert op in three_mock_queuing_contexts[0].queue

                    assert len(three_mock_queuing_contexts[1].queue) == 1
                    assert op in three_mock_queuing_contexts[1].queue

                    assert len(three_mock_queuing_contexts[2].queue) == 1
                    assert op in three_mock_queuing_contexts[2].queue

                    QueuingContext.remove_operator(op)

        assert not three_mock_queuing_contexts[0].queue
        assert not three_mock_queuing_contexts[1].queue
        assert not three_mock_queuing_contexts[2].queue

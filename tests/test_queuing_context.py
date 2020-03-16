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
from pennylane import QueuingContext


@pytest.fixture(scope="function")
def mock_queuing_context(monkeypatch):
    """A mock instance of the abstract Device class with non-empty operations"""
    with monkeypatch.context() as m:
        m.setattr(QueuingContext, '__abstractmethods__', frozenset())
        m.setattr(QueuingContext, '_append_operator', lambda self, operator: self.queue.append(operator))
        m.setattr(QueuingContext, '_remove_operator', lambda self, operator: self.queue.remove(operator))
        context = QueuingContext()
        context.queue = []

        yield context


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

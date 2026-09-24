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

"""Tests for :func:`pennylane.capture.tracing_device`."""

import contextvars

import pytest

import pennylane as qp
from pennylane.capture import get_tracing_device, tracing_device


@pytest.fixture(name="dev")
def dev_fixture():
    """A device."""
    return qp.device("null.qubit", wires=1)


class TestTracingDevice:
    """Tracing a device."""

    def test_there_is_no_device_outside_a_trace(self):
        """Nothing is being traced by default."""
        assert get_tracing_device() is None

    def test_the_published_device_is_the_one_read_back(self, dev):
        """The published device is the one being traced."""
        with tracing_device(dev) as published:
            assert get_tracing_device() is dev
            assert published is dev

    def test_the_device_is_unpublished_on_the_way_out(self, dev):
        """The device is unpublished on the way out."""
        with tracing_device(dev):
            pass
        assert get_tracing_device() is None

    def test_an_exception_still_unpublishes(self, dev):
        """Runtime errors can be caught"""
        with pytest.raises(RuntimeError, match="tracing failed"):
            with tracing_device(dev):
                raise RuntimeError("tracing failed")

        assert get_tracing_device() is None

    def test_a_nested_trace_shadows_and_restores(self, dev):
        """An inner trace wins while it is being traced."""
        inner = qp.device("null.qubit", wires=2)

        with tracing_device(dev):
            with tracing_device(inner):
                assert get_tracing_device() is inner
            assert get_tracing_device() is dev

    def test_none_can_be_published(self, dev):
        """Publishing ``None`` says there is no device."""
        with tracing_device(dev):
            with tracing_device(None):
                assert get_tracing_device() is None
            assert get_tracing_device() is dev

    def test_the_device_does_not_leak_between_contexts(self, dev):
        """The device is local to the context."""
        seen = []

        def publish_and_read():
            with tracing_device(dev):
                seen.append(get_tracing_device())

        contextvars.copy_context().run(publish_and_read)

        assert seen == [dev]
        assert get_tracing_device() is None

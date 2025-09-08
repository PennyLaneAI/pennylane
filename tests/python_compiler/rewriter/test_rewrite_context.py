# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the RewriteContext class and associated utilities."""

from uuid import UUID

import pytest

import pennylane as qml
from pennylane.compiler.python_compiler.rewriter import AbstractWire, RewriteContext, WireQubitMap

pytestmark = pytest.mark.external


class TestAbstractWire:
    """Unit tests for the AbstractWire class."""

    def test_init(self):
        """Test that AbstractWires are initialized correctly."""
        wire = AbstractWire()
        assert isinstance(wire.id, UUID)

    def test_equality(self):
        """Test that comparing two AbstractWires works correctly."""
        wire1 = AbstractWire()
        wire2 = AbstractWire()

        assert wire1 != wire2

    def test_hash(self):
        """Test that the hash of different AbstractWires is unique."""
        wire1 = AbstractWire()
        wire2 = AbstractWire()

        assert hash(wire1) != hash(wire2)


class TestWireQubitMap:
    """Unit tests for the WireQubitMap class."""


class TestRewriteContext:
    """Unit tests for the RewriteContext class."""

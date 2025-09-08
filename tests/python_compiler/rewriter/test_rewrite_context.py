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

    def test_init(self):
        """Test that the WireQubitMap's initialization is correct."""

    def test_contains(self):
        """Test that the __contains__ dunder method works correctly."""

    def test_len(self):
        """Test that the __len__ dunder method works correctly."""

    def test_len_invalid_length(self):
        """Test that __len__ raises an error if the lengths of the dictionaries do not match."""

    def test_getitem(self):
        """Test that the __getitem__ dunder method works correctly."""

    def test_getitem_invalid_ssa_value(self):
        """Test that an error is raised if the input to __getitem__ is an SSAValue
        but its type is not QubitType."""

    def test_getitem_invalid_wire(self):
        """Test that an error is raised if WireQubitMap.wires is defined but the static wire
        label is not in the wires tuple."""

    def test_setitem(self):
        """Test that the __setitem__ dunder method works correctly."""

    def test_setitem_invalid_ssa_value(self):
        """Test that an error is raised if the input to __setitem__ is an SSAValue
        but its type is not QubitType."""

    def test_setitem_ssa_key_with_invalid_val(self):
        """Test that an error is raised by __setitem__ if the key is a valid SSAValue but
        the value is not a valid wire label."""

    def test_setitem_wire_key_invalid_val(self):
        """Test that an error is raised by __setitem__ if the key is a valid wire label
        but the value is not a qubit SSAValue."""

    def test_setitem_wire_key_not_in_wires(self):
        """Test that an error is raised by __setitem__ if the key is a wire label that is
        not in the wires tuple."""

    def test_setitem_invalid_key_type(self):
        """Test that an error is raised by __setitem__ if the key is not a qubit or a wire label."""

    def test_pop(self):
        """Test that the pop method works correctly."""

    def test_pop_invalid_ssa_value(self):
        """Test that an error is raised by pop if the key is not a valid SSAValue."""

    def test_update_qubit(self):
        """Test that the update_qubit method works correctly."""

    def test_update_qubit_invalid(self):
        """Test that an error is raised by update_qubit if the old qubit is not in the map."""


class TestRewriteContext:
    """Unit tests for the RewriteContext class."""

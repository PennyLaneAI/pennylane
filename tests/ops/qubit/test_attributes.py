# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the available qubit state preparation operations.
"""
import pytest
import pennylane as qml

from pennylane.ops.qubit.attributes import Attribute

# Dummy attribute
new_attribute = Attribute(["PauliX", "PauliY", "PauliZ", "Hadamard", "RZ"])


class TestAttribute:
    """Test addition and inclusion of operations and subclasses in attributes."""

    def test_invalid_input(self):
        """Test that anything that is not a string or Operation throws an error."""
        # Test something that is not an object
        with pytest.raises(TypeError, match="can be checked for attribute inclusion"):
            assert 3 not in new_attribute

        # Test a dummy object that is not an Operation.
        with pytest.raises(TypeError, match="can be checked for attribute inclusion"):
            assert object() not in new_attribute

    def test_string_inclusion(self):
        """Test that we can check inclusion using strings."""
        assert "PauliX" in new_attribute
        assert "RX" not in new_attribute

    def test_operation_class_inclusion(self):
        """Test that we can check inclusion using Operations."""
        assert qml.PauliZ(0) in new_attribute
        assert qml.RX(0.5, wires=0) not in new_attribute

    def test_operation_subclass_inclusion(self):
        """Test that we can check inclusion using subclasses of Operations, whether
        or not anything has been instantiated."""
        assert qml.RZ in new_attribute
        assert qml.RX not in new_attribute

    def test_invalid_addition(self):
        """Test that an error is raised if we try to add something
        other than an Operation or a string."""
        with pytest.raises(TypeError, match="can be added to an attribute"):
            new_attribute.add(0)

        assert len(new_attribute) == 5

        with pytest.raises(TypeError, match="can be added to an attribute"):
            new_attribute.add(object())

        assert len(new_attribute) == 5

    def test_inclusion_after_addition(self):
        """Test that we can add operators to the set in multiple ways."""
        new_attribute.add("RX")
        new_attribute.add(qml.PhaseShift(0.5, wires=0))
        new_attribute.add(qml.RY)

        assert "RX" in new_attribute
        assert "PhaseShift" in new_attribute
        assert "RY" in new_attribute
        assert len(new_attribute) == 8

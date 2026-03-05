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
r"""Tests for the base classes used when tracking qubits for resource estimation."""

import pytest

import pennylane.estimator as qre
import pennylane.labs.estimator_beta as qre_exp

from pennylane.queuing import AnnotatedQueue
from pennylane.allocation import AllocateState
from pennylane.labs.estimator_beta.wires_manager import Allocate, Deallocate, MarkClean


class TestAllocate:
    """Test the methods and attributes of the Allocate class"""

    def test_init(self):
        """Test that the Allocate class is instantiated as expected when there is no active recording."""

        assert Allocate(4).num_wires == 4

    def test_repr(self):
        """Test that correct representation is returned for Allocate class"""

        wires = 3
        exp_string = f"Allocate({wires})"
        assert repr(Allocate(wires)) == exp_string

    def test_init_recording(self):
        """Test that the Allocate class is instantiated as expected when there is active recording."""
        with AnnotatedQueue() as q:
            ops = [Allocate(2), Allocate(4)]
        assert q.queue == ops

    def test_equal(self):
        """Test that the equal function works as expected."""
        alloc_1 = Allocate(num_wires=5)
        alloc_2 = Allocate(num_wires=5)

        assert alloc_1 == alloc_2

        alloc_3 = Allocate(num_wires=5)
        alloc_4 = Allocate(num_wires=10)

        assert alloc_3 != alloc_4

        alloc_5 = Allocate(num_wires=5)
        not_an_alloc = Deallocate(num_wires=5)
        assert alloc_5 != not_an_alloc

    def test_mul(self):
        """Test that the multiplication works with Allocate"""
        original_wires = Allocate(5)
        new_wires = original_wires * 3
        assert new_wires.num_wires == 15

    def test_type_error(self):
        """Test that an error is raised when wrong type is provided for multiplication"""
        with pytest.raises(NotImplementedError):
            _ = Allocate(5) * 4.2


class TestDeallocate:
    """Test the methods and attributes of the Deallocate class"""

    def test_init(self):
        """Test that the Deallocate class is instantiated as expected when there is no recording."""

        for i in range(3):
            assert Deallocate(i).num_wires == i

    def test_repr(self):
        """Test that correct representation is returned for Deallocate class"""

        wires = 3
        exp_string = f"Deallocate({wires})"
        assert repr(Deallocate(wires)) == exp_string

    def test_init_recording(self):
        """Test that the Deallocate class is instantiated as expected when there is active recording."""
        with AnnotatedQueue() as q:
            ops = [Deallocate(2), Deallocate(4), Deallocate(8)]

        assert q.queue == ops

    def test_mul(self):
        """Test that the multiplication works with Deallocate"""
        original_wires = Deallocate(5)
        new_wires = original_wires * 3
        assert new_wires.num_wires == 15

    def test_type_error(self):
        """Test that an error is raised when wrong type is provided for multiplication"""
        with pytest.raises(NotImplementedError):
            _ = Deallocate(5) * 4.2

    def test_equal(self):
        """Test that the equal function works as expected."""
        alloc_1 = Deallocate(num_wires=5)
        alloc_2 = Deallocate(num_wires=5)

        assert alloc_1 == alloc_2

        alloc_3 = Deallocate(num_wires=5)
        alloc_4 = Deallocate(num_wires=10)

        assert alloc_3 != alloc_4

        alloc_5 = Deallocate(num_wires=5)
        not_an_alloc = Allocate(num_wires=5)
        assert alloc_5 != not_an_alloc


class TestMarkQubits:
    """Test the methods and attributes of the MarkQubits and MarkClean class"""
    
    def test_init(self):
        pass

    def test_queue(self):
        pass

    def test_equal(self):
        pass

    def test_MarkClean_repr(self):
        pass

# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY_STATE KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains tests for classes needed to track auxilliary wires.
"""
import copy

import pytest

from pennylane.estimator.wires_manager import Allocate, Deallocate, WireResourceManager
from pennylane.queuing import AnnotatedQueue


# pylint: disable= no-self-use
class TestWireResourceManager:
    """Test the methods and attributes of the WireResourceManager class"""

    wire_manager_quantities = (
        WireResourceManager(zeroed=2),
        WireResourceManager(zeroed=4, any_state=2, algo_wires=20),
        WireResourceManager(zeroed=2, any_state=2, algo_wires=10, tight_budget=True),
    )

    wire_manager_parameters = (
        (2, 0, 0, False),
        (4, 2, 20, False),
        (2, 2, 10, True),
    )

    @pytest.mark.parametrize(
        "wire_manager, attribute_tup", zip(wire_manager_quantities, wire_manager_parameters)
    )
    def test_init(self, wire_manager, attribute_tup):
        """Test that the WireResourceManager class is instantiated as expected."""
        zeroed, any_state, logic_wires, tight_budget = attribute_tup

        assert wire_manager.zeroed == zeroed
        assert wire_manager.any_state == any_state
        assert wire_manager.algo_wires == logic_wires
        assert wire_manager.tight_budget == tight_budget

    @pytest.mark.parametrize(
        "wire_manager, attribute_tup", zip(wire_manager_quantities, wire_manager_parameters)
    )
    def test_equality(self, wire_manager, attribute_tup):
        """Test that the equality methods behaves as expected"""

        zeroed, any_state, algo_wires, tight_budget = attribute_tup

        wire_manager2 = WireResourceManager(
            zeroed=zeroed,
            any_state=any_state,
            algo_wires=algo_wires,
            tight_budget=tight_budget,
        )
        assert wire_manager == wire_manager2

    extra_wires = (0, 2, 4)

    wire_manager_parameters_algo = (
        (2, 0, 0, False),
        (4, 2, 2, False),
        (2, 2, 4, True),
    )

    @pytest.mark.parametrize(
        "wire_manager, attribute_tup, algo_q",
        zip(copy.deepcopy(wire_manager_quantities), wire_manager_parameters_algo, extra_wires),
    )
    def test_repr(self, wire_manager, attribute_tup, algo_q):
        """Test that the WireResourceManager representation is correct."""

        zeroed, any_state, _, tight_budget = attribute_tup

        expected_string = (
            f"WireResourceManager(zeroed={zeroed}, any_state={any_state}, algo_wires={algo_q}, "
            f"tight_budget={tight_budget})"
        )

        wire_manager.algo_wires = algo_q
        assert repr(wire_manager) == expected_string

    @pytest.mark.parametrize(
        "wire_manager, attribute_tup, algo_q",
        zip(copy.deepcopy(wire_manager_quantities), wire_manager_parameters_algo, extra_wires),
    )
    def test_str(self, wire_manager, attribute_tup, algo_q):
        """Test that the WireResourceManager string is correct."""

        zeroed, any_state, _, tight_budget = attribute_tup

        expected_string = (
            f"WireResourceManager(zeroed wires={zeroed}, any_state wires={any_state}, "
            f"algorithmic wires={algo_q}, tight budget={tight_budget})"
        )
        wire_manager.algo_wires = algo_q
        assert str(wire_manager) == expected_string

    @pytest.mark.parametrize(
        "wire_manager, algo_q",
        zip(copy.deepcopy(wire_manager_quantities), extra_wires),
    )
    def test_setting_algo_wires(self, wire_manager, algo_q):
        """Test that the logic wires are set correctly."""

        wire_manager.algo_wires = algo_q
        assert wire_manager.algo_wires == algo_q

    @pytest.mark.parametrize(
        "wire_manager, attribute_tup, algo_q",
        zip(copy.deepcopy(wire_manager_quantities), wire_manager_parameters_algo, extra_wires),
    )
    def test_total_wires(self, wire_manager, attribute_tup, algo_q):
        """Test that the total wires returned are correct."""

        wire_manager.algo_wires = algo_q
        total_wires = attribute_tup[0] + attribute_tup[1] + algo_q
        assert wire_manager.total_wires == total_wires

    @pytest.mark.parametrize(("wires", "zeroed", "any_state"), [(2, 2, 4), (4, 0, 6), (6, 0, 8)])
    def test_grab_zeroed(self, wires, zeroed, any_state):
        """Test that the zeroed wires are grabbed and converted to any_state wires."""

        wire_manager = WireResourceManager(zeroed=4, any_state=2, tight_budget=False)
        wire_manager.grab_zeroed(wires)
        assert wire_manager.zeroed == zeroed
        assert wire_manager.any_state == any_state

    def test_error_grab_zeroed(self):
        """Test that an error is raised when the number of zeroed wires required is greater
        than the available wires."""

        wire_manager = WireResourceManager(zeroed=4, any_state=2, tight_budget=True)
        with pytest.raises(ValueError, match="Grabbing more wires than available zeroed wires."):
            wire_manager.grab_zeroed(6)

    def test_free_wires(self):
        """Test that the any_state wires are freed properly."""

        wire_manager = WireResourceManager(zeroed=4, any_state=2)
        wire_manager.free_wires(2)
        assert wire_manager.zeroed == 6
        assert wire_manager.any_state == 0

    def test_error_free_wires(self):
        """Test that an error is raised when the number of wires being freed is greater
        than the available any_state wires."""

        wire_manager = WireResourceManager(zeroed=4, any_state=2, tight_budget=True)
        with pytest.raises(ValueError, match="Freeing more wires than available any_state wires."):
            wire_manager.free_wires(6)


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

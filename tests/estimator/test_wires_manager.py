# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This module contains tests for classes needed to track auxilliary wires.
"""
import copy

import pytest

from pennylane.estimator import Allocate, Deallocate
from pennylane.estimator.wires_manager import WireResourceManager
from pennylane.queuing import AnnotatedQueue


# pylint: disable= no-self-use
class TestWireResourceManager:
    """Test the methods and attributes of the WireResourceManager class"""

    qm_quantities = (
        WireResourceManager(clean=2),
        WireResourceManager(clean=4, dirty=2, algo=20),
        WireResourceManager(clean=2, dirty=2, algo=10, tight_budget=True),
    )

    qm_parameters = (
        (2, 0, 0, False),
        (4, 2, 20, False),
        (2, 2, 10, True),
    )

    @pytest.mark.parametrize("qm, attribute_tup", zip(qm_quantities, qm_parameters))
    def test_init(self, qm, attribute_tup):
        """Test that the WireResourceManager class is instantiated as expected."""
        clean_wires, dirty_wires, logic_wires, tight_budget = attribute_tup

        assert qm.clean_wires == clean_wires
        assert qm.dirty_wires == dirty_wires
        assert qm.algo_wires == logic_wires
        assert qm.tight_budget == tight_budget

    @pytest.mark.parametrize("qm, attribute_tup", zip(qm_quantities, qm_parameters))
    def test_equality(self, qm, attribute_tup):
        """Test that the equality methods behaves as expected"""

        clean_wires, dirty_wires, algo_wires, tight_budget = attribute_tup

        qm2 = WireResourceManager(
            clean=clean_wires,
            dirty=dirty_wires,
            algo=algo_wires,
            tight_budget=tight_budget,
        )
        assert qm == qm2

    extra_wires = (0, 2, 4)

    qm_parameters_algo = (
        (2, 0, 0, False),
        (4, 2, 2, False),
        (2, 2, 4, True),
    )

    @pytest.mark.parametrize(
        "qm, attribute_tup, algo_q",
        zip(copy.deepcopy(qm_quantities), qm_parameters_algo, extra_wires),
    )
    def test_repr(self, qm, attribute_tup, algo_q):
        """Test that the WireResourceManager representation is correct."""

        clean_wires, dirty_wires, logic_wires, tight_budget = attribute_tup

        expected_string = (
            f"WireResourceManager(clean={clean_wires}, dirty={dirty_wires}, algo={logic_wires}, "
            f"tight_budget={tight_budget})"
        )

        qm.algo_wires = algo_q
        assert repr(qm) == expected_string

    @pytest.mark.parametrize(
        "qm, attribute_tup, algo_q",
        zip(copy.deepcopy(qm_quantities), qm_parameters_algo, extra_wires),
    )
    def test_str(self, qm, attribute_tup, algo_q):
        """Test that the WireResourceManager string is correct."""

        clean_wires, dirty_wires, logic_wires, tight_budget = attribute_tup

        expected_string = (
            f"WireResourceManager(clean wires={clean_wires}, dirty wires={dirty_wires}, "
            f"algorithmic wires={logic_wires}, tight budget={tight_budget})"
        )
        qm.algo_wires = algo_q
        assert str(qm) == expected_string

    @pytest.mark.parametrize(
        "qm, algo_q",
        zip(copy.deepcopy(qm_quantities), extra_wires),
    )
    def test_algo_wires(self, qm, algo_q):
        """Test that the logic wires are set correctly."""

        qm.algo_wires = algo_q
        assert qm.algo_wires == algo_q

    @pytest.mark.parametrize(
        "qm, attribute_tup, algo_q",
        zip(copy.deepcopy(qm_quantities), qm_parameters_algo, extra_wires),
    )
    def test_total_wires(self, qm, attribute_tup, algo_q):
        """Test that the total wires returned are correct."""

        qm.algo_wires = algo_q
        total_wires = attribute_tup[0] + attribute_tup[1] + algo_q
        assert qm.total_wires == total_wires

    @pytest.mark.parametrize(
        ("wires", "clean_wires", "dirty_wires"), [(2, 2, 4), (4, 0, 6), (6, 0, 8)]
    )
    def test_grab_clean_wires(self, wires, clean_wires, dirty_wires):
        """Test that the clean wires are grabbed and converted to dirty wires."""

        qm = WireResourceManager(clean=4, dirty=2, tight_budget=False)
        qm.grab_clean_wires(wires)
        assert qm.clean_wires == clean_wires
        assert qm.dirty_wires == dirty_wires

    def test_error_grab_clean_wires(self):
        """Test that an error is raised when the number of clean wires required is greater
        than the available wires."""

        qm = WireResourceManager(clean=4, dirty=2, tight_budget=True)
        with pytest.raises(ValueError, match="Grabbing more wires than available clean wires."):
            qm.grab_clean_wires(6)

    def test_free_wires(self):
        """Test that the dirty wires are freed properly."""

        qm = WireResourceManager(clean=4, dirty=2)
        qm.free_wires(2)
        assert qm.clean_wires == 6
        assert qm.dirty_wires == 0

    def test_error_free_wires(self):
        """Test that an error is raised when the number of wires being freed is greater
        than the available dirty wires."""

        qm = WireResourceManager(clean=4, dirty=2, tight_budget=True)
        with pytest.raises(ValueError, match="Freeing more wires than available dirty wires."):
            qm.free_wires(6)


class TestAllocate:
    """Test the methods and attributes of the Allocate class"""

    def test_init(self):
        """Test that the Allocate class is instantiated as expected when there is no active recording."""

        for i in range(3):
            assert Allocate(i).num_wires == i

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

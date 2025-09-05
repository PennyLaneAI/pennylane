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
This module contains tests for classes needed to track auxilliary qubits.
"""
import copy

import pytest

import pennylane as qml
from pennylane.estimator import AllocWires, FreeWires
from pennylane.estimator.wires_manager import WireResourceManager


# pylint: disable= no-self-use
class TestWireResourceManager:
    """Test the methods and attributes of the WireResourceManager class"""

    qm_quantities = (
        WireResourceManager(work_wires=2),
        WireResourceManager(work_wires={"clean": 4, "dirty": 2}, algo_wires=20),
        WireResourceManager({"clean": 2, "dirty": 2}, algo_wires=10, tight_budget=True),
    )

    qm_parameters = (
        (2, 0, 0, False),
        (4, 2, 20, False),
        (2, 2, 10, True),
    )

    @pytest.mark.parametrize("qm, attribute_tup", zip(qm_quantities, qm_parameters))
    def test_init(self, qm, attribute_tup):
        """Test that the WireResourceManager class is instantiated as expected."""
        clean_qubits, dirty_qubits, logic_qubits, tight_budget = attribute_tup

        assert qm.clean_wires == clean_qubits
        assert qm.dirty_wires == dirty_qubits
        assert qm.algo_wires == logic_qubits
        assert qm.tight_budget == tight_budget

    @pytest.mark.parametrize("qm, attribute_tup", zip(qm_quantities, qm_parameters))
    def test_equality(self, qm, attribute_tup):
        """Test that the equality methods behaves as expected"""

        clean_qubits, dirty_qubits, algo_qubits, tight_budget = attribute_tup

        qm2 = WireResourceManager(
            work_wires={"clean": clean_qubits, "dirty": dirty_qubits},
            algo_wires=algo_qubits,
            tight_budget=tight_budget,
        )
        assert qm == qm2

    extra_qubits = (0, 2, 4)

    qm_parameters_algo = (
        (2, 0, 0, False),
        (4, 2, 2, False),
        (2, 2, 4, True),
    )

    @pytest.mark.parametrize(
        "qm, attribute_tup, algo_q",
        zip(copy.deepcopy(qm_quantities), qm_parameters_algo, extra_qubits),
    )
    def test_repr(self, qm, attribute_tup, algo_q):
        """Test that the WireResourceManager representation is correct."""

        clean_qubits, dirty_qubits, logic_qubits, tight_budget = attribute_tup

        work_wires_str = repr({"clean": clean_qubits, "dirty": dirty_qubits})
        expected_string = (
            f"WireResourceManager(work_wires={work_wires_str}, algo_wires={logic_qubits}, "
            f"tight_budget={tight_budget})"
        )

        qm.algo_qubits = algo_q
        assert repr(qm) == expected_string

    @pytest.mark.parametrize(
        "qm, attribute_tup, algo_q",
        zip(copy.deepcopy(qm_quantities), qm_parameters_algo, extra_qubits),
    )
    def test_str(self, qm, attribute_tup, algo_q):
        """Test that the WireResourceManager string is correct."""

        clean_qubits, dirty_qubits, logic_qubits, tight_budget = attribute_tup

        expected_string = (
            f"WireResourceManager(clean wires={clean_qubits}, dirty wires={dirty_qubits}, "
            f"algorithmic wires={logic_qubits}, tight budget={tight_budget})"
        )
        qm.algo_qubits = algo_q
        assert str(qm) == expected_string

    @pytest.mark.parametrize(
        "qm, attribute_tup, algo_q",
        zip(copy.deepcopy(qm_quantities), qm_parameters_algo, extra_qubits),
    )
    def test_algo_qubits(self, qm, attribute_tup, algo_q):
        """Test that the logic qubits are set correctly."""

        logic_qubits = attribute_tup[2]

        qm.algo_qubits = algo_q
        assert qm.algo_qubits == logic_qubits

    @pytest.mark.parametrize(
        "qm, attribute_tup, algo_q",
        zip(copy.deepcopy(qm_quantities), qm_parameters_algo, extra_qubits),
    )
    def test_total_qubits(self, qm, attribute_tup, algo_q):
        """Test that the total qubits returned are correct."""

        qm.algo_qubits = algo_q
        total_qubits = attribute_tup[0] + attribute_tup[1] + attribute_tup[2]
        assert qm.total_qubits == total_qubits

    @pytest.mark.parametrize(
        ("wires", "clean_wires", "dirty_wires"), [(2, 2, 4), (4, 0, 6), (6, 0, 8)]
    )
    def test_grab_clean_qubits(self, wires, clean_wires, dirty_wires):
        """Test that the clean qubits are grabbed properly."""

        qm = WireResourceManager(work_wires={"clean": 4, "dirty": 2}, tight_budget=False)
        qm.grab_clean_qubits(wires)
        assert qm.clean_wires == clean_wires
        assert qm.dirty_wires == dirty_wires

    def test_error_grab_clean_qubits(self):
        """Test that an error is raised when the number of clean qubits required is greater
        than the available qubits."""

        qm = WireResourceManager(work_wires={"clean": 4, "dirty": 2}, tight_budget=True)
        with pytest.raises(ValueError, match="Grabbing more qubits than available clean qubits."):
            qm.grab_clean_qubits(6)

    def test_free_qubits(self):
        """Test that the dirty qubits are freed properly."""

        qm = WireResourceManager(work_wires={"clean": 4, "dirty": 2})
        qm.free_qubits(2)
        assert qm.clean_wires == 6
        assert qm.dirty_wires == 0

    def test_error_free_qubits(self):
        """Test that an error is raised when the number of qubits being freed is greater
        than the available dirty qubits."""

        qm = WireResourceManager(work_wires={"clean": 4, "dirty": 2}, tight_budget=True)
        with pytest.raises(ValueError, match="Freeing more qubits than available dirty qubits."):
            qm.free_qubits(6)


class TestAllocWires:
    """Test the methods and attributes of the AllocWires class"""

    def test_init(self):
        """Test that the AllocWires class is instantiated as expected when there is no active recording."""

        for i in range(3):
            assert AllocWires(i).num_wires == i

    def test_repr(self):
        """Test that correct representation is returned for AllocWires class"""

        wires = 3
        exp_string = f"AllocWires({wires})"
        assert repr(AllocWires(wires)) == exp_string

    def test_init_recording(self):
        """Test that the AllocWires class is instantiated as expected when there is active recording."""
        with qml.queuing.AnnotatedQueue() as q:
            ops = [AllocWires(2), AllocWires(4)]
        assert q.queue == ops

    def test_equal(self):
        """Test that the equal function works as expected."""
        alloc_1 = AllocWires(num_wires=5)
        alloc_2 = AllocWires(num_wires=5)

        assert alloc_1 == alloc_2

        alloc_3 = AllocWires(num_wires=5)
        alloc_4 = AllocWires(num_wires=10)

        assert alloc_3 != alloc_4

        alloc_5 = AllocWires(num_wires=5)
        not_an_alloc = FreeWires(num_wires=5)
        assert alloc_5 != not_an_alloc

    def test_mul(self):
        """Test that the multiplication works with AllocWires"""
        original_wires = AllocWires(5)
        new_wires = original_wires * 3
        assert new_wires.num_wires == 15

    def test_type_error(self):
        """Test that an error is raised when wrong type is provided for multiplication"""
        with pytest.raises(NotImplementedError):
            _ = AllocWires(5) * 4.2


class TestFreeWires:
    """Test the methods and attributes of the FreeWires class"""

    def test_init(self):
        """Test that the FreeWires class is instantiated as expected when there is no recording."""

        for i in range(3):
            assert FreeWires(i).num_wires == i

    def test_repr(self):
        """Test that correct representation is returned for FreeWires class"""

        wires = 3
        exp_string = f"FreeWires({wires})"
        assert repr(FreeWires(wires)) == exp_string

    def test_init_recording(self):
        """Test that the FreeWires class is instantiated as expected when there is active recording."""
        with qml.queuing.AnnotatedQueue() as q:
            ops = [FreeWires(2), FreeWires(4), FreeWires(8)]

        assert q.queue == ops

    def test_mul(self):
        """Test that the multiplication works with FreeWires"""
        original_wires = FreeWires(5)
        new_wires = original_wires * 3
        assert new_wires.num_wires == 15

    def test_type_error(self):
        """Test that an error is raised when wrong type is provided for multiplication"""
        with pytest.raises(NotImplementedError):
            _ = FreeWires(5) * 4.2

    def test_equal(self):
        """Test that the equal function works as expected."""
        alloc_1 = FreeWires(num_wires=5)
        alloc_2 = FreeWires(num_wires=5)

        assert alloc_1 == alloc_2

        alloc_3 = FreeWires(num_wires=5)
        alloc_4 = FreeWires(num_wires=10)

        assert alloc_3 != alloc_4

        alloc_5 = FreeWires(num_wires=5)
        not_an_alloc = AllocWires(num_wires=5)
        assert alloc_5 != not_an_alloc

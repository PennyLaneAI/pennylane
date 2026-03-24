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

from pennylane.labs.estimator_beta.wires_manager import Allocate, Deallocate, MarkClean, MarkQubits
from pennylane.queuing import AnnotatedQueue
from pennylane.wires import Wires

# pylint: disable=too-many-arguments


class TestAllocate:
    """Test the methods and attributes of the Allocate class"""

    @pytest.mark.parametrize("restored", (None, True, False))
    @pytest.mark.parametrize("state", (None, "any", "zero"))
    @pytest.mark.parametrize("num_wires", (1, 2, 3, 5, 7))
    def test_init(self, num_wires, state, restored):
        """Test that the Allocate class is instantiated as expected when there is no active recording."""
        if state is None and restored is None:
            allocate = Allocate(num_wires)
        elif state is None:
            allocate = Allocate(num_wires, restored=restored)
        elif restored is None:
            allocate = Allocate(num_wires, state=state)
        else:
            allocate = Allocate(num_wires, state, restored)

        expected_state = "zero" if state is None else state
        expected_restored = False if restored is None else restored

        assert allocate.state == expected_state
        assert allocate.restored == expected_restored
        assert allocate.num_wires == num_wires

    @pytest.mark.parametrize(
        "num_wires, state, restored, error_msg",
        (
            (0, "zero", False, "num_wires must be a positive integer,"),
            (2.3, "zero", False, "num_wires must be a positive integer,"),
            (-4, "zero", False, "num_wires must be a positive integer,"),
            (1, "NotZero", False, "'NotZero' is not a valid AllocateState"),
            (1, 999, False, "999 is not a valid AllocateState"),
            (1, "zero", "NotFalse", "Expected restored to be True or False,"),
            (1, "zero", 123, "Expected restored to be True or False,"),
        ),
    )
    def test_init_error(self, num_wires, state, restored, error_msg):
        """Test that an error is raised if incompatible values are provided for input parameters."""
        with pytest.raises(ValueError, match=error_msg):
            Allocate(num_wires, state, restored)

    @pytest.mark.parametrize(
        "num_wires, state, restored, expected_str",
        (
            (1, None, None, "Allocate(1, state=zero, restored=False)"),
            (2, "any", None, "Allocate(2, state=any, restored=False)"),
            (3, "any", False, "Allocate(3, state=any, restored=False)"),
            (4, "any", True, "Allocate(4, state=any, restored=True)"),
            (5, "zero", None, "Allocate(5, state=zero, restored=False)"),
            (6, "zero", False, "Allocate(6, state=zero, restored=False)"),
            (7, "zero", True, "Allocate(7, state=zero, restored=True)"),
            (8, None, True, "Allocate(8, state=zero, restored=True)"),
            (9, None, False, "Allocate(9, state=zero, restored=False)"),
        ),
    )
    def test_repr(self, num_wires, state, restored, expected_str):
        """Test that correct representation is returned for Allocate class"""
        if state is None and restored is None:
            allocate = Allocate(num_wires)
        elif state is None:
            allocate = Allocate(num_wires, restored=restored)
        elif restored is None:
            allocate = Allocate(num_wires, state=state)
        else:
            allocate = Allocate(num_wires, state, restored)

        assert repr(allocate) == expected_str

    @pytest.mark.parametrize(
        "alloc_ops, expected_equality",
        (
            ((Allocate(5), Allocate(5)), True),
            ((Allocate(5), Allocate(3)), False),
            (
                (
                    Allocate(7, "any", True),
                    Allocate(7, "any", True),
                ),
                True,
            ),
            (
                (
                    Allocate(3, "any", True),
                    Allocate(3, "zero", True),
                ),
                False,
            ),
            (
                (
                    Allocate(2, "any", True),
                    Allocate(2, "any", False),
                ),
                False,
            ),
            (
                (
                    Allocate(2, "zero", True),
                    Allocate(1, "any", False),
                ),
                False,
            ),
            ((Allocate(5), Deallocate(5)), False),
        ),
    )
    def test_equal(self, alloc_ops, expected_equality):
        """Test that the equal function works as expected."""
        alloc_1, alloc_2 = alloc_ops
        assert alloc_1.equal(alloc_2) == expected_equality

    def test_immutable(self):
        """Test that this class is immutable"""
        allocate = Allocate(3, "any", True)

        with pytest.raises(AttributeError, match="Allocate instances are not mutable"):
            allocate.num_wires = 5

        with pytest.raises(AttributeError, match="Allocate instances are not mutable"):
            allocate.state = "zero"

        with pytest.raises(AttributeError, match="Allocate instances are not mutable"):
            allocate.restored = False


class TestDeallocate:
    """Test the methods and attributes of the Deallocate class"""

    @pytest.mark.parametrize(
        "num_wires, allocated_register, state, restored",
        (  ## No allocated register:
            (3, None, None, None),
            (3, None, None, True),
            (3, None, None, False),
            (5, None, "zero", None),
            (5, None, "zero", True),
            (5, None, "zero", False),
            (7, None, "any", None),
            # (7, None, "any", True), # when allocating AnyState w/ restored=True, we NEED allocated_regsiter
            (7, None, "any", False),
            ## Only allocated register:
            (None, Allocate(3), None, None),
            (None, Allocate(3, restored=True), None, None),
            (None, Allocate(3, restored=False), None, None),
            (None, Allocate(5, state="zero"), None, None),
            (None, Allocate(5, state="zero", restored=True), None, None),
            (None, Allocate(5, state="zero", restored=False), None, None),
            (None, Allocate(7, state="any"), None, None),
            (None, Allocate(7, state="any", restored=True), None, None),
            (None, Allocate(7, state="any", restored=False), None, None),
            ## Both: We just ignore the values of num_wires, restored, and state in favor of allocated_register
            (5, Allocate(3), "any", True),
            (7, Allocate(3, restored=True), "any", False),
            (2, Allocate(3, restored=False), "any", True),
            (3, Allocate(5, state="zero"), "any", True),
            (3, Allocate(5, state="zero", restored=True), "any", False),
            (3, Allocate(5, state="zero", restored=False), "any", True),
            (5, Allocate(7, state="any"), "zero", True),
            (5, Allocate(7, state="any", restored=True), "zero", False),
            (5, Allocate(7, state="any", restored=False), "zero", True),
        ),
    )
    def test_init(self, num_wires, allocated_register, state, restored):
        """Test that the Allocate class is instantiated as expected when there is no active recording."""
        if state is None and restored is None:
            deallocate = Deallocate(num_wires=num_wires, allocated_register=allocated_register)
        elif state is None:
            deallocate = Deallocate(
                num_wires=num_wires, allocated_register=allocated_register, restored=restored
            )
        elif restored is None:
            deallocate = Deallocate(
                num_wires=num_wires, allocated_register=allocated_register, state=state
            )
        else:
            deallocate = Deallocate(
                num_wires=num_wires,
                allocated_register=allocated_register,
                state=state,
                restored=restored,
            )

        expected_state = (
            allocated_register.state
            if allocated_register is not None
            else "zero" if state is None else state
        )
        expected_restored = (
            allocated_register.restored
            if allocated_register is not None
            else False if restored is None else restored
        )
        expected_num_wires = (
            allocated_register.num_wires if allocated_register is not None else num_wires
        )

        assert deallocate.state == expected_state
        assert deallocate.restored == expected_restored
        assert deallocate.num_wires == expected_num_wires
        assert deallocate.allocated_register == allocated_register

    @pytest.mark.parametrize(
        "num_wires, allocated_register, state, restored, error_msg",
        (
            (
                5,
                "NotAllocate(5)",
                "zero",
                False,
                "The allocated_register must be an instance of Allocate,",
            ),
            (
                None,
                None,
                "zero",
                False,
                "At least one of `num_wires` and `allocated_register` must be provided",
            ),
            (
                5,
                None,
                "any",
                True,
                "Must provide the `allocated_register` when deallocating an ANY state register",
            ),
            (0, None, "zero", False, "num_wires must be a positive integer,"),
            (2.3, None, "zero", False, "num_wires must be a positive integer,"),
            (-4, None, "zero", False, "num_wires must be a positive integer,"),
            (1, None, "NotZero", False, "'NotZero' is not a valid AllocateState"),
            (1, None, 999, False, "999 is not a valid AllocateState"),
            (1, None, "zero", "NotFalse", "Expected restored to be True or False,"),
            (1, None, "zero", 123, "Expected restored to be True or False,"),
        ),
    )
    def test_init_error(self, num_wires, allocated_register, state, restored, error_msg):
        """Test that an error is raised if incompatible values are provided for input parameters."""
        with pytest.raises(ValueError, match=error_msg):
            Deallocate(num_wires, allocated_register, state, restored)

    @pytest.mark.parametrize(
        "num_wires, allocated_register, state, restored, expected_str",
        (  ## No allocated register:
            (3, None, None, None, "Deallocate(3, state=zero, restored=False)"),
            (3, None, None, True, "Deallocate(3, state=zero, restored=True)"),
            (3, None, None, False, "Deallocate(3, state=zero, restored=False)"),
            (5, None, "zero", None, "Deallocate(5, state=zero, restored=False)"),
            (5, None, "zero", True, "Deallocate(5, state=zero, restored=True)"),
            (5, None, "zero", False, "Deallocate(5, state=zero, restored=False)"),
            (7, None, "any", None, "Deallocate(7, state=any, restored=False)"),
            (7, None, "any", False, "Deallocate(7, state=any, restored=False)"),
            ## Only allocated register:
            (None, Allocate(3), None, None, "Deallocate(3, state=zero, restored=False)"),
            (
                None,
                Allocate(3, restored=True),
                None,
                None,
                "Deallocate(3, state=zero, restored=True)",
            ),
            (
                None,
                Allocate(3, restored=False),
                None,
                None,
                "Deallocate(3, state=zero, restored=False)",
            ),
            (
                None,
                Allocate(5, state="zero"),
                None,
                None,
                "Deallocate(5, state=zero, restored=False)",
            ),
            (
                None,
                Allocate(5, state="zero", restored=True),
                None,
                None,
                "Deallocate(5, state=zero, restored=True)",
            ),
            (
                None,
                Allocate(5, state="zero", restored=False),
                None,
                None,
                "Deallocate(5, state=zero, restored=False)",
            ),
            (
                None,
                Allocate(7, state="any"),
                None,
                None,
                "Deallocate(7, state=any, restored=False)",
            ),
            (
                None,
                Allocate(7, state="any", restored=True),
                None,
                None,
                "Deallocate(7, state=any, restored=True)",
            ),
            (
                None,
                Allocate(7, state="any", restored=False),
                None,
                None,
                "Deallocate(7, state=any, restored=False)",
            ),
            ## Both: We just ignore the values of num_wires, restored, and state in favor of allocated_register
            (5, Allocate(3), "any", True, "Deallocate(3, state=zero, restored=False)"),
            (
                7,
                Allocate(3, restored=True),
                "any",
                False,
                "Deallocate(3, state=zero, restored=True)",
            ),
            (
                2,
                Allocate(3, restored=False),
                "any",
                True,
                "Deallocate(3, state=zero, restored=False)",
            ),
            (
                3,
                Allocate(5, state="zero"),
                "any",
                True,
                "Deallocate(5, state=zero, restored=False)",
            ),
            (
                3,
                Allocate(5, state="zero", restored=True),
                "any",
                False,
                "Deallocate(5, state=zero, restored=True)",
            ),
            (
                3,
                Allocate(5, state="zero", restored=False),
                "any",
                True,
                "Deallocate(5, state=zero, restored=False)",
            ),
            (5, Allocate(7, state="any"), "zero", True, "Deallocate(7, state=any, restored=False)"),
            (
                5,
                Allocate(7, state="any", restored=True),
                "zero",
                False,
                "Deallocate(7, state=any, restored=True)",
            ),
            (
                5,
                Allocate(7, state="any", restored=False),
                "zero",
                True,
                "Deallocate(7, state=any, restored=False)",
            ),
        ),
    )
    def test_repr(self, num_wires, allocated_register, state, restored, expected_str):
        """Test that correct representation is returned for Allocate class"""
        if state is None and restored is None:
            deallocate = Deallocate(num_wires=num_wires, allocated_register=allocated_register)
        elif state is None:
            deallocate = Deallocate(
                num_wires=num_wires, allocated_register=allocated_register, restored=restored
            )
        elif restored is None:
            deallocate = Deallocate(
                num_wires=num_wires, allocated_register=allocated_register, state=state
            )
        else:
            deallocate = Deallocate(
                num_wires=num_wires,
                allocated_register=allocated_register,
                state=state,
                restored=restored,
            )

        assert repr(deallocate) == expected_str

    @pytest.mark.parametrize(
        "alloc_ops, expected_equality",
        (
            ((Deallocate(5), Allocate(5)), False),
            ((Deallocate(5), Deallocate(5)), True),
            ((Deallocate(5), Deallocate(3)), False),
            (
                (
                    Deallocate(7, state="any", restored=False),
                    Deallocate(7, state="any", restored=False),
                ),
                True,
            ),
            (
                (
                    Deallocate(3, state="any", restored=False),
                    Deallocate(3, state="zero", restored=True),
                ),
                False,
            ),
            (
                (
                    Deallocate(2, state="zero", restored=True),
                    Deallocate(2, state="zero", restored=False),
                ),
                False,
            ),
            (
                (
                    Deallocate(2, state="zero", restored=True),
                    Deallocate(1, state="any", restored=False),
                ),
                False,
            ),
            (
                (
                    Deallocate(allocated_register=Allocate(3)),
                    Deallocate(allocated_register=Allocate(3)),
                ),
                True,
            ),
            (
                (
                    Deallocate(allocated_register=Allocate(3, "any", True)),
                    Deallocate(allocated_register=Allocate(3, "any", True)),
                ),
                True,
            ),
            (
                (
                    Deallocate(allocated_register=Allocate(3, "any", True)),
                    Deallocate(allocated_register=Allocate(3, "zero", True)),
                ),
                False,
            ),
            (
                (
                    Deallocate(allocated_register=Allocate(3, "any", True)),
                    Deallocate(allocated_register=Allocate(3, "any", False)),
                ),
                False,
            ),
            (
                (
                    Deallocate(allocated_register=Allocate(3, "any", True)),
                    Deallocate(allocated_register=Allocate(3, "any", False)),
                ),
                False,
            ),
        ),
    )
    def test_equal(self, alloc_ops, expected_equality):
        """Test that the equal function works as expected."""
        alloc_1, alloc_2 = alloc_ops
        assert alloc_1.equal(alloc_2) == expected_equality

    def test_immutable(self):
        """Test that this class is immutable"""
        deallocate = Deallocate(allocated_register=Allocate(3, "any", True))

        with pytest.raises(AttributeError, match="Deallocate instances are not mutable"):
            deallocate.num_wires = 5

        with pytest.raises(AttributeError, match="Deallocate instances are not mutable"):
            deallocate.state = "zero"

        with pytest.raises(AttributeError, match="Deallocate instances are not mutable"):
            deallocate.restored = False

        with pytest.raises(AttributeError, match="Deallocate instances are not mutable"):
            deallocate.allocated_register = Allocate(5, "zero", False)


class TestMarkQubits:
    """Test the methods and attributes of the MarkQubits and MarkClean class"""

    @pytest.mark.parametrize("wires", ([1, 2, 3], None, [0], [1, "a", 2, "b"], "c"))
    def test_init(self, wires):
        """Test that an instance of MarkQubits is instantiated correctly"""
        with AnnotatedQueue() as q:
            marked_qubits = MarkQubits(wires)
            marked_clean_qubits = MarkClean(wires)

        expected_wires = Wires([] if wires is None else wires)

        # Test wires:
        assert marked_qubits.wires == Wires(expected_wires)
        assert marked_clean_qubits.wires == Wires(expected_wires)

        # Test queuing:
        assert q.queue[0] == marked_qubits
        assert q.queue[1] == marked_clean_qubits

    def test_queue(self):
        """Test the queue method of the MarkQubits class"""
        marked_qubits1 = MarkQubits(wires=[1, 2, 3])
        marked_qubits2 = MarkQubits(wires=["a", "b", "c"])
        marked_qubits3 = MarkClean(wires=[0, "d"])

        with AnnotatedQueue() as q:
            marked_qubits3.queue()
            marked_qubits1.queue()
            marked_qubits2.queue()

        assert len(q.queue) == 3
        assert q.queue[0] is marked_qubits3
        assert q.queue[1] is marked_qubits1
        assert q.queue[2] is marked_qubits2

    def test_equal(self):
        """Test the equal dunder method of the MarkQubits class"""
        marked_qubits = MarkQubits(wires=[1, 2, 3])
        assert marked_qubits.equal(marked_qubits)
        assert hash(marked_qubits) == hash(marked_qubits)

        marked_qubits2 = MarkQubits(wires=[1, 2, 3])
        assert marked_qubits.equal(marked_qubits2)
        assert hash(marked_qubits) != hash(
            marked_qubits2
        )  # we need the hash to be different for queuing

        marked_qubits3 = MarkQubits(wires=[3, 2, 1])
        assert marked_qubits.equal(marked_qubits3)
        assert hash(marked_qubits) != hash(marked_qubits3)

        marked_qubits4 = MarkQubits(wires=[3, 2])
        assert not marked_qubits.equal(marked_qubits4)

    @pytest.mark.parametrize(
        "wires, expected_str",
        (
            ([1, 2, 3], "MarkClean(Wires([1, 2, 3]))"),
            (None, "MarkClean(Wires([]))"),
            ([0], "MarkClean(Wires([0]))"),
            ([1, "a", 2, "b"], "MarkClean(Wires([1, 'a', 2, 'b']))"),
            ("c", "MarkClean(Wires(['c']))"),
        ),
    )
    def test_MarkClean_repr(self, wires, expected_str):
        """Test the repr dundar method of the MarkClean class"""
        mark_clean = MarkClean(wires)
        assert repr(mark_clean) == expected_str

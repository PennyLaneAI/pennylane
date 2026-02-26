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

"""Tests for the base classes used when tracking qubits for resource estimation."""

import pytest

import pennylane.estimator as qre
import pennylane.labs.estimator_beta as qre_exp
from pennylane.allocation import AllocateState
from pennylane.estimator import (
    GateCount,
)
from pennylane.labs.estimator_beta.wires_manager import (
    Allocate,
    Deallocate,
    MarkClean,
    _estimate_auxiliary_wires,
    _process_circuit_lst,
    estimate_wires_from_circuit,
    estimate_wires_from_resources,
)

any_state = AllocateState.ANY


# Dummy Ops to use for testing:
class AllocateOp(qre.ResourceOperator):
    """A dummy class whose decomposition allocates qubits"""

    resource_keys = {"allocate", "num_wires"}

    def __init__(self, allocate, num_wires=0, wires=None):
        self.num_wires = num_wires
        self.allocate = allocate
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        return {"num_wires": self.num_wires, "allocate": self.allocate}

    @classmethod
    def resource_rep(cls, allocate, num_wires=0):
        params = {"num_wires": num_wires, "allocate": allocate}
        return qre.CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, allocate, num_wires=0):
        return [qre.GateCount(qre.Identity.resource_rep(), 3), allocate]


class DeallocateOp(qre.ResourceOperator):
    """A dummy class whose decomposition de-allocates qubits"""

    resource_keys = {"deallocate", "num_wires"}

    def __init__(self, deallocate, num_wires=0, wires=None):
        self.num_wires = num_wires
        self.deallocate = deallocate
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        return {"num_wires": self.num_wires, "deallocate": self.deallocate}

    @classmethod
    def resource_rep(cls, deallocate, num_wires=0):
        params = {"num_wires": num_wires, "deallocate": deallocate}
        return qre.CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, deallocate, num_wires=0):
        return [qre.GateCount(qre.Identity.resource_rep()), deallocate]


class AlocOpFree(qre.ResourceOperator):
    """A dummy class whose decomposition allocates qubits, applies an operation and deallocates qubits"""

    resource_keys = {"num_wires", "allocate", "cmpr_op", "deallocate"}

    def __init__(self, num_wires, allocate, op, deallocate=None, wires=None):
        self.num_wires = num_wires
        self.allocate = allocate
        self.deallocate = deallocate or Deallocate(allocated_register=allocate)

        qre.resource_operator._dequeue(op)
        self.cmpr_op = op.resource_rep_from_op()
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        return {
            "num_wires": self.num_wires,
            "allocate": self.allocate,
            "cmpr_op": self.cmpr_op,
            "deallocate": self.deallocate,
        }

    @classmethod
    def resource_rep(cls, num_wires, allocate, cmpr_op, deallocate=None):
        deallocate = deallocate or Deallocate(allocated_register=allocate)
        params = {
            "num_wires": num_wires,
            "allocate": allocate,
            "cmpr_op": cmpr_op,
            "deallocate": deallocate,
        }
        return qre.CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, num_wires, allocate, cmpr_op, deallocate=None):
        deallocate = deallocate or Deallocate(allocated_register=allocate)
        return [
            qre.GateCount(qre.Identity.resource_rep()),
            allocate,
            qre.GateCount(cmpr_op),
            deallocate,
        ]


def any_state_allocation1():
    """Generate list of actions for testing"""
    allocate = Allocate(3, state=AllocateState.ANY, restored=True)
    deallocate = Deallocate(allocated_register=allocate)

    return [
        allocate,
        GateCount(qre.X.resource_rep(), 5),
        deallocate,
    ]


def any_state_allocation2():
    """Generate list of actions for testing"""
    allocate = Allocate(3, state=AllocateState.ANY, restored=True)
    deallocate = Deallocate(allocated_register=allocate)

    return [
        Allocate(2),
        GateCount(qre.Y.resource_rep(), 3),
        allocate,
        Deallocate(1),
        GateCount(qre.CNOT.resource_rep(), 2),
        GateCount(qre.X.resource_rep(), 5),
        deallocate,
        Deallocate(2),
    ]


def any_state_allocation3():
    """Generate list of actions for testing"""
    allocate_3 = Allocate(3, state=AllocateState.ANY, restored=True)
    deallocate_3 = Deallocate(allocated_register=allocate_3)

    allocate_4 = Allocate(4, state=AllocateState.ANY, restored=True)
    deallocate_4 = Deallocate(allocated_register=allocate_4)

    return [
        allocate_3,
        GateCount(qre.CNOT.resource_rep(), 2),
        allocate_4,
        GateCount(qre.X.resource_rep(), 5),
        deallocate_3,
        Allocate(2),
        deallocate_4,
    ]


def nested_any_state_allocation1():
    z = qre.Z.resource_rep()
    return [
        GateCount(AlocOpFree.resource_rep(2, Allocate(3, any_state, restored=True), z), 5),
        Allocate(5),
        GateCount(AlocOpFree.resource_rep(4, Allocate(3, any_state, restored=True), z), 3),
        Deallocate(3),
        GateCount(AlocOpFree.resource_rep(4, Allocate(4, any_state, restored=True), z), 7),
    ]


def nested_any_state_allocation2():
    z = qre.Z.resource_rep()
    allocate = Allocate(2, any_state, True)
    return [
        allocate,
        GateCount(qre.X.resource_rep(), 10),
        GateCount(AlocOpFree.resource_rep(2, Allocate(5, any_state, restored=True), z), 5),
        Deallocate(allocated_register=allocate),
    ]


def nested_any_state_allocation3():
    z = qre.Z.resource_rep()
    allocate = Allocate(2, any_state, True)
    return [  # 2, 2
        allocate,  # (2, 0, 2)
        GateCount(qre.X.resource_rep(), 10),
        GateCount(
            AlocOpFree.resource_rep(2, Allocate(3, any_state, restored=True), z), 5
        ),  # (1, 0, 0) => (3, 0, 2)
        Allocate(5),  # (7, 0, 7)
        GateCount(
            AlocOpFree.resource_rep(4, Allocate(3, any_state, restored=True), z), 3
        ),  # (0, 0, 0) => (7, 0, 7)
        Deallocate(3),  # (7, 0, 4)
        GateCount(
            AlocOpFree.resource_rep(4, Allocate(4, any_state, restored=True), z), 7
        ),  # (2, 0, 0) => (7, 0, 4)
        Deallocate(allocated_register=allocate),  # (7, 0, 2)
    ]


# ------- Tests: -------
class TestEstimateAuxiliaryWires:
    """Test the private helper functions"""

    def test_error_when_deallocating_any_state_without_allocation(self):
        """Test that an error is raised when a circuit attempts to deallocate qubits in the
        Any state before they were allocated."""
        allocate = Allocate(5, state=AllocateState.ANY, restored=True)
        deallocate = Deallocate(
            5, allocated_register=allocate, state=AllocateState.ANY, restored=True
        )

        with pytest.raises(
            ValueError, match="Trying to deallocate an ANY state register before it was allocated"
        ):
            lst_actions = [
                Allocate(5),  # Allocated in the zero state
                GateCount(qre.Z.resource_rep(), 5),
                Deallocate(2),  # Deallocated in the zero state
                GateCount(qre.X.resource_rep(), 3),
                Allocate(3),
                deallocate,
                allocate,
            ]
            _estimate_auxiliary_wires(
                list_actions=lst_actions,
                scalar=3,
                num_available_any_state_aux=10,
                num_active_qubits=5,
            )

    def test_error_when_not_deallocating_any_state_allocation(self):
        """Test that an error is raised when a circuit allocates an Any state to deallocate qubits in the
        Any state before they were allocated."""
        with pytest.raises(
            ValueError,
            match="Did NOT deallocate and restore all ANY state allocations as promised:",
        ):
            lst_actions = [
                Allocate(5),  # Allocated in the zero state
                Allocate(
                    5, state=AllocateState.ANY, restored=True
                ),  # Allocate with a promise to restore
                GateCount(qre.Z.resource_rep(), 5),
                Deallocate(2),  # Deallocated in the zero state
                GateCount(qre.X.resource_rep(), 3),
                Allocate(3),
            ]
            _estimate_auxiliary_wires(
                list_actions=lst_actions,
                scalar=3,
                num_available_any_state_aux=10,
                num_active_qubits=5,
            )

    @pytest.mark.parametrize(  # All expected results were computed by hand
        "list_actions, scalar, num_active, num_aux, expected_results",
        (
            (
                [  # Plain example with no allocation
                    GateCount(qre.resource_rep(qre.X)),
                ],
                1,  # Scalar
                1,  # number of active qubits
                0,  # number of any state auxiliaries,
                (0, 0, 0),  # max alloc, max dealloc, total
            ),
            (
                [  # Allocation only
                    Allocate(3),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Allocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                ],
                1,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (5, 0, 5),  # max alloc, max dealloc, total
            ),
            (
                [  # Allocation + Scaling
                    Allocate(3),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Allocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                ],
                7,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (35, 0, 35),  # max alloc, max dealloc, total
            ),
            (
                [  # Allocation and deallocation with scaling
                    Allocate(3),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Deallocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                ],
                5,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (7, 0, 5),  # max alloc, max dealloc, total
            ),
            (
                [  # Deallocation only
                    Deallocate(4),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Deallocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                ],
                1,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (0, -6, -6),  # max alloc, max dealloc, total
            ),
            (
                [  # Deallocation + scaling
                    Deallocate(4),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Deallocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                ],
                4,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (0, -24, -24),  # max alloc, max dealloc, total
            ),
            (
                [  # Deallocation and Allocation
                    Deallocate(4),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Allocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                ],
                1,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (0, -4, -2),  # max alloc, max dealloc, total
            ),
            (
                [  # Deallocation and Allocation + scaling
                    Deallocate(4),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Allocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                ],
                6,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (0, -14, -12),  # max alloc, max dealloc, total
            ),
            (
                [  # Deallocation & Allocation with underflow
                    Deallocate(4),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Allocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                    Allocate(5),
                    GateCount(qre.resource_rep(qre.X)),
                    Deallocate(1),
                ],
                1,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (3, -4, 2),  # max alloc, max dealloc, total
            ),
            (
                [  # Deallocation & Allocation with underflow + scaling
                    Deallocate(4),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Allocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                    Allocate(5),
                    GateCount(qre.resource_rep(qre.X)),
                    Deallocate(1),
                ],
                3,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (7, -4, 6),  # max alloc, max dealloc, total
            ),
            (
                [  # Allocation & Deallocation with overflow
                    Allocate(4),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Deallocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                    Deallocate(5),
                    GateCount(qre.resource_rep(qre.X)),
                    Allocate(1),
                ],
                1,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (4, -3, -2),  # max alloc, max dealloc, total
            ),
            (
                [  # Allocation & Deallocation with overflow + scaling
                    Allocate(4),
                    GateCount(qre.resource_rep(qre.CNOT)),
                    Deallocate(2),
                    GateCount(qre.resource_rep(qre.Z), 2),
                    Deallocate(5),
                    GateCount(qre.resource_rep(qre.X)),
                    Allocate(1),
                ],
                3,  # Scalar
                2,  # number of active qubits
                0,  # number of any state auxiliaries,
                (4, -7, -6),  # max alloc, max dealloc, total
            ),
        ),
    )
    def test_estimate_auxiliary_wires(
        self, list_actions, scalar, num_active, num_aux, expected_results
    ):
        """Test qubit tracking WITHOUT nested allocation and deallocation"""
        results = _estimate_auxiliary_wires(
            list_actions=list_actions,
            scalar=scalar,
            num_available_any_state_aux=num_aux,
            num_active_qubits=num_active,
        )
        assert results == expected_results

    @pytest.mark.parametrize(  # All expected results were computed by hand
        "generate_actions, scalar, num_active, num_aux, expected_results",
        (
            (
                any_state_allocation1,
                1,
                0,
                0,
                (3, 0, 0),
            ),
            (
                any_state_allocation1,
                1,
                3,
                3,
                (3, 0, 0),
            ),
            (
                any_state_allocation1,
                1,
                5,
                8,
                (0, 0, 0),
            ),
            (
                any_state_allocation1,
                1,
                1,
                5,
                (0, 0, 0),
            ),
            (
                any_state_allocation1,
                1,
                2,
                4,
                (1, 0, 0),
            ),
            (
                any_state_allocation1,
                5,
                0,
                0,
                (3, 0, 0),
            ),
            (
                any_state_allocation1,
                5,
                3,
                3,
                (3, 0, 0),
            ),
            (
                any_state_allocation1,
                5,
                5,
                8,
                (0, 0, 0),
            ),
            (
                any_state_allocation1,
                5,
                1,
                5,
                (0, 0, 0),
            ),
            (
                any_state_allocation1,
                5,
                2,
                4,
                (1, 0, 0),
            ),
            # 2nd allocation function
            (
                any_state_allocation2,
                1,
                0,
                0,
                (5, -1, -1),
            ),
            (
                any_state_allocation2,
                1,
                3,
                3,
                (5, -1, -1),
            ),
            (
                any_state_allocation2,
                1,
                5,
                8,
                (2, -1, -1),
            ),
            (
                any_state_allocation2,
                1,
                1,
                5,
                (2, -1, -1),
            ),
            (
                any_state_allocation2,
                1,
                2,
                4,
                (3, -1, -1),
            ),
            (
                any_state_allocation2,
                5,
                0,
                0,
                (5, -5, -5),
            ),
            (
                any_state_allocation2,
                5,
                3,
                3,
                (5, -5, -5),
            ),
            (
                any_state_allocation2,
                5,
                5,
                8,
                (2, -5, -5),
            ),
            (
                any_state_allocation2,
                5,
                1,
                5,
                (2, -5, -5),
            ),
            (
                any_state_allocation2,
                5,
                2,
                4,
                (3, -5, -5),
            ),
            # 3rd allocation function
            (
                any_state_allocation3,
                1,
                0,
                0,
                (7, 0, 2),
            ),
            (
                any_state_allocation3,
                1,
                3,
                3,
                (7, 0, 2),
            ),
            (
                any_state_allocation3,
                1,
                5,
                8,
                (6, 0, 2),
            ),
            (
                any_state_allocation3,
                1,
                1,
                5,
                (5, 0, 2),
            ),
            (
                any_state_allocation3,
                1,
                2,
                4,
                (6, 0, 2),
            ),
            (
                any_state_allocation3,
                5,
                0,
                0,
                (15, 0, 10),
            ),
            (
                any_state_allocation3,
                5,
                3,
                3,
                (15, 0, 10),
            ),
            (
                any_state_allocation3,
                5,
                5,
                8,
                (14, 0, 10),
            ),
            (
                any_state_allocation3,
                5,
                1,
                5,
                (13, 0, 10),
            ),
            (
                any_state_allocation3,
                5,
                2,
                4,
                (14, 0, 10),
            ),
        ),
    )
    def test_simple_any_state_allocation(
        self, generate_actions, scalar, num_active, num_aux, expected_results
    ):
        """Test qubit tracking with Any state allocation WITHOUT nested allocation and deallocation"""
        results = _estimate_auxiliary_wires(
            list_actions=generate_actions(),
            scalar=scalar,
            num_available_any_state_aux=num_aux,
            num_active_qubits=num_active,
        )
        assert results == expected_results

    @pytest.mark.parametrize(  # All expected results were computed by hand
        "lst_actions, scalar, num_aux, num_active, expected_results",
        (
            (
                [  # Nested allocation and deallocation
                    Allocate(1),
                    GateCount(qre.X.resource_rep()),
                    GateCount(
                        qre.Prod.resource_rep(
                            (
                                (AllocateOp.resource_rep(Allocate(1)), 2),
                                (qre.CNOT.resource_rep(), 1),
                                (qre.Y.resource_rep(), 2),
                                (DeallocateOp.resource_rep(Deallocate(1)), 1),
                            ),
                            num_wires=1,
                        ),
                        3,
                    ),
                    GateCount(qre.Z.resource_rep(), 7),
                    GateCount(
                        qre.Prod.resource_rep(
                            (
                                (AllocateOp.resource_rep(Allocate(1)), 1),
                                (qre.Y.resource_rep(), 2),
                                (qre.CNOT.resource_rep(), 1),
                                (DeallocateOp.resource_rep(Deallocate(1)), 2),
                            ),
                            num_wires=1,
                        ),
                        3,
                    ),
                    Deallocate(1),
                ],
                1,
                0,
                4,
                (5, 0, 0),
            ),
            (
                [  # Nested allocation and deallocation
                    Allocate(1),
                    GateCount(qre.X.resource_rep()),
                    GateCount(
                        qre.Prod.resource_rep(
                            (
                                (AllocateOp.resource_rep(Allocate(1)), 2),
                                (qre.CNOT.resource_rep(), 1),
                                (qre.Y.resource_rep(), 2),
                                (DeallocateOp.resource_rep(Deallocate(1)), 1),
                            ),
                            num_wires=1,
                        ),
                        3,
                    ),
                    GateCount(qre.Z.resource_rep(), 7),
                    GateCount(
                        qre.Prod.resource_rep(
                            (
                                (AllocateOp.resource_rep(Allocate(1)), 1),
                                (qre.Y.resource_rep(), 2),
                                (qre.CNOT.resource_rep(), 1),
                                (DeallocateOp.resource_rep(Deallocate(1)), 2),
                            ),
                            num_wires=1,
                        ),
                        3,
                    ),
                    Deallocate(1),
                ],
                10,
                0,
                4,
                (5, 0, 0),
            ),
            (
                [  # Nested allocation and deallocation with underflow
                    Allocate(5),
                    GateCount(qre.X.resource_rep()),
                    GateCount(AllocateOp.resource_rep(allocate=Allocate(4))),
                    GateCount(qre.Z.resource_rep(), 5),
                    GateCount(DeallocateOp.resource_rep(deallocate=Deallocate(2)), 2),
                    Deallocate(5),
                    GateCount(DeallocateOp.resource_rep(deallocate=Deallocate(1)), 5),
                    GateCount(AllocateOp.resource_rep(allocate=Allocate(3))),
                ],
                1,
                0,
                6,
                (9, -5, -2),
            ),
            (
                [  # Nested allocation and deallocation with underflow + scaling
                    Allocate(5),
                    GateCount(qre.X.resource_rep()),
                    GateCount(AllocateOp.resource_rep(allocate=Allocate(4))),
                    GateCount(qre.Z.resource_rep(), 5),
                    GateCount(DeallocateOp.resource_rep(deallocate=Deallocate(2)), 2),
                    Deallocate(5),
                    GateCount(DeallocateOp.resource_rep(deallocate=Deallocate(1)), 5),
                    GateCount(AllocateOp.resource_rep(allocate=Allocate(3))),
                ],
                10,
                0,
                6,
                (9, -23, -20),
            ),
            (
                [  # Nested allocation and deallocation with overflow
                    Deallocate(5),  # -5
                    GateCount(qre.CNOT.resource_rep(), 3),
                    GateCount(AllocateOp.resource_rep(allocate=Allocate(2)), 5),  # +5
                    GateCount(qre.Z.resource_rep(), 5),
                    GateCount(DeallocateOp.resource_rep(deallocate=Deallocate(2)), 2),  # +1
                    GateCount(
                        qre.Prod.resource_rep(  # (+20, 0, +12)    # +21
                            (
                                (AllocateOp.resource_rep(allocate=Allocate(3)), 4),  # +12
                                (qre.Y.resource_rep(), 10),
                                (DeallocateOp.resource_rep(deallocate=Deallocate(4)), 2),  # -8
                            ),
                            num_wires=3,
                        ),
                        count=3,
                    ),
                    GateCount(DeallocateOp.resource_rep(deallocate=Deallocate(1)), 5),  # +16
                ],
                1,
                0,
                6,
                (21, -5, 8),
            ),
            (
                [  # Nested allocation and deallocation with overflow + scaling
                    Deallocate(5),  # -5
                    GateCount(qre.CNOT.resource_rep(), 3),
                    GateCount(AllocateOp.resource_rep(allocate=Allocate(2)), 5),  # +5
                    GateCount(qre.Z.resource_rep(), 5),
                    GateCount(DeallocateOp.resource_rep(deallocate=Deallocate(2)), 2),  # +1
                    GateCount(
                        qre.Prod.resource_rep(  # (+20, 0, +12)    # +13
                            (
                                (AllocateOp.resource_rep(allocate=Allocate(3)), 4),  # +12
                                (qre.Y.resource_rep(), 10),
                                (DeallocateOp.resource_rep(deallocate=Deallocate(4)), 2),  # -8
                            ),
                            num_wires=3,
                        ),
                        count=3,
                    ),
                    GateCount(DeallocateOp.resource_rep(deallocate=Deallocate(1)), 5),  # +8
                ],
                7,
                0,
                6,
                (69, -5, 56),
            ),
        ),
    )
    def test_nested_allocation_and_deallocation(
        self, lst_actions, scalar, num_aux, num_active, expected_results
    ):
        """Test that qubit tracking works as expected for circuits with operators whos'
        decompositions require auxiliary qubits."""
        results = _estimate_auxiliary_wires(
            list_actions=lst_actions,
            scalar=scalar,
            num_available_any_state_aux=num_aux,
            num_active_qubits=num_active,
        )
        assert results == expected_results

    @pytest.mark.parametrize(  # All expected results were computed by hand
        "generate_actions, scalar, num_active, num_aux, expected_results",
        (
            (nested_any_state_allocation1, 1, 2, 2, (6, 0, 2)),
            (nested_any_state_allocation1, 1, 2, 4, (5, 0, 2)),
            (nested_any_state_allocation1, 1, 2, 7, (5, 0, 2)),
            (nested_any_state_allocation2, 1, 2, 2, (5, 0, 0)),
            (nested_any_state_allocation2, 1, 2, 4, (3, 0, 0)),
            (nested_any_state_allocation2, 1, 2, 7, (0, 0, 0)),
            (nested_any_state_allocation3, 1, 2, 2, (7, 0, 2)),
            (nested_any_state_allocation3, 1, 2, 4, (5, 0, 2)),
            (nested_any_state_allocation3, 1, 2, 7, (5, 0, 2)),
        ),
    )
    def test_nested_any_state_allocation(
        self, generate_actions, scalar, num_active, num_aux, expected_results
    ):
        """Test qubit tracking with Any state allocation WITHOUT nested allocation and deallocation"""
        results = _estimate_auxiliary_wires(
            list_actions=generate_actions(),
            scalar=scalar,
            num_available_any_state_aux=num_aux,
            num_active_qubits=num_active,
        )
        assert results == expected_results

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

import pennylane as qml
import pennylane.estimator as qre
from pennylane.allocation import AllocateState
from pennylane.estimator import (
    GateCount,
    Resources,
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
from pennylane.wires import Wires

# pylint: disable=unused-argument,too-many-arguments,arguments-differ

any_state = AllocateState.ANY


# Dummy Ops and Funcs to use for testing:
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
        return [allocate]


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
        return [deallocate]


class AllocOpFree(qre.ResourceOperator):
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
    """Test function for nested any state allocation"""
    z = qre.Z.resource_rep()
    return [
        GateCount(AllocOpFree.resource_rep(2, Allocate(3, any_state, restored=True), z), 5),
        Allocate(5),
        GateCount(AllocOpFree.resource_rep(4, Allocate(3, any_state, restored=True), z), 3),
        Deallocate(3),
        GateCount(AllocOpFree.resource_rep(4, Allocate(4, any_state, restored=True), z), 7),
    ]


def nested_any_state_allocation2():
    """Test function for nested any state allocation"""
    z = qre.Z.resource_rep()
    allocate = Allocate(2, any_state, True)
    return [
        allocate,
        GateCount(qre.X.resource_rep(), 10),
        GateCount(AllocOpFree.resource_rep(2, Allocate(5, any_state, restored=True), z), 5),
        Deallocate(allocated_register=allocate),
    ]


def nested_any_state_allocation3():
    """Test function for nested any state allocation"""
    z = qre.Z.resource_rep()
    allocate = Allocate(2, any_state, True)
    return [
        allocate,
        GateCount(qre.X.resource_rep(), 10),
        GateCount(AllocOpFree.resource_rep(2, Allocate(3, any_state, restored=True), z), 5),
        Allocate(5),
        GateCount(AllocOpFree.resource_rep(4, Allocate(3, any_state, restored=True), z), 3),
        Deallocate(3),
        GateCount(AllocOpFree.resource_rep(4, Allocate(4, any_state, restored=True), z), 7),
        Deallocate(allocated_register=allocate),
    ]


# ------- Tests: -------
class TestProcessCircuitLst:
    """Test that the private _process_circuit_lst function works as expected."""

    def test_error_incompatible_object(self):
        """Test that an error is raised when an object is encountered that is not
        a valid type (Operator, ResourceOperator, MeasurementProcess, MarkQubits)."""

        with pytest.raises(ValueError, match="Circuit must contain only instances of"):
            _process_circuit_lst([qre.X(0), "InvalidOp"])

    def test_error_marking_qubits_not_in_circuit(self):
        """Test that an error is raised if we attempt to mark qubits that haven't been
        listed by any operator in the circuit."""
        with pytest.raises(ValueError, match=r"Attempted to mark qubits Wires\(\[5, 6\]\)"):
            _process_circuit_lst(
                [
                    MarkClean([0, 1, 2]),
                    qre.CNOT(wires=[0, 1]),
                    qre.QFT(wires=[2, 3, 4]),
                    MarkClean(wires=[4, 5, 6]),
                ]
            )

    @pytest.mark.parametrize(
        "circ, expected_processed_circ, expected_circ_wires",
        (
            (  # Explicit Wire Labels
                [
                    qre.Z(wires=0),
                    qml.Z(wires=1),
                    qre.RZ(wires="a"),
                    qml.CNOT(wires=[2, "b"]),
                    qml.RX(1.23, wires="c"),
                ],
                [
                    (qre.Z.resource_rep(), Wires(0)),
                    (qre.Z.resource_rep(), Wires(1)),
                    (qre.RZ.resource_rep(), Wires("a")),
                    (qre.CNOT.resource_rep(), Wires([2, "b"])),
                    (qre.RX.resource_rep(), Wires("c")),
                ],
                Wires([0, 1, "a", 2, "b", "c"]),
            ),
            (  # Dynamically Generated
                [
                    qre.X(),
                    qre.CNOT(),
                    qre.Toffoli(),
                    qre.MultiControlledX(num_ctrl_wires=3),
                    qre.QFT(num_wires=5),
                    qre.ControlledSequence(qre.Z(), num_control_wires=3),
                    qre.CCZ(),
                    qre.CZ(),
                    qre.Z(),
                ],
                [
                    (qre.X.resource_rep(), Wires("__generated_wire0__")),
                    (
                        qre.CNOT.resource_rep(),
                        Wires(["__generated_wire0__", "__generated_wire1__"]),
                    ),
                    (
                        qre.Toffoli.resource_rep(),
                        Wires(
                            ["__generated_wire0__", "__generated_wire1__", "__generated_wire2__"]
                        ),
                    ),
                    (
                        qre.MultiControlledX.resource_rep(3, 0),
                        Wires(
                            [
                                "__generated_wire0__",
                                "__generated_wire1__",
                                "__generated_wire2__",
                                "__generated_wire3__",
                            ]
                        ),
                    ),
                    (
                        qre.QFT.resource_rep(5),
                        Wires(
                            [
                                "__generated_wire0__",
                                "__generated_wire1__",
                                "__generated_wire2__",
                                "__generated_wire3__",
                                "__generated_wire4__",
                            ]
                        ),
                    ),
                    (
                        qre.ControlledSequence.resource_rep(qre.Z.resource_rep(), 3),
                        Wires(
                            [
                                "__generated_wire0__",
                                "__generated_wire1__",
                                "__generated_wire2__",
                                "__generated_wire3__",
                            ]
                        ),
                    ),
                    (
                        qre.CCZ.resource_rep(),
                        Wires(
                            ["__generated_wire0__", "__generated_wire1__", "__generated_wire2__"]
                        ),
                    ),
                    (qre.CZ.resource_rep(), Wires(["__generated_wire0__", "__generated_wire1__"])),
                    (qre.Z.resource_rep(), Wires("__generated_wire0__")),
                ],
                Wires(
                    [
                        "__generated_wire0__",
                        "__generated_wire1__",
                        "__generated_wire2__",
                        "__generated_wire3__",
                        "__generated_wire4__",
                    ]
                ),
            ),
            (  # Mixed Explicit and Dynamically Generated
                [
                    qre.CNOT(wires=[0, 1]),
                    qre.QFT(num_wires=5),
                    qml.CZ(wires=[1, 2]),
                    MarkClean(wires=[0, 1, 2]),
                    qre.MultiControlledX(num_ctrl_wires=3),
                    qre.Z(),
                    qml.Z(wires=1),
                ],
                [
                    (qre.CNOT.resource_rep(), Wires([0, 1])),
                    (qre.QFT.resource_rep(5), Wires([f"__generated_wire{i}__" for i in range(5)])),
                    (qre.CZ.resource_rep(), Wires([1, 2])),
                    (MarkClean(wires=[0, 1, 2]), Wires([0, 1, 2])),
                    (
                        qre.MultiControlledX.resource_rep(3, 0),
                        Wires([f"__generated_wire{i}__" for i in range(4)]),
                    ),
                    (qre.Z.resource_rep(), Wires("__generated_wire0__")),
                    (qre.Z.resource_rep(), Wires(1)),
                ],
                Wires(
                    [
                        0,
                        1,
                        "__generated_wire0__",
                        "__generated_wire1__",
                        "__generated_wire2__",
                        "__generated_wire3__",
                        "__generated_wire4__",
                        2,
                    ]
                ),
            ),
        ),
    )
    def test_process_circuit(self, circ, expected_processed_circ, expected_circ_wires):
        """Test that the processed circuit is correct."""
        actual_processed_circ, actual_circ_wires = _process_circuit_lst(circ)

        assert actual_circ_wires == expected_circ_wires
        for elem1, elem2 in zip(actual_processed_circ, expected_processed_circ):
            assert (
                elem1[0].equal(elem2[0])
                if isinstance(elem1[0], MarkClean)
                else elem1[0] == elem2[0]
            )
            assert elem1[1] == elem2[1]


class TestEstimateAuxiliaryWires:
    """Test the private helper function _estimate_auxiliary_wires"""

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
            match="Failed to uncompute and restore all `ANY state` allocations.",
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

    def test_error_when_not_enough_any_state_aux_provided(self):
        """Test that an error is raised if the number of local_num_available_any_state_aux is negative."""

        with pytest.raises(
            ValueError, match="`local_num_available_any_state_aux` shouldn't be negative,"
        ):
            _estimate_auxiliary_wires(
                list_actions=[GateCount(qre.resource_rep(qre.X))],
                scalar=1,
                num_available_any_state_aux=0,  # should be atleast 1 (corresponding to the number of qubits X acts on)
                num_active_qubits=1,  # X acts on one qubit
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
                1,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                2,  # number of any state auxiliaries,
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
                4,
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
                4,
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
                6,
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
                6,
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
                6,
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
                6,
                6,
                (69, -5, 56),
            ),
        ),
    )
    def test_nested_allocation_and_deallocation(
        self, lst_actions, scalar, num_aux, num_active, expected_results
    ):
        """Test that qubit tracking works as expected for circuits with operators whose
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

    def test_estimator_allocation_and_deallocation(self):
        """Test that we can accurately track instaces of Allocate and Deallocate
        defined in the ``pennylane.estimator`` module."""

        list_actions = [  # Allocation and deallocation with scaling
            qre.Allocate(3),
            GateCount(qre.resource_rep(qre.CNOT)),
            qre.Deallocate(2),
            GateCount(qre.resource_rep(qre.Z), 2),
        ]
        scalar = 5  # Scalar
        num_active = 2  # number of active qubits
        num_aux = 2  # number of any state auxiliaries,
        expected_results = (7, 0, 5)  # max alloc, max dealloc, total

        results = _estimate_auxiliary_wires(
            list_actions=list_actions,
            scalar=scalar,
            num_available_any_state_aux=num_aux,
            num_active_qubits=num_active,
        )
        assert results == expected_results


class TestEstimateWiresFromCircuit:
    """Test that we can correctly estimate the number of wires
    from a list of operators."""

    @pytest.mark.parametrize(
        "circuit, expected_wire_counts",  # [Op1, ..., OpN], (algo, any_state, zeroed)
        (
            (  # Test with all labels: There are as many wires as unique wire labels
                [
                    qre.Hadamard(wires=0),
                    qre.CNOT(wires=[0, 1]),
                    qre.Toffoli(wires=[0, 1, 2]),
                ],
                (3, 0, 0),
            ),
            (
                [
                    qre.Hadamard(wires=0),
                    qre.CNOT(wires=[2, 3]),
                    qre.X(wires=1),
                    qre.Y(wires=4),
                    qre.Z(wires=1),
                    qre.Toffoli(wires=[2, 3, 4]),
                ],
                (5, 0, 0),
            ),
            (
                [
                    qre.QFT(wires=range(5)),
                    qre.Hadamard(wires=1),
                    qre.Z(wires="a"),
                    qre.Hadamard(wires=1),
                    qre.CNOT(wires=[3, 2]),
                    qre.RX(wires="b"),
                    qre.Hadamard(wires=2),
                    qre.RZ(wires="c"),
                ],
                (8, 0, 0),
            ),
            (  # Test with no labels: There are as many wires as the most required wires for any single operator
                [
                    qre.Hadamard(),
                    qre.CNOT(),
                    qre.Toffoli(),
                ],
                (3, 0, 0),
            ),
            (
                [
                    qre.Hadamard(),
                    qre.CNOT(),
                    qre.X(),
                    qre.Y(),
                    qre.Z(),
                    qre.Toffoli(),
                ],
                (3, 0, 0),
            ),
            (
                [
                    qre.QFT(num_wires=5),
                    qre.Hadamard(),
                    qre.Z(),
                    qre.Hadamard(),
                    qre.CNOT(),
                    qre.RX(),
                    qre.Hadamard(),
                    qre.RZ(),
                ],
                (5, 0, 0),
            ),
            (  # Test with mixed labels: There are as many wires as unique wire labels + most required wires for any single (unlabled) operator
                [
                    qre.Hadamard(wires=0),
                    qre.CNOT(),
                    qre.Toffoli(),
                ],
                (4, 0, 0),
            ),
            (
                [
                    qre.Hadamard(),
                    qre.CNOT(wires=[2, 3]),
                    qre.X(),
                    qre.Y(),
                    qre.Z(),
                    qre.Toffoli(wires=[2, 3, 4]),
                ],
                (4, 0, 0),
            ),
            (
                [
                    qre.QFT(wires=range(5)),
                    qre.Hadamard(),
                    qre.Z(),
                    qre.Hadamard(wires=1),
                    qre.CNOT(),
                    qre.RX(wires="b"),
                    qre.Hadamard(),
                    qre.RZ(),
                ],
                (8, 0, 0),
            ),
        ),
    )
    def test_algo_wires(self, circuit, expected_wire_counts):
        """Test that the number of algorithmic wires is correctly determined
        from the wire labels of operators."""
        wire_counts = estimate_wires_from_circuit(circuit, zeroed=0, any_state=0)
        assert wire_counts == expected_wire_counts

    @pytest.mark.parametrize(
        "circuit, expected_wire_counts",  # [Op1, ..., OpN], (algo, any_state, zeroed)
        (
            (
                [
                    AllocateOp(Allocate(3)),
                    qre.CNOT(wires=[0, 1]),
                    AllocateOp(Allocate(2)),
                    qre.Z(),
                    qml.Z(wires=1),
                ],
                (3, 5, 0),
            ),
            (
                [
                    qml.QFT(wires=[0, 1, 2, 3, 4]),
                    AllocateOp(Allocate(3)),
                    qre.CNOT(),
                    DeallocateOp(Deallocate(2)),
                    qre.Z(),
                ],
                (7, 1, 2),
            ),
            (
                [
                    qre.QFT(num_wires=5, wires=range(5)),
                    AllocateOp(Allocate(3)),
                    qre.CNOT(wires=[2, 3]),
                    DeallocateOp(Deallocate(2)),
                    qre.Z(wires=0),
                    AllocateOp(Allocate(3)),
                    qml.PhaseShift(1.23, wires=1),
                ],
                (5, 4, 0),
            ),
            (  # Test nested allocate and deallocate
                [
                    AllocateOp(Allocate(1)),
                    qre.X(wires=[0]),
                    qre.Prod(
                        (
                            (
                                qre.Prod(
                                    (
                                        (AllocateOp(Allocate(1)), 2),
                                        qre.CNOT(),
                                        (qre.Y(), 2),
                                        DeallocateOp(Deallocate(1)),
                                    ),
                                ),
                                3,
                            ),
                        ),
                        wires=[0, 1, 2],
                    ),
                    qre.QFT(wires=[0, 1, 2]),
                    qre.Prod(
                        (
                            (
                                qre.Prod(
                                    (
                                        AllocateOp(Allocate(1)),
                                        (qre.Y(), 2),
                                        qre.CNOT(),
                                        (DeallocateOp(Deallocate(1)), 2),
                                    ),
                                ),
                                3,
                            ),
                        ),
                        wires=[0, 1, 2],
                    ),
                ],
                (3, 1, 4),
            ),
        ),
    )
    def test_allocate_deallocate(self, circuit, expected_wire_counts):
        """Test that the number of allocated qubits (zeroed or any state) is correct."""
        wire_counts = estimate_wires_from_circuit(circuit, zeroed=0, any_state=0)
        assert wire_counts == expected_wire_counts

    @pytest.mark.parametrize(
        "circuit, expected_wire_counts",  # [Op1, ..., OpN], (algo, any_state, zeroed)
        (
            (  # Borrow idle
                [  # 4 available idle qubits of 6 qubits total
                    qml.CNOT(wires=["busy_wire1", "busy_wire2"]),
                    AllocOpFree(
                        num_wires=1, allocate=Allocate(3), op=qre.X(), wires=["busy_wire2"]
                    ),
                    qre.QFT(num_wires=4),
                ],
                (6, 0, 0),  # The idle qubits are used in place of the allocation!
            ),
            (
                [  # 4 available idle qubits of 6 qubits total
                    AllocateOp(Allocate(2)),
                    qml.CNOT(wires=["busy_wire1", "busy_wire2"]),
                    AllocOpFree(
                        num_wires=1, allocate=Allocate(5), op=qre.X(), wires=["busy_wire2"]
                    ),
                    qre.QFT(num_wires=4),
                    DeallocateOp(Deallocate(1)),
                ],
                (6, 1, 2),  # We use as many idle qubits as possible, then allocate the rest
            ),
            (  # Borrow + Mark clean
                [  # 4 + 1 available idle qubits of 6 qubits total
                    AllocateOp(Allocate(2)),
                    qml.CNOT(wires=["busy_wire1", "busy_wire2"]),
                    MarkClean(wires=["busy_wire1"]),  # busy_wire1 is treated as clean and idle
                    AllocOpFree(
                        num_wires=1, allocate=Allocate(5), op=qre.X(), wires=["busy_wire2"]
                    ),
                    qre.QFT(num_wires=4),
                    DeallocateOp(Deallocate(1)),
                ],
                (6, 1, 1),  # We needed to allocate one fewer wire!
            ),
            (
                [
                    qre.QFT(wires=[0, 1, 2, 3, 4]),
                    MarkClean(wires=[0, 1, 2, 3, 4]),  # Marked the qubits as clean
                    AllocOpFree(num_wires=1, allocate=Allocate(3), op=qre.X(), wires=[0, 1]),
                    qre.QFT(wires=[0, 1, 2, 3, 4]),
                    MarkClean(wires=[0, 1, 2, 3, 4]),  # Same qubits can be marked and reused
                    qre.Prod(
                        (
                            AllocateOp(Allocate(2)),
                            qre.Toffoli(wires=[2, 3, 4]),
                            DeallocateOp(Deallocate(2)),
                        )
                    ),
                ],
                (5, 0, 0),
            ),
        ),
    )
    def test_borrow_idle(self, circuit, expected_wire_counts):
        """Test that idle algorithmic qubits are used over qubit allocation where ever possible."""
        wire_counts = estimate_wires_from_circuit(circuit, zeroed=0, any_state=0)
        assert wire_counts == expected_wire_counts

    @pytest.mark.parametrize(
        "circuit, expected_wire_counts",  # [Op1, ..., OpN], (algo, any_state, zeroed)
        (
            (  # Borrow "any" state
                [
                    qre.CNOT(wires=[1, 2]),
                    AllocOpFree(
                        num_wires=2,
                        allocate=Allocate(5, state=AllocateState.ANY, restored=True),
                        op=qre.CZ(),
                        wires=[1, 2],
                    ),
                ],
                (2, 0, 5),  # No available qubits to borrow
            ),
            (
                [
                    qre.QFT(num_wires=4),
                    AllocOpFree(
                        num_wires=2,
                        allocate=Allocate(5, state=AllocateState.ANY, restored=True),
                        op=qre.CZ(),
                        wires=[1, 2],
                    ),
                ],
                (6, 0, 1),  # 4 qubits were borrowed from QFT because it acts on a distinct register
            ),
            (
                [  # Can borrow algorithmic wires and allocated wires from a higher scope
                    qml.QFT(wires=[0, 1, 2, 3, 4, 5]),
                    AllocOpFree(
                        num_wires=2,
                        allocate=Allocate(4, state=AllocateState.ANY, restored=True),
                        op=qre.Z(),
                        wires=[0, 1],
                    ),  # We borrow wires 2, 3, 4, 5
                    AllocateOp(Allocate(3)),  # This request 3 new clean qubits
                    qre.Prod(
                        (
                            AllocOpFree(
                                num_wires=4,
                                allocate=Allocate(5, state=AllocateState.ANY, restored=True),
                                op=qre.QFT(num_wires=9),
                                wires=[0, 1, 2, 3],  # borrow wires 4, 5 + the 3 allocated
                            ),
                            qre.Toffoli(wires=[3, 4, 5]),
                        ),
                    ),
                ],
                (6, 3, 0),
            ),
        ),
    )
    def test_borrow_any_state(self, circuit, expected_wire_counts):
        """Test that idle algorithmic qubits or allocated qubits from a higher scope are
        used as auxiliaries when we allocate any state qubits with a promise to restore them."""
        wire_counts = estimate_wires_from_circuit(circuit, zeroed=0, any_state=0)
        assert wire_counts == expected_wire_counts

    def test_deallocate_error(self):
        """Test that deallocating more qubits than we initially allocated results in an error"""
        circ_lst = [
            AllocateOp(Allocate(2)),
            qre.CNOT(),
            DeallocateOp(Deallocate(3)),
            AllocateOp(Allocate(5)),
        ]

        with pytest.raises(ValueError, match="Deallocated more qubits than available to allocate."):
            estimate_wires_from_circuit(circ_lst)

    @pytest.mark.parametrize(
        "circuit, initial_zeored, initial_any_state, expected_wire_counts",
        (
            (
                [
                    qre.QFT(num_wires=5, wires=range(5)),
                    AllocateOp(Allocate(3)),
                    qre.CNOT(wires=[2, 3]),
                    DeallocateOp(Deallocate(2)),
                    qre.Z(wires=0),
                    AllocateOp(Allocate(3)),
                    qml.PhaseShift(1.23, wires=1),
                ],
                0,
                2,
                (5, 6, 0),
            ),
            (
                [
                    qre.QFT(num_wires=5, wires=range(5)),
                    AllocateOp(Allocate(3)),
                    qre.CNOT(wires=[2, 3]),
                    DeallocateOp(Deallocate(2)),
                    qre.Z(wires=0),
                    AllocateOp(Allocate(3)),
                    qml.PhaseShift(1.23, wires=1),
                ],
                6,
                0,
                (5, 4, 2),
            ),
            (
                [
                    qre.QFT(num_wires=5, wires=range(5)),
                    AllocateOp(Allocate(3)),
                    qre.CNOT(wires=[2, 3]),
                    DeallocateOp(Deallocate(2)),
                    qre.Z(wires=0),
                    AllocateOp(Allocate(3)),
                    qml.PhaseShift(1.23, wires=1),
                ],
                6,
                2,
                (5, 6, 2),
            ),
            (
                [  # Can borrow algorithmic wires and allocated wires from a higher scope
                    qml.QFT(wires=[0, 1, 2, 3, 4, 5]),
                    AllocOpFree(
                        num_wires=4,
                        allocate=Allocate(4, state=AllocateState.ANY, restored=True),
                        op=qre.Z(),
                        wires=[0, 1, 4, 5],
                    ),  # We borrow wires 2, 3,
                    AllocateOp(Allocate(3)),  # This requests 3 new clean qubits
                    qre.Prod(
                        (
                            AllocOpFree(
                                num_wires=5,
                                allocate=Allocate(6, state=AllocateState.ANY, restored=True),
                                op=qre.QFT(num_wires=9),
                                wires=[1, 2, 3, 4, 5],  # borrow wire "0" + the 3 allocated before
                            ),
                            qre.Toffoli(wires=[3, 4, 5]),
                        ),
                    ),
                    DeallocateOp(Deallocate(2)),
                ],
                0,
                0,
                (6, 1, 4),
            ),
            (
                [  # Can borrow algorithmic wires and allocated wires from a higher scope
                    qml.QFT(wires=[0, 1, 2, 3, 4, 5]),
                    AllocOpFree(
                        num_wires=4,
                        allocate=Allocate(4, state=AllocateState.ANY, restored=True),
                        op=qre.Z(),
                        wires=[0, 1, 4, 5],
                    ),  # We borrow wires 2, 3,
                    AllocateOp(Allocate(3)),  # This requests 3 new clean qubits
                    qre.Prod(
                        (
                            AllocOpFree(
                                num_wires=5,
                                allocate=Allocate(6, state=AllocateState.ANY, restored=True),
                                op=qre.QFT(num_wires=9),
                                wires=[1, 2, 3, 4, 5],  # borrow wire "0" + the 3 allocated before
                            ),
                            qre.Toffoli(wires=[3, 4, 5]),
                        ),
                    ),
                    DeallocateOp(Deallocate(2)),
                ],
                2,
                0,
                (6, 1, 4),
            ),
            (
                [  # Can borrow algorithmic wires and allocated wires from a higher scope
                    qml.QFT(wires=[0, 1, 2, 3, 4, 5]),
                    AllocOpFree(
                        num_wires=4,
                        allocate=Allocate(4, state=AllocateState.ANY, restored=True),
                        op=qre.Z(),
                        wires=[0, 1, 4, 5],
                    ),  # We borrow wires 2, 3,
                    AllocateOp(Allocate(3)),  # This requests 3 new clean qubits
                    qre.Prod(
                        (
                            AllocOpFree(
                                num_wires=5,
                                allocate=Allocate(6, state=AllocateState.ANY, restored=True),
                                op=qre.QFT(num_wires=9),
                                wires=[1, 2, 3, 4, 5],  # borrow wire "0" + the 3 allocated before
                            ),
                            qre.Toffoli(wires=[3, 4, 5]),
                        ),
                    ),
                    DeallocateOp(Deallocate(2)),
                ],
                7,
                0,
                (6, 1, 6),
            ),
            (
                [  # Can borrow algorithmic wires and allocated wires from a higher scope
                    qml.QFT(wires=[0, 1, 2, 3, 4, 5]),
                    AllocOpFree(
                        num_wires=4,
                        allocate=Allocate(4, state=AllocateState.ANY, restored=True),
                        op=qre.Z(),
                        wires=[0, 1, 4, 5],
                    ),  # We borrow wires 2, 3,
                    AllocateOp(Allocate(3)),  # This requests 3 new clean qubits
                    qre.Prod(
                        (
                            AllocOpFree(
                                num_wires=5,
                                allocate=Allocate(6, state=AllocateState.ANY, restored=True),
                                op=qre.QFT(num_wires=9),
                                wires=[1, 2, 3, 4, 5],  # borrow wire "0" + the 3 allocated before
                            ),
                            qre.Toffoli(wires=[3, 4, 5]),
                        ),
                    ),
                    DeallocateOp(Deallocate(2)),
                ],
                0,
                2,
                (6, 3, 2),
            ),
            (
                [  # Can borrow algorithmic wires and allocated wires from a higher scope
                    qml.QFT(wires=[0, 1, 2, 3, 4, 5]),
                    AllocOpFree(
                        num_wires=4,
                        allocate=Allocate(4, state=AllocateState.ANY, restored=True),
                        op=qre.Z(),
                        wires=[0, 1, 4, 5],
                    ),  # We borrow wires 2, 3,
                    AllocateOp(Allocate(3)),  # This requests 3 new clean qubits
                    qre.Prod(
                        (
                            AllocOpFree(
                                num_wires=5,
                                allocate=Allocate(6, state=AllocateState.ANY, restored=True),
                                op=qre.QFT(num_wires=9),
                                wires=[1, 2, 3, 4, 5],  # borrow wire "0" + the 3 allocated before
                            ),
                            qre.Toffoli(wires=[3, 4, 5]),
                        ),
                    ),
                    DeallocateOp(Deallocate(2)),
                ],
                0,
                5,
                (6, 6, 2),
            ),
            (
                [  # Can borrow algorithmic wires and allocated wires from a higher scope
                    qml.QFT(wires=[0, 1, 2, 3, 4, 5]),
                    AllocOpFree(
                        num_wires=4,
                        allocate=Allocate(4, state=AllocateState.ANY, restored=True),
                        op=qre.Z(),
                        wires=[0, 1, 4, 5],
                    ),  # We borrow wires 2, 3,
                    AllocateOp(Allocate(3)),  # This requests 3 new clean qubits
                    qre.Prod(
                        (
                            AllocOpFree(
                                num_wires=5,
                                allocate=Allocate(6, state=AllocateState.ANY, restored=True),
                                op=qre.QFT(num_wires=9),
                                wires=[1, 2, 3, 4, 5],  # borrow wire "0" + the 3 allocated before
                            ),
                            qre.Toffoli(wires=[3, 4, 5]),
                        ),
                    ),
                    DeallocateOp(Deallocate(2)),
                ],
                1,
                1,
                (6, 2, 3),
            ),
        ),
    )
    def test_zeroed_any_state_wires(
        self, circuit, initial_zeored, initial_any_state, expected_wire_counts
    ):
        """Test that we can correctly resolve the number of qubits when we provide an intial
        number of zeroed and any state qubits."""
        wire_counts = estimate_wires_from_circuit(
            circuit, zeroed=initial_zeored, any_state=initial_any_state
        )
        assert wire_counts == expected_wire_counts


class TestEstimateWiresFromResources:
    """Test that we can correctly estimate the wires from a Resources object."""

    @pytest.mark.parametrize(
        "resources, expected_wire_counts",  # {Op1: n_1, ...} (num_any_state, num_zeroed)
        (
            (
                Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=4,
                    gate_types={
                        AllocateOp.resource_rep(Allocate(3)): 1,
                        qre.CNOT.resource_rep(): 2,
                        AllocateOp.resource_rep(Allocate(1)): 2,
                        qre.Z.resource_rep(): 2,
                    },
                ),
                (5, 0),
            ),
            (
                Resources(
                    zeroed_wires=3,
                    any_state_wires=1,
                    algo_wires=4,
                    gate_types={
                        AllocateOp.resource_rep(Allocate(3)): 1,
                        qre.CNOT.resource_rep(): 2,
                        AllocateOp.resource_rep(Allocate(1)): 2,
                        qre.Z.resource_rep(): 2,
                    },
                ),
                (6, 0),
            ),
            (
                Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=5,
                    gate_types={
                        qre.QFT.resource_rep(5): 1,
                        AllocateOp.resource_rep(Allocate(2)): 1,
                        qre.CNOT.resource_rep(): 2,
                        AllocateOp.resource_rep(Allocate(1)): 1,
                        DeallocateOp.resource_rep(Deallocate(2)): 1,
                        qre.Z.resource_rep(): 2,
                    },
                ),
                (1, 2),
            ),
            (
                Resources(
                    zeroed_wires=2,
                    any_state_wires=1,
                    algo_wires=5,
                    gate_types={
                        qre.QFT.resource_rep(5): 1,
                        AllocateOp.resource_rep(Allocate(2)): 1,
                        qre.CNOT.resource_rep(): 2,
                        AllocateOp.resource_rep(Allocate(1)): 1,
                        DeallocateOp.resource_rep(Deallocate(2)): 1,
                        qre.Z.resource_rep(): 2,
                    },
                ),
                (2, 2),
            ),
            (
                Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=3,
                    gate_types={
                        AllocateOp.resource_rep(Allocate(1)): 1,
                        qre.X.resource_rep(): 1,
                        qre.Prod.resource_rep(
                            (
                                (AllocateOp.resource_rep(Allocate(1)), 2),
                                (qre.CNOT.resource_rep(), 1),
                                (qre.Y.resource_rep(), 2),
                                (DeallocateOp.resource_rep(Deallocate(1)), 1),
                            ),
                            num_wires=3,
                        ): 3,
                        qre.QFT.resource_rep(3): 1,
                        qre.Prod.resource_rep(
                            (
                                (AllocateOp.resource_rep(Allocate(1)), 1),
                                (qre.Y.resource_rep(), 2),
                                (qre.CNOT.resource_rep(), 1),
                                (DeallocateOp.resource_rep(Deallocate(1)), 2),
                            ),
                            num_wires=3,
                        ): 3,
                    },
                ),
                (1, 4),
            ),
            (
                Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=6,
                    gate_types={
                        AllocateOp.resource_rep(Allocate(2)): 2,
                        qre.CNOT.resource_rep(): 1,
                        AllocOpFree.resource_rep(
                            num_wires=1, allocate=Allocate(5), cmpr_op=qre.X.resource_rep()
                        ): 5,
                        qre.QFT.resource_rep(num_wires=4): 1,
                        DeallocateOp.resource_rep(Deallocate(1)): 3,
                    },
                ),
                (1, 8),
            ),
            (
                Resources(
                    zeroed_wires=5,
                    any_state_wires=3,
                    algo_wires=6,
                    gate_types={
                        AllocateOp.resource_rep(Allocate(2)): 2,
                        qre.CNOT.resource_rep(): 1,
                        AllocOpFree.resource_rep(
                            num_wires=1, allocate=Allocate(5), cmpr_op=qre.X.resource_rep()
                        ): 5,
                        qre.QFT.resource_rep(num_wires=4): 1,
                        DeallocateOp.resource_rep(Deallocate(1)): 3,
                    },
                ),
                (4, 8),
            ),
            (
                Resources(
                    zeroed_wires=10,
                    any_state_wires=3,
                    algo_wires=6,
                    gate_types={
                        AllocateOp.resource_rep(Allocate(2)): 2,
                        qre.CNOT.resource_rep(): 1,
                        AllocOpFree.resource_rep(
                            num_wires=1, allocate=Allocate(5), cmpr_op=qre.X.resource_rep()
                        ): 5,
                        qre.QFT.resource_rep(num_wires=4): 1,
                        DeallocateOp.resource_rep(Deallocate(1)): 3,
                    },
                ),
                (4, 9),
            ),
            (
                Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(5, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                (0, 1),
            ),
            (
                Resources(
                    zeroed_wires=4,
                    any_state_wires=0,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(10, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                (0, 6),
            ),
            (
                Resources(
                    zeroed_wires=10,
                    any_state_wires=0,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(10, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                (0, 10),
            ),
            (
                Resources(
                    zeroed_wires=0,
                    any_state_wires=6,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(10, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                (6, 0),
            ),
            (
                Resources(
                    zeroed_wires=3,
                    any_state_wires=3,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(10, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                (3, 3),
            ),
        ),
    )
    def test_allocate_deallocate(self, resources, expected_wire_counts):
        """Test that the number of allocated qubits (any state or zeroed) is correct."""
        wire_counts = estimate_wires_from_resources(resources, zeroed=0, any_state=0)
        assert wire_counts == expected_wire_counts

    def test_deallocate_error(self):
        """Test that an error is raised if more qubits are deallocated than available."""
        circ_res = Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=4,
            gate_types={
                AllocateOp.resource_rep(Allocate(2)): 1,
                qre.CNOT.resource_rep(): 2,
                DeallocateOp.resource_rep(Deallocate(1)): 3,
                AllocateOp.resource_rep(Allocate(5)): 1,
            },
        )

        with pytest.raises(ValueError, match="Deallocated more qubits than available to allocate."):
            estimate_wires_from_resources(circ_res)

    @pytest.mark.parametrize(
        "resources, initial_zeored, initial_any_state, expected_wire_counts",
        (
            (
                Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(5, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                0,
                0,
                (0, 1),
            ),
            (
                Resources(
                    zeroed_wires=1,
                    any_state_wires=0,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(10, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                3,
                0,
                (0, 6),
            ),
            (
                Resources(
                    zeroed_wires=4,
                    any_state_wires=0,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(10, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                6,
                0,
                (0, 10),
            ),
            (
                Resources(
                    zeroed_wires=0,
                    any_state_wires=3,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(10, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                0,
                3,
                (6, 0),
            ),
            (
                Resources(
                    zeroed_wires=1,
                    any_state_wires=2,
                    algo_wires=6,
                    gate_types={
                        qre.QFT.resource_rep(num_wires=4): 1,
                        AllocOpFree.resource_rep(
                            num_wires=2,
                            allocate=Allocate(10, state=AllocateState.ANY, restored=True),
                            cmpr_op=qre.CZ.resource_rep(),
                        ): 3,
                        qre.Z.resource_rep(): 5,
                    },
                ),
                2,
                1,
                (3, 3),
            ),
        ),
    )
    def test_zeroed_any_state_wires(
        self, resources, initial_zeored, initial_any_state, expected_wire_counts
    ):
        """Test that we can correctly resolve the number of qubits when we provide an intial
        number of zeroed and any state qubits."""
        wire_counts = estimate_wires_from_resources(
            resources, zeroed=initial_zeored, any_state=initial_any_state
        )
        assert wire_counts == expected_wire_counts

# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the tracking capabilities of default qutrit mixed.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.resource import Resources


class TestTracking:
    """Testing the tracking capabilities of DefaultQutritMixed."""

    def test_tracker_set_upon_initialization(self):
        """Test that a new tracker is initialized with each device."""
        tracker_1 = qml.device("default.qutrit.mixed").tracker
        tracker_2 = qml.device("default.qutrit.mixed").tracker
        assert tracker_1 is not tracker_2

    def test_tracker_not_updated_if_not_active(self):
        """Test that the tracker is not updated if not active."""
        dev = qml.device("default.qutrit.mixed")
        assert len(dev.tracker.totals) == 0

        dev.execute(qml.tape.QuantumScript())
        assert len(dev.tracker.totals) == 0
        assert len(dev.tracker.history) == 0

    def test_tracking(self):
        """Test that the new default qutrit mixed integrates with the tracker."""
        qs = qml.tape.QuantumScript([], [qml.expval(qml.GellMann(0, 3))])

        dev = qml.device("default.qutrit.mixed")
        with qml.Tracker(dev) as tracker:
            dev.execute(qs)

        assert tracker.history == {
            "batches": [1],
            "executions": [1],
            "simulations": [1],
            "results": [1.0],
            "resources": [Resources(num_wires=1)],
            "errors": [{}],
        }
        assert tracker.totals == {
            "batches": 1,
            "executions": 1,
            "results": 1.0,
            "simulations": 1,
        }
        assert tracker.latest == {
            "executions": 1,
            "simulations": 1,
            "results": 1,
            "resources": Resources(num_wires=1),
            "errors": {},
        }

    def test_tracking_resources(self):
        """Test that resources are tracked for the new default qutrit mixed device."""
        qs = qml.tape.QuantumScript(
            [
                qml.THadamard(0),
                qml.THadamard(1),
                qml.TAdd(wires=[0, 2]),
                qml.TRZ(1.23, 1, subspace=(0, 2)),
                qml.TAdd(wires=[1, 2]),
                qml.THadamard(0),
            ],
            [qml.expval(qml.GellMann(1, 8)), qml.expval(qml.GellMann(2, 7))],
        )

        expected_resources = Resources(
            num_wires=3,
            num_gates=6,
            gate_types={"THadamard": 3, "TAdd": 2, "TRZ": 1},
            gate_sizes={1: 4, 2: 2},
            depth=3,
        )

        dev = qml.device("default.qutrit.mixed")
        with qml.Tracker(dev) as tracker:
            dev.execute(qs)

        assert len(tracker.history["resources"]) == 1
        assert tracker.history["resources"][0] == expected_resources

    def test_tracking_batched_execution(self):
        """Test the number of times the device is executed over a QNode's
        lifetime is tracked by the device's tracker."""

        dev_1 = qml.device("default.qutrit.mixed", wires=2)

        def circuit_1(x, y):
            qml.TRX(x, wires=[0])
            qml.TRY(y, wires=[1])
            qml.TAdd(wires=[0, 1])
            return qml.expval(qml.GellMann(0, 3) @ qml.GellMann(1, 4))

        node_1 = qml.QNode(circuit_1, dev_1)
        num_evals_1 = 10

        with qml.Tracker(dev_1, persistent=True) as tracker1:
            for _ in range(num_evals_1):
                node_1(0.432, np.array([0.12, 0.5, 3.2]))
        assert tracker1.totals["executions"] == 3 * num_evals_1
        assert tracker1.totals["simulations"] == num_evals_1

        # test a second instance of a default qutrit mixed device
        dev_2 = qml.device("default.qutrit.mixed", wires=2)

        def circuit_2(x):
            qml.TRX(x, wires=[0])
            qml.TAdd(wires=[0, 1])
            return qml.expval(qml.GellMann(0, 3) @ qml.GellMann(1, 4))

        node_2 = qml.QNode(circuit_2, dev_2)
        num_evals_2 = 5

        with qml.Tracker(dev_2) as tracker2:
            for _ in range(num_evals_2):
                node_2(np.array([0.432, 0.61, 8.2]))
        assert tracker2.totals["simulations"] == num_evals_2
        assert tracker2.totals["executions"] == 3 * num_evals_2

        # test a new circuit on an existing instance of a qutrit mixed device
        def circuit_3(y):
            qml.TRY(y, wires=[1])
            qml.TAdd(wires=[0, 1])
            return qml.expval(qml.GellMann(0, 3) @ qml.GellMann(1, 4))

        node_3 = qml.QNode(circuit_3, dev_1)
        num_evals_3 = 7

        with tracker1:
            for _ in range(num_evals_3):
                node_3(np.array([0.12, 1.214]))
        assert tracker1.totals["simulations"] == num_evals_1 + num_evals_3
        assert tracker1.totals["executions"] == 3 * num_evals_1 + 2 * num_evals_3


shot_testing_combos = [
    # expval combinations
    ([qml.expval(qml.GellMann(0, 1))], 1, 10),
    ([qml.expval(qml.GellMann(0, 1)), qml.expval(qml.GellMann(0, 2))], 2, 20),
    # Hamiltonian test cases
    ([qml.expval(qml.Hamiltonian([1, 1], [qml.GellMann(0, 1), qml.GellMann(1, 5)]))], 1, 10),
    # op arithmetic test cases
    ([qml.expval(qml.sum(qml.GellMann(0, 1), qml.GellMann(1, 4)))], 2, 20),
    (
        [
            qml.expval(qml.prod(qml.GellMann(0, 1), qml.GellMann(1, 4))),
            qml.expval(qml.prod(qml.GellMann(1, 4), qml.GellMann(2, 7))),
        ],
        2,
        20,
    ),
    # computational basis measurements
    ([qml.sample(wires=(0, 1))], 1, 10),
    ([qml.sample(wires=(0, 1)), qml.expval(qml.GellMann(0, 1))], 2, 20),
]


class TestExecuteTracker:
    """Test that tracker tracks default qutrit mixed execute number of shots"""

    # pylint: disable=too-few-public-methods

    @pytest.mark.parametrize("mps, expected_exec, expected_shots", shot_testing_combos)
    def test_single_expval(self, mps, expected_exec, expected_shots):
        """Test tracker tracks default qutrit mixed execute number of shots for single measurements"""
        dev = qml.device("default.qutrit.mixed")
        tape = qml.tape.QuantumScript([], mps, shots=10)

        with dev.tracker:
            dev.execute(tape)

        assert dev.tracker.totals["executions"] == expected_exec
        assert dev.tracker.totals["simulations"] == 1
        assert dev.tracker.totals["shots"] == expected_shots

        tape = qml.tape.QuantumScript([qml.TRX((1.2, 2.3, 3.4), wires=0)], mps, shots=10)

        with dev.tracker:
            dev.execute(tape)

        assert dev.tracker.totals["executions"] == 3 * expected_exec
        assert dev.tracker.totals["simulations"] == 1
        assert dev.tracker.totals["shots"] == 3 * expected_shots

    def test_multiple_expval_with_prods(self):
        """
        Test tracker tracks default qutrit mixed execute number of shots for new and old opmath tensors.
        """
        mps, expected_exec, expected_shots = (
            [qml.expval(qml.GellMann(0, 1)), qml.expval(qml.GellMann(0, 1) @ qml.GellMann(1, 5))],
            2,
            20,
        )
        dev = qml.device("default.qutrit.mixed")
        tape = qml.tape.QuantumScript([], mps, shots=10)

        with dev.tracker:
            dev.execute(tape)

        assert dev.tracker.totals["executions"] == expected_exec
        assert dev.tracker.totals["simulations"] == 1
        assert dev.tracker.totals["shots"] == expected_shots

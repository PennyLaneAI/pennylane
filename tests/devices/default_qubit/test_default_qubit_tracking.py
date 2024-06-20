# Copyright 2023 Xanadu Quantum Technologies Inc.

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
Tests for the tracking capabilities of default qubit.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.resource import Resources


class TestTracking:
    """Testing the tracking capabilities of DefaultQubit."""

    def test_tracker_set_upon_initialization(self):
        """Test that a new tracker is intialized with each device."""
        assert qml.device("default.qubit").tracker is not qml.device("default.qubit").tracker

    def test_tracker_not_updated_if_not_active(self):
        """Test that the tracker is not updated if not active."""
        dev = qml.device("default.qubit")
        assert len(dev.tracker.totals) == 0

        dev.execute(qml.tape.QuantumScript())
        assert len(dev.tracker.totals) == 0
        assert len(dev.tracker.history) == 0

    def test_tracking_batch(self):
        """Test that the new default qubit integrates with the tracker."""

        qs = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])

        dev = qml.device("default.qubit")
        config = ExecutionConfig(gradient_method="adjoint")
        with qml.Tracker(dev) as tracker:
            dev.execute(qs)
            dev.compute_derivatives(qs, config)
            dev.execute([qs, qs])  # and a second time

        assert tracker.history == {
            "batches": [1, 1],
            "executions": [1, 1, 1],
            "simulations": [1, 1, 1],
            "results": [1.0, 1.0, 1.0],
            "resources": [Resources(num_wires=1), Resources(num_wires=1), Resources(num_wires=1)],
            "derivative_batches": [1],
            "derivatives": [1],
            "errors": [{}, {}, {}],
        }
        assert tracker.totals == {
            "batches": 2,
            "executions": 3,
            "results": 3.0,
            "simulations": 3,
            "derivative_batches": 1,
            "derivatives": 1,
        }
        assert tracker.latest == {
            "executions": 1,
            "simulations": 1,
            "results": 1,
            "resources": Resources(num_wires=1),
            "errors": {},
        }

    def test_tracking_execute_and_derivatives(self):
        """Test that the execute_and_compute_* calls are being tracked for the
        new default qubit device"""

        qs = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])
        dev = qml.device("default.qubit")
        config = ExecutionConfig(gradient_method="adjoint")

        with qml.Tracker(dev) as tracker:
            dev.compute_derivatives(qs, config)
            dev.execute_and_compute_derivatives([qs] * 2, config)
            dev.compute_jvp([qs] * 3, [(0,)] * 3, config)
            dev.execute_and_compute_jvp([qs] * 4, [(0,)] * 4, config)
            dev.compute_vjp([qs] * 5, [(0,)] * 5, config)
            dev.execute_and_compute_vjp([qs] * 6, [(0,)] * 6, config)

        assert tracker.history == {
            "executions": [2, 4, 6],
            "derivatives": [1, 2],
            "derivative_batches": [1],
            "execute_and_derivative_batches": [1],
            "jvps": [3, 4],
            "jvp_batches": [1],
            "execute_and_jvp_batches": [1],
            "vjps": [5, 6],
            "vjp_batches": [1],
            "execute_and_vjp_batches": [1],
            "resources": [Resources(num_wires=1)] * 12,
            "errors": [{}] * 12,
        }

    def test_tracking_resources(self):
        """Test that resources are tracked for the new default qubit device."""
        qs = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.CNOT(wires=[0, 2]),
                qml.RZ(1.23, 1),
                qml.CNOT(wires=[1, 2]),
                qml.Hadamard(0),
            ],
            [qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliY(2))],
        )

        expected_resources = Resources(
            num_wires=3,
            num_gates=6,
            gate_types={"Hadamard": 3, "CNOT": 2, "RZ": 1},
            gate_sizes={1: 4, 2: 2},
            depth=3,
        )

        dev = qml.device("default.qubit")
        with qml.Tracker(dev) as tracker:
            dev.execute(qs)

        assert len(tracker.history["resources"]) == 1
        assert tracker.history["resources"][0] == expected_resources

    def test_tracking_batched_execution(self):
        """Test the number of times the device is executed over a QNode's
        lifetime is tracked by the device's tracker."""

        dev_1 = qml.device("default.qubit", wires=2)

        def circuit_1(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node_1 = qml.QNode(circuit_1, dev_1)
        num_evals_1 = 10

        with qml.Tracker(dev_1, persistent=True) as tracker1:
            for _ in range(num_evals_1):
                node_1(0.432, np.array([0.12, 0.5, 3.2]))
        assert tracker1.totals["executions"] == 3 * num_evals_1
        assert tracker1.totals["simulations"] == num_evals_1

        # test a second instance of a default qubit device
        dev_2 = qml.device("default.qubit", wires=2)

        def circuit_2(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node_2 = qml.QNode(circuit_2, dev_2)
        num_evals_2 = 5

        with qml.Tracker(dev_2) as tracker2:
            for _ in range(num_evals_2):
                node_2(np.array([0.432, 0.61, 8.2]))
        assert tracker2.totals["simulations"] == num_evals_2
        assert tracker2.totals["executions"] == 3 * num_evals_2

        # test a new circuit on an existing instance of a qubit device
        def circuit_3(y):
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node_3 = qml.QNode(circuit_3, dev_1)
        num_evals_3 = 7

        with tracker1:
            for _ in range(num_evals_3):
                node_3(np.array([0.12, 1.214]))
        assert tracker1.totals["simulations"] == num_evals_1 + num_evals_3
        assert tracker1.totals["executions"] == 3 * num_evals_1 + 2 * num_evals_3


H0 = qml.Hamiltonian([1.0, 1.0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)])


shot_testing_combos = [
    # expval combinations
    ([qml.expval(qml.X(0))], 1, 10),
    ([qml.expval(qml.X(0)), qml.expval(qml.Y(0))], 2, 20),
    # Hamiltonian test cases
    ([qml.expval(qml.Hamiltonian([1, 0.5, 1], [qml.X(0), qml.Y(0), qml.X(1)]))], 2, 20),
    ([qml.expval(qml.Hamiltonian([1, 1], [qml.X(0), qml.X(1)], grouping_type="qwc"))], 1, 10),
    ([qml.expval(qml.Hamiltonian([1, 1], [qml.X(0), qml.Y(0)], grouping_type="qwc"))], 2, 20),
    # op arithmetic test cases
    ([qml.expval(qml.sum(qml.X(0), qml.Y(0)))], 2, 20),
    ([qml.expval(qml.sum(qml.X(0), qml.X(0) @ qml.X(1)))], 1, 10),
    ([qml.expval(qml.sum(qml.X(0), qml.Hadamard(0)))], 2, 20),
    ([qml.expval(qml.sum(qml.X(0), qml.Y(1) @ qml.X(1), grouping_type="qwc"))], 1, 10),
    (
        [
            qml.expval(qml.prod(qml.X(0), qml.X(1))),
            qml.expval(qml.prod(qml.X(1), qml.X(2))),
        ],
        1,
        10,
    ),
    # computational basis measurements
    ([qml.probs(wires=(0, 1)), qml.sample(wires=(0, 1))], 1, 10),
    ([qml.probs(wires=(0, 1)), qml.sample(wires=(0, 1)), qml.expval(qml.X(0))], 2, 20),
    # classical shadows
    ([qml.shadow_expval(H0)], 10, 10),
    ([qml.shadow_expval(H0), qml.probs(wires=(0, 1))], 11, 20),
    ([qml.classical_shadow(wires=(0, 1))], 10, 10),
]


@pytest.mark.parametrize("mps, expected_exec, expected_shots", shot_testing_combos)
def test_single_expval(mps, expected_exec, expected_shots):
    dev = qml.device("default.qubit")
    tape = qml.tape.QuantumScript([], mps, shots=10)

    with dev.tracker:
        dev.execute(tape)

    assert dev.tracker.totals["executions"] == expected_exec
    assert dev.tracker.totals["simulations"] == 1
    assert dev.tracker.totals["shots"] == expected_shots

    if not isinstance(
        tape.measurements[0], (qml.measurements.ShadowExpvalMP, qml.measurements.ClassicalShadowMP)
    ):
        tape = qml.tape.QuantumScript([qml.RX((1.2, 2.3, 3.4), wires=0)], mps, shots=10)

        with dev.tracker:
            dev.execute(tape)

        assert dev.tracker.totals["executions"] == 3 * expected_exec
        assert dev.tracker.totals["simulations"] == 1
        assert dev.tracker.totals["shots"] == 3 * expected_shots


def test_multiple_expval_with_prod():
    mps, expected_exec, expected_shots = (
        [qml.expval(qml.PauliX(0)), qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1)))],
        1,
        10,
    )
    dev = qml.device("default.qubit")
    tape = qml.tape.QuantumScript([], mps, shots=10)

    with dev.tracker:
        dev.execute(tape)

    assert dev.tracker.totals["executions"] == expected_exec
    assert dev.tracker.totals["simulations"] == 1
    assert dev.tracker.totals["shots"] == expected_shots

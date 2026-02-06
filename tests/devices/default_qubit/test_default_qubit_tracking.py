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

import pennylane as qp
from pennylane.devices import ExecutionConfig
from pennylane.resource import SpecsResources


class TestTracking:
    """Testing the tracking capabilities of DefaultQubit."""

    def test_tracker_set_upon_initialization(self):
        """Test that a new tracker is intialized with each device."""
        assert qp.device("default.qubit").tracker is not qp.device("default.qubit").tracker

    def test_tracker_not_updated_if_not_active(self):
        """Test that the tracker is not updated if not active."""
        dev = qp.device("default.qubit")
        assert len(dev.tracker.totals) == 0

        dev.execute(qp.tape.QuantumScript())
        assert len(dev.tracker.totals) == 0
        assert len(dev.tracker.history) == 0

    def test_tracking_batch(self):
        """Test that the new default qubit integrates with the tracker."""

        qs = qp.tape.QuantumScript([], [qp.expval(qp.PauliZ(0))])

        dev = qp.device("default.qubit")
        config = ExecutionConfig(gradient_method="adjoint")
        with qp.Tracker(dev) as tracker:
            dev.execute(qs)
            dev.compute_derivatives(qs, config)
            dev.execute([qs, qs])  # and a second time

        assert tracker.history == {
            "batches": [1, 1],
            "executions": [1, 1, 1],
            "simulations": [1, 1, 1],
            "results": [1.0, 1.0, 1.0],
            "resources": [
                SpecsResources(
                    num_allocs=1,
                    gate_types={},
                    gate_sizes={},
                    measurements={"expval(PauliZ)": 1},
                    depth=0,
                ),
                SpecsResources(
                    num_allocs=1,
                    gate_types={},
                    gate_sizes={},
                    measurements={"expval(PauliZ)": 1},
                    depth=0,
                ),
                SpecsResources(
                    num_allocs=1,
                    gate_types={},
                    gate_sizes={},
                    measurements={"expval(PauliZ)": 1},
                    depth=0,
                ),
            ],
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
            "resources": SpecsResources(
                num_allocs=1,
                gate_types={},
                gate_sizes={},
                measurements={"expval(PauliZ)": 1},
                depth=0,
            ),
            "errors": {},
        }

    def test_tracking_execute_and_derivatives(self):
        """Test that the execute_and_compute_* calls are being tracked for the
        new default qubit device"""

        qs = qp.tape.QuantumScript([], [qp.expval(qp.PauliZ(0))])
        dev = qp.device("default.qubit")
        config = ExecutionConfig(gradient_method="adjoint")

        with qp.Tracker(dev) as tracker:
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
            "resources": [
                SpecsResources(
                    num_allocs=1,
                    gate_types={},
                    gate_sizes={},
                    measurements={"expval(PauliZ)": 1},
                    depth=0,
                )
            ]
            * 12,
            "errors": [{}] * 12,
        }

    def test_tracking_resources(self):
        """Test that resources are tracked for the new default qubit device."""
        qs = qp.tape.QuantumScript(
            [
                qp.Hadamard(0),
                qp.Hadamard(1),
                qp.CNOT(wires=[0, 2]),
                qp.RZ(1.23, 1),
                qp.CNOT(wires=[1, 2]),
                qp.Hadamard(0),
            ],
            [qp.expval(qp.PauliZ(1)), qp.expval(qp.PauliY(2))],
        )

        expected_resources = SpecsResources(
            num_allocs=3,
            gate_types={"Hadamard": 3, "CNOT": 2, "RZ": 1},
            gate_sizes={1: 4, 2: 2},
            measurements={"expval(PauliZ)": 1, "expval(PauliY)": 1},
            depth=3,
        )

        dev = qp.device("default.qubit")
        with qp.Tracker(dev) as tracker:
            dev.execute(qs)

        assert len(tracker.history["resources"]) == 1
        assert tracker.history["resources"][0] == expected_resources

    def test_tracking_batched_execution(self):
        """Test the number of times the device is executed over a QNode's
        lifetime is tracked by the device's tracker."""

        dev_1 = qp.device("default.qubit", wires=2)

        def circuit_1(x, y):
            qp.RX(x, wires=[0])
            qp.RY(y, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

        node_1 = qp.QNode(circuit_1, dev_1)
        num_evals_1 = 10

        with qp.Tracker(dev_1, persistent=True) as tracker1:
            for _ in range(num_evals_1):
                node_1(0.432, np.array([0.12, 0.5, 3.2]))
        assert tracker1.totals["executions"] == 3 * num_evals_1
        assert tracker1.totals["simulations"] == num_evals_1

        # test a second instance of a default qubit device
        dev_2 = qp.device("default.qubit", wires=2)

        def circuit_2(x):
            qp.RX(x, wires=[0])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

        node_2 = qp.QNode(circuit_2, dev_2)
        num_evals_2 = 5

        with qp.Tracker(dev_2) as tracker2:
            for _ in range(num_evals_2):
                node_2(np.array([0.432, 0.61, 8.2]))
        assert tracker2.totals["simulations"] == num_evals_2
        assert tracker2.totals["executions"] == 3 * num_evals_2

        # test a new circuit on an existing instance of a qubit device
        def circuit_3(y):
            qp.RY(y, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))

        node_3 = qp.QNode(circuit_3, dev_1)
        num_evals_3 = 7

        with tracker1:
            for _ in range(num_evals_3):
                node_3(np.array([0.12, 1.214]))
        assert tracker1.totals["simulations"] == num_evals_1 + num_evals_3
        assert tracker1.totals["executions"] == 3 * num_evals_1 + 2 * num_evals_3


H0 = qp.Hamiltonian([1.0, 1.0], [qp.PauliZ(0) @ qp.PauliZ(1), qp.PauliX(0) @ qp.PauliX(1)])


shot_testing_combos = [
    # expval combinations
    ([qp.expval(qp.X(0))], 1, 10),
    ([qp.expval(qp.X(0)), qp.expval(qp.Y(0))], 2, 20),
    # Hamiltonian test cases
    ([qp.expval(qp.Hamiltonian([1, 0.5, 1], [qp.X(0), qp.Y(0), qp.X(1)]))], 2, 20),
    ([qp.expval(qp.Hamiltonian([1, 1], [qp.X(0), qp.X(1)], grouping_type="qwc"))], 1, 10),
    ([qp.expval(qp.Hamiltonian([1, 1], [qp.X(0), qp.Y(0)], grouping_type="qwc"))], 2, 20),
    # op arithmetic test cases
    ([qp.expval(qp.sum(qp.X(0), qp.Y(0)))], 2, 20),
    ([qp.expval(qp.sum(qp.X(0), qp.X(0) @ qp.X(1)))], 1, 10),
    ([qp.expval(qp.sum(qp.X(0), qp.Hadamard(0)))], 2, 20),
    ([qp.expval(qp.sum(qp.X(0), qp.Y(1) @ qp.X(1), grouping_type="qwc"))], 1, 10),
    (
        [
            qp.expval(qp.prod(qp.X(0), qp.X(1))),
            qp.expval(qp.prod(qp.X(1), qp.X(2))),
        ],
        1,
        10,
    ),
    # computational basis measurements
    ([qp.probs(wires=(0, 1)), qp.sample(wires=(0, 1))], 1, 10),
    ([qp.probs(wires=(0, 1)), qp.sample(wires=(0, 1)), qp.expval(qp.X(0))], 2, 20),
    # classical shadows
    ([qp.shadow_expval(H0)], 10, 10),
    ([qp.shadow_expval(H0), qp.probs(wires=(0, 1))], 11, 20),
    ([qp.classical_shadow(wires=(0, 1))], 10, 10),
]


@pytest.mark.parametrize("mps, expected_exec, expected_shots", shot_testing_combos)
def test_single_expval(mps, expected_exec, expected_shots):
    dev = qp.device("default.qubit")
    tape = qp.tape.QuantumScript([], mps, shots=10)

    with dev.tracker:
        dev.execute(tape)

    assert dev.tracker.totals["executions"] == expected_exec
    assert dev.tracker.totals["simulations"] == 1
    assert dev.tracker.totals["shots"] == expected_shots

    if not isinstance(
        tape.measurements[0], (qp.measurements.ShadowExpvalMP, qp.measurements.ClassicalShadowMP)
    ):
        tape = qp.tape.QuantumScript([qp.RX((1.2, 2.3, 3.4), wires=0)], mps, shots=10)

        with dev.tracker:
            dev.execute(tape)

        assert dev.tracker.totals["executions"] == 3 * expected_exec
        assert dev.tracker.totals["simulations"] == 1
        assert dev.tracker.totals["shots"] == 3 * expected_shots


def test_multiple_expval_with_prod():
    mps, expected_exec, expected_shots = (
        [qp.expval(qp.PauliX(0)), qp.expval(qp.prod(qp.PauliX(0), qp.PauliY(1)))],
        1,
        10,
    )
    dev = qp.device("default.qubit")
    tape = qp.tape.QuantumScript([], mps, shots=10)

    with dev.tracker:
        dev.execute(tape)

    assert dev.tracker.totals["executions"] == expected_exec
    assert dev.tracker.totals["simulations"] == 1
    assert dev.tracker.totals["shots"] == expected_shots

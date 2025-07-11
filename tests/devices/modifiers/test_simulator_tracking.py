# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for qml.devices.modifiers.simulator_tracking.
"""
from typing import Optional

# pylint: disable=unused-argument, too-few-public-methods, missing-class-docstring
import pennylane as qml
from pennylane.devices.modifiers import simulator_tracking


def test_tracking_execute():
    """Test the tracking behavior of execute with no shots."""

    @simulator_tracking
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            results = []
            for c in circuits:
                if len(c.measurements) == 1:
                    results.append(0.0)
                else:
                    results.append(tuple(0.0 for _ in c.measurements))
            return tuple(results)

    dev = DummyDev()

    tape1 = qml.tape.QuantumScript([qml.X(0)], [qml.expval(qml.X(0)), qml.expval(qml.Y(0))])
    tape2 = qml.tape.QuantumScript(
        [qml.S(0), qml.T(1)], [qml.expval(qml.X(0) + qml.Y(0))], shots=50
    )
    with dev.tracker:
        out = dev.execute((tape1, tape2))

    assert out == ((0.0, 0.0), 0.0)
    assert len(dev.tracker.history) == 7
    assert dev.tracker.history["batches"] == [1]
    assert dev.tracker.history["simulations"] == [1, 1]
    assert dev.tracker.history["executions"] == [2, 2]
    assert dev.tracker.history["results"] == [(0.0, 0.0), 0.0]
    assert dev.tracker.history["resources"] == [tape1.specs["resources"], tape2.specs["resources"]]
    assert dev.tracker.history["shots"] == [100]


def test_tracking_compute_derivatives():
    """Test the compute_derivatives tracking behavior."""

    @simulator_tracking
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return 0.0

        def compute_derivatives(
            self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None
        ):
            return 0.0

    dev = DummyDev()

    t = qml.tape.QuantumScript()
    with dev.tracker:
        out = dev.compute_derivatives((t, t, t))

    assert out == 0.0
    assert dev.tracker.history["derivative_batches"] == [1]
    assert dev.tracker.history["derivatives"] == [3]


def test_tracking_execute_and_compute_derivatives():
    """Test tracking the execute_and_compute_derivatives method."""

    @simulator_tracking
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return 0.0

        def execute_and_compute_derivatives(
            self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None
        ):
            return 0.0, 0.0

    t = qml.tape.QuantumScript(
        [qml.RX(1.2, wires=0)], [qml.expval(qml.X(0)), qml.probs(wires=(0, 1))]
    )
    dev = DummyDev()
    with dev.tracker:
        out = dev.execute_and_compute_derivatives((t, t, t))

    assert out == (0.0, 0.0)
    assert dev.tracker.history["execute_and_derivative_batches"] == [1]
    assert dev.tracker.history["executions"] == [3]
    assert dev.tracker.history["derivatives"] == [3]
    r = t.specs["resources"]
    assert dev.tracker.history["resources"] == [r, r, r]


def test_tracking_compute_jvp():
    """Test the compute_jvp tracking behavior."""

    @simulator_tracking
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return 0.0

        def compute_jvp(
            self, circuits, tangents, execution_config: Optional[qml.devices.ExecutionConfig] = None
        ):
            return 0.0

    dev = DummyDev()

    t = qml.tape.QuantumScript()
    with dev.tracker:
        out = dev.compute_jvp((t, t, t), (0.0, 0.0, 0.0))

    assert out == 0.0
    assert dev.tracker.history["jvp_batches"] == [1]
    assert dev.tracker.history["jvps"] == [3]


def test_tracking_execute_and_compute_jvp():
    """Test tracking the execute_and_compute_jvp method."""

    @simulator_tracking
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return 0.0

        def execute_and_compute_jvp(
            self, circuits, tangents, execution_config: Optional[qml.devices.ExecutionConfig] = None
        ):
            return 0.0, 0.0

    t = qml.tape.QuantumScript(
        [qml.RX(1.2, wires=0)], [qml.expval(qml.X(0)), qml.probs(wires=(0, 1))]
    )
    dev = DummyDev()
    with dev.tracker:
        out = dev.execute_and_compute_jvp((t, t, t), (1, 1, 1))

    assert out == (0.0, 0.0)

    assert dev.tracker.history["execute_and_jvp_batches"] == [1]
    assert dev.tracker.history["executions"] == [3]
    assert dev.tracker.history["jvps"] == [3]
    r = t.specs["resources"]
    assert dev.tracker.history["resources"] == [r, r, r]


def test_tracking_compute_vjp():
    """Test the compute_vjp tracking behavior."""

    @simulator_tracking
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return 0.0

        def compute_vjp(
            self,
            circuits,
            cotangents,
            execution_config: Optional[qml.devices.ExecutionConfig] = None,
        ):
            return 0.0

    dev = DummyDev()

    t = qml.tape.QuantumScript()
    with dev.tracker:
        out = dev.compute_vjp((t, t, t), (0.0, 0.0, 0.0))

    assert out == 0.0
    assert dev.tracker.history["vjp_batches"] == [1]
    assert dev.tracker.history["vjps"] == [3]


def test_tracking_execute_and_compute_vjp():
    """Test tracking the execute_and_compute_derivatives method."""

    @simulator_tracking
    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
            return 0.0

        def execute_and_compute_vjp(
            self,
            circuits,
            cotangents,
            execution_config: Optional[qml.devices.ExecutionConfig] = None,
        ):
            return 0.0, 0.0

    t = qml.tape.QuantumScript(
        [qml.Rot(1.2, 2.3, 3.4, wires=0)], [qml.expval(qml.X(0)), qml.probs(wires=(0, 1))]
    )
    dev = DummyDev()
    with dev.tracker:
        out = dev.execute_and_compute_vjp((t, t, t), (1, 1, 1))

    assert out == (0.0, 0.0)

    assert dev.tracker.history["execute_and_vjp_batches"] == [1]
    assert dev.tracker.history["executions"] == [3]
    assert dev.tracker.history["vjps"] == [3]
    r = t.specs["resources"]
    assert dev.tracker.history["resources"] == [r, r, r]

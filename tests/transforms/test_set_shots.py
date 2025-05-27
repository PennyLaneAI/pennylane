# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the ``set_shots`` transformer"""

import pytest

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.transforms.set_shots import null_postprocessing, set_shots


class TestSetShots:
    """Unit tests for the set_shots preprocessing transform."""

    def test_no_change(self):
        """Test that set_shots returns a tape with the same shots if unchanged."""
        tape = QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))], shots=10)
        batch, post_fn = set_shots(tape, 10)
        assert batch[0] is not None
        assert batch[0].shots.total_shots == 10
        assert tape.shots.total_shots == 10
        assert batch[0].operations == tape.operations
        assert batch[0].measurements == tape.measurements
        assert post_fn == null_postprocessing
        assert post_fn(batch) == batch[0]

    def test_changes_shots(self):
        """Test that set_shots returns a new tape with updated shots."""
        tape = QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))], shots=5)
        batch, post_fn = set_shots(tape, 20)
        new_tape = batch[0]
        assert new_tape is not tape
        assert new_tape.shots.total_shots == 20
        assert tape.shots.total_shots == 5
        assert new_tape.operations == tape.operations
        assert new_tape.measurements == tape.measurements
        assert post_fn == null_postprocessing

    def test_none_to_int(self):
        """Test that set_shots can set shots from None to an integer."""
        tape = QuantumScript([qml.PauliX(0)], [qml.probs(0)], shots=None)
        batch, _ = set_shots(tape, 100)
        assert batch[0].shots.total_shots == 100

    def test_int_to_none(self):
        """Test that set_shots can set shots from an integer to None."""
        tape = QuantumScript([qml.PauliX(0)], [qml.probs(0)], shots=50)
        batch, _ = set_shots(tape, None)
        assert batch[0].shots.total_shots is None

    def test_preserves_measurements_and_ops(self):
        """Test that set_shots preserves operations and measurements."""
        ops = [qml.RX(0.1, 0), qml.CNOT([0, 1])]
        measurements = [qml.expval(qml.PauliZ(1)), qml.probs(0)]
        tape = QuantumScript(ops, measurements, shots=5)
        batch, _ = set_shots(tape, 10)
        new_tape = batch[0]
        assert new_tape.operations == ops
        assert new_tape.measurements == measurements

    @pytest.mark.integration
    def test_circuit_specification(self):
        """Test that a handy shot specification works."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.sample(qml.Z(0))

        res = circuit(shots=10)
        assert len(res) == 10

    @pytest.mark.integration
    def test_circuit_shots_overridden_correctly(self):
        """Test using a tracker to ensure that the shots in a circuit are overridden correctly.

        We should have situations where:

        1. device originally has finite shots and we override it with None
        2. device originally is analytic and we override with finite shots.

        """

        dev = qml.device("default.qubit", wires=1, shots=10)

        @qml.qnode(dev, diff_method=None)
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.probs(wires=0)

        # 1. Device originally has finite shots, override with None
        with qml.Tracker(dev) as tracker:
            circuit()
        assert tracker.history["shots"][-1] == 10

        new_circuit = set_shots(circuit, shots=None)
        with qml.Tracker(dev) as tracker:
            new_circuit()
        assert not "shots" in tracker.history

    @pytest.mark.integration
    def test_override_none_with_finite_shots(self):
        """Test that we can override an analytic device with finite shots."""

        # 2. Device originally is analytic, override with finite shots
        dev = qml.device("default.qubit", wires=1, shots=None)

        # pylint: disable=function-redefined
        @qml.qnode(dev, diff_method=None)
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.probs(wires=0)

        with qml.Tracker(dev) as tracker:
            circuit()
        assert not "shots" in tracker.history

        new_circuit = set_shots(circuit, shots=20)
        with qml.Tracker(dev) as tracker:
            new_circuit()
        assert tracker.history["shots"][-1] == 20

    @pytest.mark.integration
    def test_best_diff_method_shots_to_none(self):
        """Test that we can override a device with shots to analytic with diff_method='best' adjusted."""
        # Test that when overriding from shots to analytic, backprop is used
        dev_shots = qml.device("default.qubit", wires=1, shots=100)

        @qml.qnode(dev_shots, diff_method="best")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        param = qml.numpy.array(0.5, requires_grad=True)

        # With shots, parameter-shift should be used for gradient
        with qml.Tracker(dev_shots) as tracker_shots:
            qml.grad(circuit)(param)

        # Store the history for verification
        shots_history = tracker_shots.history

        # Parameter-shift rule requires 2*num_params + 1 = 3 executions
        assert len(shots_history["executions"]) == 3

        # Override to analytic (shots=None) - should switch to backprop/adjoint
        circuit_analytic = set_shots(circuit, shots=None)

        with qml.Tracker(dev_shots) as tracker_analytic:
            qml.grad(circuit_analytic)(param)

        # Analytic gradient should require only 1 execution
        assert len(tracker_analytic.history["executions"]) == 1

    @pytest.mark.integration
    def test_best_diff_method_none_to_shots(self):
        """Test that we can override an analytic device with finite shots while using diff_method='best'."""
        # Start with an analytic device
        dev_analytic = qml.device("default.qubit", wires=1, shots=None)

        @qml.qnode(dev_analytic, diff_method="best")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        param = qml.numpy.array(0.5, requires_grad=True)

        # With analytic device, we expect backprop/adjoint (single execution)
        with qml.Tracker(dev_analytic) as tracker_analytic:
            qml.grad(circuit)(param)

        # Store the history for verification
        analytic_history = tracker_analytic.history

        # Analytic gradient should require only 1 execution
        assert len(analytic_history["executions"]) == 1

        # Override to finite shots - should switch to parameter-shift
        circuit_with_shots = set_shots(circuit, shots=100)

        with qml.Tracker(dev_analytic) as tracker_shots:
            qml.grad(circuit_with_shots)(param)

        # Parameter-shift rule requires 2*num_params + 1 = 3 executions
        assert len(tracker_shots.history["executions"]) == 3

    @pytest.mark.system
    def test_userwarnings(self):
        """
        Test that using set_shots with a QNode that has a shots value
        raises a UserWarning.
        """

        @qml.qnode(qml.device("default.qubit"))
        def c():
            return qml.sample(wires=0)

        with pytest.warns(
            qml.exceptions.PennyLaneUserWarning,
            match=r"Both 'shots=' parameter and 'set_shots' transform are specified\. The 'shots=' parameter will take precedence and override the transform\.",
        ):
            set_shots(c, shots=10)(shots=20)

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

from functools import partial

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

    def test_returns_tuple(self):
        """Test that set_shots returns a tuple of (QuantumScriptBatch, postprocessing_fn)."""
        tape = QuantumScript([qml.PauliX(0)], [qml.probs(0)], shots=1)
        result = set_shots(tape, 2)
        assert isinstance(result, tuple)
        assert isinstance(result[0], (list, tuple))
        assert callable(result[1])

    @pytest.mark.xfail(
        reason="This test is expected to fail until the pipeline is updated to use the new set_shots transform"
    )
    @pytest.mark.integration
    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("shots", [None, 1, 10])
    @pytest.mark.parametrize("interface", qml.math.SUPPORTED_INTERFACE_NAMES)
    def test_compatibility_finite_shots(self, interface, shots):
        """Test that setting shots works with default.qubit with default setting up"""
        dev = qml.device("default.qubit", wires=2)

        @partial(set_shots, shots=shots)
        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        x = qml.numpy.array(0.5)
        assert len(circuit(x)) == 2

    @pytest.mark.integration
    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("shots", [(10, 10, 10), ((10, 3),)])
    @pytest.mark.parametrize("interface", qml.math.SUPPORTED_INTERFACE_NAMES)
    def test_shot_vector(self, interface, shots):
        """Test that setting shots with shot vectors works with default setting up."""
        dev = qml.device("default.qubit", wires=2)

        @partial(set_shots, shots=shots)
        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        x = qml.numpy.array(0.5)
        assert len(circuit(x)) == 3

    @pytest.mark.integration
    def test_toplevel_accessible(self):
        """Test that qml.set_shots is available at top-level."""
        assert hasattr(qml, "set_shots")

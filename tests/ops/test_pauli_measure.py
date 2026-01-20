# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the pauli_measure module"""

import pytest

import pennylane as qml
from pennylane import queuing
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.ops import MeasurementValue, PauliMeasure
from pennylane.wires import Wires


def test_id_is_deprecated():
    """Tests that the 'id' argument is deprecated and renamed."""

    with pytest.warns(
        PennyLaneDeprecationWarning, match="The 'id' argument has been renamed to 'meas_uid'"
    ):
        op = PauliMeasure("XY", wires=[0, 1], id="blah")
    assert op.meas_uid == "blah"


class TestPauliMeasure:
    """Tests for the pauli_measure function."""

    def test_pauli_measure(self):
        """Tests that the pauli_measure is applied correctly."""

        with queuing.AnnotatedQueue() as q:
            m = qml.pauli_measure("XY", wires=[0, 1])

        assert isinstance(m, MeasurementValue)
        assert len(q.queue) == 1
        assert isinstance(q.queue[0], PauliMeasure)
        measure_op = q.queue[0]
        assert m.measurements[0] is measure_op
        assert measure_op.pauli_word == "XY"
        assert measure_op.postselect is None
        assert repr(measure_op) == "PauliMeasure('XY', wires=[0, 1])"

    def test_invalid_arguments(self):
        """Tests that the correct error is raised."""

        with pytest.raises(ValueError, match="The given Pauli word"):
            qml.pauli_measure("ABC", wires=[0, 1, 2])

        with pytest.raises(ValueError, match="The number of wires"):
            qml.pauli_measure("XYX", wires=[0, 1])

    def test_label(self):
        """Tests the label of a PauliMeasure."""

        m = PauliMeasure("XY", wires=Wires([0, 1]))
        assert m.label() == "┤↗XY├"
        assert m.label(wire=1) == "┤↗Y├"
        assert m.label(wire=0) == "┤↗X├"

    def test_hash(self):
        """Test that the hash for PauliMeasure is defined correctly."""

        m1 = PauliMeasure("XY", wires=[0, 1], id="id1")
        m2 = PauliMeasure("XY", wires=[1, 2], id="id1")
        assert hash(m1) != hash(m2)

        m3 = PauliMeasure("XZ", wires=[0, 1], id="id1")
        assert hash(m1) != hash(m3)

        m4 = PauliMeasure("XY", wires=[0, 1], id="id2")
        assert hash(m1) != hash(m4)

        m5 = PauliMeasure("XY", wires=[0, 1], id="id1")
        assert hash(m1) == hash(m5)

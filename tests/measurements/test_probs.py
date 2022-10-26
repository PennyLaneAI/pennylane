# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the probs module"""

import pytest

import pennylane as qml
from pennylane.measurements import MeasurementProcess, MeasurementShapeError, Probability
from pennylane.queuing import AnnotatedQueue


class TestProbs:
    """Tests for the probs function"""

    @pytest.mark.parametrize("wires", [[0], [2, 1], ["a", "c", 3]])
    def test_numeric_type(self, wires):
        """Test that the numeric type is correct."""
        res = qml.probs(wires=wires)
        assert res.numeric_type is float

    @pytest.mark.parametrize("wires", [[0], [2, 1], ["a", "c", 3]])
    @pytest.mark.parametrize("shots", [None, 10])
    def test_shape(self, wires, shots):
        """Test that the shape is correct."""
        dev = qml.device("default.qubit", wires=3, shots=shots)
        res = qml.probs(wires=wires)
        assert res.shape(dev) == (1, 2 ** len(wires))

    @pytest.mark.parametrize("wires", [[0], [2, 1], ["a", "c", 3]])
    def test_shape_shot_vector(self, wires):
        """Test that the shape is correct with the shot vector too."""
        res = qml.probs(wires=wires)
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)
        assert res.shape(dev) == (len(shot_vector), 2 ** len(wires))

    @pytest.mark.parametrize(
        "measurement",
        [qml.probs(wires=[0]), qml.state(), qml.sample(qml.PauliZ(0))],
    )
    def test_shape_no_device_error(self, measurement):
        """Test that an error is raised if a device is not passed when querying
        the shape of certain measurements."""
        with pytest.raises(
            MeasurementShapeError,
            match="The device argument is required to obtain the shape of the measurement process",
        ):
            measurement.shape()


class TestBetaProbs:
    """Tests for annotating the return types of the probs function"""

    @pytest.mark.parametrize("wires", [[0], [0, 1], [1, 0, 2]])
    def test_annotating_probs(self, wires):
        """Test annotating probs"""
        with AnnotatedQueue() as q:
            qml.probs(wires)

        assert len(q.queue) == 1

        meas_proc = q.queue[0]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == Probability

# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for measurements for the legacy return types"""
# pylint: disable=too-few-public-methods
import pytest

import pennylane as qml
from pennylane.measurements import MeasurementProcess, MeasurementShapeError, Shots


dev = qml.device("default.qubit", wires=3)


class TestShape:
    """Unit tests for the `shape` methods of measurement processes"""

    def test_measurement_shape_undefined_error(self):
        """Test that an error is raised if a measurement process does not have a shape method"""

        class DummyMeasurement(MeasurementProcess):
            """Dummy measurement process with no `shape` method"""

        mp = DummyMeasurement()
        shots = Shots(None)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The shape of the measurement DummyMeasurement is not defined",
        ):
            _ = mp.shape(dev, shots)

    @pytest.mark.parametrize(
        "mp",
        [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(0)),
            qml.purity(wires=0),
            qml.vn_entropy(wires=0),
            qml.mutual_info(0, 1),
        ],
    )
    @pytest.mark.parametrize("shots", [None, 10])
    def test_single_value_measurement_process_shape(self, mp, shots):
        """Test that the shape of a single-value measurements is as expected"""
        dev.shots = shots
        num_shots = Shots(shots)

        assert mp.shape(dev, num_shots) == (1,)

    @pytest.mark.parametrize(
        "mp",
        [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(0)),
            qml.purity(wires=0),
            qml.vn_entropy(wires=0),
            qml.mutual_info(0, 1),
        ],
    )
    def test_single_value_measurement_process_shape_shot_vector(self, mp):
        """Test that the shape of a single-value measurement with a shot vector is as expected"""
        shot_vector = [1, 1, 2, 3, 1]
        dev.shots = shot_vector
        shots = Shots(shot_vector)

        assert mp.shape(dev, shots) == (len(shot_vector),)

    def test_sample_shape_no_shots_error(self):
        """Test that the SampleMP.shape raises an error when no shots are being used."""
        dev.shots = None
        shots = Shots(None)
        mp = qml.sample()

        with pytest.raises(MeasurementShapeError, match="Shots are required to obtain the shape"):
            _ = mp.shape(dev, shots)

    def test_sample_shape_no_obs_with_shot_vector(self):
        """Test that the SampleMP.shape raises an error when no shots are being used."""
        shot_vector = [1, 2, 3]
        dev.shots = shot_vector
        shots = Shots(shot_vector)
        mp = qml.sample()

        with pytest.raises(
            MeasurementShapeError,
            match="Getting the output shape of a measurement returning samples along with",
        ):
            _ = mp.shape(dev, shots)

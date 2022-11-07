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
"""Unit tests for the mid_measure module"""

import pytest

import pennylane as qml
from pennylane.measurements import MeasurementValue, MeasurementValueError


class TestMeasure:
    """Tests for the measure function"""

    def test_many_wires_error(self):
        """Test that an error is raised if multiple wires are passed to
        measure."""
        with pytest.raises(
            qml.QuantumFunctionError,
            match="Only a single qubit can be measured in the middle of the circuit",
        ):
            qml.measure(wires=[0, 1])


class TestMeasurementValue:
    """Tests for the MeasurementValue class"""

    @pytest.mark.parametrize("val_pair", [(0, 1), (1, 0), (-1, 1)])
    @pytest.mark.parametrize("control_val_idx", [0, 1])
    def test_measurement_value_assertion(self, val_pair, control_val_idx):
        """Test that asserting the value of a measurement works well."""
        zero_case = val_pair[0]
        one_case = val_pair[1]
        mv = MeasurementValue(measurement_id="1234", zero_case=zero_case, one_case=one_case)
        mv == val_pair[control_val_idx]
        assert mv._control_value == val_pair[control_val_idx]  # pylint: disable=protected-access

    @pytest.mark.parametrize("val_pair", [(0, 1), (1, 0), (-1, 1)])
    @pytest.mark.parametrize("num_inv, expected_idx", [(1, 0), (2, 1), (3, 0)])
    def test_measurement_value_inversion(self, val_pair, num_inv, expected_idx):
        """Test that inverting the value of a measurement works well even with
        multiple inversions.

        Double-inversion should leave the control value of the measurement
        value in place.
        """
        zero_case = val_pair[0]
        one_case = val_pair[1]
        mv = MeasurementValue(measurement_id="1234", zero_case=zero_case, one_case=one_case)
        for _ in range(num_inv):
            mv_new = mv.__invert__()

            # Check that inversion involves creating a copy
            assert mv_new is not mv

            mv = mv_new

        assert mv._control_value == val_pair[expected_idx]  # pylint: disable=protected-access

    def test_measurement_value_assertion_error_wrong_type(self):
        """Test that the return_type related info is updated for a
        measurement."""
        mv1 = MeasurementValue(measurement_id="1111")
        mv2 = MeasurementValue(measurement_id="2222")

        with pytest.raises(
            MeasurementValueError,
            match="The equality operator is used to assert measurement outcomes, but got a value with type",
        ):
            mv1 == mv2

    def test_measurement_value_assertion_error(self):
        """Test that the return_type related info is updated for a
        measurement."""
        mv = MeasurementValue(measurement_id="1234")

        with pytest.raises(MeasurementValueError, match="Unknown measurement value asserted"):
            mv == -1

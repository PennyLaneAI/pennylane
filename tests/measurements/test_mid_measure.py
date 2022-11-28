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
from pennylane.measurements import MeasurementValue
import pennylane.numpy as np


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


class TestMeasurementValueManipulation:
    def test_apply_function_to_measurement(self):

        m = MeasurementValue(["m"], lambda v: v)

        sin_of_m = m._apply(np.sin)
        assert sin_of_m[0] == 0.0
        assert sin_of_m[1] == np.sin(1)

    def test_add_to_measurements(self):
        m0 = MeasurementValue(["m0"], lambda v: v)
        m1 = MeasurementValue(["m1"], lambda v: v)
        sum_of_measurements = m0 + m1
        assert sum_of_measurements[0] == 0
        assert sum_of_measurements[1] == 1
        assert sum_of_measurements[2] == 1
        assert sum_of_measurements[3] == 2

    def test_equality_with_scalar(self):
        m = MeasurementValue(["m"], lambda v: v)
        m_eq = m == 0
        assert m_eq[0] is True  # confirming value is actually eq to True, not just truthy
        assert m_eq[1] is False

    def test_inversion(self):
        m = MeasurementValue(["m"], lambda v: v)
        m_inversion = ~m
        assert m_inversion[0] is True
        assert m_inversion[1] is False

    def test_lt(self):
        m = MeasurementValue(["m"], lambda v: v)
        m_inversion = m < 0.5
        assert m_inversion[0] is True
        assert m_inversion[1] is False

    def test_lt_with_other_measurement_value(self):
        m1 = MeasurementValue(["m1"], lambda v: v)
        m2 = MeasurementValue(["m2"], lambda v: v)
        compared = m1 < m2
        assert compared[0] is False
        assert compared[1] is True
        assert compared[2] is False
        assert compared[3] is False

    def test_gt(self):
        m = MeasurementValue(["m"], lambda v: v)
        m_inversion = m > 0.5
        assert m_inversion[0] is False
        assert m_inversion[1] is True

    def test_gt_with_other_measurement_value(self):
        m1 = MeasurementValue(["m1"], lambda v: v)
        m2 = MeasurementValue(["m2"], lambda v: v)
        compared = m1 > m2
        assert compared[0] is False
        assert compared[1] is False
        assert compared[2] is True
        assert compared[3] is False

    def test_eq(self):
        m = MeasurementValue(["m"], lambda v: v)
        m_eq = m == 1
        assert m_eq[0] is False
        assert m_eq[1] is True

    def test_eq_with_other_measurement_value(self):
        m1 = MeasurementValue(["m1"], lambda v: v)
        m2 = MeasurementValue(["m2"], lambda v: v)
        compared = m1 == m2
        assert compared[0] is True
        assert compared[1] is False
        assert compared[2] is False
        assert compared[3] is True

    def test_merge_measurements_values_dependant_on_same_measurement(self):
        m0 = MeasurementValue(["m"], lambda v: v)
        m1 = MeasurementValue(["m"], lambda v: v)
        combined = m0 + m1
        assert combined[0] == 0
        assert combined[1] == 2

    def test_combine_measurement_value_with_non_measurement(self):
        m0 = MeasurementValue(["m"], lambda v: v)
        out = m0 + 10
        assert out[0] == 10
        assert out[1] == 11

    def test_str(self):
        m = MeasurementValue(["m"], lambda v: v)
        assert str(m) == "if m=0 => 0\nif m=1 => 1"

    def test_complex_str(self):
        a = MeasurementValue(["a"], lambda v: v)
        b = MeasurementValue(["b"], lambda v: v)
        assert (
            str(a + b)
            == """if a=0,b=0 => 0
if a=0,b=1 => 1
if a=1,b=0 => 1
if a=1,b=1 => 2"""
        )

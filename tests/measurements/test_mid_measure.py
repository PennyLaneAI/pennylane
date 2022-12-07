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
import pennylane.numpy as np
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


class TestMeasurementValueManipulation:
    """Test all the dunder methods associated with the MeasurementValue class"""

    def test_apply_function_to_measurement(self):
        """Test the general _apply method that can apply an arbitrary function to a measurement."""

        m = MeasurementValue(["m"], lambda v: v)

        sin_of_m = m._apply(np.sin)
        assert sin_of_m[0] == 0.0
        assert sin_of_m[1] == np.sin(1)

    def test_and_with_bool(self):
        """Test the __add__ dunder method between MeasurementValue and scalar."""
        m = MeasurementValue(["m"], lambda v: v)
        m_add = m & False
        assert not m_add[0]
        assert not m_add[1]

    def test_and_to_measurements(self):
        """Test the __add__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue(["m0"], lambda v: v)
        m1 = MeasurementValue(["m1"], lambda v: v)
        sum_of_measurements = m0 & m1
        assert not sum_of_measurements[0]
        assert not sum_of_measurements[1]
        assert not sum_of_measurements[2]
        assert sum_of_measurements[3]

    def test_or_with_bool(self):
        """Test the __or__ dunder method between MeasurementValue and scalar."""
        m = MeasurementValue(["m"], lambda v: v)
        m_add = m | False
        assert not m_add[0]
        assert m_add[1]

    def test_or_to_measurements(self):
        """Test the __or__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue(["m0"], lambda v: v)
        m1 = MeasurementValue(["m1"], lambda v: v)
        sum_of_measurements = m0 | m1
        assert not sum_of_measurements[0]
        assert sum_of_measurements[1]
        assert sum_of_measurements[2]
        assert sum_of_measurements[3]

    def test_add_with_scalar(self):
        """Test the __add__ dunder method between MeasurementValue and scalar."""
        m = MeasurementValue(["m"], lambda v: v)
        m_add = m + 5
        assert m_add[0] == 5
        assert m_add[1] == 6

    def test_add_to_measurements(self):
        """Test the __add__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue(["m0"], lambda v: v)
        m1 = MeasurementValue(["m1"], lambda v: v)
        sum_of_measurements = m0 + m1
        assert sum_of_measurements[0] == 0
        assert sum_of_measurements[1] == 1
        assert sum_of_measurements[2] == 1
        assert sum_of_measurements[3] == 2

    def test_radd_with_scalar(self):
        """Test the __radd__ dunder method between a scalar and a MeasurementValue."""
        m = MeasurementValue(["m"], lambda v: v)
        m_add = 5 + m
        assert m_add[0] == 5
        assert m_add[1] == 6

    def test_sub_with_scalar(self):
        """Test the __sub__ dunder method between MeasurementValue and scalar."""
        m = MeasurementValue(["m"], lambda v: v)
        m_add = m - 5
        assert m_add[0] == -5
        assert m_add[1] == -4

    def test_sub_to_measurements(self):
        """Test the __sub__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue(["m0"], lambda v: v)
        m1 = MeasurementValue(["m1"], lambda v: v)
        sum_of_measurements = m0 - m1
        assert sum_of_measurements[0] == 0
        assert sum_of_measurements[1] == -1
        assert sum_of_measurements[2] == 1
        assert sum_of_measurements[3] == 0

    def test_rsub_with_scalar(self):
        """Test the __rsub__ dunder method between a scalar and a MeasurementValue."""
        m = MeasurementValue(["m"], lambda v: v)
        m_add = 5 - m
        assert m_add[0] == 5
        assert m_add[1] == 4

    def test_mul_with_scalar(self):
        """Test the __mul__ dunder method between a MeasurementValue and a scalar"""
        m = MeasurementValue(["m"], lambda v: v)
        m_mul = m * 5
        assert m_mul[0] == 0
        assert m_mul[1] == 5

    def test_mul_with_measurement(self):
        """Test the __mul__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue(["m0"], lambda v: v)
        m1 = MeasurementValue(["m1"], lambda v: v)
        mul_of_measurements = m0 * m1
        assert mul_of_measurements[0] == 0
        assert mul_of_measurements[1] == 0
        assert mul_of_measurements[2] == 0
        assert mul_of_measurements[3] == 1

    def test_rmul_with_scalar(self):
        """Test the __rmul__ dunder method between a scalar and a MeasurementValue."""
        m = MeasurementValue(["m"], lambda v: v)
        m_mul = 5 * m
        assert m_mul[0] == 0
        assert m_mul[1] == 5

    def test_truediv_with_scalar(self):
        """Test the __truediv__ dunder method between a MeasurementValue and a scalar"""
        m = MeasurementValue(["m"], lambda v: v)
        m_mul = m / 5.0
        assert m_mul[0] == 0
        assert m_mul[1] == 1 / 5.0

    def test_truediv_with_measurement(self):
        """Test the __truediv__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue(["m0"], lambda v: v) + 3.0
        m1 = MeasurementValue(["m1"], lambda v: v) + 5.0
        mul_of_measurements = m0 / m1
        assert mul_of_measurements[0] == 3.0 / 5.0
        assert mul_of_measurements[1] == 3.0 / 6.0
        assert mul_of_measurements[2] == 4.0 / 5.0
        assert mul_of_measurements[3] == 4.0 / 6.0

    def test_rtruediv_with_scalar(self):
        """Test the __rtruediv__ dunder method between a scalar and a MeasurementValue."""
        m = MeasurementValue(["m"], lambda v: v) + 3.0
        m_mul = 5 / m
        assert m_mul[0] == 5 / 3.0
        assert m_mul[1] == 5 / 4.0

    def test_inversion(self):
        """Test the __inv__ dunder method."""
        m = MeasurementValue(["m"], lambda v: v)
        m_inversion = ~m
        assert m_inversion[0] is True
        assert m_inversion[1] is False

    def test_lt(self):
        """Test the __lt__ dunder method between a MeasurementValue and a float."""
        m = MeasurementValue(["m"], lambda v: v)
        m_inversion = m < 0.5
        assert m_inversion[0] is True
        assert m_inversion[1] is False

    def test_lt_with_other_measurement_value(self):
        """Test the __lt__ dunder method between a two MeasurementValues"""
        m1 = MeasurementValue(["m1"], lambda v: v)
        m2 = MeasurementValue(["m2"], lambda v: v)
        compared = m1 < m2
        assert compared[0] is False
        assert compared[1] is True
        assert compared[2] is False
        assert compared[3] is False

    def test_gt(self):
        """Test the __gt__ dunder method between a MeasurementValue and a flaot."""
        m = MeasurementValue(["m"], lambda v: v)
        m_inversion = m > 0.5
        assert m_inversion[0] is False
        assert m_inversion[1] is True

    def test_gt_with_other_measurement_value(self):
        """Test the __gt__ dunder method between two MeasurementValues."""
        m1 = MeasurementValue(["m1"], lambda v: v)
        m2 = MeasurementValue(["m2"], lambda v: v)
        compared = m1 > m2
        assert compared[0] is False
        assert compared[1] is False
        assert compared[2] is True
        assert compared[3] is False

    def test_le(self):
        """Test the __le__ dunder method between a MeasurementValue and a float."""
        m = MeasurementValue(["m"], lambda v: v)
        m_inversion = m <= 0.5
        assert m_inversion[0] is True
        assert m_inversion[1] is False

    def test_le_with_other_measurement_value(self):
        """Test the __le__ dunder method between a two MeasurementValues"""
        m1 = MeasurementValue(["m1"], lambda v: v)
        m2 = MeasurementValue(["m2"], lambda v: v)
        compared = m1 <= m2
        assert compared[0] is True
        assert compared[1] is True
        assert compared[2] is False
        assert compared[3] is True

    def test_ge(self):
        """Test the __ge__ dunder method between a MeasurementValue and a flaot."""
        m = MeasurementValue(["m"], lambda v: v)
        m_inversion = m >= 0.5
        assert m_inversion[0] is False
        assert m_inversion[1] is True

    def test_ge_with_other_measurement_value(self):
        """Test the __ge__ dunder method between two MeasurementValues."""
        m1 = MeasurementValue(["m1"], lambda v: v)
        m2 = MeasurementValue(["m2"], lambda v: v)
        compared = m1 >= m2
        assert compared[0] is True
        assert compared[1] is False
        assert compared[2] is True
        assert compared[3] is True

    def test_equality_with_scalar(self):
        """Test the __eq__ dunder method between a MeasurementValue and an integer."""
        m = MeasurementValue(["m"], lambda v: v)
        m_eq = m == 0
        assert m_eq[0] is True  # confirming value is actually eq to True, not just truthy
        assert m_eq[1] is False

    def test_equality_with_scalar_opposite(self):
        """Test the __eq__ dunder method between a MeasurementValue and an integer."""
        m = MeasurementValue(["m"], lambda v: v)
        m_eq = m == 1
        assert m_eq[0] is False
        assert m_eq[1] is True

    def test_eq_with_other_measurement_value(self):
        """Test the __eq__ dunder method between two MeasurementValues."""
        m1 = MeasurementValue(["m1"], lambda v: v)
        m2 = MeasurementValue(["m2"], lambda v: v)
        compared = m1 == m2
        assert compared[0] is True
        assert compared[1] is False
        assert compared[2] is False
        assert compared[3] is True

    def test_merge_measurements_values_dependant_on_same_measurement(self):
        """Test that the _merge operation does not create more than 2 branches when combining two MeasurementValues
        that are based on the same measurement."""
        m0 = MeasurementValue(["m"], lambda v: v)
        m1 = MeasurementValue(["m"], lambda v: v)
        combined = m0 + m1
        assert combined[0] == 0
        assert combined[1] == 2

    def test_combine_measurement_value_with_non_measurement(self):
        """Test that we can use dunder methods to combine a MeasurementValue with the underlying "primitive"
        of that measurement value."""
        m0 = MeasurementValue(["m"], lambda v: v)
        out = m0 + 10
        assert out[0] == 10
        assert out[1] == 11

    def test_branches_method(self):
        """Test the __eq__ dunder method between two MeasurementValues."""
        m1 = MeasurementValue(["m1"], lambda v: v)
        m2 = MeasurementValue(["m2"], lambda v: v)
        compared = m1 == m2
        branches = compared.branches()
        assert branches[(0, 0)] is True
        assert branches[(0, 1)] is False
        assert branches[(1, 0)] is False
        assert branches[(1, 1)] is True

    def test_str(self):
        """Test that the output of the __str__ dunder method is as expected"""
        m = MeasurementValue(["m"], lambda v: v)
        assert str(m) == "if m=0 => 0\nif m=1 => 1"

    def test_complex_str(self):
        """Test that the output of the __str__ dunder method is as expected
        w.r.t a more complicated MeasurementValue"""
        a = MeasurementValue(["a"], lambda v: v)
        b = MeasurementValue(["b"], lambda v: v)
        assert (
            str(a + b)
            == """if a=0,b=0 => 0
if a=0,b=1 => 1
if a=1,b=0 => 1
if a=1,b=1 => 2"""
        )

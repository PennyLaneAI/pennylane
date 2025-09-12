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

from itertools import product

import pytest

import pennylane as qml
import pennylane.numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.measurements import MeasurementValue, MidMeasureMP
from pennylane.wires import Wires

# pylint: disable=too-few-public-methods, too-many-public-methods


class TestMeasure:
    """Tests for the measure function"""

    def test_many_wires_error(self):
        """Test that an error is raised if multiple wires are passed to
        measure."""
        with pytest.raises(
            QuantumFunctionError,
            match="Only a single qubit can be measured in the middle of the circuit",
        ):
            qml.measure(wires=[0, 1])

    def test_hash(self):
        """Test that the hash for `MidMeasureMP` is defined correctly."""
        m1 = MidMeasureMP(Wires(0), id="m1")
        m2 = MidMeasureMP(Wires(0), id="m2")
        m3 = MidMeasureMP(Wires(1), id="m1")
        m4 = MidMeasureMP(Wires(0), id="m1")

        assert m1.hash != m2.hash
        assert m1.hash != m3.hash
        assert m1.hash == m4.hash

    @pytest.mark.parametrize(
        "postselect, reset, expected",
        [
            (None, False, "┤↗├"),
            (None, True, "┤↗│  │0⟩"),
            (0, False, "┤↗₀├"),
            (0, True, "┤↗₀│  │0⟩"),
            (1, False, "┤↗₁├"),
            (1, True, "┤↗₁│  │0⟩"),
        ],
    )
    def test_label(self, postselect, reset, expected):
        """Test that the label for a MidMeasureMP is correct"""
        mp = MidMeasureMP(0, postselect=postselect, reset=reset)

        label = mp.label()
        assert label == expected


mp1 = MidMeasureMP(Wires(0), id="m0")
mp2 = MidMeasureMP(Wires(1), id="m1")
mp3 = MidMeasureMP(Wires(2), id="m2")


class TestMeasurementValueManipulation:
    """Test all the dunder methods associated with the MeasurementValue class"""

    def test_error_on_boolean_conversion(self):
        """Test that an error is raised if a measurement value if used as a boolean."""

        m = MeasurementValue([mp1], lambda v: v)

        with pytest.raises(ValueError, match="The truth value of a MeasurementValue"):
            if m:
                return

    def test_apply_function_to_measurement(self):
        """Test the general _apply method that can apply an arbitrary function to a measurement."""

        m = MeasurementValue([mp1], lambda v: v)

        sin_of_m = m._apply(np.sin)  # pylint: disable=protected-access
        assert sin_of_m[0] == 0.0
        assert sin_of_m[1] == np.sin(1)

    def test_and_with_bool(self):
        """Test the __add__ dunder method between MeasurementValue and scalar."""
        m = MeasurementValue([mp1], lambda v: v)
        m_add = m & False
        assert not m_add[0]
        assert not m_add[1]

    def test_and_to_measurements(self):
        """Test the __add__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue([mp1], lambda v: v)
        m1 = MeasurementValue([mp2], lambda v: v)
        sum_of_measurements = m0 & m1
        assert not sum_of_measurements[0]
        assert not sum_of_measurements[1]
        assert not sum_of_measurements[2]
        assert sum_of_measurements[3]

    def test_or_with_bool(self):
        """Test the __or__ dunder method between MeasurementValue and scalar."""
        m = MeasurementValue([mp1], lambda v: v)
        m_add = m | False
        assert not m_add[0]
        assert m_add[1]

    def test_or_to_measurements(self):
        """Test the __or__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue([mp1], lambda v: v)
        m1 = MeasurementValue([mp2], lambda v: v)
        sum_of_measurements = m0 | m1
        assert not sum_of_measurements[0]
        assert sum_of_measurements[1]
        assert sum_of_measurements[2]
        assert sum_of_measurements[3]

    def test_add_with_scalar(self):
        """Test the __add__ dunder method between MeasurementValue and scalar."""
        m = MeasurementValue([mp1], lambda v: v)
        m_add = m + 5
        assert m_add[0] == 5
        assert m_add[1] == 6

    def test_add_to_measurements(self):
        """Test the __add__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue([mp1], lambda v: v)
        m1 = MeasurementValue([mp2], lambda v: v)
        sum_of_measurements = m0 + m1
        assert sum_of_measurements[0] == 0
        assert sum_of_measurements[1] == 1
        assert sum_of_measurements[2] == 1
        assert sum_of_measurements[3] == 2

    def test_radd_with_scalar(self):
        """Test the __radd__ dunder method between a scalar and a MeasurementValue."""
        m = MeasurementValue([mp1], lambda v: v)
        m_add = 5 + m
        assert m_add[0] == 5
        assert m_add[1] == 6

    def test_sub_with_scalar(self):
        """Test the __sub__ dunder method between MeasurementValue and scalar."""
        m = MeasurementValue([mp1], lambda v: v)
        m_add = m - 5
        assert m_add[0] == -5
        assert m_add[1] == -4

    def test_sub_to_measurements(self):
        """Test the __sub__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue([mp1], lambda v: v)
        m1 = MeasurementValue([mp2], lambda v: v)
        sum_of_measurements = m0 - m1
        assert sum_of_measurements[0] == 0
        assert sum_of_measurements[1] == -1
        assert sum_of_measurements[2] == 1
        assert sum_of_measurements[3] == 0

    def test_rsub_with_scalar(self):
        """Test the __rsub__ dunder method between a scalar and a MeasurementValue."""
        m = MeasurementValue([mp1], lambda v: v)
        m_add = 5 - m
        assert m_add[0] == 5
        assert m_add[1] == 4

    def test_mul_with_scalar(self):
        """Test the __mul__ dunder method between a MeasurementValue and a scalar"""
        m = MeasurementValue([mp1], lambda v: v)
        m_mul = m * 5
        assert m_mul[0] == 0
        assert m_mul[1] == 5

    def test_mul_with_measurement(self):
        """Test the __mul__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue([mp1], lambda v: v)
        m1 = MeasurementValue([mp2], lambda v: v)
        mul_of_measurements = m0 * m1
        assert mul_of_measurements[0] == 0
        assert mul_of_measurements[1] == 0
        assert mul_of_measurements[2] == 0
        assert mul_of_measurements[3] == 1

    def test_rmul_with_scalar(self):
        """Test the __rmul__ dunder method between a scalar and a MeasurementValue."""
        m = MeasurementValue([mp1], lambda v: v)
        m_mul = 5 * m
        assert m_mul[0] == 0
        assert m_mul[1] == 5

    def test_truediv_with_scalar(self):
        """Test the __truediv__ dunder method between a MeasurementValue and a scalar"""
        m = MeasurementValue([mp1], lambda v: v)
        m_mul = m / 5.0
        assert m_mul[0] == 0
        assert m_mul[1] == 1 / 5.0

    def test_truediv_with_measurement(self):
        """Test the __truediv__ dunder method between two MeasurementValues."""
        m0 = MeasurementValue([mp1], lambda v: v) + 3.0
        m1 = MeasurementValue([mp2], lambda v: v) + 5.0
        mul_of_measurements = m0 / m1
        assert mul_of_measurements[0] == 3.0 / 5.0
        assert mul_of_measurements[1] == 3.0 / 6.0
        assert mul_of_measurements[2] == 4.0 / 5.0
        assert mul_of_measurements[3] == 4.0 / 6.0

    def test_rtruediv_with_scalar(self):
        """Test the __rtruediv__ dunder method between a scalar and a MeasurementValue."""
        m = MeasurementValue([mp1], lambda v: v) + 3.0
        m_mul = 5 / m
        assert m_mul[0] == 5 / 3.0
        assert m_mul[1] == 5 / 4.0

    def test_inversion(self):
        """Test the __inv__ dunder method."""
        m = MeasurementValue([mp1], lambda v: v)
        m_inversion = ~m
        assert qml.math.allclose(m_inversion[0], True)
        assert qml.math.allclose(m_inversion[1], False)
        values = {mp1: True}
        assert qml.math.allclose(m_inversion.concretize(values), False)
        values = {mp1: False}
        assert qml.math.allclose(m_inversion.concretize(values), True)
        values = {mp1: np.random.rand(10) < 0.5}
        assert all(m_inversion.concretize(values) != np.array(values.values()))

    def test_lt(self):
        """Test the __lt__ dunder method between a MeasurementValue and a float."""
        m = MeasurementValue([mp1], lambda v: v)
        m_inversion = m < 0.5
        assert m_inversion[0] is True
        assert m_inversion[1] is False

    def test_lt_with_other_measurement_value(self):
        """Test the __lt__ dunder method between two MeasurementValues"""
        m1 = MeasurementValue([mp1], lambda v: v)
        m2 = MeasurementValue([mp2], lambda v: v)
        compared = m1 < m2
        assert compared[0] is False
        assert compared[1] is True
        assert compared[2] is False
        assert compared[3] is False

    def test_gt(self):
        """Test the __gt__ dunder method between a MeasurementValue and a float."""
        m = MeasurementValue([mp1], lambda v: v)
        m_inversion = m > 0.5
        assert m_inversion[0] is False
        assert m_inversion[1] is True

    def test_gt_with_other_measurement_value(self):
        """Test the __gt__ dunder method between two MeasurementValues."""
        m1 = MeasurementValue([mp1], lambda v: v)
        m2 = MeasurementValue([mp2], lambda v: v)
        compared = m1 > m2
        assert compared[0] is False
        assert compared[1] is False
        assert compared[2] is True
        assert compared[3] is False

    def test_le(self):
        """Test the __le__ dunder method between a MeasurementValue and a float."""
        m = MeasurementValue([mp1], lambda v: v)
        m_inversion = m <= 0.5
        assert m_inversion[0] is True
        assert m_inversion[1] is False

    def test_le_with_other_measurement_value(self):
        """Test the __le__ dunder method between two MeasurementValues"""
        m1 = MeasurementValue([mp1], lambda v: v)
        m2 = MeasurementValue([mp2], lambda v: v)
        compared = m1 <= m2
        assert compared[0] is True
        assert compared[1] is True
        assert compared[2] is False
        assert compared[3] is True

    def test_ge(self):
        """Test the __ge__ dunder method between a MeasurementValue and a flaot."""
        m = MeasurementValue([mp1], lambda v: v)
        m_inversion = m >= 0.5
        assert m_inversion[0] is False
        assert m_inversion[1] is True

    def test_ge_with_other_measurement_value(self):
        """Test the __ge__ dunder method between two MeasurementValues."""
        m1 = MeasurementValue([mp1], lambda v: v)
        m2 = MeasurementValue([mp2], lambda v: v)
        compared = m1 >= m2
        assert compared[0] is True
        assert compared[1] is False
        assert compared[2] is True
        assert compared[3] is True

    def test_equality_with_scalar(self):
        """Test the __eq__ dunder method between a MeasurementValue and an integer."""
        m = MeasurementValue([mp1], lambda v: v)
        m_eq = m == 0
        assert m_eq[0] is True  # confirming value is actually eq to True, not just truthy
        assert m_eq[1] is False

    def test_equality_with_scalar_opposite(self):
        """Test the __eq__ dunder method between a MeasurementValue and an integer."""
        m = MeasurementValue([mp1], lambda v: v)
        m_eq = m == 1
        assert m_eq[0] is False
        assert m_eq[1] is True

    def test_eq_with_other_measurement_value(self):
        """Test the __eq__ dunder method between two MeasurementValues."""
        m1 = MeasurementValue([mp1], lambda v: v)
        m2 = MeasurementValue([mp2], lambda v: v)
        compared = m1 == m2
        assert compared[0] is True
        assert compared[1] is False
        assert compared[2] is False
        assert compared[3] is True

    def test_non_equality_with_scalar(self):
        """Test the __ne__ dunder method between a MeasurementValue and an integer."""
        m = MeasurementValue([mp1], lambda v: v)
        m_eq = m != 0
        assert m_eq[0] is False  # confirming value is actually eq to True, not just truthy
        assert m_eq[1] is True

    def test_non_equality_with_scalar_opposite(self):
        """Test the __ne__ dunder method between a MeasurementValue and an integer."""
        m = MeasurementValue([mp1], lambda v: v)
        m_eq = m != 1
        assert m_eq[0] is True
        assert m_eq[1] is False

    def test_non_eq_with_other_measurement_value(self):
        """Test the __ne__ dunder method between two MeasurementValues."""
        m1 = MeasurementValue([mp1], lambda v: v)
        m2 = MeasurementValue([mp2], lambda v: v)
        compared = m1 != m2
        assert compared[0] is False
        assert compared[1] is True
        assert compared[2] is True
        assert compared[3] is False

    def test_merge_measurements_values_dependant_on_same_measurement(self):
        """Test that the _merge operation does not create more than 2 branches when combining two MeasurementValues
        that are based on the same measurement."""
        m0 = MeasurementValue([mp1], lambda v: v)
        m1 = MeasurementValue([mp1], lambda v: v)
        combined = m0 + m1
        assert combined[0] == 0
        assert combined[1] == 2

    def test_combine_measurement_value_with_non_measurement(self):
        """Test that we can use dunder methods to combine a MeasurementValue with the underlying "primitive"
        of that measurement value."""
        m0 = MeasurementValue([mp1], lambda v: v)
        out = m0 + 10
        assert out[0] == 10
        assert out[1] == 11

    def test_branches_method(self):
        """Test the __eq__ dunder method between two MeasurementValues."""
        m1 = MeasurementValue([mp1], lambda v: v)
        m2 = MeasurementValue([mp2], lambda v: v)
        compared = m1 == m2
        branches = compared.branches
        assert branches[(0, 0)] is True
        assert branches[(0, 1)] is False
        assert branches[(1, 0)] is False
        assert branches[(1, 1)] is True

    def test_str(self):
        """Test that the output of the __str__ dunder method is as expected"""
        m = MeasurementValue([mp1], lambda v: v)
        assert str(m) == "if m0=0 => 0\nif m0=1 => 1"

    def test_complex_str(self):
        """Test that the output of the __str__ dunder method is as expected
        w.r.t a more complicated MeasurementValue"""
        a = MeasurementValue([mp1], lambda v: v)
        b = MeasurementValue([mp2], lambda v: v)
        assert str(a + b) == (
            "if m0=0,m1=0 => 0\nif m0=0,m1=1 => 1\nif m0=1,m1=0 => 1\nif m0=1,m1=1 => 2"
        )

    def test_repr(self):
        """Test that the output of the __repr__ dunder method is as expected."""
        m = qml.measure(0)
        assert repr(m) == "MeasurementValue(wires=[0])"

    def test_complex_repr(self):
        """Test that the output of the __repr__ dunder method is as expected
        w.r.t a more complicated MeasurementValue"""
        a = MeasurementValue([mp1], lambda v: v)
        b = MeasurementValue([mp2], lambda v: v)
        assert repr(a + b) == "MeasurementValue(wires=[0, 1])"

    def test_map_wires(self):
        """Test that map_wires works as expected."""
        a = MeasurementValue([mp1], lambda v: v)
        b = a.map_wires({0: "b"})
        [new_meas] = b.measurements
        assert new_meas.wires == Wires(["b"])
        assert new_meas.id == mp1.id

    def test_mod(self):
        """Test the __mod__ dunder method between two measurement values"""
        m1 = MeasurementValue([mp1], lambda v: v)
        mod_val = m1 % 2
        assert mod_val[0] == 0
        assert mod_val[1] == 1

    def test_xor(self):
        """Test the __xor__ dunder method between two measurement values"""
        m1 = MeasurementValue([mp1], lambda v: v)
        m2 = MeasurementValue([mp2], lambda v: v)
        compared = m1 ^ m2
        assert compared[0] == 0
        assert compared[1] == 1
        assert compared[2] == 1
        assert compared[3] == 0


unary_dunders = ["__invert__"]


measurement_value_binary_dunders = [
    "__add__",
    "__mul__",
    "__radd__",
    "__rmul__",
    "__rsub__",
    "__sub__",
    "__mod__",
]

boolean_binary_dunders = [
    "__and__",
    "__eq__",
    "__ge__",
    "__gt__",
    "__le__",
    "__lt__",
    "__ne__",
    "__or__",
    "__xor__",
]

binary_dunders = measurement_value_binary_dunders + boolean_binary_dunders

divisions = ["__rtruediv__", "__truediv__"]


class TestMeasurementCompositeValueManipulation:
    """Test composite application of dunder methods associated with the MeasurementValue class"""

    @pytest.mark.parametrize("unary_name", unary_dunders)
    @pytest.mark.parametrize("binary1_name, binary2_name", product(binary_dunders, binary_dunders))
    def test_composition_between_measurement_values(self, unary_name, binary1_name, binary2_name):
        """Test the composition of dunder methods."""
        m0 = MeasurementValue([mp1], lambda v: v)
        m1 = MeasurementValue([mp2], lambda v: v)

        # 1. Apply a unary dunder method
        unary = getattr(m0, unary_name)
        m0 = unary()
        assert isinstance(m0, MeasurementValue)

        # 2. Apply first binary dunder method
        binary_dunder1 = getattr(m0, binary1_name)
        sum_of_measurements = binary_dunder1(m1)
        assert isinstance(sum_of_measurements, MeasurementValue)

        # 3. Apply a unary dunder method on the new MV
        unary = getattr(sum_of_measurements, unary_name)
        m0 = unary()
        assert isinstance(m0, MeasurementValue)

        # 4. Apply second binary dunder method
        binary_dunder2 = getattr(m0, binary2_name)

        m2 = MeasurementValue([mp1], lambda v: v)
        boolean_of_measurements = binary_dunder2(m2)

        assert isinstance(boolean_of_measurements, MeasurementValue)

    @pytest.mark.parametrize("mv_dunder_name", measurement_value_binary_dunders)
    @pytest.mark.parametrize("boolean_dunder_name", boolean_binary_dunders)
    @pytest.mark.parametrize("scalar", [MeasurementValue([mp2], lambda v: v), 0, 1.0, 1.0 + 0j])
    @pytest.mark.parametrize("boolean", [MeasurementValue([mp3], lambda v: v), True, False, None])
    def test_composition_measurement_values_and_boolean(
        self, mv_dunder_name, boolean_dunder_name, scalar, boolean
    ):  # pylint: disable=too-many-arguments
        """Test the composition of dunder methods, applying one whose argument is scalar and one whose argument
        is a boolean."""
        m0 = MeasurementValue([mp1], lambda v: v)

        # 1. Apply first binary dunder method between m0 and scalar
        binary_dunder1 = getattr(m0, mv_dunder_name)
        sum_of_measurements = binary_dunder1(scalar)
        assert isinstance(sum_of_measurements, MeasurementValue)

        # 2. Apply second binary dunder method between m0 and boolean
        binary_dunder2 = getattr(m0, boolean_dunder_name)
        boolean_of_measurements = binary_dunder2(boolean)
        assert isinstance(boolean_of_measurements, MeasurementValue)

    @pytest.mark.parametrize("div", divisions)
    @pytest.mark.parametrize("other", [MeasurementValue([mp3], lambda v: v) + 5, np.pi])
    @pytest.mark.parametrize("binary", binary_dunders)
    def test_composition_with_division(self, binary, div, other):
        """Test the composition of dunder methods with division."""
        # 1. Apply a binary dundar
        m0 = MeasurementValue([mp1], lambda v: v)
        m1 = MeasurementValue([mp2], lambda v: v)

        binary_dunder = getattr(m0, binary)
        m0 = binary_dunder(m1)

        # 2. Apply a division method
        division_dunder = getattr(m0, div)
        res = division_dunder(other)
        assert isinstance(res, MeasurementValue)


class TestMeasurementValueItems:
    """Test that a MeasurementValue returns its items correctly."""

    # pylint: disable=protected-access

    funcs_and_expected_single = [
        ((lambda v: v), [0, 1]),
        ((lambda v: 1 - v), [1, 0]),
    ]

    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("func, expected", funcs_and_expected_single)
    def test_items_single_mp(self, func, expected, postselect):
        """Test the full items. Note that postselect should not affect the
        output at all."""
        mp = MidMeasureMP(0, postselect=postselect)
        mv = MeasurementValue([mp], func)
        items = list(mv.items())
        assert items == [((0,), expected[0]), ((1,), expected[1])]

    funcs_and_expected_multi = [
        ((lambda *v: sum(v) == 1), [0, 1, 1, 0, 1, 0, 0, 0]),
        ((lambda *v: v[0]), [0, 0, 0, 0, 1, 1, 1, 1]),
    ]

    @pytest.mark.parametrize("func, expected", funcs_and_expected_multi)
    def test_items_multiple_mps(self, func, expected):
        """Test the full items."""
        MP0 = MidMeasureMP(0)
        MP1 = MidMeasureMP(1)
        MP2 = MidMeasureMP(2)
        branches3 = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]
        mv = MeasurementValue([MP0, MP1, MP2], func)
        items = list(mv.items())
        assert len(items) == len(branches3) == len(expected)
        for item, branch, exp in zip(items, branches3, expected):
            assert item == (branch, exp)

    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("func, expected", funcs_and_expected_single)
    def test_postselected_items_single_mp(self, postselect, func, expected):
        """Test the full items."""
        mp = MidMeasureMP(0, postselect=postselect)
        mv = MeasurementValue([mp], func)
        items = list(mv.postselected_items())
        if postselect is None:
            assert items == [((0,), expected[0]), ((1,), expected[1])]
        else:
            all_items = [((), expected[0]), ((), expected[1])]
            assert items == [all_items[postselect]]

    branches3 = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]

    postselects_and_branches = [
        ((None, 1, None), [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]),
        ((None, 1, 0), [(0, 1, 0), (1, 1, 0)]),
        ((1, 1, 1), [(1, 1, 1)]),
        ((0, None, None), [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]),
        ((None, None, None), branches3),
    ]

    @pytest.mark.parametrize("postselects, branches", postselects_and_branches)
    @pytest.mark.parametrize("func, expected", funcs_and_expected_multi)
    def test_postselected_items_multiple_mps(self, func, expected, postselects, branches):
        """Test the full items."""
        MP0 = MidMeasureMP(0, postselect=postselects[0])
        MP1 = MidMeasureMP(1, postselect=postselects[1])
        MP2 = MidMeasureMP(2, postselect=postselects[2])

        mv = MeasurementValue([MP0, MP1, MP2], func)
        items = list(mv.postselected_items())
        assert len(items) == len(branches)
        for item, branch in zip(items, branches):
            pruned_branch = tuple(b for i, b in enumerate(branch) if postselects[i] is None)
            assert item == (pruned_branch, expected[self.branches3.index(branch)])

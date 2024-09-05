# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for BooleanFn utility class.
"""
import pytest

import pennylane as qml


class TestBooleanFn:
    @pytest.mark.parametrize(
        "fn, arg, expected",
        [
            (lambda x: True, -1, True),
            (lambda x: True, 10, True),
            (lambda x: x < 4, -1, True),
            (lambda x: x < 4, 10, False),
        ],
    )
    def test_basic_functionality(self, fn, arg, expected):
        """Test initialization and calling of BooleanFn."""
        crit = qml.BooleanFn(fn)
        assert crit(arg) == expected

    def test_not(self):
        """Test that logical negation works."""
        crit = qml.BooleanFn(lambda x: x < 4)
        ncrit = ~crit
        assert crit(-2) and not ncrit(-2)
        assert not crit(10) and ncrit(10)

    def test_and(self):
        """Test that logical conjunction works."""
        crit_0 = qml.BooleanFn(lambda x: x > 4)
        crit_1 = qml.BooleanFn(lambda x: x < 9)
        crit = crit_0 & crit_1
        assert not crit(-2)
        assert crit(6)
        assert not crit(10)

    def test_or(self):
        """Test that logical or works."""
        crit_0 = qml.BooleanFn(lambda x: x < 4)
        crit_1 = qml.BooleanFn(lambda x: x > 9)
        crit = crit_0 | crit_1
        assert crit(-2)
        assert not crit(6)
        assert crit(10)

    def test_repr(self):
        """Test the repr is right."""

        def greater_than_five(x):
            return x > 5

        func = qml.BooleanFn(greater_than_five)

        assert repr(func) == "BooleanFn(greater_than_five)"

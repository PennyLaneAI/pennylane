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
"""
Tests for the :class:`pennylane.data.data_manager.params.ParamArg` class.
"""


import pytest

from pennylane.data.data_manager.params import ParamArg, format_param_args


class TestParamArg:
    """Tests for ``ParamArg``."""

    def test_values(self):
        """Tests that ``values()`` returns all the possible values of ``ParamArg``."""

        assert set(ParamArg.values()) == {"full", "default"}

    @pytest.mark.parametrize(
        "val, expect",
        [
            (ParamArg.DEFAULT, True),
            (ParamArg.FULL, True),
            ("full", True),
            ("default", True),
            (None, False),
            ("DEFAULT", False),
        ],
    )
    def test_is_arg(self, val, expect):
        """Tests that ``is_arg()`` returns True iff the argument is a member of ``ParamArg`` or
        one of its values."""
        assert ParamArg.is_arg(val) == expect

    @pytest.mark.parametrize("val", [ParamArg.FULL, ParamArg.DEFAULT])
    def test_str(self, val):
        """Test that __str__ returns the value of the enum."""

        assert str(val) == val.value


@pytest.mark.parametrize(
    ("param", "details", "expected"),
    [
        ("layout", 1, ["1"]),
        ("layout", [1], ["1"]),
        ("layout", ["foo", "bar", "baz"], ["foo", "bar", "baz"]),
        ("layout", [1, 4], ["1x4"]),
        ("layout", [[1, 4]], ["1x4"]),
        ("bondlength", [1, 1.100], ["1.0", "1.1"]),
        ("random", "foo", ["foo"]),
        ("random", ["foo", "bar"], ["foo", "bar"]),
        ("random", [ParamArg.FULL, ParamArg.DEFAULT], ParamArg.FULL),
        ("random", [ParamArg.DEFAULT, ParamArg.FULL], ParamArg.DEFAULT),
        ("random", ParamArg.DEFAULT.value, ParamArg.DEFAULT),
        ("random", ParamArg.DEFAULT, ParamArg.DEFAULT),
        ("random", [ParamArg.DEFAULT], ParamArg.DEFAULT),
        ("random", ["whatever", ParamArg.DEFAULT.value], ParamArg.DEFAULT),
    ],
)
def test_format_param_args(param, details, expected):
    """Test that format_param_args behaves as expected."""
    assert format_param_args(param, details) == expected

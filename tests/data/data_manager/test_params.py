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

from pennylane.data.data_manager.params import (
    Description,
    ParamArg,
    format_param_args,
    format_params,
    provide_defaults,
)

pytestmark = pytest.mark.data


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


@pytest.mark.parametrize(
    "param, details, match_error",
    [
        (
            "layout",
            (1, None),
            r"Invalid layout value of '\(1, None\)'. Must be a string or a tuple of ints.",
        ),
        ("bondlength", None, r"Invalid bondlength 'None'. Must be a string, int or float."),
        ("foo", 1, r"Invalid type 'int' for parameter 'foo'"),
    ],
)
def test_format_param_args_errors(param, details, match_error):
    """Test that format_param_args raises the expected errors
    for incorrect inputs."""
    with pytest.raises(TypeError, match=match_error):
        format_param_args(param, details)


def test_format_params():
    """Test that format_params calls format_param_args with each parameter."""
    assert format_params(
        layout=[1, 4], bondlength=["0.5", "0.6"], z="full", y=ParamArg.DEFAULT
    ) == [
        {"name": "layout", "values": ["1x4"]},
        {"name": "bondlength", "values": ["0.5", "0.6"]},
        {"name": "z", "values": ParamArg.FULL},
        {"name": "y", "values": ParamArg.DEFAULT},
    ]


class TestDescription:
    """Tests for Description."""

    def test_str(self):
        """Test that __str__ is equivalent to dict __str__."""
        params = {"foo": "bar", "x": "y"}
        assert str(Description(params)) == str(params)

    def test_repr(self):
        """Test that __repr__ is equivalent to dict __repr__."""
        params = {"foo": "bar", "x": "y"}
        assert repr(Description(params)) == f"Description({repr(params)})"


@pytest.mark.parametrize(
    "data_name, params, expected_params",
    [
        (
            "qchem",
            [{"name": "molname", "values": ["H2"]}],
            [
                {"name": "molname", "values": ["H2"]},
                {"default": True, "name": "basis"},
                {"default": True, "name": "bondlength"},
            ],
        ),
        (
            "qchem",
            [{"name": "molname", "values": ["H2"]}, {"name": "bondlength", "values": ["0.82"]}],
            [
                {"name": "molname", "values": ["H2"]},
                {"name": "bondlength", "values": ["0.82"]},
                {"default": True, "name": "basis"},
            ],
        ),
        (
            "qspin",
            [{"name": "sysname", "values": ["BoseHubbard"]}],
            [
                {"name": "sysname", "values": ["BoseHubbard"]},
                {"default": True, "name": "periodicity"},
                {"default": True, "name": "lattice"},
                {"default": True, "name": "layout"},
            ],
        ),
        (
            "qspin",
            [
                {"name": "sysname", "values": ["BoseHubbard"]},
                {"name": "periodicity", "values": ["closed"]},
            ],
            [
                {"name": "sysname", "values": ["BoseHubbard"]},
                {"name": "periodicity", "values": ["closed"]},
                {"default": True, "name": "lattice"},
                {"default": True, "name": "layout"},
            ],
        ),
        (
            "qspin",
            [
                {"name": "sysname", "values": ["BoseHubbard"]},
                {"name": "periodicity", "values": ["closed"]},
                {"name": "lattice", "values": ["chain"]},
            ],
            [
                {"name": "sysname", "values": ["BoseHubbard"]},
                {"name": "periodicity", "values": ["closed"]},
                {"name": "lattice", "values": ["chain"]},
                {"default": True, "name": "layout"},
            ],
        ),
        ("other", [], []),
    ],
)
def test_provide_defaults(data_name, params, expected_params):
    assert provide_defaults(data_name, params) == expected_params

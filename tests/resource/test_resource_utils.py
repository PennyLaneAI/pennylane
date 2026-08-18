# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for resource utility functions"""

import pytest

import pennylane as qp
from pennylane.resource._utils import (
    get_marker_level_map,
    make_level_name_unique,
    preprocess_level_input,
)


@pytest.mark.parametrize(
    "level,output,expect_warnings",
    [
        (0, [0], False),
        ([0, 1], [0, 1], False),
        ([0, 1, 1, 1], [0, 1], True),
        ((0, 1), [0, 1], False),
        (range(3, 0, -1), [1, 2, 3], True),
        ("foo", [2], False),
        (["foo", "bar"], [2, 3], False),
        ((1, "foo", "baz", 4, "bar"), [1, 2, 3, 4, 5], True),
        ("all", [0, 1, 2, 3, 4, 5, 6], False),
        ("user", [6], False),
    ],
)
def test_preprocess_levels(level, output, expect_warnings):
    """Test that _preprocess_level_input works correctly"""
    marker_to_level = {
        "foo": 2,
        "bar": 3,
        # Assume unnamed level at 4
        "baz": 5,
    }
    # Assume that there are 6 passes in the pipeline total

    if expect_warnings:
        with pytest.warns(
            UserWarning,
            match="The 'level' argument to qp.specs for QJIT'd QNodes has been sorted to be in ascending "
            "order with no duplicate levels.",
        ):
            assert preprocess_level_input(level, marker_to_level, 6) == output
    else:
        assert preprocess_level_input(level, marker_to_level, 6) == output


def test_preprocess_levels_invalid():
    with pytest.raises(ValueError, match="out of bounds"):
        preprocess_level_input(-10, {}, 5)

    with pytest.raises(ValueError, match="out of bounds"):
        preprocess_level_input(10, {}, 5)

    with pytest.raises(ValueError, match="Invalid level"):
        preprocess_level_input([1, 2, 3.14], {}, 5)

    with pytest.raises(ValueError, match="Marker name 'foo' not found"):
        preprocess_level_input("foo", {}, 5)


def test_get_marker_level_map():
    """Test that the marker to level mapping is correct"""
    pipeline = qp.CompilePipeline()

    pipeline.add_marker("m0")
    pipeline += qp.transform(pass_name="cancel_inverses")
    pipeline.add_marker("m1")
    pipeline.add_marker("m2")
    pipeline += qp.transform(pass_name="cancel_inverses")
    pipeline += qp.transform(pass_name="cancel_inverses")
    pipeline.add_marker("m3")

    expected_mapping = {
        "m0": 0,
        "m1": 1,
        "m2": 1,
        "m3": 3,
    }

    assert get_marker_level_map(pipeline) == expected_mapping

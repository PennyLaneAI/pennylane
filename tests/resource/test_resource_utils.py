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

# pylint: disable=redefined-outer-name


@pytest.fixture
def example_pipeline():
    pipeline = qp.CompilePipeline([qp.transforms.cancel_inverses for _ in range(6)])

    pipeline.add_marker("foo", 2)
    pipeline.add_marker("bar", 3)
    pipeline.add_marker("baz", 5)

    return pipeline


def test_make_level_name_unique():
    existing_levels = {"foo", "foo-2", "bar"}

    assert make_level_name_unique("foo", existing_levels) == "foo-3"
    assert make_level_name_unique("bar", existing_levels) == "bar-2"
    assert make_level_name_unique("baz", existing_levels) == "baz"


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
def test_preprocess_levels(level, output, expect_warnings, example_pipeline):
    """Test that _preprocess_level_input works correctly"""

    # 6 total transforms, with markers at levels 2, 3, and 5

    if expect_warnings:
        with pytest.warns(
            UserWarning,
            match="The 'level' argument to qp.specs for QJIT'd QNodes has been sorted to be in ascending "
            "order with no duplicate levels.",
        ):
            assert preprocess_level_input(level, example_pipeline) == output
    else:
        assert preprocess_level_input(level, example_pipeline) == output


def test_preprocess_levels_invalid(example_pipeline):
    with pytest.raises(ValueError, match="out of bounds"):
        preprocess_level_input(1, qp.CompilePipeline())

    with pytest.raises(ValueError, match="out of bounds"):
        preprocess_level_input(-10, example_pipeline)

    with pytest.raises(ValueError, match="out of bounds"):
        preprocess_level_input(10, example_pipeline)

    with pytest.raises(ValueError, match="Invalid level"):
        preprocess_level_input([1, 2, 3.14], example_pipeline)

    with pytest.raises(ValueError, match="Marker name 'potato' not found"):
        preprocess_level_input("potato", example_pipeline)


def test_preprocess_levels_tape_transforms():
    """Test that a warning is raised if the user has applied tape transforms."""

    @qp.transform
    def dummy_transform(tape):
        """Returns a tape-only transform that can be used for testing"""
        return (tape,), lambda res: res[0]

    pipeline = qp.CompilePipeline([dummy_transform, qp.transforms.cancel_inverses])

    with pytest.raises(
        ValueError,
        match=r"Specs encountered the following tape transforms: .*dummy_transform.*\. Tape transforms are no longer supported by specs.",
    ):
        preprocess_level_input("all", pipeline)


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

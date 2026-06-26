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
from pennylane.resource.utils import (
    get_last_tape_transform_level,
    preprocess_level_input,
)


@pytest.mark.parametrize(
    "level,output,expect_warnings",
    [
        (0, [0], False),
        (slice(3), [0, 1, 2], False),
        (slice(1, 3), [1, 2], False),
        (slice(1, 4, 2), [1, 3], False),
        ([0, 1], [0, 1], False),
        ([0, 1, 1, 1], [0, 1], True),
        ((0, 1), [0, 1], False),
        (range(3, 0, -1), [1, 2, 3], True),
        ("foo", [2], False),
        (["foo", "bar"], [2, 3], False),
        ((1, "foo", "baz", 4, "bar"), [1, 2, 3, 4, 5], True),
        ("all", [0, 1, 2, 3, 4, 5, 6], False),
        ("all-mlir", [4, 5, 6], False),
        ("user", [6], False),
    ],
)
def test_preprocess_levels(level, output, expect_warnings):
    """Test that _preprocess_level_input works correctly"""
    marker_to_level = {
        "foo": 2,
        "bar": 3,
        # Treat MLIR lowering as level 4
        "baz": 5,
    }

    if expect_warnings:
        with pytest.warns(
            UserWarning,
            match="The 'level' argument to qp.specs for QJIT'd QNodes has been sorted to be in ascending "
            "order with no duplicate levels.",
        ):
            assert preprocess_level_input(level, marker_to_level, 5, 4) == output
    else:
        assert preprocess_level_input(level, marker_to_level, 5, 4) == output


@pytest.mark.parametrize(
    "num_tapes, expected",
    [
        (  # If there are no tape transforms, the "Before Tape Transforms" level should be skipped
            0,
            list(range(5)),
        ),
        (2, list(range(6))),
        (5, list(range(6))),
    ],
)
def test_preprocess_levels_all(num_tapes, expected):
    # Assume there are always 4 transforms in the pipeline
    assert preprocess_level_input("all", {}, 4, num_tapes) == expected


def test_preprocess_levels_invalid():
    with pytest.raises(ValueError, match="out of bounds"):
        preprocess_level_input(-10, {}, 5, 0)

    with pytest.raises(ValueError, match="out of bounds"):
        preprocess_level_input(10, {}, 5, 0)

    with pytest.raises(ValueError, match="Invalid level"):
        preprocess_level_input([1, 2, 3.14], {}, 5, 0)

    with pytest.raises(ValueError, match="Marker name 'foo' not found"):
        preprocess_level_input("foo", {}, 5, 0)


def test_get_last_tape_transform_level():
    """Test that _get_last_tape_transform_level works correctly"""

    @qp.transform
    def dummy_transform(tape):
        return (tape,), lambda res: res[0]

    # If there are no transforms, the last transform level should be 0
    assert get_last_tape_transform_level(qp.CompilePipeline()) == 0
    # If there are *any* tape transforms, this should return the number of tape transforms
    # since there is an implied level 0 for "Before Tape Transforms"
    assert get_last_tape_transform_level(qp.CompilePipeline(dummy_transform)) == 1
    assert get_last_tape_transform_level(qp.CompilePipeline(dummy_transform, dummy_transform)) == 2

    # MLIR passes should not be counted
    assert (
        get_last_tape_transform_level(
            qp.CompilePipeline(dummy_transform, qp.transform(pass_name="cancel_inverses"))
        )
        == 1
    )

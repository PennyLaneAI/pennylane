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
"""Contains tests for 'get_compile_pipeline'."""

# pylint: disable=not-an-iterable
import re
import warnings

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.transforms import CompilePipeline
from pennylane.core.transforms.compile_pipeline import DEVICE_PREPROCESSING
from pennylane.exceptions import TransformError
from pennylane.workflow import get_compile_pipeline
from pennylane.workflow._levels import (
    level_to_number,
    levels_to_numbers,
    split_device_preprocessing,
    validate_level,
)


def _circuit_with_marker():
    """A QNode with two user transforms and a ``"blah"`` marker between them."""

    @qp.transforms.merge_rotations
    @qp.marker("blah")
    @qp.transforms.cancel_inverses
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit(x):
        qp.RX(x, wires=0)
        return qp.expval(qp.Z(0))

    return circuit


USER_PIPELINE = CompilePipeline(qp.transforms.cancel_inverses, qp.transforms.merge_rotations)


@pytest.mark.catalyst
class TestValidation:
    """Tests for validation errors."""

    def test_input_is_not_qnode(self):
        """Tests that an error is raised when the input is not a QNode."""

        def circuit():
            return qp.expval(qp.Z(0))

        with pytest.raises(
            ValueError, match="get_compile_pipeline can only be applied to a qjit'd QNode"
        ):
            get_compile_pipeline(circuit)

    def test_input_is_not_qjit(self):
        """Tests that an error is raised when the input is a QNode that is not qjit'd."""

        with pytest.raises(
            ValueError, match="get_compile_pipeline can only be applied to a qjit'd QNode"
        ):
            get_compile_pipeline(_circuit_with_marker())

    def test_input_is_not_qjit_qnode(self):
        """Tests when the input is QJIT'd but not a QNode."""

        @qp.qjit
        def inc(x):
            return x + 1

        with pytest.raises(
            ValueError, match="get_compile_pipeline can only be applied to a qjit'd QNode"
        ):
            get_compile_pipeline(inc)

    @pytest.mark.parametrize("level", ["gradient", "all-mlir"])
    def test_removed_level(self, level):
        """Tests that the 'gradient' and 'all-mlir' levels are no longer supported."""

        with pytest.raises(ValueError, match=f"The level '{level}' is no longer supported."):
            get_compile_pipeline(qp.qjit(_circuit_with_marker()), level=level)

    @pytest.mark.parametrize(
        "level", [0.5, True, None, {0, 1}, iter([0, 1]), b"\x01", bytearray(b"\x01")]
    )
    def test_invalid_level_type(self, level):
        """Tests validation for incorrect level types."""

        with pytest.raises(ValueError, match="Invalid level"):
            get_compile_pipeline(qp.qjit(_circuit_with_marker()), level=level)

    @pytest.mark.parametrize("level", [[], ()])
    def test_empty_sequence(self, level):
        """Tests that an empty Sequence of levels is rejected."""

        with pytest.raises(ValueError, match="does not select any level"):
            get_compile_pipeline(qp.qjit(_circuit_with_marker()), level=level)

    @pytest.mark.parametrize(
        "level", [[0, 0.5], [0, True], ["all"], ["top", "all-user"], [slice(0, 1)]]
    )
    def test_invalid_level_in_sequence(self, level):
        """Tests validation for incorrect levels in a Sequence."""

        with pytest.raises(ValueError, match="Invalid level .* in the level Sequence"):
            get_compile_pipeline(qp.qjit(_circuit_with_marker()), level=level)

    def test_missing_marker_label(self):
        """Tests that a string that doesn't have a marker raises an exception."""

        with pytest.raises(
            ValueError,
            match=re.escape("Marker name 'my_marker' not found in the compile pipeline."),
        ):
            get_compile_pipeline(qp.qjit(_circuit_with_marker()), level="my_marker")

    @pytest.mark.parametrize(
        "level_slice", [slice("blah", None), slice(0, 1.5), slice(0, None, 1.0), slice(0, True)]
    )
    def test_slice_with_invalid_bounds(self, level_slice):
        """Tests that a slice with bounds or step other than int or None is rejected."""

        with pytest.raises(ValueError, match="the bounds and step of a slice must be int or None"):
            get_compile_pipeline(qp.qjit(_circuit_with_marker()), level=level_slice)

    @pytest.mark.parametrize("level_slice", [slice(4, None), slice(5, 9)])
    def test_slice_start_beyond_device_level(self, level_slice):
        """Tests that a slice starting after the device level is rejected, as for an int level."""

        with pytest.raises(ValueError, match="the start of a slice must be at most 3"):
            get_compile_pipeline(qp.qjit(_circuit_with_marker()), level=level_slice)

    @pytest.mark.parametrize("level", [np.array([0, 1]), np.zeros((1, 2), dtype=int), np.array(1)])
    def test_numpy_array(self, level):
        """Tests that NumPy arrays are not accepted as levels."""

        with pytest.raises(ValueError, match="Invalid level"):
            get_compile_pipeline(qp.qjit(_circuit_with_marker()), level=level)

    @pytest.mark.parametrize("args", [(), (0.5,), (np.array([0.5]),), ([0.5, 0.1],), ("blah",)])
    def test_previous_call_signature(self, args):
        """Tests that calling the returned compile pipeline with the QNode arguments, as in the
        previous ``get_compile_pipeline(qnode, level)(*args)`` signature, raises an error."""

        pipeline = get_compile_pipeline(qp.qjit(_circuit_with_marker()))

        with pytest.raises(TypeError, match="now returns the CompilePipeline directly"):
            pipeline(*args)

    @pytest.mark.parametrize("level", [-1, 4, [0, 4]])
    def test_level_out_of_bounds(self, level):
        """Tests that an integer level beyond the device level raises an exception."""

        with pytest.raises(ValueError, match="The 'level' argument is out of bounds"):
            get_compile_pipeline(qp.qjit(_circuit_with_marker()), level=level)


@pytest.mark.catalyst
class TestUserLevel:
    """Tests 'user' level transforms."""

    def test_user_level_pipeline_with_markers(self):
        """Tests that markers are properly retained."""

        dev = qp.device("null.qubit")

        pipeline = CompilePipeline()
        pipeline += qp.transforms.merge_rotations
        pipeline.add_marker("after-merge-rotations")
        pipeline += qp.transforms.cancel_inverses
        pipeline.add_marker("after-cancel-inverses")

        @pipeline
        @qp.qnode(dev)
        def circuit():
            return qp.expval(qp.Z(0))

        circuit = qp.qjit(circuit)

        assert get_compile_pipeline(circuit, level="user") == pipeline

    def test_user_level_with_final_transform(self):
        """Tests that a final transform is part of the user level."""

        @qp.gradients.metric_tensor
        @qp.transforms.merge_rotations(atol=1e-5)
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            return qp.expval(qp.Z(0))

        circuit = qp.qjit(circuit)

        cp = get_compile_pipeline(circuit, level="user")

        assert len(cp) == 4
        assert cp[0].tape_transform == qp.transforms.cancel_inverses.tape_transform
        assert cp[1].tape_transform == qp.transforms.merge_rotations.tape_transform
        assert cp[2].tape_transform == qp.gradients.metric_tensor.expand_transform
        assert cp[3].tape_transform == qp.gradients.metric_tensor.tape_transform

    def test_no_user_levels(self):
        """Ensures an empty compile pipeline if no user transforms."""

        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            return qp.expval(qp.Z(0))

        circuit = qp.qjit(circuit)

        assert get_compile_pipeline(circuit, level="user") == CompilePipeline()


@pytest.mark.catalyst
class TestLevels:
    """Tests the levels other than 'user'."""

    @staticmethod
    def _circuit():
        return qp.qjit(_circuit_with_marker())

    def test_device_level(self):
        """Tests that the device preprocessing is a single level after the user transforms, and
        that the compile pipeline does not depend on the QNode arguments."""

        cp = get_compile_pipeline(self._circuit(), level="device")

        expected = USER_PIPELINE + DEVICE_PREPROCESSING
        expected.add_marker("blah", 1)
        assert cp == expected
        assert len(cp) == 3
        assert str(cp).splitlines()[-2] == "  [2] Device Preprocessing"

    def test_device_level_keeps_final_marker(self):
        """Tests that a marker after the last user transform is kept at the user and device
        levels."""

        circuit = qp.qjit(qp.marker(_circuit_with_marker(), "end"))

        assert "end" in get_compile_pipeline(circuit, level="user").markers
        assert get_compile_pipeline(circuit, level="device").get_marker_level("end") == 2

    def test_marker_level(self):
        """Tests that a string corresponding to a marker level can be used, and that the marker is
        kept at the end of the compile pipeline."""

        cp = get_compile_pipeline(self._circuit(), level="blah")

        expected = CompilePipeline(qp.transforms.cancel_inverses)
        expected.add_marker("blah")
        assert cp == expected

    @pytest.mark.parametrize(
        "level, marker_level",
        [(1, 1), (slice(0, 1), 1), (slice(1, 2), 0), (slice(1, None), 0), (slice(2, None), None)],
    )
    def test_markers_of_selected_transforms(self, level, marker_level):
        """Tests that a marker at the start or the end of the selected transforms is kept, and that
        the other markers are dropped."""

        cp = get_compile_pipeline(self._circuit(), level=level)
        assert cp.get_marker_level("blah") == marker_level

    def test_level_is_top(self):
        """Tests that level is top returns an empty pipeline."""

        assert get_compile_pipeline(self._circuit(), level="top") == CompilePipeline()

    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    def test_level_is_integer(self, level):
        """Tests that an integer level is the number of transforms, with the device preprocessing
        counting as one."""

        cp = get_compile_pipeline(self._circuit(), level=level)

        full_pipeline = USER_PIPELINE + DEVICE_PREPROCESSING
        assert list(cp) == list(full_pipeline[:level])

    @pytest.mark.parametrize(
        "level_slice",
        [
            slice(2, 5),
            slice(1, None),
            slice(None, 3),
            slice(1, 2),
            slice(None, 1),
        ],
    )
    def test_level_is_slice(self, level_slice):
        """Tests that slice levels index the user transforms followed by the device
        preprocessing."""

        cp = get_compile_pipeline(self._circuit(), level=level_slice)

        full_pipeline = USER_PIPELINE + DEVICE_PREPROCESSING
        assert list(cp) == list(full_pipeline[level_slice])

    @pytest.mark.parametrize("level_slice", [slice(-1, None), slice(None, -1), slice(-2, -1)])
    def test_slice_with_negative_bounds(self, level_slice):
        """Tests that a slice with negative bounds is rejected."""

        with pytest.raises(ValueError, match="the bounds of a slice must be non-negative"):
            get_compile_pipeline(self._circuit(), level=level_slice)

    @pytest.mark.parametrize("level_slice", [slice(None, None, -1), slice(0, None, 0)])
    def test_slice_with_non_positive_step(self, level_slice):
        """Tests that a slice with a step that is not positive is rejected, since the device
        preprocessing is applied after the other transforms."""

        with pytest.raises(ValueError, match="the step of a slice must be positive"):
            get_compile_pipeline(self._circuit(), level=level_slice)

    def test_slice_with_step_greater_than_one(self):
        """Tests that a slice with a step greater than 1 skips transforms, with a warning."""

        with pytest.warns(UserWarning, match="has a step greater than 1"):
            cp = get_compile_pipeline(self._circuit(), level=slice(0, None, 2))

        assert list(cp) == list((USER_PIPELINE + DEVICE_PREPROCESSING)[::2])

    def test_slice_with_step_greater_than_one_and_markers(self):
        """Tests that a slice with a step greater than 1 raises a single warning, which mentions
        that the markers are dropped."""

        with pytest.warns(UserWarning) as record:
            cp = get_compile_pipeline(self._circuit(), level=slice(0, None, 2))

        messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
        assert len(messages) == 1
        assert "Markers are dropped from the result" in messages[0]
        assert not cp.markers

    @pytest.mark.parametrize("level_slice", [slice(1, None), slice(1, 100)])
    def test_slice_without_stop_includes_device_preprocessing(self, level_slice):
        """Tests that a slice without a stop, or with a stop past the device level, includes the
        device preprocessing."""

        cp = get_compile_pipeline(self._circuit(), level=level_slice)
        assert list(cp) == list((USER_PIPELINE + DEVICE_PREPROCESSING)[1:])
        assert cp[-1] == DEVICE_PREPROCESSING

    def test_slice_starting_at_device_level(self):
        """Tests that a slice starting at the device level is empty."""

        assert get_compile_pipeline(self._circuit(), level=slice(3, None)) == (CompilePipeline())

    def test_slice_with_unit_step(self):
        """Tests that a slice with an explicit step of 1 is the same as without a step."""

        circuit = self._circuit()
        assert get_compile_pipeline(circuit, level=slice(0, None, 1)) == get_compile_pipeline(
            circuit, level=slice(0, None)
        )

    @pytest.mark.parametrize("level", [np.int64(1), [np.int32(1), np.int64(3)]])
    def test_numpy_integer_levels(self, level):
        """Tests that NumPy integers are accepted as integer levels."""

        circuit = self._circuit()
        expected = [1, 3] if isinstance(level, list) else 1
        assert get_compile_pipeline(circuit, level=level) == get_compile_pipeline(
            circuit, level=expected
        )

    def test_sequence_of_levels(self):
        """Tests that a Sequence of levels returns one compile pipeline per level."""

        pipelines = get_compile_pipeline(self._circuit(), level=(0, "blah", "device"))

        assert isinstance(pipelines, list)
        assert [len(pipeline) for pipeline in pipelines] == [0, 1, 3]

    def test_single_entry_sequence(self):
        """Tests that a Sequence with a single level returns a list."""

        pipelines = get_compile_pipeline(self._circuit(), level=["user"])
        assert [len(pipeline) for pipeline in pipelines] == [2]

    def test_unsorted_sequence(self):
        """Tests that a Sequence of levels is sorted, with duplicates removed."""

        with pytest.warns(UserWarning, match="The 'level' argument has been sorted"):
            pipelines = get_compile_pipeline(self._circuit(), level=["device", 0, "top"])

        assert [len(pipeline) for pipeline in pipelines] == [0, 3]

    @pytest.mark.parametrize("level, lengths", [("all-user", [0, 1, 2]), ("all", [0, 1, 2, 3])])
    def test_all_levels(self, level, lengths):
        """Tests that 'all-user' and 'all' return the compile pipeline at every stage."""

        pipelines = get_compile_pipeline(self._circuit(), level=level)
        assert [len(pipeline) for pipeline in pipelines] == lengths


@pytest.mark.catalyst
def test_qjit_with_autograph():
    """Tests that the compile pipeline of a qjit'd QNode is retrieved when autograph converts the
    QNode into a function."""

    circuit = qp.qjit(_circuit_with_marker(), autograph=True)

    expected = CompilePipeline(USER_PIPELINE)
    expected.add_marker("blah", 1)
    assert get_compile_pipeline(circuit) == expected


def test_device_preprocessing_repr():
    """Tests that the device preprocessing placeholder is displayed as such."""

    cp = USER_PIPELINE + DEVICE_PREPROCESSING

    assert repr(DEVICE_PREPROCESSING) == "<Device Preprocessing>"
    assert "[2] Device Preprocessing" in str(cp)
    assert "[2] <Device Preprocessing>" in repr(cp)


class TestDevicePreprocessingCannotBeApplied:
    """Tests that the device preprocessing placeholder raises an error when it is applied, rather
    than when the transformed QNode is executed."""

    @staticmethod
    def _qnode():
        @qp.qnode(qp.device("default.qubit", wires=1))
        def circuit(x):
            qp.RX(x, wires=0)
            return qp.expval(qp.Z(0))

        return circuit

    def test_decorator(self):
        """Tests applying the compile pipeline as a decorator."""

        device_pipeline = USER_PIPELINE + DEVICE_PREPROCESSING

        with pytest.raises(TransformError, match="placeholder and cannot be applied"):

            @device_pipeline
            @qp.qnode(qp.device("default.qubit", wires=1))
            def _(x):
                qp.RX(x, wires=0)
                return qp.expval(qp.Z(0))

    def test_qnode(self):
        """Tests applying the compile pipeline to a QNode."""

        device_pipeline = USER_PIPELINE + DEVICE_PREPROCESSING

        with pytest.raises(TransformError, match="placeholder and cannot be applied"):
            device_pipeline(self._qnode())

    def test_tapes(self):
        """Tests applying the compile pipeline to tapes."""

        device_pipeline = USER_PIPELINE + DEVICE_PREPROCESSING
        tape = qp.tape.QuantumScript([qp.RX(0.5, wires=0)], [qp.expval(qp.Z(0))])

        with pytest.raises(TransformError, match="placeholder and cannot be applied"):
            device_pipeline([tape])

    @pytest.mark.catalyst
    def test_qjit(self):
        """Tests that, with qjit, the error is about the placeholder rather than about a missing
        pass name."""

        circuit = qp.qjit(_circuit_with_marker(), capture=True)
        device_pipeline = get_compile_pipeline(circuit, level="device")

        with pytest.raises(TransformError, match="placeholder and cannot be applied"):
            qp.qjit(device_pipeline(self._qnode()), capture=True)


@pytest.mark.parametrize(
    "level, expected, expect_warning",
    [
        (0, [0], False),
        ("top", [0], False),
        ("user", [6], False),
        ("device", [7], False),
        (7, [7], False),
        ("foo", [2], False),
        ("all-user", [0, 1, 2, 3, 4, 5, 6], False),
        ("all", [0, 1, 2, 3, 4, 5, 6, 7], False),
        ([0, 1], [0, 1], False),
        ((0, 1), [0, 1], False),
        (["foo", "bar"], [2, 3], False),
        (["top", "user"], [0, 6], False),
        (["top", 1], [0, 1], False),
        ([0, 1, 1, 1], [0, 1], True),
        (["top", 0], [0], True),
        (range(3, 0, -1), [1, 2, 3], True),
        ((1, "foo", "baz", 4, "bar"), [1, 2, 3, 4, 5], True),
    ],
)
def test_levels_to_numbers(level, expected, expect_warning):
    """Tests that a level is converted to the sorted numbers of the levels it selects."""

    # 6 user transforms, with markers at levels 2, 3, and 5, so the device level is 7
    pipeline = CompilePipeline([qp.transforms.cancel_inverses for _ in range(6)])
    pipeline.add_marker("foo", 2)
    pipeline.add_marker("bar", 3)
    pipeline.add_marker("baz", 5)

    if expect_warning:
        with pytest.warns(UserWarning, match="has been sorted to be in ascending order"):
            assert levels_to_numbers(level, pipeline) == expected
    else:
        assert levels_to_numbers(level, pipeline) == expected


@pytest.mark.parametrize(
    "level, match",
    [
        (0.5, "Invalid level"),
        ([], "does not select any level"),
        (["typo"], "Marker name 'typo' not found"),
        (slice(-1, None), "the bounds of a slice must be non-negative"),
    ],
)
def test_validate_level_rejects_invalid_levels(level, match):
    """Tests that invalid levels are rejected."""

    with pytest.raises(ValueError, match=match):
        validate_level(level, USER_PIPELINE)


@pytest.mark.parametrize("level", ["all", [2, 0, 0], slice(1, None, 2)])
def test_validate_level_accepts_valid_levels(level):
    """Tests that valid levels are accepted without any warning, as they are not converted."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_level(level, USER_PIPELINE)


def test_level_to_number_invalid_type():
    """Tests that a single level that is neither a str nor an int is rejected."""

    with pytest.raises(ValueError, match="Invalid level 1.5"):
        level_to_number(1.5, USER_PIPELINE)


def test_split_device_preprocessing():
    """Tests that the device preprocessing is split off the end of a compile pipeline, keeping
    the markers placed after the user transforms."""

    pipeline = USER_PIPELINE + DEVICE_PREPROCESSING
    pipeline.add_marker("end", 2)

    user_pipeline, has_device_preprocessing = split_device_preprocessing(pipeline)
    assert has_device_preprocessing
    assert list(user_pipeline) == list(USER_PIPELINE)
    assert user_pipeline.get_marker_level("end") == 2

    assert split_device_preprocessing(USER_PIPELINE) == (USER_PIPELINE, False)

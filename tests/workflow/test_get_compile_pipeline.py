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

import re

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.transforms import CompilePipeline
from pennylane.core.transforms.compile_pipeline import _DEVICE_PREPROCESSING
from pennylane.exceptions import TransformError
from pennylane.workflow import get_compile_pipeline
from pennylane.workflow.get_compile_pipeline import _levels_to_numbers


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


class TestValidation:
    """Tests for validation errors."""

    def test_input_is_not_qnode(self):
        """Tests that an error is raised when the input is not a QNode."""

        def circuit():
            return qp.expval(qp.Z(0))

        with pytest.raises(
            ValueError,
            match=re.escape("Can only retrieve the compile pipeline of a QNode or a qjit'd QNode"),
        ):
            get_compile_pipeline(circuit)

    @pytest.mark.catalyst
    def test_input_is_not_qjit_qnode(self):
        """Tests when the input is QJIT'd but not a QNode."""

        @qp.qjit
        def inc(x):
            return x + 1

        with pytest.raises(
            ValueError,
            match=re.escape("Can only retrieve the compile pipeline of a QNode or a qjit'd QNode"),
        ):
            get_compile_pipeline(inc)

    def test_gradient_level(self):
        """Tests that the 'gradient' level is no longer supported."""

        with pytest.raises(ValueError, match="The level 'gradient' is no longer supported."):
            get_compile_pipeline(_circuit_with_marker(), level="gradient")

    @pytest.mark.parametrize(
        "level", [0.5, True, None, {0, 1}, iter([0, 1]), b"\x01", bytearray(b"\x01")]
    )
    def test_invalid_level_type(self, level):
        """Tests validation for incorrect level types."""

        with pytest.raises(ValueError, match="Invalid level"):
            get_compile_pipeline(_circuit_with_marker(), level=level)

    @pytest.mark.parametrize("level", [[], ()])
    def test_empty_sequence(self, level):
        """Tests that an empty Sequence of levels is rejected."""

        with pytest.raises(ValueError, match="does not select any level"):
            get_compile_pipeline(_circuit_with_marker(), level=level)

    @pytest.mark.parametrize(
        "level", [[0, 0.5], [0, True], ["all"], ["top", "all-user"], [slice(0, 1)]]
    )
    def test_invalid_level_in_sequence(self, level):
        """Tests validation for incorrect levels in a Sequence."""

        with pytest.raises(ValueError, match="Invalid level .* in the level Sequence"):
            get_compile_pipeline(_circuit_with_marker(), level=level)

    def test_missing_marker_label(self):
        """Tests that a string that doesn't have a marker raises an exception."""

        with pytest.raises(
            ValueError,
            match=re.escape("Marker name 'my_marker' not found in the compile pipeline."),
        ):
            get_compile_pipeline(_circuit_with_marker(), level="my_marker")

    @pytest.mark.parametrize("level", [-1, 4, [0, 4]])
    def test_level_out_of_bounds(self, level):
        """Tests that an integer level beyond the device level raises an exception."""

        with pytest.raises(ValueError, match="The 'level' argument is out of bounds"):
            get_compile_pipeline(_circuit_with_marker(), level=level)


@pytest.mark.parametrize("use_qjit", [False, pytest.param(True, marks=[pytest.mark.catalyst])])
class TestUserLevel:
    """Tests 'user' level transforms."""

    def test_user_level_pipeline_with_markers(self, use_qjit):
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

        if use_qjit:
            circuit = qp.qjit(circuit)

        assert get_compile_pipeline(circuit, level="user") == pipeline

    def test_user_level_with_final_transform(self, use_qjit):
        """Tests that a final transform is part of the user level."""

        @qp.gradients.metric_tensor
        @qp.transforms.merge_rotations(atol=1e-5)
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            return qp.expval(qp.Z(0))

        if use_qjit:
            circuit = qp.qjit(circuit)

        cp = get_compile_pipeline(circuit, level="user")

        assert len(cp) == 4
        assert cp[0].tape_transform == qp.transforms.cancel_inverses.tape_transform
        assert cp[1].tape_transform == qp.transforms.merge_rotations.tape_transform
        assert cp[2].tape_transform == qp.gradients.metric_tensor.expand_transform
        assert cp[3].tape_transform == qp.gradients.metric_tensor.tape_transform

    def test_no_user_levels(self, use_qjit):
        """Ensures an empty compile pipeline if no user transforms."""

        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            return qp.expval(qp.Z(0))

        if use_qjit:
            circuit = qp.qjit(circuit)

        assert get_compile_pipeline(circuit, level="user") == CompilePipeline()

    def test_informative_transforms(self, use_qjit):
        """Tests that informative transforms are part of the user level."""

        @qp.adjoint_metric_tensor  # INFORMATIVE!
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("null.qubit"))
        def circuit(angles):
            qp.RX(angles[0], wires=0)
            return qp.expval(qp.Z(0))

        if use_qjit:
            circuit = qp.qjit(circuit)

        pipeline = get_compile_pipeline(circuit)
        assert len(pipeline) == 3
        assert pipeline[0].tape_transform == qp.transforms.merge_rotations.tape_transform
        assert pipeline[1].tape_transform == qp.gradients.adjoint_metric_tensor.expand_transform
        assert pipeline[2].tape_transform == qp.gradients.adjoint_metric_tensor.tape_transform


@pytest.mark.parametrize("use_qjit", [False, pytest.param(True, marks=[pytest.mark.catalyst])])
class TestLevels:
    """Tests the levels other than 'user'."""

    @staticmethod
    def _circuit(use_qjit):
        circuit = _circuit_with_marker()
        return qp.qjit(circuit) if use_qjit else circuit

    def test_device_level(self, use_qjit):
        """Tests that the device preprocessing is a single level after the user transforms, and
        that the compile pipeline does not depend on the QNode arguments."""

        cp = get_compile_pipeline(self._circuit(use_qjit), level="device")

        expected = USER_PIPELINE + _DEVICE_PREPROCESSING
        expected.add_marker("blah", 1)
        assert cp == expected
        assert len(cp) == 3
        assert str(cp).splitlines()[-2] == "  [2] Device Preprocessing"

    def test_device_level_keeps_final_marker(self, use_qjit):
        """Tests that a marker after the last user transform is kept at the user and device
        levels."""

        circuit = qp.marker(_circuit_with_marker(), "end")
        if use_qjit:
            circuit = qp.qjit(circuit)

        assert "end" in get_compile_pipeline(circuit, level="user").markers
        assert get_compile_pipeline(circuit, level="device").get_marker_level("end") == 2

    def test_marker_level(self, use_qjit):
        """Tests that a string corresponding to a marker level can be used."""

        cp = get_compile_pipeline(self._circuit(use_qjit), level="blah")
        assert cp == CompilePipeline(qp.transforms.cancel_inverses)

    def test_level_is_top(self, use_qjit):
        """Tests that level is top returns an empty pipeline."""

        assert get_compile_pipeline(self._circuit(use_qjit), level="top") == CompilePipeline()

    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    def test_level_is_integer(self, level, use_qjit):
        """Tests that an integer level is the number of transforms, with the device preprocessing
        counting as one."""

        cp = get_compile_pipeline(self._circuit(use_qjit), level=level)

        full_pipeline = USER_PIPELINE + _DEVICE_PREPROCESSING
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
    def test_level_is_slice(self, level_slice, use_qjit):
        """Tests that slice levels index the user transforms followed by the device
        preprocessing."""

        cp = get_compile_pipeline(self._circuit(use_qjit), level=level_slice)

        full_pipeline = USER_PIPELINE + _DEVICE_PREPROCESSING
        assert list(cp) == list(full_pipeline[level_slice])

    @pytest.mark.parametrize("level_slice", [slice(-1, None), slice(None, -1), slice(-2, -1)])
    def test_slice_with_negative_bounds(self, level_slice, use_qjit):
        """Tests that a slice with negative bounds is rejected."""

        with pytest.raises(ValueError, match="the bounds of a slice must be non-negative"):
            get_compile_pipeline(self._circuit(use_qjit), level=level_slice)

    @pytest.mark.parametrize("level_slice", [slice(None, None, -1), slice(0, None, 0)])
    def test_slice_with_non_positive_step(self, level_slice, use_qjit):
        """Tests that a slice with a step that is not positive is rejected, since the device
        preprocessing is applied after the other transforms."""

        with pytest.raises(ValueError, match="the step of a slice must be positive"):
            get_compile_pipeline(self._circuit(use_qjit), level=level_slice)

    def test_slice_with_step_greater_than_one(self, use_qjit):
        """Tests that a slice with a step greater than 1 skips transforms, with a warning."""

        with pytest.warns(UserWarning, match="has a step greater than 1"):
            cp = get_compile_pipeline(self._circuit(use_qjit), level=slice(0, None, 2))

        assert list(cp) == list((USER_PIPELINE + _DEVICE_PREPROCESSING)[::2])

    def test_slice_with_unit_step(self, use_qjit):
        """Tests that a slice with an explicit step of 1 is the same as without a step."""

        circuit = self._circuit(use_qjit)
        assert get_compile_pipeline(circuit, level=slice(0, None, 1)) == get_compile_pipeline(
            circuit, level=slice(0, None)
        )

    @pytest.mark.parametrize("level", [np.int64(1), [np.int32(1), np.int64(3)]])
    def test_numpy_integer_levels(self, level, use_qjit):
        """Tests that NumPy integers are accepted as integer levels."""

        circuit = self._circuit(use_qjit)
        expected = [1, 3] if isinstance(level, list) else 1
        assert get_compile_pipeline(circuit, level=level) == get_compile_pipeline(
            circuit, level=expected
        )

    def test_sequence_of_levels(self, use_qjit):
        """Tests that a Sequence of levels returns one compile pipeline per level."""

        pipelines = get_compile_pipeline(self._circuit(use_qjit), level=(0, "blah", "device"))

        assert isinstance(pipelines, list)
        assert [len(pipeline) for pipeline in pipelines] == [0, 1, 3]

    def test_single_entry_sequence(self, use_qjit):
        """Tests that a Sequence with a single level returns a list."""

        pipelines = get_compile_pipeline(self._circuit(use_qjit), level=["user"])
        assert [len(pipeline) for pipeline in pipelines] == [2]

    def test_unsorted_sequence(self, use_qjit):
        """Tests that a Sequence of levels is sorted, with duplicates removed."""

        with pytest.warns(UserWarning, match="The 'level' argument has been sorted"):
            pipelines = get_compile_pipeline(self._circuit(use_qjit), level=["device", 0, "top"])

        assert [len(pipeline) for pipeline in pipelines] == [0, 3]

    @pytest.mark.parametrize("level, lengths", [("all-user", [0, 1, 2]), ("all", [0, 1, 2, 3])])
    def test_all_levels(self, level, lengths, use_qjit):
        """Tests that 'all-user' and 'all' return the compile pipeline at every stage."""

        pipelines = get_compile_pipeline(self._circuit(use_qjit), level=level)
        assert [len(pipeline) for pipeline in pipelines] == lengths


@pytest.mark.catalyst
def test_qjit_with_autograph():
    """Tests that the compile pipeline of a qjit'd QNode is retrieved when autograph converts the
    QNode into a function."""

    circuit = qp.qjit(_circuit_with_marker(), autograph=True)

    expected = CompilePipeline(USER_PIPELINE)
    expected.add_marker("blah", 1)
    assert get_compile_pipeline(circuit) == expected


def test_device_preprocessing_cannot_be_applied():
    """Tests that the device preprocessing placeholder cannot be applied to a QNode."""

    circuit = _circuit_with_marker()
    device_pipeline = get_compile_pipeline(circuit, level="device")

    @device_pipeline
    @qp.qnode(qp.device("default.qubit", wires=1))
    def new_circuit(x):
        qp.RX(x, wires=0)
        return qp.expval(qp.Z(0))

    with pytest.raises(TransformError, match="placeholder and cannot be applied"):
        new_circuit(0.5)


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
            assert _levels_to_numbers(level, pipeline) == expected
    else:
        assert _levels_to_numbers(level, pipeline) == expected

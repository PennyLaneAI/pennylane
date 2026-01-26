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

import pytest

import pennylane as qml
from pennylane.transforms.core.compile_pipeline import CompilePipeline
from pennylane.workflow import get_compile_pipeline


class TestValidation:
    """Tests for validation errors."""

    @pytest.mark.parametrize(
        "unsupported_level",
        (
            [0],
            [0, 1],
            (0,),
            (0, 1),
        ),
    )
    def test_incorrect_level_type(self, unsupported_level):
        """Tests validation for incorrect level types."""

        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"'level={unsupported_level}' of type '{type(unsupported_level)}' is not supported."
            ),
        ):
            _ = get_compile_pipeline(circuit, level=unsupported_level)()


class TestUserLevel:
    """Tests 'user' level transforms."""

    def test_user_level_pipeline(self):
        """Tests the contents of a user level pipeline."""

        dev = qml.device("reference.qubit")

        @qml.transforms.merge_rotations(atol=1e-5)
        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.Z(0))

        cp = get_compile_pipeline(circuit, level="user")()
        assert len(cp) == 2
        assert cp[0].tape_transform == qml.transforms.cancel_inverses.tape_transform
        assert cp[1].tape_transform == qml.transforms.merge_rotations.tape_transform

    def test_user_level_with_final_transform(self):
        """Tests that a final transform is correctly re-appended."""

        dev = qml.device("reference.qubit")

        @qml.gradients.metric_tensor
        @qml.transforms.merge_rotations(atol=1e-5)
        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.Z(0))

        cp = get_compile_pipeline(circuit, level="user")()

        assert len(cp) == 4
        assert cp[0].tape_transform == qml.transforms.cancel_inverses.tape_transform
        assert cp[1].tape_transform == qml.transforms.merge_rotations.tape_transform
        assert cp[2].tape_transform == qml.gradients.metric_tensor.expand_transform
        assert cp[3].tape_transform == qml.gradients.metric_tensor.tape_transform

    def test_no_user_levels(self):
        """Ensures an empty compile pipeline if no user transforms."""
        dev = qml.device("reference.qubit")

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.Z(0))

        cp = get_compile_pipeline(circuit, level="user")()

        assert cp == CompilePipeline()


class TestGradientLevel:
    """Tests 'device' level transforms."""


class TestDeviceLevel:
    """Tests 'device' level transforms."""

# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the `marker` function"""

import pytest

import pennylane as qml
from pennylane.transforms.core.compile_pipeline import ProtectedLevel


class TestMarkerQNode:
    """Tests the integration with compile pipeline."""

    def test_uniqueness_checking(self):
        """Test an error is raised if a level is not unique."""

        with pytest.raises(
            ValueError,
            match="Found multiple markers for level 'something'. Markers must be unique.",
        ):

            @qml.marker(level="something")
            @qml.transforms.merge_rotations
            @qml.marker(level="something")
            @qml.qnode(qml.device("null.qubit"))
            def c():
                return qml.state()

    @pytest.mark.parametrize("protected_level_str", [level.value for level in ProtectedLevel])
    def test_protected_levels(self, protected_level_str):
        """Test an error is raised for using a protected level."""

        with pytest.raises(
            ValueError, match=f"Found marker for protected level '{protected_level_str}'"
        ):

            @qml.marker(level=protected_level_str)
            @qml.qnode(qml.device("null.qubit"))
            def c():
                return qml.state()

    def test_simple_qnode(self):
        """Tests that markers are placed in the qnode's compilation pipeline."""

        @qml.marker(level="after-cancel-inverses")
        @qml.transforms.cancel_inverses
        @qml.marker("after-merge-rotations")
        @qml.transforms.merge_rotations
        @qml.qnode(qml.device("null.qubit"))
        def c():
            return qml.state()

        assert c.compile_pipeline.markers == ["after-merge-rotations", "after-cancel-inverses"]
        assert c.compile_pipeline.get_marker_level("after-merge-rotations") == 1
        assert c.compile_pipeline.get_marker_level("after-cancel-inverses") == 2

    def test_marker_missing_level(self):
        """Tests when the marker is missing level."""

        with pytest.raises(ValueError, match="marker requires a 'level' argument."):

            @qml.marker()
            @qml.transforms.cancel_inverses
            @qml.qnode(qml.device("null.qubit"))
            def c():
                return qml.state()

    def test_marker_is_first_decorator(self):
        """Tests when the marker is the first decorator."""

        @qml.marker("after-cancel-inverses")
        @qml.transforms.cancel_inverses
        @qml.marker("after-merge-rotations")
        @qml.transforms.merge_rotations
        @qml.marker("nothing-applied")
        @qml.qnode(qml.device("null.qubit"))
        def c():
            return qml.state()

        assert c.compile_pipeline.markers == [
            "nothing-applied",
            "after-merge-rotations",
            "after-cancel-inverses",
        ]
        assert c.compile_pipeline.get_marker_level("nothing-applied") == 0
        assert c.compile_pipeline.get_marker_level("after-merge-rotations") == 1
        assert c.compile_pipeline.get_marker_level("after-cancel-inverses") == 2

    def test_marker_embedded_before_pipeline_decorator(self):
        """Tests that markers applied before the pipeline decorator are included."""

        pipeline = qml.CompilePipeline()
        pipeline.add_marker("no-transforms")
        pipeline += qml.transforms.cancel_inverses
        pipeline.add_marker("after-cancel-inverses")

        assert pipeline.markers == ["no-transforms", "after-cancel-inverses"]
        assert pipeline.get_marker_level("no-transforms") == 0
        assert pipeline.get_marker_level("after-cancel-inverses") == 1

        @qml.marker("after-pipeline")
        @pipeline
        @qml.qnode(qml.device("null.qubit"))
        def c():
            return qml.state()

        assert c.compile_pipeline.markers == [
            "no-transforms",
            "after-cancel-inverses",
            "after-pipeline",
        ]
        assert c.compile_pipeline.get_marker_level("no-transforms") == 0
        assert c.compile_pipeline.get_marker_level("after-cancel-inverses") == 1
        assert c.compile_pipeline.get_marker_level("after-pipeline") == 1

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

import pennylane as qp
from pennylane.transforms.core.compile_pipeline import ProtectedLevel

jax = pytest.importorskip("jax")
pytestmark = [pytest.mark.jax, pytest.mark.capture]


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestMarkerQNode:
    """Tests the integration with compile pipeline."""

    def test_applied_not_on_qnode(self):
        """Test exception when applied on something that's not a QNode."""

        def qn():
            qp.H(0)
            return qp.state()

        with pytest.raises(ValueError, match="Object to mark must be a QNode."):
            qp.marker(qn, "marker0")

    def test_function(self):
        """Tests that it can be used functionally."""

        @qp.qnode(qp.device("null.qubit"))
        def c():
            return qp.state()

        qp.marker(c, "marker0")
        qp.marker(c, label="marker1")

        assert len(c.compile_pipeline.markers) == 2

    def test_uniqueness_checking(self):
        """Test an error is raised if a level is not unique."""

        with pytest.raises(
            ValueError,
            match="Found multiple markers for level 'something'. Markers must be unique.",
        ):

            @qp.marker(label="something")
            @qp.transforms.merge_rotations
            @qp.marker(label="something")
            @qp.qnode(qp.device("null.qubit"))
            def c():
                return qp.state()

    @pytest.mark.parametrize("protected_level_str", [level.value for level in ProtectedLevel])
    def test_protected_levels(self, protected_level_str):
        """Test an error is raised for using a protected level."""

        with pytest.raises(
            ValueError, match=f"Found marker for protected level '{protected_level_str}'"
        ):

            @qp.marker(label=protected_level_str)
            @qp.qnode(qp.device("null.qubit"))
            def c():
                return qp.state()

    def test_simple_qnode(self):
        """Tests that markers are placed in the qnode's compilation pipeline."""

        @qp.marker(label="after-cancel-inverses")
        @qp.transforms.cancel_inverses
        @qp.marker("after-merge-rotations")
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("null.qubit"))
        def c():
            return qp.state()

        assert c.compile_pipeline.markers == ["after-merge-rotations", "after-cancel-inverses"]
        assert c.compile_pipeline.get_marker_level("after-merge-rotations") == 1
        assert c.compile_pipeline.get_marker_level("after-cancel-inverses") == 2

    def test_marker_missing_level(self):
        """Tests when the marker is missing level."""

        with pytest.raises(ValueError, match="marker requires a 'label' argument."):

            @qp.marker()
            @qp.transforms.cancel_inverses
            @qp.qnode(qp.device("null.qubit"))
            def c():
                return qp.state()

    def test_marker_is_first_decorator(self):
        """Tests when the marker is the first decorator."""

        @qp.marker("after-cancel-inverses")
        @qp.transforms.cancel_inverses
        @qp.marker("after-merge-rotations")
        @qp.transforms.merge_rotations
        @qp.marker("nothing-applied")
        @qp.qnode(qp.device("null.qubit"))
        def c():
            return qp.state()

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

        pipeline = qp.CompilePipeline()
        pipeline += qp.transforms.cancel_inverses
        pipeline.add_marker("after-cancel-inverses")

        assert pipeline.markers == ["after-cancel-inverses"]
        assert pipeline.get_marker_level("after-cancel-inverses") == 1

        @qp.marker("after-pipeline")
        @pipeline
        @qp.marker("before-pipeline")
        @qp.qnode(qp.device("null.qubit"))
        def c():
            return qp.state()

        assert c.compile_pipeline.markers == [
            "before-pipeline",
            "after-cancel-inverses",
            "after-pipeline",
        ]
        assert c.compile_pipeline.get_marker_level("before-pipeline") == 0
        assert c.compile_pipeline.get_marker_level("after-cancel-inverses") == 1
        assert c.compile_pipeline.get_marker_level("after-pipeline") == 1

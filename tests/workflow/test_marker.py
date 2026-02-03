# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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


class TestMarkerQNode:
    """Tests the integration with compile pipeline."""

    def test_uniqueness_checking(self):
        """Test an error is raised if a level is not unique."""

        with pytest.raises(
            ValueError,
            match="Found multiple markers for level something. Markers should be unique.",
        ):

            @qml.marker(level="something")
            @qml.marker(level="something")
            @qml.qnode(qml.device("null.qubit"))
            def c():
                return qml.state()

    def test_protected_levels(self):
        """Test an error is raised for using a protected level."""

        with pytest.raises(ValueError, match="Found marker for protected level gradient"):

            @qml.marker(level="gradient")
            @qml.qnode(qml.device("null.qubit"))
            def c():
                return qml.state()

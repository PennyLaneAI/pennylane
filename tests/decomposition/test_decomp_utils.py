# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Unit tests for utility functions in the ``decomposition`` module.
"""

import pytest

import pennylane as qml


@pytest.mark.unit
def test_toggle_graph_decomposition():
    """Test the toggling of the graph-based decomposition system."""

    assert not qml.decomposition.enabled_graph()

    qml.decomposition.enable_graph()
    assert qml.decomposition.enabled_graph()

    qml.decomposition.disable_graph()
    assert not qml.decomposition.enabled_graph()

    qml.decomposition.enable_graph()
    assert qml.decomposition.enabled_graph()

    qml.decomposition.disable_graph()
    assert not qml.decomposition.enabled_graph()

    qml.decomposition.enable_graph()
    assert qml.decomposition.enabled_graph()

    qml.decomposition.disable_graph()
    assert not qml.decomposition.enabled_graph()

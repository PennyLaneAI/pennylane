# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for deprecations in pennylane/dla/"""
# These tests can be completely removed in v0.41

import pytest

import pennylane as qml


def test_lie_closure_deprecation_warning():
    """Test deprecation warning when calling qml.pauli.lie_closure"""
    with pytest.warns(qml.PennyLaneDeprecationWarning, match="Please call"):
        _ = qml.pauli.lie_closure([qml.X(0)])


def test_structure_constants_deprecation_warning():
    """Test deprecation warning when calling qml.pauli.structure_constants"""
    with pytest.warns(qml.PennyLaneDeprecationWarning, match="Please call"):
        _ = qml.pauli.structure_constants([qml.X(0)])


def test_center_deprecation_warning():
    """Test deprecation warning when calling qml.pauli.center"""
    with pytest.warns(qml.PennyLaneDeprecationWarning, match="Please call"):
        _ = qml.pauli.center([qml.X(0)])

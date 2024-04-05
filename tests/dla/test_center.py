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
"""Tests for pennylane/dla/center.py functionality"""

from copy import copy
import pytest
import numpy as np

import pennylane as qml
from pennylane.dla import center

TRIVIAL_CENTERS = (
    ([qml.I()], [qml.I()]), # just the identity
    ([qml.I(), qml.X(0)], [qml.I(), qml.X(0)]), # identity and some other operator
    ([qml.X(0), qml.X(1)], [qml.X(0), qml.X(1)]), # two non-overlapping wires
    ([qml.X(0), qml.Y(1)], [qml.X(0), qml.Y(1)]), # two non-overlapping wires with different ops
    ([qml.X(0), qml.Y(0), qml.Z(0), qml.I()], [qml.I()]), # non-trivial DLA, but trivial center
)

@pytest.mark.parametrize("ops, true_res", TRIVIAL_CENTERS)
def test_trivial_center(ops, true_res):
    """Test a trivial centers with Identity operators or non-overlapping wires"""
    res = center(ops)
    assert res == true_res

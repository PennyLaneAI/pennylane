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
"""
Tests for the MPSPrep template.
"""

import numpy as np

import pennylane as qml


def test_standard_validity():
    """Check the operation using the assert_valid function."""
    op = qml.MPSPrep(mps=[1.0, 2.0, 3.0], wires=[0, 1, 2])
    qml.ops.functions.assert_valid(op, skip_differentiation=True)


def test_access_to_param():
    """tests that the parameter is accessible."""
    mps = [np.ones([2, 2]), np.ones([3, 3]), np.ones([1])]
    op = qml.MPSPrep(mps=mps, wires=[0, 1, 2])

    for arr1, arr2 in zip(mps, op.mps):
        assert np.allclose(arr1, arr2)

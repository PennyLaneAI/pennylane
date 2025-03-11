# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for qml.grad and qml.jacobian
"""
import pytest

import pennylane as qml


def test_informative_error_on_bad_shape():
    """Test that an informative error is raised if taking the jacobian of a non-array."""

    def f(x):
        return (2 * x,)

    with pytest.raises(ValueError, match="autograd can only differentiate with"):
        qml.jacobian(f)(qml.numpy.array(2.0))

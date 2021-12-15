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
"""Unit tests for the identity operator."""
import pennylane as qml
import numpy as np


def test_identity_eigvals(tol):
    """Test identity eigenvalues are correct"""
    res = qml.Identity._eigvals()
    expected = np.array([1, 1])
    assert np.allclose(res, expected, atol=tol, rtol=0)


def test_matrix_representation(tol):
    """Test the matrix representation"""
    res_static = qml.Identity.compute_matrix()
    res_dynamic = qml.Identity(wires=0).matrix()
    expected = np.array([[1., 0.], [0., 1.]])
    assert np.allclose(res_static, expected, atol=tol)
    assert np.allclose(res_dynamic, expected, atol=tol)

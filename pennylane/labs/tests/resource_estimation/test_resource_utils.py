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
Test the utility functions for resource estimation.
"""

import numpy as np
import pytest

from pennylane.labs.resource_estimation.resource_utils import approx_poly_degree


def test_approx_poly_degree():
    """Test the approx_poly_degree function"""
    x_vec = np.array([1, 2, 3, 4, 5])
    y_vec = np.array([1, 4, 9, 16, 25])
    coeffs, loss = approx_poly_degree(x_vec, y_vec, max_degree=3)

    assert np.allclose(coeffs, np.array([0, 0, 1]))
    assert np.allclose(loss, 0)


@pytest.mark.parametrize("basis", ["chebyshev", "legendre", "hermite"])
def test_approx_poly_degree_basis(basis):
    """Test the approx_poly_degree function with different bases"""
    x_vec = np.random.RandomState(123).rand(10)
    y_vec = np.random.RandomState(863).rand(10)
    e_tol = 1e-2
    coeffs, loss = approx_poly_degree(x_vec, y_vec, basis=basis, error_tol=e_tol)

    assert isinstance(coeffs, np.ndarray)
    assert isinstance(loss, float)
    assert loss <= e_tol

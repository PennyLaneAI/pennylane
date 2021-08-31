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
Unit tests for functions needed to computing integrals over basis functions.
"""
import numpy as np
from pennylane.hf.integrals import primitive_norm, contracted_norm


def test_gaussian_norm():
    r"""Test that the normalization constant of a Gaussian function representing s orbitals
    is :math:`(\frac {2 \alpha}{\pi})^{{3/4}}`.
    """
    l = (0, 0, 0)
    alpha = np.array([3.425250914])
    n = (2 * alpha / np.pi) ** (3 / 4)

    assert np.allclose(primitive_norm(l, alpha), n)


def test_contraction_norm():
    r"""Tests that the normalization constant of a linear combination of three  normalized
    Gaussian functions is the trivial value of :math:`1/3`.
    """
    l = (0, 0, 0)
    alpha = np.array([3.425250914, 3.425250914, 3.425250914])
    c = np.array([1.79444183, 1.79444183, 1.79444183])

    assert np.allclose(contracted_norm(l, alpha, c), 0.33333333)

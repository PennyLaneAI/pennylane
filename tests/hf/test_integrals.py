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
import pytest
from pennylane.hf.integrals import contracted_norm, primitive_norm


@pytest.mark.parametrize(
    ("l", "alpha", "n"),
    [
        # normalization constant for an s orbital is :math:`(\frac {2 \alpha}{\pi})^{{3/4}}`.
        ((0, 0, 0), np.array([3.425250914]), np.array([1.79444183])),
    ],
)
def test_gaussian_norm(l, alpha, n):
    r"""Test that the computed normalization constant of a Gaussian function is correct."""
    assert np.allclose(primitive_norm(l, alpha), n)


@pytest.mark.parametrize(
    ("l", "alpha", "a", "n"),
    [
        # normalization constant for a contracted Gaussian function composed of three normalized
        # s orbital is :math:`1/3`.
        (
            (0, 0, 0),
            np.array([3.425250914, 3.425250914, 3.425250914]),
            np.array([1.79444183, 1.79444183, 1.79444183]),
            np.array([0.33333333]),
        )
    ],
)
def test_contraction_norm(l, alpha, a, n):
    r"""Test that the computed normalization constant of a contracted Gaussian function is correct."""
    assert np.allclose(contracted_norm(l, alpha, a), n)

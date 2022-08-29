# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for functions needed for estimating complexity of measuring expectation values.
"""
import pytest
import pennylane as qml
from pennylane import numpy as np


coeffs = [np.array([-0.32707061, 0.7896887]), np.array([0.18121046])]
error = 0.0016  # chemical accuracy
shots = 419217  # computed manually


@pytest.mark.parametrize(
    ("coefficients", "error", "shots"),
    [
        (coeffs, error, shots),
    ],
)
def test_estimate_samples(coefficients, error, shots):
    r"""Test that the estimate_samples function returns the correct number of measurements."""
    m = qml.resource.estimate_samples(coefficients, error=error)

    assert m == shots


@pytest.mark.parametrize(
    ("coefficients", "error", "shots"),
    [
        (coeffs, error, shots),
    ],
)
def test_estimate_error(coefficients, error, shots):
    r"""Test that the estimate_error function returns the correct error."""
    e = qml.resource.estimate_error(coefficients, shots=shots)

    assert np.allclose(e, error)

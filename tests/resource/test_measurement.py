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
Unit tests for functions needed for estimating the complexity of measuring expectation values.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np

coeffs = [np.array([-0.32707061, 0.7896887]), np.array([0.18121046])]
error = 0.0016  # chemical accuracy
shots = 419218  # computed manually
variances = [0.73058343, 0.03283723]  # obtained with the upper bound var(pauli_word) = 1


@pytest.mark.parametrize(
    ("coefficients", "err", "shots_", "var"),
    [
        (coeffs, error, shots, variances),
    ],
)
def test_estimate_shots(coefficients, err, shots_, var):
    r"""Test that the estimate_shots function returns the correct number of measurements."""
    m_novar = qml.resource.estimate_shots(coefficients, error=err)
    m_var = qml.resource.estimate_shots(coefficients, variances=var, error=err)

    assert m_novar == shots_
    assert m_var == shots_


@pytest.mark.parametrize(
    ("coefficients", "err", "shots_", "var"),
    [
        (coeffs, error, shots, variances),
    ],
)
def test_estimate_error(coefficients, err, shots_, var):
    r"""Test that the estimate_error function returns the correct error."""
    e_novar = qml.resource.estimate_error(coefficients, shots=shots_)
    e_var = qml.resource.estimate_error(coefficients, variances=var, shots=shots_)

    assert np.allclose(e_novar, err)
    assert np.allclose(e_var, err)
